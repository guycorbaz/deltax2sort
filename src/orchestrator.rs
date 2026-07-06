use crate::app_config::AppConfig;
use crate::hardware::{Position, RobotController, WorkspaceLimits};
use crate::vision::DetectedObject;
use anyhow::Result;
use log::{debug, error, info, warn};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, mpsc, watch};

/// Maximum age of a detection before its pick is discarded: after this long
/// the belt has carried the object away from the detected position, so
/// executing the pick would grab at empty belt.
const PICK_TTL: Duration = Duration::from_secs(3);

#[derive(Debug, Clone)]
pub enum RobotCommand {
    MoveTo(Position),
    Gripper(bool),
    Home,
    Wait(Duration),
}

/// Everything the rest of the system (UI, vision loop) may ask of the
/// orchestrator. Sent over an unbounded channel so safety messages
/// (`EStop`) can never be dropped because a queue is full.
#[derive(Debug)]
pub enum OrchestratorMsg {
    /// Schedule a full pick-and-place sequence for a detected object.
    Pick(DetectedObject),
    /// Recovery/operator homing: runs even while paused (it is the required
    /// step to leave the E-stopped state) and takes priority over the queue.
    Home,
    /// Operator gripper toggle. Runs even while paused so a part held after a
    /// failed pick or an E-stop can be released by hand. Best-effort, not part
    /// of a pick sequence.
    SetGripper(bool),
    Pause,
    Resume,
    /// Clear the queue and pause. The hardware halt itself is done by the
    /// caller through the `EmergencyStop` handles — this message only makes
    /// sure no further commands are fed to the robot.
    EStop,
    /// Graceful shutdown: finish the command in flight, park the robot
    /// (gripper off, raise to z_travel), then stop the loop so the task can be
    /// awaited instead of aborted when the process exits.
    Shutdown,
}

/// Confirmed orchestrator state, published over a `watch` channel so the UI
/// shows what the system IS doing, never what a button hoped it would do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrchestratorState {
    /// Not executing picks; Resume (UI Start) will begin sorting.
    Paused,
    /// Executing queued commands and accepting picks.
    Running,
    /// E-stop happened: paused AND Resume is refused until a Home succeeds.
    EStopped,
}

/// One pick-and-place as an atomic group: its commands plus the expiry of the
/// detection that produced it. Expiry is re-checked before the group STARTS, so
/// a sequence that waited behind others past its TTL is dropped whole rather
/// than descending onto belt the object has left. Once started it always runs
/// to completion — never abort mid-sequence (the gripper may hold a part).
struct PickSequence {
    object_id: u64,
    expiry: Instant,
    started: bool,
    steps: VecDeque<RobotCommand>,
}

pub struct InstructionQueue {
    groups: VecDeque<PickSequence>,
}

impl InstructionQueue {
    pub fn new() -> Self {
        Self {
            groups: VecDeque::new(),
        }
    }

    fn push_sequence(&mut self, seq: PickSequence) {
        self.groups.push_back(seq);
    }

    pub fn clear(&mut self) {
        self.groups.clear();
    }

    /// Total pending commands across all groups.
    pub fn len(&self) -> usize {
        self.groups.iter().map(|g| g.steps.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// The next command to execute at `now`, dropping any not-yet-started group
    /// whose expiry has passed. `None` when nothing remains. A group already in
    /// flight (`started`) is never dropped, even if it expires mid-sequence.
    pub fn next_command(&mut self, now: Instant) -> Option<RobotCommand> {
        while let Some(front) = self.groups.front_mut() {
            if !front.started && now >= front.expiry {
                let dropped = self.groups.pop_front().expect("front exists");
                warn!(
                    "Orchestrator: dropping expired pick sequence for object {} before it started \
                     (belt has carried it away)",
                    dropped.object_id
                );
                continue;
            }
            front.started = true;
            match front.steps.pop_front() {
                Some(cmd) => return Some(cmd),
                None => {
                    self.groups.pop_front(); // exhausted → advance to next group
                }
            }
        }
        None
    }
}

// Not on the active path yet: interception needs robot position feedback
// (docs/TODO.md); schedule_pick targets last-seen positions until then.
#[allow(dead_code)]
pub struct TrajectoryPlanner {
    robot_speed_mm_s: f32,
    /// Signed belt speed in robot coordinates: positive means objects
    /// travel toward +Y.
    conveyor_speed_mm_s: f32,
    /// Y coordinate of the line where picks happen.
    pick_line_y: f32,
}

impl TrajectoryPlanner {
    pub fn new(conveyor_speed_mm_s: f32, robot_speed_mm_s: f32) -> Self {
        Self {
            robot_speed_mm_s,
            conveyor_speed_mm_s,
            pick_line_y: 0.0,
        }
    }

    /// Predict where the object crosses the pick line and whether the robot can
    /// be there in time. Returns the intercept as `(x, y)` in robot mm — XY
    /// only: the pick height (z) always comes from configuration, never from
    /// vision (which reports the belt plane as z = 0). Returns None when the
    /// object is moving away from (or already past) the pick line, or is
    /// unreachable in time.
    // Unit-tested but not called on the active path yet (see struct note).
    #[allow(dead_code)]
    pub fn calculate_intercept(
        &self,
        robot_pos: Position,
        object: &DetectedObject,
    ) -> Option<(f32, f32)> {
        let pos = object.world_pos?;
        if self.conveyor_speed_mm_s == 0.0 {
            return None;
        }
        let time_to_line = (self.pick_line_y - pos.y) / self.conveyor_speed_mm_s;
        if time_to_line <= 0.0 {
            // Already past the pick line (or moving the wrong way).
            return None;
        }
        // The belt only moves along Y, so X is unchanged at intercept.
        let (target_x, target_y) = (pos.x, self.pick_line_y);
        let dx = target_x - robot_pos.x;
        let dy = target_y - robot_pos.y;
        let robot_time = (dx * dx + dy * dy).sqrt() / self.robot_speed_mm_s;
        (robot_time < time_to_line).then_some((target_x, target_y))
    }
}

pub struct Orchestrator {
    rx: mpsc::UnboundedReceiver<OrchestratorMsg>,
    queue: InstructionQueue,
    paused: bool,
    /// Set by EStop; cleared only by a successful Home. While set, Resume is
    /// refused (the operator must re-home after an emergency stop).
    needs_home: bool,
    /// A Home was requested (UI); executed with priority, even while paused.
    pending_home: bool,
    /// A manual gripper toggle was requested (UI); executed best-effort, even
    /// while paused. `None` = nothing pending.
    pending_gripper: Option<bool>,
    /// Set by `Shutdown`: the loop parks the robot and exits at the next turn.
    shutting_down: bool,
    /// Last position successfully commanded, so shutdown can raise straight up
    /// to z_travel without guessing X/Y. `None` until the first move.
    last_pos: Option<Position>,
    state_tx: watch::Sender<OrchestratorState>,
    /// Publishes a concise, operator-facing message when a hardware command
    /// fails. On a kiosk Pi with no terminal the log is invisible, so every
    /// failure that pauses sorting must also reach the UI banner.
    error_tx: watch::Sender<Option<String>>,
    // Not yet used by schedule_pick: belt-motion compensation needs robot
    // position feedback (docs/TODO.md).
    #[allow(dead_code)]
    planner: TrajectoryPlanner,
    limits: WorkspaceLimits,
    z_travel: f32,
    z_pick: f32,
    /// Drop position per recognised class. A pick is scheduled only for a
    /// class present here; anything else (unrecognised, or recognised but
    /// unassigned) is left on the belt to reach the end catch bin.
    class_drops: std::collections::HashMap<String, Position>,
    /// Object ids whose pick was declined for a RETRYABLE reason (it went stale
    /// in a busy queue while the object is still on the belt). The vision loop
    /// re-arms those tracks so they are emitted again. `None` = no re-arm sink.
    declined_tx: Option<mpsc::UnboundedSender<u64>>,
    robot: Arc<Mutex<Box<dyn RobotController>>>,
}

impl Orchestrator {
    pub fn new(
        config: &AppConfig,
        robot: Arc<Mutex<Box<dyn RobotController>>>,
    ) -> (
        mpsc::UnboundedSender<OrchestratorMsg>,
        watch::Receiver<OrchestratorState>,
        watch::Receiver<Option<String>>,
        Self,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        // SAFETY-relevant default: the orchestrator starts PAUSED, so no
        // pick can move the robot before the operator presses Start.
        let (state_tx, state_rx) = watch::channel(OrchestratorState::Paused);
        let (error_tx, error_rx) = watch::channel(None);
        let orchestrator = Self {
            rx,
            queue: InstructionQueue::new(),
            paused: true,
            needs_home: false,
            pending_home: false,
            pending_gripper: None,
            shutting_down: false,
            last_pos: None,
            state_tx,
            error_tx,
            planner: TrajectoryPlanner::new(
                config.conveyor.speed_mm_s,
                // F parameter is mm/min.
                config.robot.feed_rate as f32 / 60.0,
            ),
            limits: config.robot.workspace_limits(),
            z_travel: config.robot.z_travel,
            z_pick: config.robot.z_pick,
            class_drops: config
                .sorting
                .class_drop_positions()
                .into_iter()
                .map(|(class, (x, y, z))| (class, Position { x, y, z }))
                .collect(),
            declined_tx: None,
            robot,
        };
        (tx, state_rx, error_rx, orchestrator)
    }

    /// Push a concise failure message to the operator banner. The detailed
    /// chain still goes to the log; this is the one line the operator sees.
    fn report_error(&self, msg: String) {
        let _ = self.error_tx.send(Some(msg));
    }

    /// Where to report retryable pick declines so the vision loop can re-arm
    /// the track. Set once at startup, before the loop is spawned.
    pub fn set_declined_sink(&mut self, tx: mpsc::UnboundedSender<u64>) {
        self.declined_tx = Some(tx);
    }

    /// Actuate the gripper without letting a failure abort the caller. Used
    /// for recovery (release a held part) and for the manual toggle: opening
    /// the gripper is not motion, so it stays within the no-auto-retry rule.
    async fn set_gripper_best_effort(&self, on: bool) {
        let verb = if on { "engage" } else { "release" };
        if let Err(e) = self.robot.lock().await.set_gripper(on).await {
            error!("Orchestrator: gripper {verb} failed: {e:#}");
            self.report_error(format!("Gripper {verb} failed: {e:#}"));
        }
    }

    /// Leave the robot in a safe state before the process exits: drop any held
    /// part and lift off the belt. Best-effort — a failure must not block the
    /// exit. Skipped when E-stopped: after M112 the robot ignores commands
    /// until re-homed, so the blocking waits would only delay shutdown.
    async fn park_for_shutdown(&mut self) {
        if self.needs_home {
            warn!("Orchestrator: shutdown while E-stopped — robot halted, skipping park");
            return;
        }
        info!("Orchestrator: parking for shutdown (gripper off, raise to z_travel)");
        self.set_gripper_best_effort(false).await;
        if let Some(p) = self.last_pos {
            let park = Position {
                x: p.x,
                y: p.y,
                z: self.z_travel,
            };
            if let Err(e) = self.robot.lock().await.move_to(park).await {
                warn!("Orchestrator: shutdown raise to z_travel failed: {e:#}");
            }
        }
    }

    fn state(&self) -> OrchestratorState {
        if self.needs_home {
            OrchestratorState::EStopped
        } else if self.paused {
            OrchestratorState::Paused
        } else {
            OrchestratorState::Running
        }
    }

    /// Publish the current confirmed state to the UI watch channel.
    fn publish_state(&self) {
        let _ = self.state_tx.send(self.state());
    }

    /// Main loop. Consumes messages and executes queued commands one at a
    /// time. The robot mutex is held only for the duration of a single
    /// hardware call — never across a Wait — so the UI (Home, E-Stop paths)
    /// is never starved.
    pub async fn run(mut self) {
        info!("Orchestrator loop started");
        loop {
            // Drain everything that has arrived so control messages
            // (Pause/EStop) take effect before the next command runs.
            while let Ok(msg) = self.rx.try_recv() {
                self.handle_message(msg);
            }

            // Graceful shutdown: the command in flight has already returned
            // (execution is sequential), so park now and leave the loop.
            if self.shutting_down {
                self.park_for_shutdown().await;
                break;
            }

            // Operator homing runs first and even while paused: it is the
            // recovery action that clears the E-stopped state.
            if self.pending_home {
                self.pending_home = false;
                match self.execute(RobotCommand::Home).await {
                    Ok(()) => {
                        if self.needs_home {
                            info!("Orchestrator: home complete — E-stop state cleared");
                            self.needs_home = false;
                        }
                    }
                    Err(e) => {
                        error!("Orchestrator: home failed: {:#}", e);
                        self.report_error(format!("Home failed: {e:#}"));
                    }
                }
                self.publish_state();
                continue;
            }

            // Manual gripper toggle: runs even while paused so a held part can
            // be released by hand (e.g. after re-homing out of an E-stop).
            if let Some(on) = self.pending_gripper.take() {
                info!("Orchestrator: manual gripper {}", if on { "ON" } else { "OFF" });
                self.set_gripper_best_effort(on).await;
                continue;
            }

            if self.paused || self.queue.is_empty() {
                // Nothing to execute: block until the next message.
                match self.rx.recv().await {
                    Some(msg) => {
                        self.handle_message(msg);
                        continue;
                    }
                    None => break, // all senders dropped: shutdown
                }
            }

            // next_command() drops any expired-before-start sequence (dropping
            // whole groups, never mid-sequence) and returns the next command;
            // the body runs only on a command failure.
            if let Some(cmd) = self.queue.next_command(Instant::now())
                && let Err(e) = self.execute(cmd).await
            {
                error!(
                    "Orchestrator: command failed: {:#}. Clearing queue and pausing; \
                     send Resume to continue.",
                    e
                );
                self.report_error(format!("Robot command failed: {e:#}. Sorting paused."));
                self.queue.clear();
                self.paused = true;
                // The pending Gripper(false) was just discarded with the
                // queue: release best-effort so suction can't hold a part
                // indefinitely. The robot answered until now, so this is
                // not the post-M112 case that could stall.
                self.set_gripper_best_effort(false).await;
                self.publish_state();
            }
        }
        info!("Orchestrator loop stopped (all senders dropped)");
    }

    fn handle_message(&mut self, msg: OrchestratorMsg) {
        match msg {
            OrchestratorMsg::Pick(object) => {
                if self.paused {
                    // Accepting picks while paused would build a backlog of
                    // long-gone objects that gets executed on Resume.
                    warn!(
                        "Orchestrator: paused — dropping pick for object {}",
                        object.id
                    );
                    return;
                }
                // Clock read happens only here at the message boundary;
                // schedule_pick stays pure and testable.
                self.schedule_pick(object, Instant::now());
            }
            OrchestratorMsg::Home => {
                info!("Orchestrator: home requested");
                self.pending_home = true;
            }
            OrchestratorMsg::SetGripper(on) => {
                info!("Orchestrator: manual gripper {} requested", if on { "ON" } else { "OFF" });
                self.pending_gripper = Some(on);
            }
            OrchestratorMsg::Pause => {
                info!(
                    "Orchestrator: paused ({} command(s) queued)",
                    self.queue.len()
                );
                self.paused = true;
                self.publish_state();
            }
            OrchestratorMsg::Resume => {
                if self.needs_home {
                    // Safety interlock: after an E-stop the operator must
                    // re-home before sorting can restart.
                    warn!("Orchestrator: Resume refused — home required after E-stop");
                } else {
                    info!("Orchestrator: resumed");
                    self.paused = false;
                }
                self.publish_state();
            }
            OrchestratorMsg::Shutdown => {
                info!("Orchestrator: graceful shutdown requested");
                self.shutting_down = true;
            }
            OrchestratorMsg::EStop => {
                warn!(
                    "Orchestrator: E-STOP — dropping {} queued command(s) and pausing",
                    self.queue.len()
                );
                // No blocking gripper release here: M112 has already gone to
                // the robot preemptively, so a blocking M05 would likely wait
                // out the full 30 s feedback deadline and stall Home recovery.
                // If a cell wants the gripper to open on E-stop it does so at
                // the hardware halt (release_gripper_on_estop prepends M05 to
                // the M112 write); otherwise the operator uses the manual
                // gripper toggle after re-homing.
                self.queue.clear();
                self.paused = true;
                self.needs_home = true;
                self.publish_state();
            }
        }
    }

    async fn execute(&mut self, cmd: RobotCommand) -> Result<()> {
        match cmd {
            // Sleep WITHOUT holding the robot lock.
            RobotCommand::Wait(duration) => {
                tokio::time::sleep(duration).await;
                Ok(())
            }
            RobotCommand::MoveTo(pos) => {
                self.robot.lock().await.move_to(pos).await?;
                // Remember where we are so shutdown can raise straight up.
                self.last_pos = Some(pos);
                Ok(())
            }
            RobotCommand::Gripper(on) => self.robot.lock().await.set_gripper(on).await,
            RobotCommand::Home => self.robot.lock().await.home().await,
        }
    }

    /// Queue the pick-and-place sequence for `object`. `now` is passed in
    /// (rather than read internally) so staleness is testable with explicit
    /// instants.
    fn schedule_pick(&mut self, object: DetectedObject, now: Instant) {
        let age = now.saturating_duration_since(object.seen_at);
        if age > PICK_TTL {
            warn!(
                "Orchestrator: dropping stale pick for object {} (seen {:.1} s ago, TTL {} s)",
                object.id,
                age.as_secs_f32(),
                PICK_TTL.as_secs()
            );
            // Retryable: the object is likely still on the belt (the pick only
            // went stale because the queue was busy). Ask the vision loop to
            // re-arm the track so it is emitted again with a fresh detection.
            if let Some(tx) = &self.declined_tx {
                let _ = tx.send(object.id);
            }
            return;
        }
        let Some(pos) = object.world_pos else {
            warn!(
                "Orchestrator: object {} has no world position — skipping",
                object.id
            );
            return;
        };
        // Route by recognised class: only objects whose class is assigned to a
        // bin are picked. An unrecognised object (class None) or a recognised
        // one with no bin assignment is deliberately left on the belt to reach
        // the end catch bin — "do nothing" is the safe default.
        let drop_pos = match &object.class {
            Some(class) => match self.class_drops.get(class) {
                Some(&drop) => drop,
                None => {
                    debug!(
                        "Orchestrator: object {} class '{}' has no bin assignment — leaving it on the belt",
                        object.id, class
                    );
                    return;
                }
            },
            None => {
                debug!(
                    "Orchestrator: object {} unrecognised — leaving it on the belt",
                    object.id
                );
                return;
            }
        };
        // Z comes from configuration, not from vision (which reports the
        // belt plane as z = 0).
        let pick = Position {
            x: pos.x,
            y: pos.y,
            z: self.z_pick,
        };
        let travel = Position {
            z: self.z_travel,
            ..pick
        };
        if !self.limits.contains(pick) || !self.limits.contains(travel) {
            warn!(
                "Orchestrator: pick target {:?} outside workspace — skipping object {}",
                pick, object.id
            );
            return;
        }
        info!(
            "Orchestrator: scheduling pick for object {} at ({:.1}, {:.1})",
            object.id, pick.x, pick.y
        );
        let steps = VecDeque::from(vec![
            RobotCommand::MoveTo(travel),                    // approach from above
            RobotCommand::MoveTo(pick),                      // descend
            RobotCommand::Gripper(true),                     // grab
            RobotCommand::Wait(Duration::from_millis(150)),  // let suction settle
            RobotCommand::MoveTo(travel),                    // lift
            RobotCommand::MoveTo(drop_pos),                  // to the assigned bin
            RobotCommand::Gripper(false),                    // release
        ]);
        self.queue.push_sequence(PickSequence {
            object_id: object.id,
            // Re-checked at execution time so a sequence stuck behind others is
            // dropped rather than run onto belt the object has left.
            expiry: object.seen_at + PICK_TTL,
            started: false,
            steps,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_config::{Assignment, BinConfig};
    use opencv::core::Rect;
    use std::time::SystemTime;

    const ORIGIN: Position = Position {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    /// The class every test object carries, assigned to a bin at [`DROP`].
    const PART: &str = "part";

    /// Config with a single bin "bin-a" at (150, 0, -100) and the `PART` class
    /// assigned to it — so `object_at` objects are picked and dropped there,
    /// matching the pre-routing tests' expectations.
    fn test_config() -> AppConfig {
        let mut cfg = AppConfig::default();
        cfg.sorting.bins = vec![BinConfig {
            id: "bin-a".into(),
            x: 150.0,
            y: 0.0,
            z: -100.0,
        }];
        cfg.sorting.assignments = vec![Assignment {
            class: PART.into(),
            bin: "bin-a".into(),
        }];
        cfg
    }

    fn object_at(x: f32, y: f32) -> DetectedObject {
        object_of_class(x, y, Some(PART.to_string()))
    }

    fn object_of_class(x: f32, y: f32, class: Option<String>) -> DetectedObject {
        DetectedObject {
            id: 1,
            rect: Rect::new(0, 0, 10, 10),
            world_pos: Some(Position { x, y, z: 0.0 }),
            class,
            confidence: 0.0,
            timestamp: SystemTime::now(),
            seen_at: Instant::now(),
        }
    }

    fn test_orchestrator() -> Orchestrator {
        let robot: Arc<Mutex<Box<dyn RobotController>>> =
            Arc::new(Mutex::new(Box::new(crate::hardware::MockRobot::new())));
        let (_tx, _state_rx, _error_rx, orch) = Orchestrator::new(&test_config(), robot);
        orch
    }

    /// Orchestrator plus a handle to the mock robot's command log, so tests
    /// can assert exactly what the robot was told (order included).
    fn test_orchestrator_with_log() -> (
        Orchestrator,
        std::sync::Arc<std::sync::Mutex<Vec<crate::hardware::MockRobotCommand>>>,
    ) {
        let mock = crate::hardware::MockRobot::new();
        let log = mock.command_log();
        let robot: Arc<Mutex<Box<dyn RobotController>>> = Arc::new(Mutex::new(Box::new(mock)));
        let (_tx, _state_rx, _error_rx, orch) = Orchestrator::new(&test_config(), robot);
        (orch, log)
    }

    #[tokio::test]
    async fn pick_sequence_commands_the_robot_in_order() {
        use crate::hardware::MockRobotCommand::*;
        let (mut orch, log) = test_orchestrator_with_log();
        orch.handle_message(OrchestratorMsg::Resume);
        let obj = object_at(50.0, 0.0);
        let now = obj.seen_at + Duration::from_secs(1);
        orch.schedule_pick(obj, now);
        // Drain the queue through execute (the run loop, minus timing/locks).
        while let Some(cmd) = orch.queue.next_command(Instant::now()) {
            orch.execute(cmd).await.unwrap();
        }
        let travel = Position {
            x: 50.0,
            y: 0.0,
            z: orch.z_travel,
        };
        let pick = Position {
            x: 50.0,
            y: 0.0,
            z: orch.z_pick,
        };
        assert_eq!(
            *log.lock().unwrap(),
            vec![
                MoveTo(travel),      // approach above
                MoveTo(pick),        // descend
                Gripper(true),       // grab (Wait is not a robot command)
                MoveTo(travel),      // lift
                MoveTo(DROP),        // to the assigned bin (bin-a)
                Gripper(false), // release over the drop
            ],
        );
    }

    #[test]
    fn intercept_upstream_object_is_reachable() {
        // Belt moves toward +Y at 100 mm/s; object 100 mm upstream of the
        // pick line arrives in 1 s; robot needs 50/250 = 0.2 s.
        let planner = TrajectoryPlanner::new(100.0, 250.0);
        let (x, y) = planner
            .calculate_intercept(ORIGIN, &object_at(50.0, -100.0))
            .expect("object should be interceptable");
        // XY only — no z is returned (the pick height comes from config, not
        // from vision's z = 0).
        assert_eq!(x, 50.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn intercept_object_past_pick_line_is_none() {
        let planner = TrajectoryPlanner::new(100.0, 250.0);
        // Object already downstream of the pick line: previously abs()
        // produced a bogus positive time here.
        assert!(
            planner
                .calculate_intercept(ORIGIN, &object_at(50.0, 100.0))
                .is_none()
        );
    }

    #[test]
    fn intercept_with_stopped_belt_is_none() {
        let planner = TrajectoryPlanner::new(0.0, 250.0);
        assert!(
            planner
                .calculate_intercept(ORIGIN, &object_at(50.0, -100.0))
                .is_none()
        );
    }

    #[test]
    fn intercept_unreachable_in_time_is_none() {
        // Object arrives at the line in 0.1 s but the robot is 320 mm away.
        let planner = TrajectoryPlanner::new(100.0, 250.0);
        let far_robot = Position {
            x: -160.0,
            y: 0.0,
            z: 0.0,
        };
        assert!(
            planner
                .calculate_intercept(far_robot, &object_at(160.0, -10.0))
                .is_none()
        );
    }

    #[test]
    fn intercept_object_without_world_pos_is_none() {
        let planner = TrajectoryPlanner::new(100.0, 250.0);
        let mut obj = object_at(0.0, -100.0);
        obj.world_pos = None;
        assert!(planner.calculate_intercept(ORIGIN, &obj).is_none());
    }

    #[test]
    fn fresh_pick_is_scheduled() {
        let mut orch = test_orchestrator();
        let object = object_at(50.0, 0.0);
        // 1 s old: within PICK_TTL (explicit instants, no clock reads).
        let now = object.seen_at + Duration::from_secs(1);
        orch.schedule_pick(object, now);
        assert_eq!(orch.queue.len(), 7, "full pick-and-place sequence queued");
    }

    #[test]
    fn stale_pick_is_dropped() {
        let mut orch = test_orchestrator();
        let object = object_at(50.0, 0.0);
        // 4 s old: past the 3 s PICK_TTL — the belt has moved on.
        let now = object.seen_at + Duration::from_secs(4);
        orch.schedule_pick(object, now);
        assert!(orch.queue.is_empty(), "stale pick must not be queued");
    }

    #[test]
    fn a_stale_pick_signals_a_decline_and_a_fresh_one_does_not() {
        let mut orch = test_orchestrator();
        let (tx, mut rx) = mpsc::unbounded_channel();
        orch.set_declined_sink(tx);

        // Fresh pick: scheduled, no decline signalled.
        let fresh = object_at(50.0, 0.0);
        let seen = fresh.seen_at;
        orch.schedule_pick(fresh, seen + Duration::from_secs(1));
        assert!(rx.try_recv().is_err(), "a fresh pick must not signal a decline");
        assert_eq!(orch.queue.len(), 7);

        // Stale pick (object id 1): dropped AND signalled so the track re-arms.
        let stale = object_at(50.0, 0.0);
        let stale_seen = stale.seen_at;
        orch.schedule_pick(stale, stale_seen + Duration::from_secs(4));
        assert_eq!(rx.try_recv().ok(), Some(1), "a stale pick signals a decline");
    }

    #[test]
    fn unassigned_or_unrecognised_pick_does_not_signal_a_decline() {
        let mut orch = test_orchestrator();
        let (tx, mut rx) = mpsc::unbounded_channel();
        orch.set_declined_sink(tx);
        // Unrecognised object: dropped (catch bin) but NOT retryable.
        let obj = object_of_class(50.0, 0.0, None);
        let seen = obj.seen_at;
        orch.schedule_pick(obj, seen + Duration::from_secs(1));
        assert!(
            rx.try_recv().is_err(),
            "an unsortable object must not be re-armed"
        );
    }

    #[test]
    fn unrecognised_object_is_not_scheduled() {
        // class None → not sorted, rides the belt to the catch bin.
        let mut orch = test_orchestrator();
        let object = object_of_class(50.0, 0.0, None);
        let now = object.seen_at + Duration::from_secs(1);
        orch.schedule_pick(object, now);
        assert!(orch.queue.is_empty(), "unrecognised object must not be picked");
    }

    #[test]
    fn recognised_but_unassigned_class_is_not_scheduled() {
        // Known class, but no bin assigned to it → also passes to the catch bin.
        let mut orch = test_orchestrator();
        let object = object_of_class(50.0, 0.0, Some("no-bin-for-this".into()));
        let now = object.seen_at + Duration::from_secs(1);
        orch.schedule_pick(object, now);
        assert!(
            orch.queue.is_empty(),
            "a class with no bin assignment must not be picked"
        );
    }

    #[test]
    fn objects_route_to_their_assigned_bin() {
        use crate::app_config::{Assignment, BinConfig};
        // Two classes → two different bins.
        let mut cfg = AppConfig::default();
        cfg.sorting.bins = vec![
            BinConfig { id: "left".into(), x: -120.0, y: 0.0, z: -100.0 },
            BinConfig { id: "right".into(), x: 120.0, y: 0.0, z: -100.0 },
        ];
        cfg.sorting.assignments = vec![
            Assignment { class: "red".into(), bin: "left".into() },
            Assignment { class: "blue".into(), bin: "right".into() },
        ];
        let robot: Arc<Mutex<Box<dyn RobotController>>> =
            Arc::new(Mutex::new(Box::new(crate::hardware::MockRobot::new())));
        let (_tx, _s, _e, mut orch) = Orchestrator::new(&cfg, robot);

        let red = object_of_class(20.0, 0.0, Some("red".into()));
        let now = red.seen_at + Duration::from_secs(1);
        orch.schedule_pick(red, now);
        // The 6th command (index 5) is the drop move.
        let red_drop = drop_move(&mut orch);
        assert_eq!(red_drop, Position { x: -120.0, y: 0.0, z: -100.0 });

        let blue = object_of_class(20.0, 0.0, Some("blue".into()));
        orch.schedule_pick(blue, now);
        let blue_drop = drop_move(&mut orch);
        assert_eq!(blue_drop, Position { x: 120.0, y: 0.0, z: -100.0 });
    }

    /// Drain a scheduled pick sequence and return its drop-move target (the
    /// second-to-last command; the last is the gripper release).
    fn drop_move(orch: &mut Orchestrator) -> Position {
        let mut moves = Vec::new();
        while let Some(cmd) = orch.queue.next_command(Instant::now()) {
            if let RobotCommand::MoveTo(p) = cmd {
                moves.push(p);
            }
        }
        // moves: travel, pick, travel(lift), drop → the last MoveTo is the drop.
        *moves.last().expect("a pick sequence has move commands")
    }

    #[test]
    fn pick_while_paused_is_dropped() {
        let mut orch = test_orchestrator();
        orch.handle_message(OrchestratorMsg::Pause);
        orch.handle_message(OrchestratorMsg::Pick(object_at(50.0, 0.0)));
        assert!(orch.queue.is_empty(), "paused pick must not be queued");
        // A fresh pick after Resume goes through.
        orch.handle_message(OrchestratorMsg::Resume);
        orch.handle_message(OrchestratorMsg::Pick(object_at(50.0, 0.0)));
        assert_eq!(orch.queue.len(), 7);
    }

    #[test]
    fn orchestrator_starts_paused_so_boot_picks_cannot_move_the_robot() {
        let mut orch = test_orchestrator();
        assert_eq!(orch.state(), OrchestratorState::Paused);
        // A pick arriving before the operator pressed Start is dropped.
        orch.handle_message(OrchestratorMsg::Pick(object_at(50.0, 0.0)));
        assert!(orch.queue.is_empty(), "no motion before operator Start");
        // Operator Start (Resume) enables picking.
        orch.handle_message(OrchestratorMsg::Resume);
        assert_eq!(orch.state(), OrchestratorState::Running);
    }

    #[test]
    fn estop_refuses_resume_until_home_succeeds() {
        let mut orch = test_orchestrator();
        orch.handle_message(OrchestratorMsg::Resume);
        orch.handle_message(OrchestratorMsg::EStop);
        assert_eq!(orch.state(), OrchestratorState::EStopped);
        // Resume is refused while re-home is pending.
        orch.handle_message(OrchestratorMsg::Resume);
        assert_eq!(orch.state(), OrchestratorState::EStopped);
        assert!(orch.paused, "must stay paused after refused Resume");
        // Home request is registered; simulate the run loop completing it.
        orch.handle_message(OrchestratorMsg::Home);
        assert!(orch.pending_home);
        orch.pending_home = false;
        orch.needs_home = false; // what the run loop does on home success
        orch.handle_message(OrchestratorMsg::Resume);
        assert_eq!(orch.state(), OrchestratorState::Running);
    }

    #[test]
    fn state_watch_publishes_confirmed_transitions() {
        let robot: Arc<Mutex<Box<dyn RobotController>>> =
            Arc::new(Mutex::new(Box::new(crate::hardware::MockRobot::new())));
        let (_tx, state_rx, _error_rx, mut orch) = Orchestrator::new(&test_config(), robot);
        assert_eq!(*state_rx.borrow(), OrchestratorState::Paused);
        orch.handle_message(OrchestratorMsg::Resume);
        assert_eq!(*state_rx.borrow(), OrchestratorState::Running);
        orch.handle_message(OrchestratorMsg::EStop);
        assert_eq!(*state_rx.borrow(), OrchestratorState::EStopped);
    }

    #[tokio::test]
    async fn shutdown_records_last_position_and_parks() {
        let mut orch = test_orchestrator();
        // A successful move records where the arm is.
        let pos = Position {
            x: 40.0,
            y: 20.0,
            z: orch.z_pick,
        };
        orch.execute(RobotCommand::MoveTo(pos)).await.unwrap();
        assert_eq!(orch.last_pos, Some(pos));
        // The message flags the loop; park raises to z_travel best-effort.
        orch.handle_message(OrchestratorMsg::Shutdown);
        assert!(orch.shutting_down);
        orch.park_for_shutdown().await; // against the mock: must not panic
    }

    #[tokio::test]
    async fn shutdown_while_estopped_skips_park() {
        let mut orch = test_orchestrator();
        orch.handle_message(OrchestratorMsg::EStop);
        assert!(orch.needs_home);
        // Post-M112 the robot would ignore commands: park must return without
        // issuing any (best-effort, non-blocking shutdown).
        orch.park_for_shutdown().await;
    }

    #[test]
    fn manual_gripper_toggle_is_registered_even_while_paused() {
        let mut orch = test_orchestrator();
        assert!(orch.pending_gripper.is_none());
        // Paused (default) must not block the recovery toggle.
        orch.handle_message(OrchestratorMsg::SetGripper(false));
        assert_eq!(orch.pending_gripper, Some(false));
        // A later request overrides the pending one.
        orch.handle_message(OrchestratorMsg::SetGripper(true));
        assert_eq!(orch.pending_gripper, Some(true));
    }

    #[test]
    fn report_error_reaches_the_operator_banner_channel() {
        let robot: Arc<Mutex<Box<dyn RobotController>>> =
            Arc::new(Mutex::new(Box::new(crate::hardware::MockRobot::new())));
        let (_tx, _state_rx, error_rx, orch) = Orchestrator::new(&test_config(), robot);
        // Starts empty so a fresh boot shows no banner.
        assert!(error_rx.borrow().is_none());
        orch.report_error("Robot command failed: boom. Sorting paused.".to_string());
        assert_eq!(
            error_rx.borrow().as_deref(),
            Some("Robot command failed: boom. Sorting paused.")
        );
    }

    fn sequence(object_id: u64, expiry: Instant, steps: Vec<RobotCommand>) -> PickSequence {
        PickSequence {
            object_id,
            expiry,
            started: false,
            steps: VecDeque::from(steps),
        }
    }

    #[test]
    fn instruction_queue_fifo_and_clear() {
        let now = Instant::now();
        let far = now + Duration::from_secs(60);
        let mut q = InstructionQueue::new();
        assert!(q.is_empty());
        q.push_sequence(sequence(1, far, vec![RobotCommand::Home, RobotCommand::Gripper(true)]));
        q.push_sequence(sequence(2, far, vec![RobotCommand::Gripper(false)]));
        assert_eq!(q.len(), 3, "total commands across groups");
        // FIFO across and within groups.
        assert!(matches!(q.next_command(now), Some(RobotCommand::Home)));
        assert!(matches!(q.next_command(now), Some(RobotCommand::Gripper(true))));
        assert!(matches!(q.next_command(now), Some(RobotCommand::Gripper(false))));
        assert!(q.next_command(now).is_none());
        q.push_sequence(sequence(3, far, vec![RobotCommand::Home]));
        q.clear();
        assert!(q.next_command(now).is_none() && q.is_empty());
    }

    #[test]
    fn expired_sequence_is_dropped_before_it_starts() {
        // Enqueued fresh, but by execution time the TTL has passed: the whole
        // sequence is dropped, not run onto belt the object has left (#3).
        let mut orch = test_orchestrator();
        let obj = object_at(50.0, 0.0);
        let seen = obj.seen_at;
        orch.schedule_pick(obj, seen + Duration::from_secs(1));
        assert_eq!(orch.queue.len(), 7);
        assert!(
            orch.queue.next_command(seen + Duration::from_secs(5)).is_none(),
            "an expired-before-start sequence must be dropped whole"
        );
        assert!(orch.queue.is_empty());
    }

    #[test]
    fn a_started_sequence_is_never_dropped_mid_flight() {
        // Once the first command is taken, later expiry must NOT abort the rest
        // (the gripper may be holding a part).
        let mut orch = test_orchestrator();
        let obj = object_at(50.0, 0.0);
        let seen = obj.seen_at;
        orch.schedule_pick(obj, seen + Duration::from_secs(1));
        // First command taken while fresh → group started.
        assert!(orch.queue.next_command(seen + Duration::from_secs(1)).is_some());
        // TTL passes mid-sequence: the remaining commands still run.
        assert!(
            orch.queue.next_command(seen + Duration::from_secs(5)).is_some(),
            "a started sequence runs to completion despite expiry"
        );
    }

    #[test]
    fn expired_group_is_skipped_to_reach_a_still_valid_one() {
        let now = Instant::now();
        let mut q = InstructionQueue::new();
        // First group already expired, second still valid.
        q.push_sequence(sequence(
            1,
            now - Duration::from_secs(1),
            vec![RobotCommand::MoveTo(ORIGIN)],
        ));
        q.push_sequence(sequence(
            2,
            now + Duration::from_secs(60),
            vec![RobotCommand::Gripper(true)],
        ));
        // The expired group is dropped whole; the next command comes from the
        // still-valid group.
        assert!(matches!(q.next_command(now), Some(RobotCommand::Gripper(true))));
        assert!(q.next_command(now).is_none());
    }

    // --- Run-loop integration tests -------------------------------------
    //
    // These drive the real async `run()` loop with in-code fakes (project
    // rule: no new deps, no real hardware). They cover what the unit tests
    // above cannot: message drain ordering, pause/resume, E-stop preemption,
    // command-failure recovery, home recovery, manual gripper and shutdown —
    // all through the loop itself, not by calling handlers directly.
    //
    // Determinism: `#[tokio::test]` uses a current-thread runtime, so the
    // spawned orchestrator makes no progress until the test awaits. Sending
    // every message and dropping the sender *before* awaiting therefore means
    // the loop's first drain sees them all in order (mpsc is FIFO). Dropping
    // the sender is also how the loop terminates (recv → None).

    use crate::hardware::MockRobotCommand;

    /// A robot boxed behind the shared async mutex, as the orchestrator holds it.
    type BoxedRobot = Arc<Mutex<Box<dyn RobotController>>>;
    /// Shared handle to a mock robot's ordered command log.
    type CmdLog = std::sync::Arc<std::sync::Mutex<Vec<MockRobotCommand>>>;

    /// A boxed `MockRobot` plus a handle to its command log.
    fn mock_robot_with_log() -> (BoxedRobot, CmdLog) {
        let mock = crate::hardware::MockRobot::new();
        let log = mock.command_log();
        (Arc::new(Mutex::new(Box::new(mock))), log)
    }

    /// Spawn `run()`, feed it `msgs`, drop the sender, and await termination
    /// (bounded, so a stuck loop fails the test instead of hanging forever).
    /// Returns the state/error watch receivers for post-run assertions.
    async fn run_with(
        robot: BoxedRobot,
        msgs: Vec<OrchestratorMsg>,
    ) -> (
        watch::Receiver<OrchestratorState>,
        watch::Receiver<Option<String>>,
    ) {
        let (tx, state_rx, error_rx, orch) = Orchestrator::new(&test_config(), robot);
        let handle = tokio::spawn(orch.run());
        for m in msgs {
            tx.send(m).expect("orchestrator receiver alive");
        }
        drop(tx); // all senders gone → loop returns after draining
        tokio::time::timeout(Duration::from_secs(10), handle)
            .await
            .expect("run loop did not terminate")
            .expect("run loop task panicked");
        (state_rx, error_rx)
    }

    /// Poll the mock log until it holds at least `n` commands (used when a test
    /// must let real work happen before sending the next message).
    async fn wait_for_commands(log: &CmdLog, n: usize) {
        for _ in 0..500 {
            if log.lock().unwrap().len() >= n {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("robot did not reach {n} commands in time");
    }

    // Config-default pick geometry for object_at(50, 0): z_travel = -50,
    // z_pick = -180, drop = (150, 0, -100).
    const TRAVEL_50: Position = Position { x: 50.0, y: 0.0, z: -50.0 };
    const PICK_50: Position = Position { x: 50.0, y: 0.0, z: -180.0 };
    const DROP: Position = Position { x: 150.0, y: 0.0, z: -100.0 };

    /// A robot whose moves always fail at the hardware — the case the queue
    /// clears and pauses on. Gripper/home/connect succeed so the best-effort
    /// release after a failed pick is observable. estop_handle is unused by
    /// the loop, so `None` is fine.
    use anyhow::anyhow;
    use async_trait::async_trait;

    struct FailingRobot {
        log: CmdLog,
    }

    impl FailingRobot {
        fn new() -> (Self, CmdLog) {
            let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            (Self { log: log.clone() }, log)
        }
        fn record(&self, cmd: MockRobotCommand) {
            self.log.lock().unwrap().push(cmd);
        }
    }

    #[async_trait]
    impl RobotController for FailingRobot {
        async fn connect(&mut self) -> Result<()> {
            Ok(())
        }
        async fn home(&mut self) -> Result<()> {
            self.record(MockRobotCommand::Home);
            Ok(())
        }
        async fn move_to(&mut self, _pos: Position) -> Result<()> {
            // Passes the orchestrator's own limit check but fails at the wire.
            Err(anyhow!("simulated hardware move failure"))
        }
        async fn set_gripper(&mut self, on: bool) -> Result<()> {
            self.record(MockRobotCommand::Gripper(on));
            Ok(())
        }
        async fn stop(&mut self) -> Result<()> {
            Ok(())
        }
        fn estop_handle(&self) -> Option<Arc<dyn crate::hardware::EmergencyStop>> {
            None
        }
    }

    #[tokio::test]
    async fn run_loop_executes_a_full_pick_end_to_end() {
        use MockRobotCommand::*;
        let (robot, log) = mock_robot_with_log();
        let (state_rx, _err) = run_with(
            robot,
            vec![
                OrchestratorMsg::Resume,
                OrchestratorMsg::Pick(object_at(50.0, 0.0)),
            ],
        )
        .await;
        assert_eq!(
            *log.lock().unwrap(),
            vec![
                MoveTo(TRAVEL_50), // approach above
                MoveTo(PICK_50),   // descend (Wait is not a robot command)
                Gripper(true),     // grab
                MoveTo(TRAVEL_50), // lift
                MoveTo(DROP),      // carry to drop
                Gripper(false),    // release
            ],
        );
        assert_eq!(*state_rx.borrow(), OrchestratorState::Running);
    }

    #[tokio::test]
    async fn run_loop_drops_pick_while_paused() {
        // Default state is Paused: a boot-time pick must not move the robot.
        let (robot, log) = mock_robot_with_log();
        let (state_rx, _err) =
            run_with(robot, vec![OrchestratorMsg::Pick(object_at(50.0, 0.0))]).await;
        assert!(
            log.lock().unwrap().is_empty(),
            "paused pick must command the robot nothing"
        );
        assert_eq!(*state_rx.borrow(), OrchestratorState::Paused);
    }

    #[tokio::test]
    async fn run_loop_pause_drops_then_resume_executes() {
        let (robot, log) = mock_robot_with_log();
        run_with(
            robot,
            vec![
                OrchestratorMsg::Resume,
                OrchestratorMsg::Pause,
                OrchestratorMsg::Pick(object_at(10.0, 0.0)), // dropped while paused
                OrchestratorMsg::Resume,
                OrchestratorMsg::Pick(object_at(50.0, 0.0)), // this one runs
            ],
        )
        .await;
        let cmds = log.lock().unwrap();
        assert_eq!(
            cmds.len(),
            6,
            "only the pick after the second Resume should run, got {cmds:?}"
        );
        assert_eq!(cmds[0], MockRobotCommand::MoveTo(TRAVEL_50));
    }

    #[tokio::test]
    async fn run_loop_estop_preempts_queued_commands() {
        // Pick then E-stop arrive in the same drain: the scheduled sequence is
        // cleared before a single command is executed.
        let (robot, log) = mock_robot_with_log();
        let (state_rx, _err) = run_with(
            robot,
            vec![
                OrchestratorMsg::Resume,
                OrchestratorMsg::Pick(object_at(50.0, 0.0)),
                OrchestratorMsg::EStop,
            ],
        )
        .await;
        assert!(
            log.lock().unwrap().is_empty(),
            "E-stop must drop the queued pick before it reaches the robot"
        );
        assert_eq!(*state_rx.borrow(), OrchestratorState::EStopped);
    }

    #[tokio::test]
    async fn run_loop_command_failure_clears_queue_and_pauses() {
        let (failing, log) = FailingRobot::new();
        let robot: BoxedRobot = Arc::new(Mutex::new(Box::new(failing)));
        let (state_rx, error_rx) = run_with(
            robot,
            vec![
                OrchestratorMsg::Resume,
                OrchestratorMsg::Pick(object_at(50.0, 0.0)),
            ],
        )
        .await;
        // The first MoveTo fails: the queue is cleared (no later pick command
        // runs) and only the best-effort gripper release reaches the robot.
        assert_eq!(*log.lock().unwrap(), vec![MockRobotCommand::Gripper(false)]);
        assert_eq!(*state_rx.borrow(), OrchestratorState::Paused);
        assert!(
            error_rx
                .borrow()
                .as_deref()
                .is_some_and(|m| m.contains("failed")),
            "the operator banner must report the failure"
        );
    }

    #[tokio::test]
    async fn run_loop_home_clears_estop_state() {
        // E-stop, then operator Home: Home runs even though we are paused and
        // E-stopped, and clears the E-stop interlock.
        let (robot, log) = mock_robot_with_log();
        let (state_rx, _err) = run_with(
            robot,
            vec![
                OrchestratorMsg::Resume,
                OrchestratorMsg::EStop,
                OrchestratorMsg::Home,
            ],
        )
        .await;
        assert_eq!(
            *log.lock().unwrap(),
            vec![MockRobotCommand::Home],
            "home must run as the E-stop recovery action"
        );
        assert_eq!(
            *state_rx.borrow(),
            OrchestratorState::Paused,
            "after homing the E-stop interlock is cleared (back to Paused)"
        );
    }

    #[tokio::test]
    async fn run_loop_manual_gripper_runs_while_paused() {
        // A manual toggle must reach the robot even from the default paused
        // state (release a part held after a failed pick / E-stop).
        let (robot, log) = mock_robot_with_log();
        run_with(robot, vec![OrchestratorMsg::SetGripper(true)]).await;
        assert_eq!(
            *log.lock().unwrap(),
            vec![MockRobotCommand::Gripper(true)],
            "manual gripper toggle must run while paused"
        );
    }

    #[tokio::test]
    async fn run_loop_shutdown_parks_after_the_pick_completes() {
        use MockRobotCommand::*;
        // Unlike the messages above, Shutdown must arrive *after* the pick has
        // run, otherwise it would (correctly) preempt it and there would be no
        // recorded position to park from. So we let the pick finish first.
        let (robot, log) = mock_robot_with_log();
        let (tx, _state_rx, _err, orch) = Orchestrator::new(&test_config(), robot);
        let handle = tokio::spawn(orch.run());
        tx.send(OrchestratorMsg::Resume).unwrap();
        tx.send(OrchestratorMsg::Pick(object_at(50.0, 0.0))).unwrap();
        wait_for_commands(&log, 6).await; // whole pick executed
        tx.send(OrchestratorMsg::Shutdown).unwrap();
        drop(tx);
        tokio::time::timeout(Duration::from_secs(10), handle)
            .await
            .expect("run loop did not terminate")
            .expect("run loop task panicked");

        let cmds = log.lock().unwrap();
        // Park raises straight up to z_travel above the last position (DROP),
        // after releasing the gripper.
        let park_raise = Position { x: 150.0, y: 0.0, z: -50.0 };
        assert_eq!(
            cmds.last().cloned(),
            Some(MoveTo(park_raise)),
            "shutdown must raise to z_travel above the last position"
        );
        assert_eq!(
            cmds[cmds.len() - 2],
            Gripper(false),
            "shutdown releases the gripper before raising"
        );
    }
}
