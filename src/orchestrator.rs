use crate::app_config::AppConfig;
use crate::hardware::{Position, RobotController, WorkspaceLimits};
use crate::vision::DetectedObject;
use anyhow::Result;
use log::{error, info, warn};
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

pub struct InstructionQueue {
    queue: VecDeque<RobotCommand>,
}

impl InstructionQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    pub fn push(&mut self, cmd: RobotCommand) {
        self.queue.push_back(cmd);
    }

    pub fn pop(&mut self) -> Option<RobotCommand> {
        self.queue.pop_front()
    }

    pub fn clear(&mut self) {
        self.queue.clear();
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
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

    /// Predict where the object crosses the pick line and whether the robot
    /// can be there in time. Returns None when the object is moving away
    /// from (or already past) the pick line, or is unreachable in time.
    // Unit-tested but not called on the active path yet (see struct note).
    #[allow(dead_code)]
    pub fn calculate_intercept(
        &self,
        robot_pos: Position,
        object: &DetectedObject,
    ) -> Option<Position> {
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
        let target = Position {
            x: pos.x,
            y: self.pick_line_y,
            z: pos.z,
        };
        let dx = target.x - robot_pos.x;
        let dy = target.y - robot_pos.y;
        let robot_time = (dx * dx + dy * dy).sqrt() / self.robot_speed_mm_s;
        (robot_time < time_to_line).then_some(target)
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
    drop_pos: Position,
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
            drop_pos: Position {
                x: config.sorting.drop_x,
                y: config.sorting.drop_y,
                z: config.sorting.drop_z,
            },
            robot,
        };
        (tx, state_rx, error_rx, orchestrator)
    }

    /// Push a concise failure message to the operator banner. The detailed
    /// chain still goes to the log; this is the one line the operator sees.
    fn report_error(&self, msg: String) {
        let _ = self.error_tx.send(Some(msg));
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

            if let Some(cmd) = self.queue.pop() {
                if let Err(e) = self.execute(cmd).await {
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
            return;
        }
        let Some(pos) = object.world_pos else {
            warn!(
                "Orchestrator: object {} has no world position — skipping",
                object.id
            );
            return;
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
        self.queue.push(RobotCommand::MoveTo(travel)); // approach from above
        self.queue.push(RobotCommand::MoveTo(pick)); // descend
        self.queue.push(RobotCommand::Gripper(true));
        self.queue
            .push(RobotCommand::Wait(Duration::from_millis(150))); // let suction settle
        self.queue.push(RobotCommand::MoveTo(travel)); // lift
        self.queue.push(RobotCommand::MoveTo(self.drop_pos));
        self.queue.push(RobotCommand::Gripper(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::ObjectClass;
    use opencv::core::Rect;
    use std::time::SystemTime;

    const ORIGIN: Position = Position {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    fn object_at(x: f32, y: f32) -> DetectedObject {
        DetectedObject {
            id: 1,
            rect: Rect::new(0, 0, 10, 10),
            world_pos: Some(Position { x, y, z: 0.0 }),
            class: ObjectClass::Unknown,
            confidence: 0.0,
            timestamp: SystemTime::now(),
            seen_at: Instant::now(),
        }
    }

    fn test_orchestrator() -> Orchestrator {
        let robot: Arc<Mutex<Box<dyn RobotController>>> =
            Arc::new(Mutex::new(Box::new(crate::hardware::MockRobot::new())));
        let (_tx, _state_rx, _error_rx, orch) = Orchestrator::new(&AppConfig::default(), robot);
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
        let (_tx, _state_rx, _error_rx, orch) = Orchestrator::new(&AppConfig::default(), robot);
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
        while let Some(cmd) = orch.queue.pop() {
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
                MoveTo(orch.drop_pos),
                Gripper(false), // release over the drop
            ],
        );
    }

    #[test]
    fn intercept_upstream_object_is_reachable() {
        // Belt moves toward +Y at 100 mm/s; object 100 mm upstream of the
        // pick line arrives in 1 s; robot needs 50/250 = 0.2 s.
        let planner = TrajectoryPlanner::new(100.0, 250.0);
        let target = planner
            .calculate_intercept(ORIGIN, &object_at(50.0, -100.0))
            .expect("object should be interceptable");
        assert_eq!(target.x, 50.0);
        assert_eq!(target.y, 0.0);
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
        let (_tx, state_rx, _error_rx, mut orch) = Orchestrator::new(&AppConfig::default(), robot);
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
        let (_tx, _state_rx, error_rx, orch) = Orchestrator::new(&AppConfig::default(), robot);
        // Starts empty so a fresh boot shows no banner.
        assert!(error_rx.borrow().is_none());
        orch.report_error("Robot command failed: boom. Sorting paused.".to_string());
        assert_eq!(
            error_rx.borrow().as_deref(),
            Some("Robot command failed: boom. Sorting paused.")
        );
    }

    #[test]
    fn instruction_queue_fifo_and_clear() {
        let mut q = InstructionQueue::new();
        assert!(q.is_empty());
        q.push(RobotCommand::Home);
        q.push(RobotCommand::Gripper(true));
        assert_eq!(q.len(), 2);
        assert!(matches!(q.pop(), Some(RobotCommand::Home)));
        q.clear();
        assert!(q.pop().is_none());
    }
}
