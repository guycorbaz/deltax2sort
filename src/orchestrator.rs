use crate::app_config::AppConfig;
use crate::hardware::{Position, RobotController, WorkspaceLimits};
use crate::vision::DetectedObject;
use anyhow::Result;
use log::{error, info, warn};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};

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
    /// Enqueue a single raw command (e.g. Home from the UI).
    Command(RobotCommand),
    Pause,
    Resume,
    /// Clear the queue and pause. The hardware halt itself is done by the
    /// caller through the `EmergencyStop` handles — this message only makes
    /// sure no further commands are fed to the robot.
    EStop,
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
    // Not yet used by schedule_pick: belt-motion compensation needs robot
    // position feedback (documentation/TODO.md).
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
    ) -> (mpsc::UnboundedSender<OrchestratorMsg>, Self) {
        let (tx, rx) = mpsc::unbounded_channel();
        let orchestrator = Self {
            rx,
            queue: InstructionQueue::new(),
            paused: false,
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
        (tx, orchestrator)
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
                    self.queue.clear();
                    self.paused = true;
                }
            }
        }
        info!("Orchestrator loop stopped (all senders dropped)");
    }

    fn handle_message(&mut self, msg: OrchestratorMsg) {
        match msg {
            OrchestratorMsg::Pick(object) => self.schedule_pick(object),
            OrchestratorMsg::Command(cmd) => self.queue.push(cmd),
            OrchestratorMsg::Pause => {
                info!("Orchestrator: paused ({} command(s) queued)", self.queue.len());
                self.paused = true;
            }
            OrchestratorMsg::Resume => {
                info!("Orchestrator: resumed");
                self.paused = false;
            }
            OrchestratorMsg::EStop => {
                warn!(
                    "Orchestrator: E-STOP — dropping {} queued command(s) and pausing",
                    self.queue.len()
                );
                self.queue.clear();
                self.paused = true;
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
            RobotCommand::MoveTo(pos) => self.robot.lock().await.move_to(pos).await,
            RobotCommand::Gripper(on) => self.robot.lock().await.set_gripper(on).await,
            RobotCommand::Home => self.robot.lock().await.home().await,
        }
    }

    fn schedule_pick(&mut self, object: DetectedObject) {
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
        self.queue.push(RobotCommand::Wait(Duration::from_millis(150))); // let suction settle
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
        }
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
        assert!(planner
            .calculate_intercept(ORIGIN, &object_at(50.0, 100.0))
            .is_none());
    }

    #[test]
    fn intercept_with_stopped_belt_is_none() {
        let planner = TrajectoryPlanner::new(0.0, 250.0);
        assert!(planner
            .calculate_intercept(ORIGIN, &object_at(50.0, -100.0))
            .is_none());
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
        assert!(planner
            .calculate_intercept(far_robot, &object_at(160.0, -10.0))
            .is_none());
    }

    #[test]
    fn intercept_object_without_world_pos_is_none() {
        let planner = TrajectoryPlanner::new(100.0, 250.0);
        let mut obj = object_at(0.0, -100.0);
        obj.world_pos = None;
        assert!(planner.calculate_intercept(ORIGIN, &obj).is_none());
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
