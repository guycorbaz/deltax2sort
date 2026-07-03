use anyhow::{Result, anyhow};
use async_trait::async_trait;
use log::{debug, info, warn};
use opencv::{core, prelude::*, videoio};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Physical workspace of the robot. Every real move is validated against
/// these bounds before any G-code is sent.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceLimits {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z_min: f32,
    pub z_max: f32,
}

impl WorkspaceLimits {
    pub fn contains(&self, p: Position) -> bool {
        p.x >= self.x_min
            && p.x <= self.x_max
            && p.y >= self.y_min
            && p.y <= self.y_max
            && p.z >= self.z_min
            && p.z <= self.z_max
    }
}

impl Default for WorkspaceLimits {
    /// Delta X2 (SP-X2) spec: X/Y in [-160, 160] mm, Z in [-200, 0] (Z0 top).
    fn default() -> Self {
        Self {
            x_min: -160.0,
            x_max: 160.0,
            y_min: -160.0,
            y_max: 160.0,
            z_min: -200.0,
            z_max: 0.0,
        }
    }
}

type SharedPort = Arc<StdMutex<Box<dyn serialport::SerialPort>>>;

/// Immediate hardware halt, callable from any thread. Implementations own a
/// dedicated serial handle (`SerialPort::try_clone`), so triggering does NOT
/// contend with the async mutexes or with a command that is mid-flight —
/// this is what makes the UI E-Stop preemptive.
pub trait EmergencyStop: Send + Sync {
    fn trigger(&self);
}

struct SerialEStop {
    port: SharedPort,
    command: &'static [u8],
    label: &'static str,
}

impl EmergencyStop for SerialEStop {
    fn trigger(&self) {
        warn!("E-STOP: sending halt to {}", self.label);
        match self.port.lock() {
            Ok(mut p) => {
                let _ = p.write_all(self.command);
                let _ = p.flush();
            }
            Err(_) => warn!("E-STOP: {} port mutex poisoned", self.label),
        }
    }
}

struct MockEStop {
    label: &'static str,
}

impl EmergencyStop for MockEStop {
    fn trigger(&self) {
        warn!("E-STOP ({}): halt triggered (mock)", self.label);
    }
}

// --- ROBOT ---

#[async_trait]
pub trait RobotController: Send + Sync {
    async fn connect(&mut self) -> Result<()>;
    async fn home(&mut self) -> Result<()>;
    async fn move_to(&mut self, pos: Position) -> Result<()>;
    async fn set_gripper(&mut self, on: bool) -> Result<()>;
    // Programmatic halt; unused today — the UI E-stop path deliberately
    // bypasses the traits via `estop_handle` (preemption).
    #[allow(dead_code)]
    async fn stop(&mut self) -> Result<()>; // E-Stop
    /// Preemptive halt handle; available once connected. See [`EmergencyStop`].
    fn estop_handle(&self) -> Option<Arc<dyn EmergencyStop>>;
}

pub struct MockRobot {
    connected: bool,
    current_pos: Position,
    limits: WorkspaceLimits,
}

impl MockRobot {
    // Only unit tests construct it without explicit limits.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::with_limits(WorkspaceLimits::default())
    }

    pub fn with_limits(limits: WorkspaceLimits) -> Self {
        Self {
            connected: false,
            current_pos: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            limits,
        }
    }
}

#[async_trait]
impl RobotController for MockRobot {
    async fn connect(&mut self) -> Result<()> {
        info!("MockRobot: Connecting...");
        sleep(Duration::from_millis(500)).await;
        self.connected = true;
        info!("MockRobot: Connected!");
        Ok(())
    }

    async fn home(&mut self) -> Result<()> {
        info!("MockRobot: Homing (G28)...");
        sleep(Duration::from_millis(1000)).await;
        // Spec: Z0 is top, homing goes to top.
        self.current_pos = Position {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        info!("MockRobot: Homed.");
        Ok(())
    }

    async fn move_to(&mut self, pos: Position) -> Result<()> {
        if !self.limits.contains(pos) {
            return Err(anyhow!(
                "MockRobot: target {:?} outside workspace {:?}",
                pos,
                self.limits
            ));
        }
        info!("MockRobot: Moving to {:?}", pos);
        sleep(Duration::from_millis(100)).await;
        self.current_pos = pos;
        Ok(())
    }

    async fn set_gripper(&mut self, on: bool) -> Result<()> {
        let state = if on { "ON (M03)" } else { "OFF (M05)" };
        info!("MockRobot: Gripper {}", state);
        sleep(Duration::from_millis(50)).await;
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        warn!("MockRobot: EMERGENCY STOP TRIGGERED!");
        Ok(())
    }

    fn estop_handle(&self) -> Option<Arc<dyn EmergencyStop>> {
        Some(Arc::new(MockEStop {
            label: "mock robot",
        }))
    }
}

pub struct DeltaX2 {
    port_name: String,
    baud_rate: u32,
    limits: WorkspaceLimits,
    feed_rate: u32,
    cmd_seq: u64,
    port: Option<SharedPort>,
    /// Second OS handle to the same device (`try_clone`), used only for
    /// emergency stop so it can be written while `port` is busy.
    estop_port: Option<SharedPort>,
}

impl DeltaX2 {
    /// Upper bound on how long a single G-code command (including the
    /// physical move) may take before we give up waiting for its FEEDBACK.
    const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

    pub fn new(port_name: &str, baud_rate: u32, limits: WorkspaceLimits, feed_rate: u32) -> Self {
        Self {
            port_name: port_name.to_string(),
            baud_rate,
            limits,
            feed_rate,
            cmd_seq: 0,
            port: None,
            estop_port: None,
        }
    }

    /// Send one G-code command and block (on a dedicated blocking thread)
    /// until the robot echoes the FEEDBACK id, meaning the command has
    /// physically completed.
    async fn write_gcode(&mut self, cmd: &str) -> Result<()> {
        let port = self
            .port
            .clone()
            .ok_or_else(|| anyhow!("Robot not connected"))?;
        self.cmd_seq += 1;
        let fb_id = format!("sync_{}", self.cmd_seq);
        let cmd = cmd.trim().to_string();
        tokio::task::spawn_blocking(move || {
            send_and_wait_feedback(&port, &cmd, &fb_id, Self::COMMAND_TIMEOUT)
        })
        .await?
    }
}

/// Serial I/O is synchronous; this runs inside `spawn_blocking` so it never
/// stalls the async executor.
fn send_and_wait_feedback(
    port: &SharedPort,
    cmd: &str,
    fb_id: &str,
    timeout: Duration,
) -> Result<()> {
    let mut p = port
        .lock()
        .map_err(|_| anyhow!("serial port mutex poisoned"))?;
    let full = format!("{} FEEDBACK:{}\n", cmd, fb_id);
    p.write_all(full.as_bytes())?;
    p.flush()?;

    let deadline = Instant::now() + timeout;
    let mut reader = BufReader::new(&mut **p);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // EOF: port closed / device unplugged. Previously this
                // spun forever on the empty line.
                return Err(anyhow!("serial port closed while waiting for '{}'", cmd));
            }
            Ok(_) => {
                let t = line.trim();
                if t == fb_id {
                    debug!("DeltaX2: command executed ({})", cmd);
                    return Ok(());
                } else if t.is_empty() || t.eq_ignore_ascii_case("ok") {
                    // 'ok' acknowledges receipt; keep waiting for the
                    // FEEDBACK echo that marks physical completion.
                } else if t.to_ascii_lowercase().contains("error") {
                    return Err(anyhow!("robot reported error for '{}': {}", cmd, t));
                } else {
                    debug!("DeltaX2: ignoring unexpected line '{}'", t);
                }
            }
            // Per-read timeout (2 s) — keep polling until the deadline.
            Err(e) if e.kind() == ErrorKind::TimedOut => {}
            Err(e) => return Err(anyhow!("reading feedback for '{}': {}", cmd, e)),
        }
        if Instant::now() >= deadline {
            return Err(anyhow!(
                "timed out after {:?} waiting for '{}' to complete",
                timeout,
                cmd
            ));
        }
    }
}

#[async_trait]
impl RobotController for DeltaX2 {
    async fn connect(&mut self) -> Result<()> {
        info!(
            "DeltaX2: Connecting to {} at {}...",
            self.port_name, self.baud_rate
        );
        let port_name = self.port_name.clone();
        let baud_rate = self.baud_rate;
        let port =
            tokio::task::spawn_blocking(move || -> Result<Box<dyn serialport::SerialPort>> {
                let mut port = serialport::new(&port_name, baud_rate)
                    .timeout(Duration::from_millis(2000))
                    .open()?;

                // The controller may reboot when the port opens.
                std::thread::sleep(Duration::from_secs(2));
                port.write_all(b"IsDelta\n")?;
                port.flush()?;

                // Scan past any boot banner until the handshake answer.
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut reader = BufReader::new(&mut port);
                loop {
                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) => return Err(anyhow!("serial port closed during handshake")),
                        Ok(_) => {
                            let t = line.trim();
                            if t == "YesDelta" {
                                break;
                            }
                            if !t.is_empty() {
                                debug!("DeltaX2: handshake, skipping line '{}'", t);
                            }
                        }
                        Err(e) if e.kind() == ErrorKind::TimedOut => {}
                        Err(e) => return Err(e.into()),
                    }
                    if Instant::now() >= deadline {
                        return Err(anyhow!(
                            "device at {} did not answer IsDelta handshake",
                            port_name
                        ));
                    }
                }
                drop(reader);
                Ok(port)
            })
            .await??;

        // Dedicated handle for the emergency stop path.
        let estop = port.try_clone()?;
        self.port = Some(Arc::new(StdMutex::new(port)));
        self.estop_port = Some(Arc::new(StdMutex::new(estop)));
        info!("DeltaX2: Validated and Connected.");
        Ok(())
    }

    async fn home(&mut self) -> Result<()> {
        info!("DeltaX2: Homing...");
        self.write_gcode("G28").await?;
        // Set absolute positioning as default after home
        self.write_gcode("G90").await?;
        Ok(())
    }

    async fn move_to(&mut self, pos: Position) -> Result<()> {
        // Validate before touching the port so a bad target can never
        // reach the hardware.
        if !self.limits.contains(pos) {
            return Err(anyhow!(
                "DeltaX2: refusing move to {:?}: outside workspace {:?}",
                pos,
                self.limits
            ));
        }
        let cmd = format!(
            "G01 X{:.2} Y{:.2} Z{:.2} F{}",
            pos.x, pos.y, pos.z, self.feed_rate
        );
        self.write_gcode(&cmd).await
    }

    async fn set_gripper(&mut self, on: bool) -> Result<()> {
        let cmd = if on { "M03" } else { "M05" };
        self.write_gcode(cmd).await
    }

    async fn stop(&mut self) -> Result<()> {
        warn!("DeltaX2: Sending emergency stop (M112)!");
        // Fire-and-forget on the dedicated handle: never waits for
        // feedback and never queues behind an in-flight command.
        let port = self
            .estop_port
            .clone()
            .or_else(|| self.port.clone())
            .ok_or_else(|| anyhow!("Robot not connected"))?;
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut p = port
                .lock()
                .map_err(|_| anyhow!("serial port mutex poisoned"))?;
            p.write_all(b"M112\n")?;
            p.flush()?;
            Ok(())
        })
        .await?
    }

    fn estop_handle(&self) -> Option<Arc<dyn EmergencyStop>> {
        Some(Arc::new(SerialEStop {
            port: self.estop_port.clone()?,
            command: b"M112\n",
            label: "robot (M112)",
        }))
    }
}

// --- CONVEYOR ---

#[async_trait]
pub trait ConveyorController: Send + Sync {
    async fn connect(&mut self) -> Result<()>;
    async fn start(&mut self, speed: i32) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    /// Preemptive halt handle; available once connected. See [`EmergencyStop`].
    fn estop_handle(&self) -> Option<Arc<dyn EmergencyStop>>;
}

pub struct SerialConveyor {
    port_name: String,
    baud_rate: u32,
    port: Option<SharedPort>,
    estop_port: Option<SharedPort>,
}

impl SerialConveyor {
    pub fn new(port_name: &str, baud_rate: u32) -> Self {
        Self {
            port_name: port_name.to_string(),
            baud_rate,
            port: None,
            estop_port: None,
        }
    }

    async fn write_cmd(&mut self, cmd: &str) -> Result<()> {
        let port = self
            .port
            .clone()
            .ok_or_else(|| anyhow!("Conveyor not connected"))?;
        let line = format!("{}\n", cmd.trim());
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut p = port
                .lock()
                .map_err(|_| anyhow!("serial port mutex poisoned"))?;
            p.write_all(line.as_bytes())?;
            p.flush()?;
            Ok(())
        })
        .await?
    }
}

#[async_trait]
impl ConveyorController for SerialConveyor {
    async fn connect(&mut self) -> Result<()> {
        info!(
            "Conveyor: Connecting to {} at {}...",
            self.port_name, self.baud_rate
        );
        let port_name = self.port_name.clone();
        let baud_rate = self.baud_rate;
        let port =
            tokio::task::spawn_blocking(move || -> Result<Box<dyn serialport::SerialPort>> {
                Ok(serialport::new(&port_name, baud_rate)
                    .timeout(Duration::from_millis(1000))
                    .open()?)
            })
            .await??;
        let estop = port.try_clone()?;
        self.port = Some(Arc::new(StdMutex::new(port)));
        self.estop_port = Some(Arc::new(StdMutex::new(estop)));
        info!("Conveyor: Connected.");
        Ok(())
    }

    async fn start(&mut self, speed: i32) -> Result<()> {
        let cmd = format!("M3 S{}", speed);
        self.write_cmd(&cmd).await
    }

    async fn stop(&mut self) -> Result<()> {
        self.write_cmd("M5").await
    }

    fn estop_handle(&self) -> Option<Arc<dyn EmergencyStop>> {
        Some(Arc::new(SerialEStop {
            port: self.estop_port.clone()?,
            command: b"M5\n",
            label: "conveyor (M5)",
        }))
    }
}

pub struct MockConveyor {
    running: bool,
}

impl MockConveyor {
    pub fn new() -> Self {
        Self { running: false }
    }
}

#[async_trait]
impl ConveyorController for MockConveyor {
    async fn connect(&mut self) -> Result<()> {
        info!("MockConveyor: Connected");
        Ok(())
    }

    async fn start(&mut self, speed: i32) -> Result<()> {
        info!("MockConveyor: Starting at speed {}", speed);
        self.running = true;
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("MockConveyor: Stopped");
        self.running = false;
        Ok(())
    }

    fn estop_handle(&self) -> Option<Arc<dyn EmergencyStop>> {
        Some(Arc::new(MockEStop {
            label: "mock conveyor",
        }))
    }
}

// --- CAMERA ---

#[async_trait]
pub trait CameraDriver: Send + Sync {
    async fn connect(&mut self) -> Result<()>;
    async fn get_frame(&mut self) -> Result<core::Mat>;
    /// Capture resolution actually in effect (after `connect` for the real
    /// camera, which may pick the nearest mode it supports). Calibration
    /// must be based on this, not on the requested configuration values.
    fn resolution(&self) -> (u32, u32);
}

pub struct OpencvCamera {
    device_id: i32,
    width: u32,
    height: u32,
    fps: u32,
    fourcc: Option<String>,
    cap: Option<videoio::VideoCapture>,
}

impl OpencvCamera {
    pub fn new(device_id: i32, width: u32, height: u32, fps: u32, fourcc: Option<String>) -> Self {
        Self {
            device_id,
            width,
            height,
            fps,
            fourcc,
            cap: None,
        }
    }
}

// SAFETY: `VideoCapture` is only ever accessed through `&mut self` methods
// and the camera is owned by a single task at a time; the impls exist only
// because `CameraDriver` requires Send + Sync. To be replaced by a dedicated
// camera thread + channel when the vision loop lands (docs/TODO.md).
unsafe impl Send for OpencvCamera {}
unsafe impl Sync for OpencvCamera {}

#[async_trait]
impl CameraDriver for OpencvCamera {
    async fn connect(&mut self) -> Result<()> {
        info!("Camera: Connecting to ID {}...", self.device_id);
        let mut cap = videoio::VideoCapture::new(self.device_id, videoio::CAP_ANY)?;

        if !videoio::VideoCapture::is_opened(&cap)? {
            return Err(anyhow!("Failed to open camera {}", self.device_id));
        }

        // Keep the driver's internal frame queue as short as possible so
        // get_frame returns the freshest frame, not one buffered seconds ago
        // (best effort — not every backend honours it).
        let _ = cap.set(videoio::CAP_PROP_BUFFERSIZE, 1.0);
        if let Some(fourcc) = &self.fourcc {
            let b = fourcc.as_bytes();
            let code = videoio::VideoWriter::fourcc(
                b[0] as char,
                b[1] as char,
                b[2] as char,
                b[3] as char,
            )?;
            cap.set(videoio::CAP_PROP_FOURCC, code as f64)?;
        }
        cap.set(videoio::CAP_PROP_FRAME_WIDTH, self.width as f64)?;
        cap.set(videoio::CAP_PROP_FRAME_HEIGHT, self.height as f64)?;
        cap.set(videoio::CAP_PROP_FPS, self.fps as f64)?;

        // Cameras silently fall back to the nearest mode they support; adopt
        // and report what is actually in effect so calibration stays honest.
        let actual_w = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as u32;
        let actual_h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as u32;
        let actual_fps = cap.get(videoio::CAP_PROP_FPS)? as u32;
        if actual_w != 0 && (actual_w, actual_h) != (self.width, self.height) {
            warn!(
                "Camera: requested {}x{} but device uses {}x{}",
                self.width, self.height, actual_w, actual_h
            );
            self.width = actual_w;
            self.height = actual_h;
        }
        if actual_fps != 0 && actual_fps != self.fps {
            warn!(
                "Camera: requested {} fps but device uses {} fps",
                self.fps, actual_fps
            );
            self.fps = actual_fps;
        }

        self.cap = Some(cap);
        info!(
            "Camera: Connected ({}x{} @ {} fps).",
            self.width, self.height, self.fps
        );
        Ok(())
    }

    async fn get_frame(&mut self) -> Result<core::Mat> {
        if let Some(cap) = &mut self.cap {
            let mut frame = core::Mat::default();
            cap.read(&mut frame)?;
            if frame.empty() {
                return Err(anyhow!("Empty frame captured"));
            }
            Ok(frame)
        } else {
            Err(anyhow!("Camera not connected"))
        }
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Mock camera honouring the configured resolution and frame rate, so mock
/// runs exercise the same `[camera]` settings as the real driver.
pub struct MockCamera {
    width: u32,
    height: u32,
    fps: u32,
}

impl MockCamera {
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        Self { width, height, fps }
    }
}

#[async_trait]
impl CameraDriver for MockCamera {
    async fn connect(&mut self) -> Result<()> {
        info!(
            "MockCamera: Connected ({}x{} @ {} fps)",
            self.width, self.height, self.fps
        );
        Ok(())
    }

    async fn get_frame(&mut self) -> Result<core::Mat> {
        let frame = core::Mat::new_rows_cols_with_default(
            self.height as i32,
            self.width as i32,
            core::CV_8UC3,
            core::Scalar::all(0.0),
        )?;
        sleep(Duration::from_millis(1000 / self.fps.max(1) as u64)).await;
        Ok(frame)
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_limits_contains() {
        let l = WorkspaceLimits::default();
        assert!(l.contains(Position {
            x: 0.0,
            y: 0.0,
            z: -100.0
        }));
        assert!(!l.contains(Position {
            x: 0.0,
            y: 0.0,
            z: 20.0 // above Z0 (top) — physically impossible
        }));
        assert!(!l.contains(Position {
            x: 200.0,
            y: 0.0,
            z: -100.0
        }));
    }

    #[tokio::test]
    async fn mock_robot_rejects_out_of_bounds_move() {
        let mut robot = MockRobot::new();
        let result = robot
            .move_to(Position {
                x: 500.0,
                y: 0.0,
                z: -50.0,
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn deltax2_validates_bounds_before_port_access() {
        // No hardware: bounds are checked before the connection state.
        let mut robot = DeltaX2::new("/dev/null", 115200, WorkspaceLimits::default(), 15000);
        let result = robot
            .move_to(Position {
                x: 0.0,
                y: 0.0,
                z: 20.0,
            })
            .await;
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("outside workspace"), "got: {msg}");

        // In-bounds move on a disconnected robot fails with 'not connected'.
        let result = robot
            .move_to(Position {
                x: 0.0,
                y: 0.0,
                z: -50.0,
            })
            .await;
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not connected"), "got: {msg}");
    }
}
