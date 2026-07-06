mod app_config;
mod hardware;
mod orchestrator;
mod vision;

slint::include_modules!();

use anyhow::Context;
use app_config::AppConfig;
use clap::Parser;
use hardware::{
    CameraDriver, ConveyorController, MockCamera, MockConveyor, MockRobot, OpencvCamera,
    RobotController,
};
use log::{error, info, warn};
use orchestrator::{Orchestrator, OrchestratorMsg, OrchestratorState};
use slint::ComponentHandle;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the configuration file
    #[arg(short, long, default_value = "Settings.toml")]
    config: String,

    /// Run with simulated robot, conveyor and camera (no hardware needed)
    #[arg(short, long, default_value_t = false)]
    mock: bool,

    /// Operator profile; overrides `[ui].profile` from the config file.
    #[arg(long, value_enum)]
    profile: Option<app_config::UiProfile>,
}

type SharedRobot = Arc<Mutex<Box<dyn RobotController>>>;
type SharedConveyor = Arc<Mutex<Box<dyn ConveyorController>>>;

/// Bring up logging to a daily-rotating file (plus stderr when configured).
/// The returned handle must outlive the program so rotation/flush keep
/// running. `RUST_LOG`, if set, overrides `logging.level` (so `RUST_LOG=debug`
/// still traces G-code on demand).
fn init_logging(cfg: &app_config::LoggingConfig) -> anyhow::Result<flexi_logger::LoggerHandle> {
    use flexi_logger::{
        Age, Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming, WriteMode,
    };
    let cleanup = if cfg.keep_days == 0 {
        Cleanup::Never
    } else {
        Cleanup::KeepLogFiles(cfg.keep_days as usize)
    };
    let mut builder = Logger::try_with_env_or_str(&cfg.level)?
        .log_to_file(
            FileSpec::default()
                .directory(&cfg.directory)
                .basename("deltax2sort"),
        )
        .rotate(Criterion::Age(Age::Day), Naming::Timestamps, cleanup)
        .append() // continue the same day's file across restarts
        // Flush every record: a crash mid-run must not lose the tail, which is
        // exactly what a debugging log is for.
        .write_mode(WriteMode::Direct);
    if cfg.to_console {
        builder = builder.duplicate_to_stderr(Duplicate::All);
    }
    Ok(builder.start()?)
}

/// One-shot startup dump of the effective configuration — the first thing to
/// check when a log is opened for debugging.
fn log_config_summary(args: &Args, c: &AppConfig) {
    info!(
        "Mode: {} | log level {:?} → {}/",
        if args.mock { "MOCK" } else { "REAL hardware" },
        c.logging.level,
        c.logging.directory
    );
    info!(
        "Robot: port {} @ {} baud, workspace X[{},{}] Y[{},{}] Z[{},{}], z_pick {} z_travel {} feed {} mm/min, home_on_connect {}, release_gripper_on_estop {}",
        c.robot.port_name, c.robot.baud_rate,
        c.robot.x_min, c.robot.x_max, c.robot.y_min, c.robot.y_max, c.robot.z_min, c.robot.z_max,
        c.robot.z_pick, c.robot.z_travel, c.robot.feed_rate, c.robot.home_on_connect,
        c.robot.release_gripper_on_estop,
    );
    info!(
        "Conveyor: port {} @ {} baud, default_speed {}, speed {} mm/s",
        c.conveyor.port_name, c.conveyor.baud_rate, c.conveyor.default_speed, c.conveyor.speed_mm_s,
    );
    info!(
        "Camera: device {} {}x{} @ {} fps, fourcc {:?}",
        c.camera.device_id, c.camera.width, c.camera.height, c.camera.fps, c.camera.fourcc,
    );
    info!(
        "Sorting: {} bin(s), {} class assignment(s) | Vision: threshold {} area [{},{}] invert {} {} mm/px",
        c.sorting.bins.len(), c.sorting.assignments.len(),
        c.vision.threshold, c.vision.min_area, c.vision.max_area, c.vision.invert, c.vision.mm_per_px,
    );
}

/// Apply the operator profile to the window: the Pi keeps the 800x480 kiosk
/// size from the `.slint`; the workstation gets a larger window and the
/// `workstation` flag that gates the (future) learning UI.
fn apply_profile(ui: &AppWindow, profile: app_config::UiProfile) {
    match profile {
        app_config::UiProfile::Pi => ui.set_workstation(false),
        app_config::UiProfile::Workstation => {
            ui.set_workstation(true);
            ui.window()
                .set_size(slint::LogicalSize::new(1280.0, 800.0));
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = AppConfig::load(&args.config)
        .with_context(|| format!("loading configuration from {}", args.config))?;

    // Logging comes up as soon as the config is known — it drives the level,
    // file directory and rotation. Hold the handle for the whole run so the
    // background rotation/flush keeps working.
    let _logger = init_logging(&config.logging).context("initialising logging")?;
    info!("Starting Delta X2 Sorting System...");
    info!("Configuration loaded from {}.", args.config);
    log_config_summary(&args, &config);

    // A `--profile` flag overrides the config's `[ui].profile`; one binary,
    // two roles (kiosk sorter vs. teaching workstation).
    let profile = args.profile.unwrap_or(config.ui.profile);
    info!(
        "UI profile: {:?} ({})",
        profile,
        if args.profile.is_some() { "from --profile" } else { "from config" }
    );

    let limits = config.robot.workspace_limits();

    // --- Hardware Init ---
    let (robot, conveyor): (SharedRobot, SharedConveyor) = if args.mock {
        info!("Initializing Hardware (Mock Mode)...");
        (
            Arc::new(Mutex::new(Box::new(MockRobot::with_limits(limits)))),
            Arc::new(Mutex::new(Box::new(MockConveyor::new()))),
        )
    } else {
        info!("Initializing Hardware (Real Mode)...");
        (
            Arc::new(Mutex::new(Box::new(hardware::DeltaX2::new(
                &config.robot.port_name,
                config.robot.baud_rate,
                limits,
                config.robot.feed_rate,
                config.robot.release_gripper_on_estop,
            )))),
            Arc::new(Mutex::new(Box::new(hardware::SerialConveyor::new(
                &config.conveyor.port_name,
                config.conveyor.baud_rate,
            )))),
        )
    };

    {
        let mut r = robot.lock().await;
        r.connect()
            .await
            .with_context(|| format!("connecting to robot at {}", config.robot.port_name))?;
        if config.robot.home_on_connect {
            r.home().await.context("homing robot after connect")?;
        }
    }
    {
        let mut c = conveyor.lock().await;
        c.connect()
            .await
            .with_context(|| format!("connecting to conveyor at {}", config.conveyor.port_name))?;
    }

    // E-stop handles own dedicated serial handles and bypass the async
    // mutexes, so the UI can halt hardware even while a command is in flight.
    let robot_estop = robot.lock().await.estop_handle();
    let conveyor_estop = conveyor.lock().await.estop_handle();

    // --- Camera ---
    let cam_cfg = &config.camera;
    let mut camera: Box<dyn CameraDriver> = if args.mock {
        info!("Using Mock Camera");
        Box::new(MockCamera::new(
            cam_cfg.width,
            cam_cfg.height,
            cam_cfg.fps,
            config.vision.invert,
        ))
    } else {
        info!("Connecting to real Camera (device {})", cam_cfg.device_id);
        Box::new(
            OpencvCamera::new(
                cam_cfg.device_id,
                cam_cfg.width,
                cam_cfg.height,
                cam_cfg.fps,
                cam_cfg.fourcc.clone(),
            )
            .context("invalid camera.fourcc")?,
        )
    };
    camera
        .connect()
        .await
        .with_context(|| format!("connecting to camera device {}", config.camera.device_id))?;

    // --- Orchestrator (starts PAUSED: no pick can move the robot before
    // the operator presses Start) ---
    let (orch_tx, orch_state, orch_errors, orch) = Orchestrator::new(&config, robot.clone());
    let orch_handle = tokio::spawn(orch.run());

    // Shutdown signal for the vision loop (the orchestrator stops via its
    // Shutdown message so it can park first). Held until the end of main.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    // Actual belt run-state, mirrored to the vision loop so it only predicts
    // belt drift while the belt is really moving (no phantom picks when
    // stopped). Set true only after a successful conveyor start.
    let (belt_running_tx, belt_running_rx) = tokio::sync::watch::channel(false);

    // Live camera feed: latest-wins watch channel from the vision loop to the
    // UI (the operator only ever needs the newest frame). None until the first
    // frame is rendered.
    let (frame_tx, frame_rx) = tokio::sync::watch::channel::<Option<vision::pipeline::FrameImage>>(None);

    // Recognition is loaded here (not inside the vision loop) so the catalogue
    // handle can later be shared with the learning UI. Disabled → None.
    let recognizer = if config.recognition.enabled {
        match vision::embedder::Recognizer::load(&config.recognition) {
            Ok(r) => Some(r),
            Err(e) => {
                warn!("Recognition disabled: {e:#}");
                None
            }
        }
    } else {
        None
    };

    // Learning: only in the workstation profile, and only with recognition on.
    // Unrecognised objects flow here for the operator to label (latest-wins).
    let learning = profile == app_config::UiProfile::Workstation && recognizer.is_some();
    let (label_tx, label_rx) = if learning {
        let (tx, rx) =
            tokio::sync::watch::channel::<Option<vision::pipeline::LabelRequest>>(None);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    // --- Vision loop: owns the camera, feeds pick-ready objects to the
    // orchestrator (camera → detect → track → recognise → pixel-to-world →
    // Pick) and publishes the annotated frame for the UI ---
    let vision_handle = vision::pipeline::spawn_vision_loop(
        camera,
        &config,
        orch_tx.clone(),
        shutdown_rx,
        belt_running_rx,
        frame_tx,
        recognizer,
        label_tx,
    );

    // Learning consumer (C1): log each unrecognised object and its nearest
    // known classes. The workstation labelling panel replaces this in C2.
    if let Some(mut label_rx) = label_rx {
        tokio::spawn(async move {
            loop {
                let request = label_rx.borrow_and_update().clone();
                if let Some(req) = request {
                    let nearest: Vec<String> = req
                        .nearest
                        .iter()
                        .map(|(c, s)| format!("{c} {:.0}%", s * 100.0))
                        .collect();
                    info!(
                        "Learning: unrecognised object — nearest known: [{}]",
                        nearest.join(", ")
                    );
                }
                if label_rx.changed().await.is_err() {
                    break;
                }
            }
        });
    }

    // --- UI ---
    let ui = AppWindow::new()?;
    ui.set_robot_status("Ready (paused)".into());
    ui.set_conveyor_status("Stopped".into());
    apply_profile(&ui, profile);
    let ui_weak = ui.as_weak();

    // Mirror the orchestrator's CONFIRMED state into the UI (never what a
    // button hoped): status text + the Start interlock after an E-stop.
    {
        let ui_handle = ui_weak.clone();
        let mut orch_state = orch_state;
        tokio::spawn(async move {
            loop {
                let state = *orch_state.borrow_and_update();
                let ui_handle2 = ui_handle.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_handle2.upgrade() {
                        // Any confirmed transition means a pending Home (if
                        // any) has been carried out: the run loop always
                        // publishes state after executing Home. Clearing it
                        // here retires the "Homing…" feedback exactly once the
                        // action is real, not when the button was tapped.
                        ui.set_home_pending(false);
                        match state {
                            OrchestratorState::Paused => {
                                ui.set_estopped(false);
                                ui.set_orchestrator_running(false);
                                ui.set_robot_status("Ready (paused)".into());
                            }
                            OrchestratorState::Running => {
                                ui.set_estopped(false);
                                ui.set_orchestrator_running(true);
                                ui.set_robot_status("Sorting".into());
                            }
                            OrchestratorState::EStopped => {
                                ui.set_estopped(true);
                                ui.set_orchestrator_running(false);
                                ui.set_is_running(false);
                                ui.set_robot_status("E-STOP — HOME REQUIRED".into());
                            }
                        }
                    }
                });
                if orch_state.changed().await.is_err() {
                    break; // orchestrator gone
                }
            }
        });
    }

    // Surface orchestrator hardware failures (robot command / home) in the
    // operator banner — the log is invisible on a kiosk Pi. Only failures are
    // published, so this never clears an error the UI itself set.
    {
        let ui_handle = ui_weak.clone();
        let mut error_rx = orch_errors;
        tokio::spawn(async move {
            loop {
                let msg = error_rx.borrow_and_update().clone();
                if let Some(text) = msg {
                    let ui_handle2 = ui_handle.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_handle2.upgrade() {
                            ui.set_error_text(text.into());
                        }
                    });
                }
                if error_rx.changed().await.is_err() {
                    break; // orchestrator gone
                }
            }
        });
    }

    // Live camera feed → UI, with a staleness watchdog: no frame for
    // FEED_TIMEOUT flips the UI to a visible "FEED LOST" state rather than
    // freezing on the last image with no indication.
    {
        let ui_handle = ui_weak.clone();
        let mut frame_rx = frame_rx;
        const FEED_TIMEOUT: Duration = Duration::from_secs(1);
        tokio::spawn(async move {
            loop {
                match tokio::time::timeout(FEED_TIMEOUT, frame_rx.changed()).await {
                    Ok(Ok(())) => {
                        let image = frame_rx.borrow_and_update().clone();
                        if let Some(buffer) = image {
                            let ui_handle2 = ui_handle.clone();
                            let _ = slint::invoke_from_event_loop(move || {
                                if let Some(ui) = ui_handle2.upgrade() {
                                    // Wrap-and-set only: the pixel buffer was
                                    // built in the vision/blocking thread.
                                    ui.set_camera_feed(slint::Image::from_rgb8(buffer));
                                    ui.set_feed_lost(false);
                                }
                            });
                        }
                    }
                    Ok(Err(_)) => break, // vision loop gone: stop updating
                    Err(_) => {
                        let ui_handle2 = ui_handle.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle2.upgrade() {
                                ui.set_feed_lost(true);
                            }
                        });
                    }
                }
            }
        });
    }

    // Start / Pause toggle: drives the conveyor AND the pick pipeline
    // (Resume/Pause) — sorting never starts without the operator.
    {
        let ui_handle = ui_weak.clone();
        let conveyor = conveyor.clone();
        let tx = orch_tx.clone();
        let belt_tx = belt_running_tx.clone();
        let speed = config.conveyor.default_speed as i32;
        ui.on_start_clicked(move || {
            let ui = match ui_handle.upgrade() {
                Some(ui) => ui,
                None => return,
            };
            // Debounce: ignore the tap if a conveyor command is still in flight.
            // `starting` is read here and frozen for this command so out-of-order
            // completions can't flip the state the wrong way.
            if ui.get_command_pending() {
                return;
            }
            let starting = !ui.get_is_running();
            info!("UI: {} requested", if starting { "Start" } else { "Pause" });
            ui.set_command_pending(true);
            ui.set_error_text("".into());
            let _ = tx.send(if starting {
                OrchestratorMsg::Resume
            } else {
                OrchestratorMsg::Pause
            });
            let conveyor = conveyor.clone();
            let ui_handle = ui_handle.clone();
            let belt_tx = belt_tx.clone();
            tokio::spawn(async move {
                let result = {
                    let mut c = conveyor.lock().await;
                    if starting {
                        c.start(speed).await
                    } else {
                        c.stop().await
                    }
                };
                // Belt is running only after a successful start; the vision
                // loop uses this to avoid drifting a stationary part (#28).
                let _ = belt_tx.send(result.is_ok() && starting);
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_handle.upgrade() {
                        ui.set_command_pending(false);
                        match result {
                            Ok(()) => {
                                ui.set_is_running(starting);
                                ui.set_conveyor_status(
                                    if starting { "Running" } else { "Stopped" }.into(),
                                );
                            }
                            Err(e) => {
                                let verb = if starting { "start" } else { "stop" };
                                error!("UI: conveyor {} failed: {:#}", verb, e);
                                // Reconcile conservatively: the belt is in an
                                // unknown state, so leave Stop reachable (via
                                // error-text) instead of trusting is_running.
                                ui.set_is_running(false);
                                ui.set_conveyor_status("ERROR".into());
                                ui.set_error_text(format!("Conveyor {} failed", verb).into());
                            }
                        }
                    }
                });
            });
        });
    }

    {
        let ui_handle = ui_weak.clone();
        let conveyor = conveyor.clone();
        let tx = orch_tx.clone();
        let belt_tx = belt_running_tx.clone();
        ui.on_stop_clicked(move || {
            let ui = match ui_handle.upgrade() {
                Some(ui) => ui,
                None => return,
            };
            if ui.get_command_pending() {
                return;
            }
            info!("UI: Stop requested");
            ui.set_command_pending(true);
            ui.set_error_text("".into());
            let _ = tx.send(OrchestratorMsg::Pause);
            // Orchestrator is now paused; the belt is no longer treated as
            // running so vision stops predicting drift.
            let _ = belt_tx.send(false);
            let conveyor = conveyor.clone();
            let ui_handle = ui_handle.clone();
            tokio::spawn(async move {
                let result = conveyor.lock().await.stop().await;
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_handle.upgrade() {
                        ui.set_command_pending(false);
                        match result {
                            Ok(()) => {
                                ui.set_is_running(false);
                                ui.set_conveyor_status("Stopped".into());
                            }
                            Err(e) => {
                                error!("UI: conveyor stop failed: {:#}", e);
                                // Keep Stop reachable (error-text) so the belt
                                // can be forced down on a retry.
                                ui.set_conveyor_status("ERROR".into());
                                ui.set_error_text("Conveyor stop failed".into());
                            }
                        }
                    }
                });
            });
        });
    }

    {
        let tx = orch_tx.clone();
        let ui_handle = ui_weak.clone();
        ui.on_home_clicked(move || {
            info!("UI: Home requested");
            // Dedicated recovery message: runs with priority even while paused
            // and clears the E-stopped state on success.
            if tx.send(OrchestratorMsg::Home).is_err() {
                error!("UI: orchestrator is gone; cannot home");
                if let Some(ui) = ui_handle.upgrade() {
                    ui.set_error_text("Cannot home: controller is gone".into());
                }
                return;
            }
            // Immediate feedback: show "Homing…" and lock the button until the
            // orchestrator confirms completion (state watch clears it). Also
            // clear any stale banner — the operator is taking a recovery action.
            if let Some(ui) = ui_handle.upgrade() {
                ui.set_home_pending(true);
                ui.set_error_text("".into());
            }
        });
    }

    // Manual gripper toggle: a recovery action so a part held after a failed
    // pick or an E-stop can be released by hand. Routed through the
    // orchestrator so all robot I/O stays single-owner; state is optimistic.
    {
        let tx = orch_tx.clone();
        let ui_handle = ui_weak.clone();
        ui.on_gripper_toggle(move |on| {
            info!("UI: gripper {} requested", if on { "engage" } else { "release" });
            if let Some(ui) = ui_handle.upgrade() {
                ui.set_error_text("".into());
            }
            if tx.send(OrchestratorMsg::SetGripper(on)).is_err() {
                error!("UI: orchestrator is gone; cannot toggle gripper");
                if let Some(ui) = ui_handle.upgrade() {
                    ui.set_error_text("Cannot toggle gripper: controller is gone".into());
                }
                return;
            }
            if let Some(ui) = ui_handle.upgrade() {
                ui.set_gripper_on(on);
            }
        });
    }

    {
        let tx = orch_tx.clone();
        let ui_handle = ui_weak.clone();
        let belt_tx = belt_running_tx.clone();
        ui.on_estop_clicked(move || {
            warn!("UI: EMERGENCY STOP TRIGGERED!");
            // 1. Halt hardware immediately, bypassing queues and locks.
            if let Some(handle) = &robot_estop {
                handle.trigger();
            }
            if let Some(handle) = &conveyor_estop {
                handle.trigger();
            }
            // 2. Drop all queued work and pause the orchestrator; the state
            // watcher will flip the UI to E-STOP / lock Start until re-home.
            let _ = tx.send(OrchestratorMsg::EStop);
            // Belt halted: vision must not predict drift.
            let _ = belt_tx.send(false);
            if let Some(ui) = ui_handle.upgrade() {
                ui.set_is_running(false);
                // Clear any in-flight command gate so Start isn't left locked
                // after a successful re-home.
                ui.set_command_pending(false);
                ui.set_error_text("".into());
                ui.set_conveyor_status("E-STOP".into());
            }
        });
    }

    info!("UI: Running event loop...");
    ui.run()?;

    // --- Graceful shutdown ---
    // Closing the window must not leave the arm at pick height with the
    // vacuum on and the camera still open. Order: stop the belt, release the
    // camera, let the orchestrator finish its current command and park, then
    // wait for both tasks so the runtime is not dropped mid-command.
    const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(20);
    info!("Shutting down: stopping conveyor.");
    if let Err(e) = conveyor.lock().await.stop().await {
        warn!("Conveyor stop on shutdown failed: {:#}", e);
    }
    // Signal the vision loop to release the camera.
    let _ = shutdown_tx.send(true);
    // Ask the orchestrator to finish in flight, park, and exit. (Does not rely
    // on all senders dropping — UI callbacks still hold clones at this point.)
    if orch_tx.send(OrchestratorMsg::Shutdown).is_err() {
        warn!("Orchestrator already gone at shutdown");
    }
    // Wait for both tasks, but never hang the exit: a wedged robot is bounded
    // by the timeout, after which the runtime drop aborts whatever remains.
    match tokio::time::timeout(SHUTDOWN_TIMEOUT, async {
        let _ = orch_handle.await;
        let _ = vision_handle.await;
    })
    .await
    {
        Ok(()) => info!("Orchestrator and vision loop stopped cleanly."),
        Err(_) => warn!(
            "Shutdown timed out after {}s; exiting anyway.",
            SHUTDOWN_TIMEOUT.as_secs()
        ),
    }
    info!("System shut down.");
    Ok(())
}
