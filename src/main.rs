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
}

type SharedRobot = Arc<Mutex<Box<dyn RobotController>>>;
type SharedConveyor = Arc<Mutex<Box<dyn ConveyorController>>>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    info!("Starting Delta X2 Sorting System...");

    let args = Args::parse();
    let config = AppConfig::load(&args.config)
        .with_context(|| format!("loading configuration from {}", args.config))?;
    info!("Configuration loaded from {}.", args.config);

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
        Box::new(MockCamera::new(cam_cfg.width, cam_cfg.height, cam_cfg.fps))
    } else {
        info!("Connecting to real Camera (device {})", cam_cfg.device_id);
        Box::new(OpencvCamera::new(
            cam_cfg.device_id,
            cam_cfg.width,
            cam_cfg.height,
            cam_cfg.fps,
            cam_cfg.fourcc.clone(),
        ))
    };
    camera
        .connect()
        .await
        .with_context(|| format!("connecting to camera device {}", config.camera.device_id))?;

    // --- Orchestrator (starts PAUSED: no pick can move the robot before
    // the operator presses Start) ---
    let (orch_tx, orch_state, orch_errors, orch) = Orchestrator::new(&config, robot.clone());
    tokio::spawn(orch.run());

    // --- Vision loop: owns the camera, feeds pick-ready objects to the
    // orchestrator (camera → detect → track → pixel-to-world → Pick) ---
    vision::pipeline::spawn_vision_loop(camera, &config, orch_tx.clone());

    // --- UI ---
    let ui = AppWindow::new()?;
    ui.set_robot_status("Ready (paused)".into());
    ui.set_conveyor_status("Stopped".into());
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
                        match state {
                            OrchestratorState::Paused => {
                                ui.set_estopped(false);
                                ui.set_robot_status("Ready (paused)".into());
                            }
                            OrchestratorState::Running => {
                                ui.set_estopped(false);
                                ui.set_robot_status("Sorting".into());
                            }
                            OrchestratorState::EStopped => {
                                ui.set_estopped(true);
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

    // Start / Pause toggle: drives the conveyor AND the pick pipeline
    // (Resume/Pause) — sorting never starts without the operator.
    {
        let ui_handle = ui_weak.clone();
        let conveyor = conveyor.clone();
        let tx = orch_tx.clone();
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
            tokio::spawn(async move {
                let result = {
                    let mut c = conveyor.lock().await;
                    if starting {
                        c.start(speed).await
                    } else {
                        c.stop().await
                    }
                };
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
            // Clear any stale banner: the operator is taking a recovery action.
            if let Some(ui) = ui_handle.upgrade() {
                ui.set_error_text("".into());
            }
            // Dedicated recovery message: runs even while paused and clears
            // the E-stopped state on success.
            if tx.send(OrchestratorMsg::Home).is_err() {
                error!("UI: orchestrator is gone; cannot home");
                if let Some(ui) = ui_handle.upgrade() {
                    ui.set_error_text("Cannot home: controller is gone".into());
                }
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

    info!("Shutting down: stopping conveyor.");
    if let Err(e) = conveyor.lock().await.stop().await {
        warn!("Conveyor stop on shutdown failed: {:#}", e);
    }
    info!("System shut down.");
    Ok(())
}
