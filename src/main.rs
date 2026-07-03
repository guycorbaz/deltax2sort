// Vision/orchestration building blocks are compiled before they are wired
// into the main loop (phased development, see implementation_plan.md);
// dead_code stays allowed until the vision loop lands.
#![allow(dead_code)]

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
use orchestrator::{Orchestrator, OrchestratorMsg, RobotCommand};
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

    // --- Camera (the frame/vision loop is not wired yet; connecting here
    // validates the configuration and device availability) ---
    let mut camera: Box<dyn CameraDriver> = if args.mock {
        info!("Using Mock Camera");
        Box::new(MockCamera::new())
    } else {
        info!(
            "Connecting to real Camera (device {})",
            config.camera.device_id
        );
        Box::new(OpencvCamera::new(
            config.camera.device_id,
            config.camera.width,
            config.camera.height,
        ))
    };
    camera
        .connect()
        .await
        .with_context(|| format!("connecting to camera device {}", config.camera.device_id))?;

    // --- Orchestrator ---
    let (orch_tx, orch) = Orchestrator::new(&config, robot.clone());
    tokio::spawn(orch.run());

    // --- UI ---
    let ui = AppWindow::new()?;
    ui.set_robot_status("Connected".into());
    ui.set_conveyor_status("Stopped".into());
    let ui_weak = ui.as_weak();

    // Start / Pause toggle: actually drives the conveyor.
    {
        let ui_handle = ui_weak.clone();
        let conveyor = conveyor.clone();
        let speed = config.conveyor.default_speed as i32;
        ui.on_start_clicked(move || {
            let starting = ui_handle
                .upgrade()
                .map(|ui| !ui.get_is_running())
                .unwrap_or(true);
            info!("UI: {} requested", if starting { "Start" } else { "Pause" });
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
                match result {
                    Ok(()) => {
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_handle.upgrade() {
                                ui.set_is_running(starting);
                                ui.set_conveyor_status(
                                    if starting { "Running" } else { "Stopped" }.into(),
                                );
                            }
                        });
                    }
                    Err(e) => error!("UI: conveyor command failed: {:#}", e),
                }
            });
        });
    }

    {
        let ui_handle = ui_weak.clone();
        let conveyor = conveyor.clone();
        ui.on_stop_clicked(move || {
            info!("UI: Stop requested");
            let conveyor = conveyor.clone();
            let ui_handle = ui_handle.clone();
            tokio::spawn(async move {
                if let Err(e) = conveyor.lock().await.stop().await {
                    error!("UI: conveyor stop failed: {:#}", e);
                    return;
                }
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_handle.upgrade() {
                        ui.set_is_running(false);
                        ui.set_conveyor_status("Stopped".into());
                    }
                });
            });
        });
    }

    {
        let tx = orch_tx.clone();
        ui.on_home_clicked(move || {
            info!("UI: Home requested");
            if tx
                .send(OrchestratorMsg::Command(RobotCommand::Home))
                .is_err()
            {
                error!("UI: orchestrator is gone; cannot home");
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
            // 2. Drop all queued work and pause the orchestrator.
            let _ = tx.send(OrchestratorMsg::EStop);
            if let Some(ui) = ui_handle.upgrade() {
                ui.set_is_running(false);
                ui.set_robot_status("E-STOP".into());
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
