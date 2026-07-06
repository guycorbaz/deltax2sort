# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Also read `_bmad-output/project-context.md`** — the maintained rules file for AI agents (safety invariants, stack constraints, testing/workflow rules). Its Safety Invariants section overrides everything else.

## Project Overview

Rust control application for a Delta X2 delta robot that sorts LEGO bricks off a conveyor belt using a USB camera and OpenCV vision. GUI is built with Slint. Development follows the phased plan in `docs/implementation_plan.md`; open work items are tracked as GitHub issues (https://github.com/guycorbaz/deltax2sort/issues) — file/close issues as you complete work.

## Commands

```bash
cargo build                 # Build (compiles ui/app_window.slint via build.rs)
cargo test                  # Unit tests: config validation, calibration, planner, limits, queue
cargo run -- --mock         # Run with mock robot/conveyor/camera (no hardware needed)
cargo run                   # Run against real hardware (serial ports + camera)
RUST_LOG=debug cargo run -- --mock            # RUST_LOG overrides [logging].level (debug traces G-code)
cd docs && latexmk -pdf manual.tex   # Build the operations manual (PDF)
```

**System dependencies** (Ubuntu): `clang libclang-dev llvm-dev libudev-dev libopencv-dev pkg-config`. If the system only ships a versioned `libclang-N.so.1` (as after OS upgrades), the build needs `LIBCLANG_PATH=$PWD/libs` with the `libs/libclang.so` symlink pointing at it. Serial access requires membership in the `dialout` group.

## Architecture

Three layers wired together in `src/main.rs` (hardware init, Slint UI callbacks, orchestrator spawn):

- **Hardware layer (`src/hardware.rs`)** — async traits `RobotController`, `ConveyorController`, `CameraDriver`, each with a real and a mock implementation selected by `--mock`; new hardware features must work through the traits so mock mode keeps working. Handles are shared as `Arc<tokio::Mutex<Box<dyn Trait>>>`. Serial I/O is synchronous (`serialport`) and always runs inside `spawn_blocking`. `WorkspaceLimits` is enforced inside `move_to` of *both* robot drivers — an out-of-bounds target errors before any G-code is sent.
- **Vision layer (`src/vision/`)** — `detector.rs` (blob detection, parameters from `[vision]` config), `calibration.rs` (affine pixel→robot transform incl. rotation; z is always 0, pick height comes from config), `tracker.rs` and `classifier.rs` (still stubs). Not yet wired into a running loop — see TODO.
- **Logic layer (`src/orchestrator.rs`)** — the `Orchestrator` consumes `OrchestratorMsg` (Pick/Command/Pause/Resume/EStop) from an **unbounded mpsc channel**; the sender returned by `Orchestrator::new` is the only way to talk to it. It executes commands one at a time, never holds the robot lock across a `Wait`, and on command failure clears its queue and pauses (requires `Resume`). Queued picks are **atomic groups** carrying their detection expiry (`seen_at + PICK_TTL`): `InstructionQueue::next_command` drops a whole group whose TTL passed *before it starts* (belt carried the object away), but a group already in flight always runs to completion — never abort mid-sequence (the gripper may hold a part). `TrajectoryPlanner::calculate_intercept` uses *signed* belt speed (`conveyor.speed_mm_s`, positive = toward +Y) and returns `None` for objects already past the pick line.

**Safety invariants to preserve when changing code:**
1. Config is validated at startup (`AppConfig::validate`) — nothing that can command the robot outside `[z_min, z_max]`/workspace may pass validation.
2. The E-stop path must stay preemptive: `EmergencyStop` handles own *cloned* serial ports (`try_clone`) and are triggered synchronously from the UI callback, bypassing the tokio mutexes and the orchestrator queue. Robot halt is `M112`, conveyor halt `M5`. When `robot.release_gripper_on_estop` is set, the robot halt write is `M05\nM112` (gripper opens as part of the same preemptive write, before the halt) — keep any gripper release on the E-stop path fire-and-forget like this; never a blocking `set_gripper` after `M112` (it would wait out the 30 s feedback deadline and stall Home recovery).
3. `DeltaX2::write_gcode` appends a unique `FEEDBACK:sync_<n>` id and blocks until the echo (= physical completion), with EOF detection and a 30 s overall deadline. Don't reintroduce unbounded waits. The feedback wait also polls a shared `estop_flag` (`AtomicBool`): the E-stop handles/`stop()` raise it so a command in flight when `M112` fires aborts within one serial read window instead of burning the 30 s deadline; `home()` clears it before `G28` (the one command the firmware accepts after `M112`). Keep this: don't let a non-`home` command clear the flag, or recovery could stall again.

Configuration lives in `Settings.toml` (`src/app_config.rs`): `[robot]` (ports, workspace, z_pick/z_travel, feed_rate), `[conveyor]` (port, default_speed raw S-value, signed speed_mm_s), `[camera]`, `[sorting]` (drop position), `[vision]` (threshold/areas/invert), `[logging]` (level/directory/to_console/keep_days — daily-rotating file log via flexi_logger, set up in `main::init_logging`). New fields need serde defaults so old config files keep parsing (there's a test for that).

## Delta X2 protocol

Authoritative reference: Appendix "Delta X2 G-code Protocol Reference" in the manual (`docs/manual/gcode.tex`). Key facts: handshake `IsDelta`→`YesDelta`; `FEEDBACK:<id>` echo marks physical completion; workspace X/Y ∈ [-160, 160] mm, Z ∈ [-200, 0] (Z0 = top, homing `G28` goes to top center); gripper `M03`/`M05`; E-stop `M112`.

## Documentation

- `docs/manual.tex` + `docs/manual/*.tex` — the operations manual (LaTeX book, one file per chapter: preamble, titlepage, overview, installation, configuration, operation, maintenance, gcode appendix; has an index via imakeidx). Update the relevant chapter when behavior/config changes, and rebuild the PDF.
- GitHub issues (https://github.com/guycorbaz/deltax2sort/issues) — the tracker of record for remaining work; the go-to place to see what is stubbed vs. real. `docs/TODO.md` is only a pointer.
- `docs/specifications.md`, `docs/implementation_plan.md`, `docs/walkthrough.md` — original requirements, phase plan and walkthrough (single canonical copies under `docs/`).
