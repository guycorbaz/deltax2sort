# deltax2sort

**A small-parts sorting system — LEGO® bricks, screws, and other small objects — built around a Delta X2 delta robot, OpenCV vision and a Rust/Slint control application.**

Parts travel on a conveyor belt under a USB camera; the application detects them, converts pixel coordinates to robot coordinates, and drives the [Delta X2](https://docs.deltaxrobot.com/products/deltax2/deltax2_basic_kit/) to pick each object off the moving belt and drop it into the right bin. LEGO bricks are the first use case; the vision/classification pipeline is meant to be retargetable to any small parts (screws, nuts, …).

📖 **Project site & documentation:** https://guycorbaz.github.io/deltax2sort/ — the full operations manual is available as [PDF](https://guycorbaz.github.io/deltax2sort/manual.pdf).

## Features

- **Delta X2 control over G-code** — handshake, motion, gripper, homing; every command blocks until the robot confirms *physical* completion (`FEEDBACK` echo) with a hard deadline.
- **Safety first** — configuration validated at startup (nothing can command the robot outside its workspace), workspace limits enforced in the driver itself, and a preemptive emergency stop that bypasses all queues and locks (`M112` robot / `M5` conveyor).
- **OpenCV vision** — blob detection with configurable thresholds, affine pixel→robot calibration including rotation. (The full camera→pick pipeline is the current work in progress — see [docs/TODO.md](docs/TODO.md).)
- **Touch-first GUI in Slint** — designed for the official Raspberry Pi 7" Touch Display (800×480).
- **Full mock mode** — `cargo run -- --mock` runs the complete application with simulated robot, conveyor and camera; no hardware needed for development.

## Hardware

| Component | Details |
|---|---|
| Robot | [Delta X2 basic kit](https://docs.deltaxrobot.com/products/deltax2/deltax2_basic_kit/), USB serial |
| Conveyor | Belt with G-code speed controller, USB serial |
| Camera | USB camera above the belt — model not chosen yet; resolution, frame rate and pixel format are configurable in `Settings.toml` |
| Target computer | Raspberry Pi 4 or 5 (ARM64 Linux) |
| Operator display | Official Raspberry Pi 7" Touch Display, 800×480 |

Development runs on any x86_64 Linux machine using mock mode.

## Building

System packages (Ubuntu/Debian):

```bash
sudo apt update && sudo apt install -y \
    clang libclang-dev llvm-dev \
    libudev-dev libopencv-dev pkg-config
```

Rust ≥ 1.85 (the crate uses the 2024 edition), installed via [rustup](https://rustup.rs). Then:

```bash
cargo build                 # build (also compiles the Slint UI)
cargo test                  # unit tests (no hardware required)
cargo run -- --mock         # run with simulated hardware
cargo run                   # run against the real robot/conveyor/camera
```

Serial access requires membership in the `dialout` group. See the [Installation chapter of the manual](https://guycorbaz.github.io/deltax2sort/manual.pdf) for details.

## Configuration

Everything lives in `Settings.toml`, validated at startup: robot serial port and workspace bounds, pick/travel heights, conveyor speed (signed, in mm/s), camera parameters (`device_id`, `width`/`height`, `fps`, optional `fourcc`), drop position and vision thresholds. New fields always ship with defaults, so existing configuration files keep working across upgrades.

## Documentation

- [Operations manual (PDF)](https://guycorbaz.github.io/deltax2sort/manual.pdf) — installation, configuration, operation, maintenance, and the Delta X2 G-code protocol reference. LaTeX sources under [`docs/manual/`](docs/manual/).
- [`docs/TODO.md`](docs/TODO.md) — the maintained backlog: what is real vs. still stubbed.
- [`docs/specifications.md`](docs/specifications.md) and [`docs/implementation_plan.md`](docs/implementation_plan.md) — original requirements and phased plan.

## Project status

The hardware layer (robot, conveyor, E-stop), configuration, orchestrator and the vision building blocks (detector, calibration) are implemented and unit-tested. The next milestone is wiring the vision loop — camera → detection → tracking → coordinate transform → pick — followed by the live camera feed in the UI. Object classification (ONNX, trained on a PC and deployed to the Pi as a model file) comes after — the classifier is a swappable model, so retargeting the sorter to screws or other parts is a matter of training a new model, not changing the code.

LEGO® is a trademark of the LEGO Group, which does not sponsor, authorize or endorse this project.
