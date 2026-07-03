# Development Guide - Delta X2 LEGO Sorting System

This guide covers the system requirements, build process, and project structure for developers working on the Delta X2 LEGO Sorting System.

## 1. System Requirements

To compile and run this project, several non-Rust system libraries are required for hardware communication and computer vision.

### OS Support

- **Linux (Ubuntu 24.04 recommended)**: Fully supported.
- **Other OS**: Not officially tested.

### Required Software Packages

Run the following command to install all necessary dependencies on Ubuntu:

```bash
sudo apt update && sudo apt install -y \
    clang \
    libclang-dev \
    llvm-dev \
    libudev-dev \
    libopencv-dev \
    pkg-config
```

#### Why these are needed

- **`clang` & `libclang-dev`**: Required by the `opencv` crate to generate Rust bindings from C++ headers.
- **`libudev-dev`**: Required by the `serialport` crate for USB/Serial device discovery on Linux.
- **`libopencv-dev`**: The core computer vision library.
- **`pkg-config`**: Helps the Rust build scripts locate the installed libraries.

## 2. Building the Project

Ensure you have the latest stable Rust toolchain installed.

```bash
# Clone the repository (if applicable)
# git clone <repo_url>
# cd deltax2sort

# Build the project
cargo build

# Run in Mock Mode (No hardware required)
cargo run -- --mock

# Run with Real Hardware
cargo run
```

## 3. Command Line Arguments

The application supports the following arguments:

- `-m`, `--mock`: Run with simulated robot, conveyor, and camera (default: false).
- `-c`, `--config <PATH>`: Specify a custom configuration file (default: `Settings.toml`).
- `-h`, `--help`: Print help information.

## 4. Configuration (`Settings.toml`)

The system uses a TOML file for hardware calibration and connection settings. If `Settings.toml` does not exist, the application will create a default one on the first run.

Key sections:

- `[robot]`: Port name, baud rate, and physical coordinate limits.
- `[conveyor]`: Serial port and speed settings.
- `[camera]`: Device ID and resolution.

## 5. Project Structure

- `src/main.rs`: Entry point, UI initialization, and background task management.
- `src/hardware.rs`: Drivers for Delta X2 robot, Conveyor, and Camera.
- `src/vision/`: Object detection, tracking, and calibration logic.
- `src/orchestrator.rs`: Trajectory planning and command queueing.
- `ui/app_window.slint`: The Slint-based graphical user interface.
- `docs/`: Detailed planning and specifications.

## 6. Troubleshooting

### Build fails on `opencv` or `clang-sys`

Ensure `libclang-dev` and `libopencv-dev` are installed. If the error mentions `llvm-config`, ensure `llvm-dev` is installed.

### Serial Port Permission Denied

On Linux, you may need to add your user to the `dialout` group to access serial ports:

```bash
sudo usermod -a -G dialout $USER
# Note: You must log out and back in for this to take effect.
```

### Slint UI issues

If the UI fails to compile, check the `build.rs` and ensured `slint-build` is listed in `[build-dependencies]` in `Cargo.toml`.
