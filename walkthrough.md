# Delta X2 Sorting System - Walkthrough (Phases 1-3)

The system is currently capable of:
1.  **Connecting to Hardware**: Robot (G-code/Serial), Conveyor (Serial), and Camera (OpenCV).
    *   *Note: Pass `--mock` to run with simulated hardware (real devices are used by default).*
2.  **Vision Processing**: Capturing frames, detecting blobs (thresholding), and transforming pixel coordinates to robot coordinates.
3.  **Orchestration**: Analyzing detections and scheduling a "Pick and Place" sequence.

## Architecture

### 1. Hardware Layer (`src/hardware.rs`)
Defines traits `RobotController`, `ConveyorController`, `CameraDriver`.
- **DeltaX2**: Implements G-code over Serial.
- **SerialConveyor**: Implements `M3/M5` over Serial.
- **OpencvCamera**: Wraps `opencv::videoio`.

### 2. Vision Layer (`src/vision/`)
- **Detector**: OTSU Thresholding + Contour finding.
- **Calibration**: Linear Ax + B transform.
- **Tracker**: (Stub) Identifies objects over time.
- **Classifier**: (Stub) placeholders for ML.

### 3. Logic Layer (`src/orchestrator.rs`)
- **InstructionQueue**: FIFO queue for `RobotCommand`s (Move, Grip, Home).
- **Orchestrator**: Main loop that pops commands and sends them to the Robot controller.

## How to Run

```bash
# Mock mode (no hardware required)
RUST_LOG=info cargo run -- --mock

# Real hardware
RUST_LOG=info cargo run
```

### Expected Output (mock mode)
```text
INFO  - Starting Delta X2 Sorting System...
INFO  - Configuration loaded from Settings.toml.
INFO  - Initializing Hardware (Mock Mode)...
INFO  - MockRobot: Connecting...
INFO  - MockRobot: Connected!
INFO  - MockRobot: Homing (G28)...
INFO  - MockRobot: Homed.
INFO  - MockConveyor: Connected
INFO  - Using Mock Camera
INFO  - MockCamera: Connected
INFO  - Orchestrator loop started
INFO  - UI: Running event loop...
```

The operator window then opens; Start/Pause and Stop drive the (mock)
conveyor, Home queues a homing command, and EMERGENCY STOP halts the
hardware and clears the orchestrator queue.

For the full manual (installation, configuration reference, operation),
see `documentation/manual.pdf`. Open work items: `documentation/TODO.md`.
