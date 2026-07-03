# Implementation Plan - Delta X2 LEGO Sorting System

Comprehensive roadmap for developing a Rust-based LEGO sorting application using a Delta X2 robot, OpenCV, and Slint.

## Status Overview

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Foundation & Hardware Abstraction | 100% Complete |
| **Phase 2** | Vision & Object Tracking | 40% Complete |
| **Phase 3** | Brain & Orchestration | 30% Complete |
| **Phase 4** | User Interface (Slint) | 30% Complete |

---

## Proposed Changes

### Phase 1: Foundation & Hardware Abstraction

Goal: Establish communication with hardware and provide mock environments for testing.

#### Step 1: Project Initialization

- [x] Initialize Rust project `deltax2sort`.
- [x] Configure `Cargo.toml` with `tokio`, `opencv`, `serialport`, `serde`, and `anyhow`.

#### Step 2: Configuration Management

- [x] Implement `AppConfig` struct in `app_config.rs`.
- [x] Implement TOML loading and saving logic with default fallbacks.
- [x] Add CLI argument support for specifying config paths (e.g., `--config custom.toml`).

#### Step 3: Robot Control Interface

- [x] Define `RobotController` trait in `hardware.rs`.
- [x] Implement `MockRobot` for hardware-less testing.
- [x] Implement `DeltaX2` struct with serial G-code communication.
- [x] Implement basic robot commands: `home`, `move_to`, `set_gripper`.
- [x] Implement command acknowledgment (waiting for "ok" from firmware).

#### Step 4: Conveyor & Camera Drivers

- [x] Define `ConveyorController` trait and implement `MockConveyor` & `SerialConveyor`.
- [x] Define `CameraDriver` trait and implement `MockCamera` & `OpencvCamera`.
- [x] Implement frame acquisition from OpenCV capture device.

---

### Phase 2: Vision System

Goal: Convert raw camera pixels into tracked, classified physical objects.

#### Step 5: Data Structures & Geometry

- [x] Define `DetectedObject` and `ObjectClass` in `vision/mod.rs`.
- [/] Implement `CoordinateTransformer` in `vision/calibration.rs`.
  - [x] Basic pixel-to-world linear scaling.
  - [ ] Add support for affine transformation (rotation/skew).
  - [ ] Implement a calibration routine (detecting 4 known points).

#### Step 6: Object Detection (The "Eyes")

- [/] Refine `BlobDetector` in `vision/detector.rs`.
  - [x] Implement grayscale conversion and Gaussian blur.
  - [x] Implement thresholding and contour finding.
  - [ ] Optimize threshold parameters for LEGO bricks against the belt.
  - [ ] Filter by aspect ratio and circularity to reduce noise.

#### Step 7: Object Tracking (The "Memory")

- [/] Enhance `Tracker` in `vision/tracker.rs`.
  - [x] Basic detection list management.
  - [ ] Implement IOU (Intersection over Union) matching between frames.
  - [ ] Implement velocity estimation (belt speed integration).
  - [ ] Add "persistence" (don't drop objects immediately on one missed frame).

#### Step 8: Classification

- [/] Implement `Classifier` in `vision/classifier.rs`.
  - [x] Define trait and mock implementation.
  - [ ] (Future) Integrate ONNX/TensorFlow Lite for deep learning classification.
  - [ ] Implement color detection (Hue histogram).

---

### Phase 3: Brain & Orchestration

Goal: Orchestrate robot movements based on vision data and safety rules.

#### Step 9: Trajectory Planning

- [/] Refine `TrajectoryPlanner` in `orchestrator.rs`.
  - [x] Naive intercept calculation (static Y).
  - [ ] Implement dynamic intercept based on real-time belt speed.
  - [ ] Add boundary checks (don't try to pick outside robot work area).

#### Step 10: Orchestrator Loop

- [/] Build professional `Orchestrator` loop.
  - [x] Command queue with priority (E-Stop vs. Pick).
  - [ ] Decouple Vision acquisition from Orchestration using Channels.
  - [ ] Handle "Robot Busy" states efficiently.

#### Step 11: Safety & Stability

- [ ] Implement `SafetyMonitor`.
  - [ ] Detect hardware timeouts/serial disconnects.
  - [ ] Soft limits for robot movement.
  - [ ] Heartbeat signal to hardware.

---

### Phase 4: User Interface (Slint)

Goal: Provide a premium dashboard for monitoring and control.

#### Step 12: Slint Integration

- [ ] Add `slint` to `Cargo.toml`.
- [ ] Create `ui/app.slint` with modern design (dark mode, glassmorphism).
- [ ] Implement state synchronization between Rust backend and Slint frontend.

#### Step 13: Live Monitoring

- [ ] Display live camera feed with tracked object overlays.
- [ ] Real-time status indicators (Robot: Connected, Belt: Running).
- [ ] Graph of sorting throughput (bricks per minute).

#### Step 14: Manual Control & Config

- [ ] Add "Home" and "E-Stop" buttons.
- [ ] UI for adjusting calibration parameters live.
- [ ] Manual conveyor speed control.

---

## Verification Plan

### Automated Tests

- [ ] `cargo test` for `app_config` (loading/saving).
- [ ] `cargo test` for `vision::calibration` (coordinate math).
- [ ] `cargo test` for `orchestrator` (trajectory logic).

### Integration & Manual

- [ ] **Hardware Test (Robot)**: Run `examples/test_robot.rs` to verify serial movement.
- [ ] **Hardware Test (Camera)**: Run `examples/test_vision.rs` to see raw feed.
- [ ] **End-to-End Mock Run**: Run `main.rs` with mock flags to verify loop logic.
- [ ] **Full Hardware Run**: Sort 10 bricks and measure success rate.
