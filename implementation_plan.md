# Implementation Plan - Delta X2 LEGO Sorting System

# Goal Description
Develop a Rust-based control application for a Delta X2 robot to sort LEGO bricks. The system allows for granular, incremental development.

## User Review Required
> [!IMPORTANT]
> **Hardware Protocol Verification**: Verify physical connections and protocols (Conveyor Serial, Robot G-code) before Step 5.

## Proposed Changes

### Phase 1: Foundation & Hardware Abstraction
Goal: Get the system running and talking to hardware (or mocks).

#### Step 1: Project Initialization
- Initialize Rust project `deltax2sort`.
- Add core dependencies: `tokio`, `log`, `anyhow`, `serde`.

#### Step 2: Configuration Module
- [config.rs] Define `AppConfig` structs for Camera, Robot (Port, Baud, Limits), and Conveyor.
- [config.rs] Implement TOML loading/saving.
- **Verify**: Unit test config parsing.

#### Step 3: Robot Interface Definition
- [hardware.rs] Define `RobotController` trait (async methods: `home`, `move_to`, `grip`).
- [hardware.rs] Implement `MockRobot` struct for testing without hardware.

#### Step 4: Robot Serial Implementation
- [hardware.rs] Implement `DeltaX2` struct using `serialport`.
- Implement G-code formatting (`G1`, `M3`).
- **Verify**: Run simple binary to home robot.

#### Step 5: Conveyor Interface
- [hardware.rs] Define `ConveyorController` trait.
- [hardware.rs] Implement `SerialConveyor` and `MockConveyor`.
- **Verify**: Send Start/Stop commands to belt.

#### Step 6: Camera Abstraction
- [hardware.rs] Define `CameraDriver` trait (`get_frame()`).
- [hardware.rs] Implement `OpencvCamera` using `opencv` crate.
- **Verify**: Capture and save a single image to disk.

### Phase 2: Vision System
Goal: Process images into "Tracked Objects".

#### Step 7: Vision Data Structures
- [vision/mod.rs] Define `Frame`, `DetectedObject`, `ObjectLabel` structs.

#### Step 8: Calibration Logic
- [vision/calibration.rs] Implement `CoordinateTransformer` struct.
- Implement `pixel_to_world` math (Affinity transform or Homography).
- **Verify**: Unit test strict coordinate conversion.

#### Step 9: Blob Detector
- [vision/mod.rs] Implement `detect_blobs(image) -> Vec<Rect>`.
- Use OpenCV basic thresholding + contour finding.
- **Verify**: Test on saved images from Step 6.

#### Step 10: Visual Odometry (Tracker part 1)
- [vision/tracker.rs] Implement `calculate_belt_offset(frame1, frame2) -> delta_y`.
- Use optical flow or template matching on belt texture.

#### Step 11: Object Tracker
- [vision/tracker.rs] Implement `TrackedObject` state machine (New -> Verified -> Lost).
- correlated detections across frames using Odometry offset.

#### Step 12: Classifier Infrastructure
- [vision/classifier.rs] Define `Classifier` trait.
- Implement `MockClassifier` (returns "Unknown").
- Implement `BrickLinkClient` stub (for future API access).

### Phase 3: Brain & Orchestration
Goal: Decide what to pick and when.

#### Step 13: Look-Ahead Trajectory Planner
- [orchestrator.rs] Implement math: `calculate_intercept(robot_pos, object_pos, belt_speed) -> pick_time`.
- Optimization: Generic heuristic to pick "closest reachable" brick.

#### Step 14: Command Queue
- [orchestrator.rs] Implement `InstructionQueue` (Priority Queue).
- Needs `pause`, `clear`, and `resume` methods for Safety events.

#### Step 15: Safety Monitor
- [orchestrator.rs] Implement `SafetyGuard` that monitors command timeouts.
- Implement E-Stop logic (clears queue, halts hardware).

#### Step 16: Main Loop Wiring
- [main.rs] Spawn tasks: Vision Loop, Orchestrator Loop, Hardware Keepalives.
- Connect via Channels (`tokio::sync::mpsc`).

### Phase 4: User Interface (Slint)
Goal: Operator control.

#### Step 17: Basic UI Layout
- [ui/app.slint] Create window with `start_btn`, `stop_btn`, `estop_btn`.
- Define `MainWindow` struct in Rust.

#### Step 18: Live Video Feed
- [ui/app.slint] Add `Image` component.
- [ui/main.rs] Convert `cv::Mat` to `slint::Image` and push to UI.

#### Step 19: Overlays
- [ui/app.slint] Add support for draw Recatngles over video.
- Pass `Vec<BoundingBox>` from Rust to Slint.

#### Step 20: Learning Interface
- [ui/app.slint] Create "Unknown Object" Popup.
- [ui/main.rs] Handle user input to save image with Label.

## Verification Plan

### Automated
- **Component Tests**: Run `cargo test` after Steps 2, 8, 9, 10, 13.
- **Integration Test**: Run `Step 16` with Mock hardware to verify loop stability.

### Manual
- **Hardware Check**: Step 4, 5, 6 (Robot moves, Belt moves, Camera sees).
- **Calibration**: Step 8 (Check math accuracy).
- **Full Run**: Step 21 (Sort 1 brick).
