# Delta X LEGO Sorting Robot - System Specifications

## 1. System Overview
The system consists of a Delta X robot arm, a conveyor belt, and a USB camera. The goal is to sort LEGO bricks into 4 distinct categories based on color and/or shape. Unknown items will pass through the conveyor to a reject/overflow container. The control application will be written in Rust with a Slint GUI.

## 2. Hardware Interfaces

### 2.1 Delta X Robot
- **Model**: **Delta X2**.
- **Communication**: USB Serial (typically `/dev/ttyUSBx` or `/dev/ttyACMxyz`).
- **Protocol**: G-code.
    - Baud Rate: 115200 (standard for Delta X).
    - Commands: `G28` (Home), `G1 X... Y... Z...` (Move), `M3`/`M5` (Pump/Gripper control - *to be confirmed*).

### 2.2 Conveyor Belt
- **Communication**: Separate USB Serial connection.
- **Protocol**: *Unknown/Custom*.
    - **Architecture**: Likely a simple serial command interface (e.g., Arduino-based or industrial controller).
    - **Requirement**: Application must support a configurable generic serial driver (Baud rate, Start/Stop command strings).

### 2.3 USB Camera
- **Input**: Video stream of the conveyor belt upstream of the robot.
- **Resolution/FPS**: High resolution preferred for shape details (1080p or 720p). Global Shutter camera recommended to reduce motion blur.
- **Library**: `opencv` (Rust `opencv` crate) and potentially `onnxruntime` for ML inference.
- **Environment**: **Fixed Lighting** is required (e.g., LED strips/ring) to ensure consistent color detection and shadow minimization.

### 2.4 Safety & Standard Compliance
- **Emergency Stop**: Software must support an immediate Stop command (E-Stop) triggered by UI or physical button (if connected).
- **Collision/Jam Protection**: System should detect command timeouts or unexpected stalls and enter a "Resulting Error" state.

## 3. Software Architecture (Rust)

The application will be multi-threaded/async to handle real-time vision and robot control simultaneously.

### 3.1 Vision & Learning Subsystem (Advanced)
- **Responsibility**: Object detection, Feature Extraction, Classification, and "Active Learning".
- **Workflow**:
    1.  **Detection**: Locate object on belt, extract bounding box image.
    2.  **Inference**: Pass image to Classifier (ML Model).
    3.  **Confidence Check**:
        -   **High Confidence**: Sort to assigned box.
        -   **Low Confidence / Unknown**: Let object pass to "Reject/Later" bin.
        -   **Data Collection**: Save image of unknown object to "Labeling Queue".
- **Human-in-the-Loop**:
    -   Operator can view "Unknown" images in UI.
    -   Operator assigns Label (e.g., "3001 - 2x4 Brick Red").
    -   **Standardization**: Integrate with **Rebrickable/BrickLink** Part IDs to standardize naming.
    -   System retrains or updates reference database.

### 3.2 Robot Control Subsystem
- **Responsibility**: Manage Serial connection to Delta X.
- **Queue**: Receive `PickTask`s and execute moves.
- **Performance**:
    -   Target **Picks Per Minute (PPM)**: Optimization for high throughput.
    -   **Look-ahead**: Scheduler should optimize pathing to assume future positions of multiple bricks.
- **Logic**:
    1.  Move to `PickPosition` (calculated from Vision + **Visual Odometry** for belt speed compensation).
    2.  Actuate Gripper (Suction on).
    3.  Move to `DropPosition` (Box 1-4).
    4.  Actuate Gripper (Suction off).
    5.  Return to Home/Wait.

### 3.3 Main Orchestrator
- **Responsibility**: Coordinate Vision and Robot.
- **Sorting Logic**:
    -   Receive detection.
    -   Check if it matches one of the 4 active filters.
    -   Calculate "Time to Intercept".
    -   Schedule Pick Task if reachable.

### 3.4 User Interface (Slint)
- **Live View**:
    -   **Video Feed**: Real-time display of the conveyor belt.
    -   **Overlays**: Draw **GREEN** bounding boxes around recognized/sortable bricks. Draw **RED** or **YELLOW** boxes around unknown items.
- **Status Dashboard**:
    -   Robot Status (Idle/Busy/Alarm).
    -   Conveyor Status.
    -   Counts per category.
- **Sorting Session Config**:
    -   **Batch Setup**: Select between 1 and 6 Lego types to sort in the current run.
    -   **Assignment**: Map "Lego Type A (Part #3001)" -> "Position 1".
- **Learning Interface**:
    -   **Notification**: "Unknown Object Detected".
    -   **Labeling Tool**: Show cropped image, ask user to select category or create new one (Searchable by BrickLink ID).
- **Controls**:
    -   **Emergency STOP**: Large, prominent Red button to immediately halt robot and conveyor.
    -   Start/Stop System.
    -   **Calibration Wizard**: Tool to calibrate Camera<->Robot mapping.
    -   **Sorting Config**: Select which color/shape goes to Box 1, 2, 3, 4.

## 4. Dependencies (Tentative)
- **GUI**: `slint`
- **Serial**: `serialport`
- **Vision**: `opencv`
- **ML/AI**: `orts` (ONNX Runtime) or `burn` (Rust Deep Learning) or `smartcore`.
- **Async Runtime**: `tokio` (for managing IO/Threads).
- **Config**: `serde`, `toml` (for saving calibration/settings).

## 5. Open Questions for User
1.  **Conveyor Control**: Is the conveyor plugged into the Robot's control box (Aux port), or is it a separate USB device? If separate, what is the protocol?
2.  **Gripper**: Is it a suction cup (Air Pump)? How is it triggered? (Usually `M3`/`M5` or digital IO).
3.  **Camera**: Do you have a specific camera model? (Standard UVC is assumed).
4.  **Sorting Criteria**: Simple color blobs, or do we need ML/Shape matching for specific LEGO shapes (e.g. 2x4 vs 2x2)?
