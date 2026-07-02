# TODO — Remaining Work

Maintained list of open work items. Last updated: 2026-07-02, after the
hardening pass (see below). Check items off and re-date when working here.

## Completed in the 2026-07-02 hardening pass (for context)

- E-stop wired end-to-end: preemptive `EmergencyStop` handles on dedicated
  serial handles (`M112` robot / `M5` conveyor), orchestrator queue cleared.
- Workspace limits enforced in `DeltaX2::move_to` and `MockRobot::move_to`;
  configuration validated at startup (`AppConfig::validate`).
- `Settings.toml` fixed (`z_pick` was -250, outside the physical Z range).
- Orchestrator restructured around an unbounded mpsc channel with
  Pause/Resume/EStop messages; robot lock no longer held across `Wait`;
  command failure clears the queue and pauses instead of killing the loop.
- Serial I/O moved to `spawn_blocking`; feedback wait has EOF detection,
  error-line detection and an overall deadline; unique feedback ids.
- Start/Stop/Home UI buttons actually drive the conveyor / orchestrator;
  fake "bricks sorted" counter removed.
- Calibration default matched to 1280x720 (offset_y was computed for 480px);
  rotation implemented; intercept math no longer targets objects already
  past the pick line. Unit tests added for config, calibration, planner,
  limits, queue.
- Classifier stub returns Unknown/0.0 instead of Brick2x4/0.9.
- Unused `config` crate dependency removed.

## High priority — path to first sorted brick

- [ ] **Wire the vision loop** (plan step 16): spawn a task
      `camera.get_frame()` → `BlobDetector::detect` → `Tracker::update` →
      `CoordinateTransformer::pixel_to_world` → `OrchestratorMsg::Pick`.
      Everything exists as building blocks; nothing is connected.
- [ ] **Live camera feed in the UI** (plan step 18): convert `cv::Mat` to
      `slint::Image`, push via `invoke_from_event_loop`; then bounding-box
      overlays (step 19, green = known, red/yellow = unknown).
- [ ] **Robot position feedback**: implement the `Position` query
      (manual, Appendix "G-code Protocol Reference") on `RobotController` so the
      trajectory planner's `calculate_intercept` can be used by
      `schedule_pick` — picks currently target the object's *last seen*
      position without belt-motion compensation.
- [ ] **Pause/Resume in the UI**: the orchestrator already understands
      `Pause`/`Resume` messages; add buttons/state. Also define the
      recovery flow after an automatic pause (command failure) and after
      E-stop (re-home + operator confirmation).
- [ ] **SafetyGuard** (plan step 15): watchdog for command timeouts /
      stalls that triggers the E-stop path automatically.
- [ ] **Real "bricks sorted" counter**: increment when a pick sequence
      completes (the orchestrator knows); currently always 0.

## Medium priority

- [ ] **Real tracker** (plan step 11): IOU/distance matching across frames
      with the belt-odometry offset; `tracker.rs` currently re-IDs every
      object on every frame. `belt_shift_y` parameter is accepted and
      ignored.
- [ ] **Visual odometry** (plan step 10): `calculate_belt_offset(frame1,
      frame2)` via optical flow or template matching; also enables
      measuring the true `conveyor.speed_mm_s`.
- [ ] **Classifier**: real color/shape classification, later ONNX
      inference; per-class drop bins (spec: up to 6 categories, currently a
      single `[sorting]` drop position).
- [ ] **Camera calibration wizard** (spec §UI): interactive pixel↔robot
      mapping; the affine transform supports rotation but nothing measures
      it. Calibration params are hardcoded defaults, not in Settings.toml.
- [ ] **Replace `unsafe impl Send/Sync` on `OpencvCamera`** with a
      dedicated camera thread + frame channel.
- [ ] **Conveyor protocol verification** (spec open question): `M3 S<x>` /
      `M5` is assumed; confirm against the real belt controller, and map
      `default_speed` (raw S value) to mm/s.
- [ ] **Learning interface** (plan step 20, spec §3.1): unknown-object
      popup, labeling queue, BrickLink/Rebrickable part-ID lookup
      (`BrickLinkClient` is a stub).

## Housekeeping

- [ ] Deduplicate documentation: `implementation_plan.md` and
      `walkthrough.md` exist both at the repo root and in `documentation/`;
      root `requirement.md` is empty. Keep one copy under `documentation/`.
- [ ] More tests: detector on fixture images (plan step 9), DeltaX2
      protocol against a scripted fake serial port, orchestrator run-loop
      integration test with mocks (plan step 16 verification).
- [ ] Remove the crate-level `#![allow(dead_code)]` in `main.rs` once the
      vision loop consumes the currently-unwired building blocks.
- [ ] Measure the real average robot speed; the planner derives it as
      `feed_rate / 60` which ignores acceleration.
- [ ] CI: `cargo fmt --check`, `clippy`, `cargo test` (needs OpenCV in the
      CI image).
- [ ] Build environment note: after an OS upgrade the system may only ship
      a versioned `libclang-*.so.N`; re-point `libs/libclang.so` and build
      with `LIBCLANG_PATH=$PWD/libs` (see manual, Troubleshooting).
