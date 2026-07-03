# TODO — Remaining Work

Maintained list of open work items. Last updated: 2026-07-03, after wiring
the vision loop (see below). Check items off and re-date when working here.

## Completed 2026-07-03 — vision loop wired (for context)

- Vision pipeline task (`src/vision/pipeline.rs`): owns the camera,
  frame → detect → track → pixel-to-world → `OrchestratorMsg::Pick`;
  OpenCV work in `spawn_blocking`, capture-error backoff, clean exit when
  the orchestrator channel closes. Spawned from `main.rs` (mock included).
- Real tracker: greedy nearest-neighbour matching against belt-shift
  predicted positions (+y), miss counting/eviction, `reported` flag —
  one physical object = exactly one `Pick`.
- Stale-pick invalidation in the orchestrator (`PICK_TTL` 3 s on the new
  monotonic `DetectedObject::seen_at`); picks arriving while paused are
  dropped.
- New config key `vision.mm_per_px` (default 0.5, validated > 0); the
  transform is built from `CameraDriver::resolution()` +
  `CalibrationParams::centered` at startup.
- Crate-level `#![allow(dead_code)]` removed; remaining stubs carry
  targeted allows.

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

- [x] **Wire the vision loop** (plan step 16) — done 2026-07-03:
      `src/vision/pipeline.rs` (`VisionPipeline` + `spawn_vision_loop`);
      one Pick per physical object, stale/paused picks dropped by the
      orchestrator.
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

- [x] **Real tracker** (plan step 11) — done 2026-07-03: greedy
      nearest-neighbour matching with belt-shift prediction (from
      `conveyor.speed_mm_s`, not odometry yet), eviction after 5 missed
      frames, single-report guarantee. Tuning constants
      (`max_match_dist_px = 60`, `min_seen = 3`) are code constants in
      `tracker.rs`/`pipeline.rs`; move to config if field tuning demands.
- [ ] **Visual odometry** (plan step 10): `calculate_belt_offset(frame1,
      frame2)` via optical flow or template matching; also enables
      measuring the true `conveyor.speed_mm_s`.
- [ ] **Classifier**: real color/shape classification, later ONNX
      inference; per-class drop bins (spec: up to 6 categories, currently a
      single `[sorting]` drop position).
- [ ] **Choose the physical camera** (model TBD, 2026-07-03): `[camera]`
      is fully configurable (device_id, width/height, fps, optional
      fourcc) and the driver adopts the mode the device actually grants —
      re-check defaults once the model is picked.
- [ ] **Camera calibration wizard** (spec §UI): interactive pixel↔robot
      mapping; the affine transform supports rotation but nothing measures
      it. Since 2026-07-03 params are built at startup from
      `CameraDriver::resolution()` + `vision.mm_per_px` (Settings.toml)
      via `CalibrationParams::centered` (rotation 0, centre = robot
      origin); the wizard should measure scale, rotation and offset.
- [ ] **Replace `unsafe impl Send/Sync` on `OpencvCamera`** with a
      dedicated camera thread + frame channel.
- [ ] **Conveyor protocol verification** (spec open question): `M3 S<x>` /
      `M5` is assumed; confirm against the real belt controller, and map
      `default_speed` (raw S value) to mm/s.
- [ ] **Learning interface** (plan step 20, spec §3.1): unknown-object
      popup, labeling queue, BrickLink/Rebrickable part-ID lookup
      (`BrickLinkClient` is a stub). Intended workflow (2026-07-03):
      capture/label/train on the x86 PC (more convenient UI, more CPU),
      then copy the resulting portable model file (e.g. ONNX) to the
      Raspberry Pi — so the classifier must load its model from a
      configurable path, never embed it.

## Housekeeping

- [ ] Generalize wording in the manual/specifications: the project targets
      small-parts sorting in general (screws, nuts, …), LEGO bricks being
      only the first use case (README and project site already updated,
      2026-07-03).
- [ ] Run `cargo clippy` for the first time and triage the existing warning
      stock (status unverified as of 2026-07-03; not part of the local gate —
      see `_bmad-output/project-context.md`).
- [ ] Decide the Raspberry Pi build path (cross-compile from x86 vs build on
      the Pi 4/5); until decided, no deployment tooling in the repo.
- [ ] Deduplicate documentation: `implementation_plan.md` and
      `walkthrough.md` exist both at the repo root and in `docs/`;
      root `requirement.md` is empty. Keep one copy under `docs/`.
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
