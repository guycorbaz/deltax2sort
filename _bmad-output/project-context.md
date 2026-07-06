---
project_name: 'deltax2sort'
user_name: 'Guy'
date: '2026-07-03'
sections_completed:
  ['safety_invariants', 'technology_stack', 'language_rules', 'framework_rules',
   'testing_rules', 'code_quality_rules', 'workflow_rules']
existing_patterns_found: 12
status: 'complete'
rule_count: 51
optimized_for_llm: true
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## ⚠ Safety Invariants — READ FIRST

This application moves a physical robot. These invariants override every
other rule in this file. Never weaken them; any change touching them is
safety-critical (unit test + human review required).

1. **Never touch real hardware unprompted**: no real serial ports, no robot
   motion, no conveyor start without the user's explicit go-ahead. All
   development and verification runs in `--mock`.
2. **Startup config validation is the motion gate**: `AppConfig::validate`
   must reject anything that could command the robot outside
   `[z_min, z_max]`/workspace. No code path may command motion from
   unvalidated values.
3. **The E-stop path stays preemptive**: `EmergencyStop` owns *cloned* serial
   ports (`try_clone`) and fires synchronously from the UI callback, bypassing
   the tokio mutexes and the orchestrator queue (robot halt `M112`, conveyor
   halt `M5`; when `robot.release_gripper_on_estop` is set the robot write is
   `M05\nM112`, opening the gripper as part of the same preemptive write).
   Never route it through channels, mutexes or async, and never do a blocking
   `set_gripper` after `M112` — both lose preemption / stall recovery.
4. **Workspace limits are enforced inside `move_to` of BOTH robot drivers**
   (real and mock): an out-of-bounds target errors before any G-code is sent.
   Every new motion path goes through `move_to`.
5. **No unbounded hardware waits**: `write_gcode` blocks on the
   `FEEDBACK:<id>` echo with EOF detection and a 30 s overall deadline — keep
   every hardware wait deadline-bounded. The wait also polls a shared
   `estop_flag` (`AtomicBool`) so an in-flight command aborts within one serial
   read window when the E-stop fires; only `home()` clears it (before `G28`),
   the command the firmware accepts after `M112`.
6. **On command failure the orchestrator clears its queue and pauses**
   (explicit `Resume` required) — never auto-retry motion after a failure.

## Technology Stack & Versions

**Source of truth: `Cargo.toml` + committed `Cargo.lock`.** Never run `cargo update`
or bump a dependency unless explicitly asked.

- **Rust, edition 2024** (requires rustc ≥ 1.85; no rust-toolchain.toml — check
  `rustc --version` before diagnosing build errors as code bugs)
- **tokio 1.x** (`full`) — async runtime
- **async-trait** — REQUIRED for the hardware traits: they are used as
  `Box<dyn Trait>` behind `Arc<tokio::Mutex<...>>`. Do NOT migrate to native
  async-fn-in-trait; it breaks dyn dispatch.
- **serialport (sync)** — deliberate choice, always wrapped in `spawn_blocking`.
  Do NOT propose tokio-serial.
- **opencv 0.9x** — bindings are generated at build time against the *system*
  OpenCV 4.x (`libopencv-dev`): don't assume upstream-doc APIs exist locally,
  don't rely on contrib modules, don't bump the crate without checking system
  compat. The build of this crate is slow — that's normal, not a hang.
  ⚠ Mock mode does NOT remove system deps: `cargo test` / `cargo run -- --mock`
  still need libopencv/libclang (classic CI trap; there is no CI today).
- **Slint** — `slint` and `slint-build` versions must stay in lockstep. UI Rust
  code is GENERATED from `ui/app_window.slint` by `build.rs`: never edit
  generated code; change the `.slint` file instead.
- **anyhow** (backtrace feature; needs `RUST_BACKTRACE=1` at runtime) — the only
  error style: no thiserror/eyre, no custom error enums.
- **log + env_logger** — the only logging: do NOT introduce tracing.
  `RUST_LOG=debug` traces G-code.
- **Zero dev-dependencies** — the test seam is the hardware traits + `--mock`;
  do not add mockall/proptest.
- **Linux-only by design** (libudev, `dialout` group, `/dev/tty*` paths). No
  Windows/macOS support intended — don't add `cfg(target_os)` portability code.
- **Deployment target: Raspberry Pi 4 and 5 + official 7" Touch Display
  (800×480, DSI, capacitive touch)**. Dev machine is x86 Ubuntu — code must
  build and run on both (aarch64 Linux included); mind the Pi's CPU budget for
  vision/UI work (no desktop-class assumptions).
- libclang build workaround after OS upgrades: see CLAUDE.md (`LIBCLANG_PATH`,
  applies to every cargo command incl. test/clippy).

## Critical Implementation Rules

### Language-Specific Rules (Rust)

- **Errors**: `anyhow::Result<T>` on every fallible function; no `.unwrap()`/
  `.expect()` outside tests and `build.rs` (a runtime panic can leave the robot
  mid-motion). Never swallow a hardware `Result` (`let _ =`, `.ok()`) — the
  orchestrator's clear-queue-and-pause recovery depends on errors propagating.
- **`.context("...")`** at fallible boundaries (serial, config load, OpenCV) in
  NEW code; today only `src/main.rs` does this — do not retrofit unprompted.
- **Blocking code**: anything that can block (serial I/O, OpenCV capture) runs
  inside `tokio::task::spawn_blocking`, never directly in an async fn. Every
  blocking wait on hardware carries an explicit deadline (model:
  `write_gcode`'s 30 s FEEDBACK deadline) — no unbounded reads or wait loops.
- **Lock discipline**: never hold a hardware `tokio::Mutex` guard across a
  *timed* wait (`sleep`, the intercept `Wait`); holding it for the duration of
  a single hardware command (`move_to`, `write_gcode`) IS the intended
  exclusivity pattern (`src/orchestrator.rs::execute`).
- **Shared handles**: hardware is `Arc<tokio::Mutex<Box<dyn Trait>>>` — clone
  the `Arc`, never open a second serial handle. Intentional exception: the
  E-stop path is synchronous, trait-less and mutex-free on `try_clone`d ports —
  never "modernize" it toward async/traits.
- **Units & frames**: every spatial value documents its unit and frame
  (mm robot-space vs px camera-space) in the signature or doc comment.
- New vision code goes under `src/vision/`.

### Framework-Specific Rules

- **Three layers, one direction**: `main.rs` wires; the orchestrator commands the
  robot; hardware only through the `src/hardware.rs` traits. The vision loop
  (`src/vision/pipeline.rs`) is wired: camera → detect → track → transform →
  `Pick`. The classifier is still a stub.
- **Mock parity**: every new trait method gets BOTH impls (real + mock);
  `cargo run -- --mock` must stay fully functional — no `todo!()` in mocks.
  Camera access only via `CameraDriver`: never open `opencv::videoio` directly.
  Mocks record the commands they receive: `MockRobot`/`MockConveyor` keep an
  ordered `command_log()` (a shared `Arc<Mutex<Vec<_>>>`) so tests can assert
  exactly what the hardware was told; a rejected `move_to` records nothing.
- **Orchestrator is the only robot commander** (sole bypass: the E-stop path —
  synchronous, on `try_clone`d ports, bypassing the tokio mutexes; never route
  E-stop through the channel, that loses preemption). Talk to it exclusively via
  the mpsc sender from `Orchestrator::new`. Keep the channel UNBOUNDED — UI/EStop
  senders must never block — but unbounded ≠ license to spam: **one physical
  object = one `Pick`**. The vision loop must track/dedupe before sending; never
  push raw per-frame detections. Detections carry timestamps; stale picks (belt
  moved on, or accumulated during Pause) must be invalidated, not executed.
- **Coordinate frames**: pixels never leave `src/vision/`. The affine transform
  in `calibration.rs` is the only pixel→robot gate; everything in a `Pick` is mm
  robot-space; z ALWAYS comes from config (calibration yields z=0) — never from
  vision. `move_to`'s workspace check is the safety net, not the contract.
- **Intercept planner is NOT on the active path yet**: `calculate_intercept`
  uses *signed* `conveyor.speed_mm_s` (positive = +Y) and returns `None` past
  the pick line — preserve both — but `schedule_pick` targets last-seen
  positions until the `Position` G-code query is implemented. Don't wire
  interception without robot position feedback.
- **Slint threading**: background→UI only via `ui.as_weak()` +
  `slint::invoke_from_event_loop`; never touch UI state from a tokio task,
  never block inside a Slint callback.
- **Camera feed (when built)**: frames are latest-wins — single-slot/watch
  channel, NEVER an unbounded queue (accumulated latency shows the operator the
  past). Build the `SharedPixelBuffer` in the vision/blocking thread; the
  event-loop closure only wraps and sets. Frame + its detections travel as ONE
  atomic message (or burn overlays into the frame) so boxes can't desync. Stale
  feed must declare itself: no frame for ~1 s → visible "FEED LOST" state,
  never a silent freeze.
- **UI shows confirmed state, not hoped state**: "Stopped" only after the
  orchestrator confirms; after E-stop, Start stays locked until re-home. The
  E-stop button keeps its size/position/z-order above any new UI element and
  its callback stays synchronous — no new feature enters that path.
- **Touch-first UI at 800×480**: the operator screen is the RPi 7" Touch
  Display — finger input only: no hover-dependent affordances, no
  keyboard-only paths. Touch targets ≥ ~45 px on this panel; E-stop stays the
  largest target on screen. The main operator view must fit 800×480 with NO
  scrolling. Downscale camera frames to display size BEFORE the Mat→Image
  conversion — never push full-resolution frames through the Pi's CPU to show
  them on an 800×480 panel.

### Testing Rules

- **`cargo test` needs the system toolchain** (libopencv/libclang) even though
  tests touch no hardware. If the environment can't run it, say so and deliver
  as explicitly unverified — never claim green without a run.
- **Tests are in-module, pure and deterministic** — no real sleeps, no real
  serial ports, no real camera, no wall-clock asserts.
- **Async/timer logic**: `#[tokio::test(start_paused = true)]` +
  `tokio::time::advance`; `tokio::time::sleep` in code, never
  `std::thread::sleep`. Time-dependent logic takes timestamps as parameters
  (the `calculate_intercept` pattern — pure geometry, that's why it's
  testable); use `Instant` for duration comparisons — `SystemTime` (what
  `DetectedObject.timestamp` is) can go backwards under NTP, keep it for
  logging only.
- **Safety logic always gets a unit test** (config validation, workspace
  limits, intercept planning — extend the pattern to tracker/classifier as
  they leave stub state).
- **The config-compat test is a frozen fixture**: never add new fields to the
  legacy TOML in `app_config.rs` — add an assertion that the new field gets
  its serde default. Never delete that test.
- **Vision tests use synthetic `Mat`s built in code** (shapes on a flat
  background) — no committed image assets. Extend `MockCamera` to serve
  deterministic frames; the real end-to-end test asserts a correct `Pick`
  exits the orchestrator channel. `cargo run -- --mock` is a HUMAN smoke test
  (it opens a Slint window) — agents verify via `cargo test`.
- **Mocks record the commands they receive** so tests assert what the robot
  was told, not merely that nothing panicked.

### Code Quality & Style Rules

- **Log levels have meaning**: `error` = needs operator attention; `warn` =
  degraded/skipped; `info` = lifecycle events only (startup, homing, state
  changes); `debug` = G-code and per-frame tracing. Nothing per-frame above
  `debug` — at 30 fps an `info` in the vision loop floods the log.
- **Comments state constraints, not narration** — sparse comments explaining
  WHY/invariants. Code and comments in English.
- **Clippy status: unverified** (never run on this repo as of writing). NOT
  part of the gate; never claim "clippy clean". If you run it: fix only
  warnings your diff introduced, log the pre-existing stock in a GitHub issue
  — no mass refactor.

### Development Workflow Rules

- **No CI — the local gate before any review/commit**: `cargo test` passes +
  `cargo fmt --check` clean (default rustfmt settings, no rustfmt.toml).
- **Development and verification happen in `--mock`** (the real-hardware
  prohibition lives in the safety section).
- **Deployment target: Raspberry Pi 4 and 5 (ARM64)** — build path
  (cross-compile vs build on the Pi) not yet decided: don't introduce
  x86-only assumptions, don't add deployment tooling (cross, Docker,
  packaging scripts) unprompted.
- **GitHub issues are the tracker of record**
  (https://github.com/guycorbaz/deltax2sort/issues): file new findings as
  issues (in English) and reference/close them when completing work;
  `docs/TODO.md` is only a pointer.
- **The manual follows behavior**: any behavior/config change updates the
  relevant chapter in `docs/manual/*.tex` and rebuilds the PDF
  (`cd docs && latexmk -pdf manual.tex`); `manual.pdf` is committed.
- **Commits**: only when asked; work directly on `main`.

---

## Usage Guidelines

**For AI Agents:**

- Read this file before implementing any code; the Safety Invariants override
  everything else.
- Follow ALL rules exactly as documented; when in doubt, prefer the more
  restrictive option.
- Update this file when new confirmed patterns emerge (via the human).

**For Humans:**

- Keep this file lean and focused on agent needs — every line must prevent a
  real mistake.
- Update when the stack, hardware targets or conventions change; review
  periodically and remove rules that become obvious or stale.

Last Updated: 2026-07-03
