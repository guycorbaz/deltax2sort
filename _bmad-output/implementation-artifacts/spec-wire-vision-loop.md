---
title: 'Wire the vision loop: camera → detection → tracking → pick'
type: 'feature'
created: '2026-07-03'
status: 'done'
baseline_commit: 'bf16f13'
context: ['{project-root}/_bmad-output/project-context.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** All vision building blocks exist (camera drivers, `BlobDetector`,
`Tracker`, `CoordinateTransformer`, `OrchestratorMsg::Pick`) but nothing is
connected — the robot never picks anything. The current `Tracker` is a stub
that re-IDs every object on every frame, so naive wiring would spam one Pick
per detection per frame into the unbounded channel.

**Approach:** Add a vision pipeline task that owns the camera and drives
frame → detect → track → pixel-to-world → Pick. Upgrade the tracker to real
distance matching with belt-shift prediction so each physical object is
reported exactly once. Stale picks are dropped at the orchestrator.

## Boundaries & Constraints

**Always:** Follow `_bmad-output/project-context.md` (safety invariants
first). One physical object = one `Pick`. OpenCV processing in
`spawn_blocking`. Camera only via `CameraDriver` (use `resolution()` +
`CalibrationParams::centered` for the transform). Pixels never leave
`src/vision/`; z comes from config. Mock mode stays fully functional.
Time-dependent logic takes instants as parameters (no internal clock reads
in testable code paths).

**Ask First:** Any new dependency; any change to the E-stop path or to
`write_gcode`; making the pipeline restart the conveyor or robot on its own.

**Never:** UI camera feed / overlays (next milestone). Intercept planning in
`schedule_pick` (needs robot position feedback — separate TODO). Visual belt
odometry (config speed is used for shift prediction). Classifier work.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy path | Object enters view, seen ≥ min_seen frames | Exactly one `Pick` with world-mm position, z=0 | N/A |
| Same object persists | Object stays in view for 100 frames | No further `Pick` (reported flag) | N/A |
| Object flickers | Missed ≤ max_missed frames then reappears nearby | Same track ID, still one `Pick` total | N/A |
| Object leaves | Not matched for > max_missed frames | Track evicted; a later new object gets a new ID | N/A |
| Stale pick | `Pick` arrives with `seen_at` older than TTL | Orchestrator drops it with `warn!` | N/A |
| Paused orchestrator | `Pick` arrives while paused | Dropped with `warn!` (no stale backlog on Resume) | N/A |
| Camera read fails | `get_frame` errors | `warn!`, backoff ~500 ms, loop continues | No tight error loop |
| Orchestrator gone | `tx.send` fails | Pipeline loop exits cleanly with `info!` | N/A |
| Empty belt | Frames with no blobs | No messages sent at all | N/A |

</frozen-after-approval>

## Code Map

- `src/vision/tracker.rs` -- stub to replace with distance matching + belt-shift prediction + `reported` flag
- `src/vision/pipeline.rs` -- NEW: `VisionPipeline` (process_frame) + `spawn_vision_loop`
- `src/vision/mod.rs` -- `DetectedObject` (add `seen_at: std::time::Instant`), export pipeline
- `src/vision/detector.rs` -- `BlobDetector::detect` (sets `seen_at`; otherwise unchanged)
- `src/vision/calibration.rs` -- `CalibrationParams::centered(w, h, mm_per_px)` — used, not modified
- `src/orchestrator.rs` -- `handle_message`/`schedule_pick`: drop paused/stale picks
- `src/app_config.rs` -- `[vision] mm_per_px` (serde default 0.5, validate > 0, frozen-fixture assertion)
- `src/hardware.rs` -- `CameraDriver::resolution()` — used, not modified
- `src/main.rs` -- spawn the pipeline; remove crate-level `#![allow(dead_code)]`

## Tasks & Acceptance

**Execution:**
- [x] `src/app_config.rs` -- add `vision.mm_per_px` (default 0.5) + validation + assertion in the frozen legacy fixture test -- transform scale must be configurable, not magic
- [x] `src/vision/mod.rs` -- add `seen_at: std::time::Instant` to `DetectedObject` (keep `SystemTime` for logging) -- monotonic staleness checks
- [x] `src/vision/tracker.rs` -- rewrite: nearest-neighbour matching against predicted positions (`last_rect` center + belt_shift_px on y), greedy, threshold `max_match_dist_px`; unmatched actives get `missed_frames += 1`, evicted past `max_missed_frames`; `take_ready(min_seen)` returns tracks seen ≥ min_seen frames with `reported == false` and marks them reported -- this is the 1-object-=-1-Pick guarantee
- [x] `src/vision/pipeline.rs` -- NEW `VisionPipeline { detector, tracker, transformer, px_per_frame math }` with pure-ish `process_frame(&mut self, frame: &Mat, dt: Duration) -> Result<Vec<DetectedObject>>` (returns pick-ready objects with `world_pos` set); `spawn_vision_loop(camera, cfg, tx)` builds the transformer from `camera.resolution()` + `mm_per_px`, loops get_frame → spawn_blocking(process_frame) → send `Pick`s; error backoff; exit when channel closed
- [x] `src/orchestrator.rs` -- in `handle_message`, drop `Pick` when paused (`warn!`); in `schedule_pick(object, now: Instant)`, drop when `now - seen_at > PICK_TTL` (const 3 s, commented) -- stale-pick invalidation
- [x] `src/main.rs` -- spawn the vision loop after the orchestrator (move the camera Box in); remove `#![allow(dead_code)]`, add targeted `#[allow(dead_code)]` only where still needed (classifier stub, planner)
- [x] tests (in-module) -- tracker: stable ID across shifted frames, single `take_ready` emission, eviction; detector: synthetic white-rect `Mat` → one detection; pipeline: synthetic frame end-to-end → one pick-ready object with correct mm position, second identical frame → none; orchestrator: stale pick and paused pick are dropped (explicit `Instant`s, no sleeps); config: mm_per_px default

**Acceptance Criteria:**
- Given a synthetic frame with one blob processed twice, when `process_frame` runs on both frames, then exactly one pick-ready object is returned overall and its `world_pos` matches the centered affine transform (mm, z = 0).
- Given the orchestrator is paused or the pick is older than the TTL, when the `Pick` arrives, then the queue length stays 0 and a warning is logged.
- Given `cargo run -- --mock`, when the app starts, then the vision loop runs against `MockCamera` (black frames → no picks) and the UI stays responsive.
- Given the full suite, when `cargo test` runs, then all tests (17 existing + new) pass with no image assets and no real sleeps in assertions.

## Spec Change Log

## Design Notes

Pipeline ownership: the loop owns the camera `Box` (no second handle, no
mutex). `Mat` is `Send`, so the pipeline moves into `spawn_blocking` and back
each frame (`let (p, out) = spawn_blocking(move || { let o = p.process_frame(..); (p, o) }).await?`).
Belt shift per frame in *pixels*: `speed_mm_s * dt / mm_per_px` on +y (default
rotation 0 ⇒ robot +Y ≡ pixel +y; document the assumption next to the math).
`get_frame().await` keeps the known debt of blocking reads inside the camera
driver (existing TODO: dedicated camera thread) — out of scope here.
Tuning constants in `Tracker::new`: `max_match_dist_px = 60`, `min_seen = 3`,
`max_missed_frames = 5` — plain consts with comments, config later if needed.

## Verification

**Commands:**
- `cargo test` -- expected: all tests pass (17 existing + ~8 new)
- `cargo fmt --check` -- expected: clean
- `RUST_LOG=info cargo run -- --mock` -- expected (HUMAN smoke test): app starts, "vision loop started" logged, no Pick spam, UI responsive

## Suggested Review Order

**The loop and its lifecycle**

- Entry point: the spawned task owning the camera — capture, dt/timestamp math, backoff, exit paths
  [`pipeline.rs:85`](../../src/vision/pipeline.rs#L85)

- Per-frame core: detect → belt-shift px → track → take_ready → pixel_to_world (z stays 0)
  [`pipeline.rs:56`](../../src/vision/pipeline.rs#L56)

- Wiring: camera Box moves into the loop right after the orchestrator spawn
  [`main.rs:118`](../../src/main.rs#L118)

**One object = one Pick (tracker)**

- Greedy nearest-neighbour matching against belt-predicted centers
  [`tracker.rs:56`](../../src/vision/tracker.rs#L56)

- Belt-drift accumulation on missed frames (review patch) — prediction follows the belt
  [`tracker.rs:12`](../../src/vision/tracker.rs#L12)

- The one-shot `reported` guarantee
  [`tracker.rs:139`](../../src/vision/tracker.rs#L139)

**Stale/paused pick protection (orchestrator)**

- Paused picks dropped at the message boundary; clock read only here
  [`orchestrator.rs:214`](../../src/orchestrator.rs#L214)

- TTL gate in schedule_pick (`PICK_TTL` 3 s, known limits in deferred work)
  [`orchestrator.rs:267`](../../src/orchestrator.rs#L267)

**Capture honesty (review patches)**

- `seen_at` = capture time, not processing time
  [`detector.rs:39`](../../src/vision/detector.rs#L39)

- Driver frame queue shrunk to 1 so frames are fresh (best effort)
  [`hardware.rs:587`](../../src/hardware.rs#L587)

**Peripherals**

- New `vision.mm_per_px` (serde default, finite-positive validation)
  [`app_config.rs:88`](../../src/app_config.rs#L88)

- Tests: pipeline end-to-end with synthetic frames
  [`pipeline.rs:158`](../../src/vision/pipeline.rs#L158)

- Tests: tracker identity/eviction/drift cases
  [`tracker.rs:152`](../../src/vision/tracker.rs#L152)

- Tests: stale/paused pick drops with explicit Instants
  [`orchestrator.rs:319`](../../src/orchestrator.rs#L319)
