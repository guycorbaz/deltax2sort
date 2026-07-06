use super::calibration::{CalibrationParams, CoordinateTransformer};
use super::detector::BlobDetector;
use super::embedder::Recognizer;
use super::tracker::Tracker;
use super::{ClassName, DetectedObject};
use anyhow::{Context, Result, anyhow};
use log::{debug, error, info, warn};
use opencv::core::{Mat, Rect, Scalar, Size};
use opencv::{imgproc, prelude::*};
use slint::{Rgb8Pixel, SharedPixelBuffer};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch};

/// Live-feed image (frame with detection overlays already burned in),
/// downscaled for the display. Latest-wins: the UI only ever shows the newest,
/// so accumulated latency can never lie to the operator about belt position.
pub type FrameImage = SharedPixelBuffer<Rgb8Pixel>;

/// Recognise the object in `rect` of `frame`: crop, embed, nearest-neighbour.
/// A failure (bad ROI, inference error) logs and yields `None` rather than
/// aborting the frame — a missed classification just leaves the object
/// unrecognised (and unsorted), never crashes the loop.
fn classify_roi(rec: &Recognizer, frame: &Mat, rect: &Rect) -> Option<ClassName> {
    match crop_roi(frame, rect).and_then(|roi| rec.classify(&roi)) {
        Ok(class) => class,
        Err(e) => {
            warn!("Vision: recognition failed for a track: {e:#}");
            None
        }
    }
}

/// Clamp `rect` to the frame and return an owned crop. Detections can sit
/// partly off-frame near the edges; clamping keeps the ROI valid.
fn crop_roi(frame: &Mat, rect: &Rect) -> Result<Mat> {
    let (fw, fh) = (frame.cols(), frame.rows());
    let x = rect.x.max(0);
    let y = rect.y.max(0);
    let w = rect.width.min(fw - x);
    let h = rect.height.min(fh - y);
    if w <= 0 || h <= 0 {
        return Err(anyhow!("empty ROI for rect {rect:?} in {fw}x{fh} frame"));
    }
    let roi = Mat::roi(frame, Rect::new(x, y, w, h))?;
    Ok(roi.try_clone()?)
}

/// Max display size the feed is downscaled to before the Mat→Image conversion
/// (the Pi panel is 800x480 and the preview occupies only part of it).
const DISPLAY_MAX_W: u32 = 640;
const DISPLAY_MAX_H: u32 = 480;

/// Overlay colours in OpenCV BGR. Recognised parts are green; anything still
/// unrecognised (today: everything — the classifier is a stub) is yellow.
fn overlay_color(class: &Option<ClassName>) -> Scalar {
    match class {
        Some(_) => Scalar::new(0.0, 255.0, 0.0, 0.0),  // green (BGR)
        None => Scalar::new(0.0, 255.0, 255.0, 0.0),   // yellow
    }
}

/// Downscale a BGR `Mat` to fit `max_w`×`max_h` (never upscaling) and convert
/// it to an RGB `SharedPixelBuffer` the Slint `Image` can wrap directly. Built
/// here, in the vision/blocking thread, so the UI closure only wraps and sets.
fn mat_to_frame_image(bgr: &Mat, max_w: u32, max_h: u32) -> Result<FrameImage> {
    let (w, h) = (bgr.cols(), bgr.rows());
    if w <= 0 || h <= 0 {
        return Err(anyhow!("cannot render an empty frame ({w}x{h})"));
    }
    let scale = (max_w as f64 / w as f64)
        .min(max_h as f64 / h as f64)
        .min(1.0);
    let dst_w = ((w as f64 * scale).round() as i32).max(1);
    let dst_h = ((h as f64 * scale).round() as i32).max(1);

    let mut resized = Mat::default();
    imgproc::resize(
        bgr,
        &mut resized,
        Size::new(dst_w, dst_h),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
    // Fresh single-channel-group Mats from resize/cvt_color are contiguous, so
    // the byte buffer is a tight w*h*3 with no row padding — copy it straight
    // into the pixel buffer.
    if !rgb.is_continuous() {
        return Err(anyhow!("converted frame is not contiguous"));
    }
    let mut buffer = SharedPixelBuffer::<Rgb8Pixel>::new(dst_w as u32, dst_h as u32);
    buffer
        .make_mut_bytes()
        .copy_from_slice(rgb.data_bytes()?);
    Ok(buffer)
}

use crate::app_config::AppConfig;
use crate::hardware::CameraDriver;
use crate::orchestrator::OrchestratorMsg;

/// Frames a track must be detected in before it is emitted as pick-ready.
const MIN_SEEN_FRAMES: u32 = 3;

/// Backoff after a failed frame capture, so a dead camera does not spin the
/// loop.
const CAPTURE_RETRY_DELAY: Duration = Duration::from_millis(500);

/// Camera → detection → tracking → world coordinates. Owns all per-frame
/// state; pure with respect to time (`dt` is a parameter), so it is
/// testable with synthetic frames and explicit durations.
pub struct VisionPipeline {
    detector: BlobDetector,
    tracker: Tracker,
    transformer: CoordinateTransformer,
    /// Signed belt speed in robot coordinates, mm/s (positive = +Y).
    belt_speed_mm_s: f32,
    /// Camera scale at the belt plane, mm per pixel.
    mm_per_px: f32,
    /// Object recogniser, when `[recognition]` is enabled and a model loaded.
    /// `None` = recognition off: every object stays unrecognised (unsorted).
    recognizer: Option<Recognizer>,
}

impl VisionPipeline {
    /// `width`/`height` are the capture resolution actually in effect
    /// (`CameraDriver::resolution()` after connect), in pixels.
    pub fn new(config: &AppConfig, width: u32, height: u32) -> Self {
        let mm_per_px = config.vision.mm_per_px;
        Self {
            detector: BlobDetector::from_config(&config.vision),
            tracker: Tracker::new(),
            transformer: CoordinateTransformer::new(CalibrationParams::centered(
                width, height, mm_per_px,
            )),
            belt_speed_mm_s: config.conveyor.speed_mm_s,
            mm_per_px,
            recognizer: None,
        }
    }

    /// Attach a recogniser (called once, at startup, when recognition is on).
    pub fn set_recognizer(&mut self, recognizer: Recognizer) {
        self.recognizer = Some(recognizer);
    }

    /// Process one frame captured `dt` after the previous one, at
    /// `captured_at` (monotonic). `belt_running` is the actual conveyor
    /// run-state: when it is stopped the prediction shift is zero, so a
    /// stationary part cannot drift out of its track and be re-emitted.
    /// Returns the pick-ready objects (each physical object exactly once over
    /// its lifetime) with `world_pos` set in robot mm; z stays 0 (belt plane)
    /// — the pick height comes from configuration, never from vision.
    pub fn process_frame(
        &mut self,
        frame: &Mat,
        dt: Duration,
        captured_at: Instant,
        belt_running: bool,
    ) -> Result<Vec<DetectedObject>> {
        let detections = self.detector.detect(frame, captured_at)?;

        // Belt motion since the last frame, in pixels. The centered
        // calibration uses rotation = 0, i.e. robot +Y is aligned with
        // pixel +y — that assumption is what lets a mm/s belt speed be
        // converted straight to a pixel +y shift. Zero while the belt is
        // stopped: the configured speed is a nominal value, not proof of
        // motion (see #28; visual odometry #13 will measure it directly).
        let belt_shift_px = if belt_running {
            self.belt_speed_mm_s * dt.as_secs_f32() / self.mm_per_px
        } else {
            0.0
        };

        self.tracker.update(detections, belt_shift_px);
        // Recognise each track once, as it becomes pick-ready, and cache the
        // class on the track (drives routing and the overlay). `rec` is bound
        // before the mutable tracker borrow so the two disjoint fields don't
        // conflict.
        if let Some(rec) = self.recognizer.as_ref() {
            self.tracker
                .classify_ready_tracks(MIN_SEEN_FRAMES, |rect| classify_roi(rec, frame, rect));
        }
        let mut ready = self.tracker.take_ready(MIN_SEEN_FRAMES);
        for obj in &mut ready {
            let cx = obj.rect.x as f32 + obj.rect.width as f32 / 2.0;
            let cy = obj.rect.y as f32 + obj.rect.height as f32 / 2.0;
            obj.world_pos = Some(self.transformer.pixel_to_world(cx, cy)?);
        }
        Ok(ready)
    }

    /// Render the current frame for the live feed: draw a bounding box (green =
    /// known, yellow = unknown) around every track visible in this frame, then
    /// downscale + convert to an RGB buffer. Must be called right after
    /// `process_frame` for the same `frame`, so the boxes match the image.
    /// Overlays are burned into the frame, so image and boxes can never desync.
    pub fn render_display_frame(&self, frame: &Mat) -> Result<FrameImage> {
        let mut annotated = frame.clone();
        for (rect, class) in self.tracker.current_overlays() {
            let color = overlay_color(&class);
            imgproc::rectangle(
                &mut annotated,
                rect,
                color,
                2, // px, at capture resolution — scaled down with the frame
                imgproc::LINE_8,
                0,
            )?;
            // Label recognised objects with their class name, just above the
            // box (unrecognised ones stay unlabelled — nothing to name yet).
            if let Some(name) = &class {
                imgproc::put_text(
                    &mut annotated,
                    name,
                    opencv::core::Point::new(rect.x, (rect.y - 8).max(12)),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    imgproc::LINE_8,
                    false,
                )?;
            }
        }
        mat_to_frame_image(&annotated, DISPLAY_MAX_W, DISPLAY_MAX_H)
    }
}

/// Spawn the vision loop: the task takes ownership of the camera and feeds
/// pick-ready objects to the orchestrator. Exits when the orchestrator
/// channel closes or `shutdown` flips to true (on which the camera is
/// released as the task returns). OpenCV work runs inside `spawn_blocking`;
/// the pipeline moves in and out each iteration (`Mat` and the pipeline are
/// `Send`).
pub fn spawn_vision_loop(
    mut camera: Box<dyn CameraDriver>,
    config: &AppConfig,
    tx: mpsc::UnboundedSender<OrchestratorMsg>,
    mut shutdown: watch::Receiver<bool>,
    belt_running: watch::Receiver<bool>,
    // Latest-wins sink for the live feed. The UI reads only the newest frame,
    // so a slow consumer drops stale frames instead of building a backlog.
    frame_tx: watch::Sender<Option<FrameImage>>,
) -> tokio::task::JoinHandle<()> {
    let (width, height) = camera.resolution();
    let mm_per_px = config.vision.mm_per_px;
    let mut pipeline = VisionPipeline::new(config, width, height);

    // Attach the recogniser if enabled; a load failure disables recognition
    // (objects stay unrecognised → unsorted) rather than aborting startup.
    if config.recognition.enabled {
        match Recognizer::load(&config.recognition) {
            Ok(rec) => {
                info!(
                    "Vision: recognition enabled ({} class(es) in the catalogue)",
                    rec.class_count()
                );
                pipeline.set_recognizer(rec);
            }
            Err(e) => warn!("Vision: recognition disabled — {e:#}"),
        }
    }

    tokio::spawn(async move {
        info!(
            "Vision loop started ({}x{} px, {} mm/px)",
            width, height, mm_per_px
        );
        let mut last_frame = Instant::now();
        loop {
            // Stop between frames on shutdown; returning drops (releases) the
            // camera. `biased` so a pending shutdown always wins over a ready
            // frame.
            let frame = tokio::select! {
                biased;
                res = shutdown.changed() => {
                    // Sender dropped (Err) also means shut down.
                    if res.is_err() || *shutdown.borrow() {
                        info!("Vision loop: shutdown — releasing camera");
                        return;
                    }
                    continue;
                }
                frame = camera.get_frame() => match frame {
                    Ok(frame) => frame,
                    Err(e) => {
                        warn!("Vision: frame capture failed: {:#} — retrying", e);
                        tokio::time::sleep(CAPTURE_RETRY_DELAY).await;
                        continue;
                    }
                },
            };
            // Stamped right after the camera read returns — the closest to
            // the true capture instant we can measure without driver
            // timestamps, and before any processing latency accrues.
            let now = Instant::now();
            let dt = now - last_frame;
            last_frame = now;

            // Snapshot the belt run-state for this frame: no prediction drift
            // while the belt is stopped (avoids phantom duplicate picks).
            let running = *belt_running.borrow();
            // Detect/track AND render the live-feed image in the same blocking
            // hop, on the same frame, so the overlay boxes always match the
            // pixels shown.
            let result = tokio::task::spawn_blocking(move || {
                let out = pipeline.process_frame(&frame, dt, now, running).and_then(|ready| {
                    let image = pipeline.render_display_frame(&frame)?;
                    Ok((ready, image))
                });
                (pipeline, out)
            })
            .await
            .context("vision processing task panicked");
            let ready = match result {
                Ok((p, out)) => {
                    pipeline = p;
                    match out {
                        Ok((ready, image)) => {
                            // Latest-wins: overwrites any frame the UI has not
                            // consumed yet. Ignore send errors (no receiver).
                            let _ = frame_tx.send(Some(image));
                            ready
                        }
                        Err(e) => {
                            warn!("Vision: frame processing failed: {:#} — retrying", e);
                            tokio::time::sleep(CAPTURE_RETRY_DELAY).await;
                            continue;
                        }
                    }
                }
                Err(e) => {
                    // A panic in OpenCV processing is a bug; stop the loop
                    // rather than risk feeding garbage picks. Terminal state:
                    // detection is dead until restart — operator attention.
                    error!("Vision loop stopping permanently: {:#}", e);
                    break;
                }
            };

            for obj in ready {
                debug!(
                    "Vision: object {} pick-ready at {:?}",
                    obj.id, obj.world_pos
                );
                if tx.send(OrchestratorMsg::Pick(obj)).is_err() {
                    info!("Vision loop exiting: orchestrator channel closed");
                    return;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{CV_8UC3, Rect, Scalar};
    use opencv::imgproc;

    /// Config matching a 640x480 synthetic camera: white (lighter-than-belt)
    /// objects, 0.5 mm/px, belt stopped so a static frame stays consistent.
    fn test_config() -> AppConfig {
        let mut cfg = AppConfig::default();
        cfg.vision.invert = false;
        cfg.vision.mm_per_px = 0.5;
        cfg.conveyor.speed_mm_s = 0.0;
        cfg
    }

    fn frame_with_white_rect(rect: Rect) -> Mat {
        let mut frame =
            Mat::new_rows_cols_with_default(480, 640, CV_8UC3, Scalar::all(0.0)).unwrap();
        imgproc::rectangle(
            &mut frame,
            rect,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )
        .unwrap();
        frame
    }

    #[test]
    fn one_blob_yields_exactly_one_pick_ready_object_with_mm_position() {
        let mut pipeline = VisionPipeline::new(&test_config(), 640, 480);
        // Blob center at pixel (420, 120); centered transform at 0.5 mm/px:
        // world = ((420-320)*0.5, (120-240)*0.5) = (50, -60) mm.
        let frame = frame_with_white_rect(Rect::new(400, 100, 40, 40));
        let dt = Duration::from_millis(33);

        let mut emissions = Vec::new();
        for _ in 0..10 {
            emissions.push(
                pipeline
                    .process_frame(&frame, dt, Instant::now(), true)
                    .unwrap(),
            );
        }

        let total: usize = emissions.iter().map(Vec::len).sum();
        assert_eq!(total, 1, "one physical object = one pick-ready emission");
        assert!(emissions[0].is_empty() && emissions[1].is_empty());
        let obj = &emissions[2][0]; // ready on the MIN_SEEN_FRAMES-th frame
        let pos = obj.world_pos.expect("world_pos must be set");
        assert!((pos.x - 50.0).abs() < 1.0, "x: {}", pos.x);
        assert!((pos.y + 60.0).abs() < 1.0, "y: {}", pos.y);
        assert_eq!(pos.z, 0.0, "vision must always report the belt plane");
    }

    #[test]
    fn render_display_frame_downscales_and_stays_tight() {
        // 1280x720 fit into 640x480 (never upscaling) → scale 0.5 → 640x360.
        let mut pipeline = VisionPipeline::new(&test_config(), 1280, 720);
        let mut frame =
            Mat::new_rows_cols_with_default(720, 1280, CV_8UC3, Scalar::all(0.0)).unwrap();
        imgproc::rectangle(
            &mut frame,
            Rect::new(600, 300, 40, 40),
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )
        .unwrap();
        // Populate the tracker so there is a box to overlay.
        pipeline
            .process_frame(&frame, Duration::from_millis(33), Instant::now(), true)
            .unwrap();

        let img = pipeline.render_display_frame(&frame).unwrap();
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 360);
        // Tight RGB buffer: exactly w*h*3 bytes, no row padding.
        assert_eq!(img.as_bytes().len(), 640 * 360 * 3);
    }

    #[test]
    fn render_display_frame_handles_an_empty_scene() {
        let pipeline = VisionPipeline::new(&test_config(), 640, 480);
        let frame = Mat::new_rows_cols_with_default(480, 640, CV_8UC3, Scalar::all(0.0)).unwrap();
        // No detections → no overlays, but a valid image is still produced.
        let img = pipeline.render_display_frame(&frame).unwrap();
        assert_eq!((img.width(), img.height()), (640, 480));
    }

    #[test]
    fn empty_frames_yield_no_objects() {
        let mut pipeline = VisionPipeline::new(&test_config(), 640, 480);
        let frame = Mat::new_rows_cols_with_default(480, 640, CV_8UC3, Scalar::all(0.0)).unwrap();
        for _ in 0..5 {
            assert!(
                pipeline
                    .process_frame(&frame, Duration::from_millis(33), Instant::now(), true)
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[tokio::test]
    async fn mock_camera_frames_drive_the_pipeline_to_a_pick() {
        use crate::hardware::{CameraDriver, MockCamera};
        // Default config (invert = true): MockCamera renders a dark blob on a
        // light belt, which the configured detector must pick up.
        let cfg = AppConfig::default();
        let mut cam = MockCamera::new(cfg.camera.width, cfg.camera.height, cfg.camera.fps, cfg.vision.invert);
        cam.connect().await.unwrap();
        let (w, h) = cam.resolution();
        let mut pipeline = VisionPipeline::new(&cfg, w, h);

        let mut picks = 0usize;
        for _ in 0..6 {
            let frame = cam.get_frame().await.unwrap();
            let ready = pipeline
                .process_frame(&frame, Duration::from_millis(33), Instant::now(), false)
                .unwrap();
            for obj in &ready {
                let p = obj.world_pos.expect("pick must carry a world position");
                // Centred blob → robot origin (within a couple of mm).
                assert!(p.x.abs() < 5.0 && p.y.abs() < 5.0, "world pos: {p:?}");
            }
            picks += ready.len();
        }
        assert_eq!(picks, 1, "the mock part is picked exactly once");
    }

    #[test]
    fn stopped_belt_does_not_re_emit_a_stationary_object() {
        // Non-zero configured belt speed, but the belt is NOT running. A part
        // that stays put must be picked exactly once — with drift applied it
        // would leave its track after ~1 s and be re-emitted repeatedly (#28).
        let mut cfg = test_config();
        cfg.conveyor.speed_mm_s = 100.0; // nominal speed, belt stopped below
        let mut pipeline = VisionPipeline::new(&cfg, 640, 480);
        let frame = frame_with_white_rect(Rect::new(300, 220, 40, 40));
        let dt = Duration::from_millis(33);

        let total: usize = (0..60)
            .map(|_| {
                pipeline
                    .process_frame(&frame, dt, Instant::now(), false) // belt stopped
                    .unwrap()
                    .len()
            })
            .sum();
        assert_eq!(total, 1, "stationary part emitted once while belt stopped");
    }
}
