use super::DetectedObject;
use super::calibration::{CalibrationParams, CoordinateTransformer};
use super::detector::BlobDetector;
use super::tracker::Tracker;
use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use opencv::core::Mat;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch};

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
        }
    }

    /// Process one frame captured `dt` after the previous one, at
    /// `captured_at` (monotonic). Returns the pick-ready objects (each
    /// physical object exactly once over its lifetime) with `world_pos` set
    /// in robot mm; z stays 0 (belt plane) — the pick height comes from
    /// configuration, never from vision.
    pub fn process_frame(
        &mut self,
        frame: &Mat,
        dt: Duration,
        captured_at: Instant,
    ) -> Result<Vec<DetectedObject>> {
        let detections = self.detector.detect(frame, captured_at)?;

        // Belt motion since the last frame, in pixels. The centered
        // calibration uses rotation = 0, i.e. robot +Y is aligned with
        // pixel +y — that assumption is what lets a mm/s belt speed be
        // converted straight to a pixel +y shift.
        let belt_shift_px = self.belt_speed_mm_s * dt.as_secs_f32() / self.mm_per_px;

        self.tracker.update(detections, belt_shift_px);
        let mut ready = self.tracker.take_ready(MIN_SEEN_FRAMES);
        for obj in &mut ready {
            let cx = obj.rect.x as f32 + obj.rect.width as f32 / 2.0;
            let cy = obj.rect.y as f32 + obj.rect.height as f32 / 2.0;
            obj.world_pos = Some(self.transformer.pixel_to_world(cx, cy)?);
        }
        Ok(ready)
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
) -> tokio::task::JoinHandle<()> {
    let (width, height) = camera.resolution();
    let mm_per_px = config.vision.mm_per_px;
    let mut pipeline = VisionPipeline::new(config, width, height);

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

            let result = tokio::task::spawn_blocking(move || {
                let out = pipeline.process_frame(&frame, dt, now);
                (pipeline, out)
            })
            .await
            .context("vision processing task panicked");
            let ready = match result {
                Ok((p, out)) => {
                    pipeline = p;
                    match out {
                        Ok(ready) => ready,
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
            emissions.push(pipeline.process_frame(&frame, dt, Instant::now()).unwrap());
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
    fn empty_frames_yield_no_objects() {
        let mut pipeline = VisionPipeline::new(&test_config(), 640, 480);
        let frame = Mat::new_rows_cols_with_default(480, 640, CV_8UC3, Scalar::all(0.0)).unwrap();
        for _ in 0..5 {
            assert!(
                pipeline
                    .process_frame(&frame, Duration::from_millis(33), Instant::now())
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
