use super::DetectedObject;
use super::ObjectClass;
use crate::app_config::VisionConfig;
use anyhow::Result;
use opencv::{
    core::{self, Mat, Size},
    imgproc,
};
use std::time::{Instant, SystemTime};

pub struct BlobDetector {
    min_area: f64,
    max_area: f64,
    threshold: f64,
    invert: bool,
}

impl BlobDetector {
    // Convenience default-config constructor; production code goes through
    // `from_config`.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::from_config(&VisionConfig::default())
    }

    pub fn from_config(cfg: &VisionConfig) -> Self {
        Self {
            min_area: cfg.min_area,
            max_area: cfg.max_area,
            threshold: cfg.threshold,
            invert: cfg.invert,
        }
    }

    /// `captured_at` is when the frame left the camera (as close to capture
    /// as the caller can measure) — NOT the processing time. It becomes
    /// `seen_at` on every detection so staleness checks include the
    /// capture-to-processing latency.
    pub fn detect(&self, frame: &Mat, captured_at: Instant) -> Result<Vec<DetectedObject>> {
        let mut gray = Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut blurred = Mat::default();
        imgproc::gaussian_blur(
            &gray,
            &mut blurred,
            Size::new(5, 5),
            0.0,
            0.0,
            core::BORDER_DEFAULT,
        )?;

        // Fixed binary threshold from config. `invert` selects whether
        // objects are darker (INV) or lighter than the belt. Adaptive/OTSU
        // thresholding is a possible future improvement (docs/TODO.md).
        let threshold_type = if self.invert {
            imgproc::THRESH_BINARY_INV
        } else {
            imgproc::THRESH_BINARY
        };
        let mut thresh = Mat::default();
        imgproc::threshold(&blurred, &mut thresh, self.threshold, 255.0, threshold_type)?;

        let mut contours = core::Vector::<core::Vector<core::Point>>::new();
        imgproc::find_contours(
            &thresh,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            core::Point::new(0, 0),
        )?;

        let mut objects = Vec::new();

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;

            if area >= self.min_area && area <= self.max_area {
                let rect = imgproc::bounding_rect(&contour)?;

                objects.push(DetectedObject {
                    id: 0, // Assigned by Tracker
                    rect,
                    world_pos: None, // Assigned by Calibrator
                    class: ObjectClass::Unknown,
                    confidence: 0.0,
                    timestamp: SystemTime::now(),
                    seen_at: captured_at,
                });
            }
        }

        Ok(objects)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Point, Rect, Scalar};

    /// Black 640x480 BGR frame with one white filled rectangle.
    /// Detection uses `invert = false` (objects LIGHTER than the belt) —
    /// the config default `invert = true` expects darker-than-belt objects.
    fn frame_with_white_rect(rect: Rect) -> Mat {
        let mut frame =
            Mat::new_rows_cols_with_default(480, 640, core::CV_8UC3, Scalar::all(0.0)).unwrap();
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

    fn light_object_config() -> VisionConfig {
        VisionConfig {
            invert: false,
            ..VisionConfig::default()
        }
    }

    #[test]
    fn white_rect_yields_one_detection() {
        let detector = BlobDetector::from_config(&light_object_config());
        let frame = frame_with_white_rect(Rect::new(100, 200, 40, 40));
        let objects = detector.detect(&frame, Instant::now()).unwrap();
        assert_eq!(objects.len(), 1);
        // Blur widens the blob by a pixel or two; the center must hold.
        let r = objects[0].rect;
        let center = Point::new(r.x + r.width / 2, r.y + r.height / 2);
        assert!((center.x - 120).abs() <= 2, "center x: {}", center.x);
        assert!((center.y - 220).abs() <= 2, "center y: {}", center.y);
    }

    #[test]
    fn empty_frame_yields_no_detections() {
        let detector = BlobDetector::from_config(&light_object_config());
        let frame =
            Mat::new_rows_cols_with_default(480, 640, core::CV_8UC3, Scalar::all(0.0)).unwrap();
        assert!(detector.detect(&frame, Instant::now()).unwrap().is_empty());
    }
}
