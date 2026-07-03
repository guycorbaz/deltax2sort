use super::DetectedObject;
use super::ObjectClass;
use crate::app_config::VisionConfig;
use anyhow::Result;
use opencv::{
    core::{self, Mat, Size},
    imgproc,
};
use std::time::SystemTime;

pub struct BlobDetector {
    min_area: f64,
    max_area: f64,
    threshold: f64,
    invert: bool,
}

impl BlobDetector {
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

    pub fn detect(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
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
                });
            }
        }

        Ok(objects)
    }
}
