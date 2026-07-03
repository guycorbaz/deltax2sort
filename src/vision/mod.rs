pub mod calibration;
pub mod classifier;
pub mod detector;
pub mod pipeline;
pub mod tracker;

use crate::hardware::Position;
use opencv::core::Rect;
use std::time::{Instant, SystemTime};

#[derive(Debug, Clone)]
pub enum ObjectClass {
    Unknown,
    // Classifier is a stub (docs/TODO.md): concrete brick classes are not
    // produced yet, only Unknown.
    #[allow(dead_code)]
    Brick2x4,
    #[allow(dead_code)]
    Brick2x2,
    #[allow(dead_code)]
    Brick4x4,
    #[allow(dead_code)]
    Plate2x4,
}

#[derive(Debug, Clone)]
pub struct DetectedObject {
    /// Stable track id assigned by the tracker (0 before tracking).
    pub id: u64,
    /// Bounding box in pixel coordinates (camera frame).
    pub rect: Rect,
    /// Object center in robot coordinates (mm); z is always 0 (belt plane),
    /// the pick height comes from configuration.
    pub world_pos: Option<Position>,
    // class/confidence are populated once the classifier leaves stub state;
    // timestamp is carried for future logging/UI. None is read yet.
    #[allow(dead_code)]
    pub class: ObjectClass,
    #[allow(dead_code)]
    pub confidence: f32,
    /// Wall-clock capture time — for logging only (can go backwards under
    /// NTP); staleness checks must use `seen_at`.
    #[allow(dead_code)]
    pub timestamp: SystemTime,
    /// Monotonic capture time, used for pick staleness checks.
    pub seen_at: Instant,
}
