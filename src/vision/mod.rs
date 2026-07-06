pub mod calibration;
pub mod catalog;
pub mod classifier;
pub mod detector;
pub mod pipeline;
pub mod tracker;

use crate::hardware::Position;
use opencv::core::Rect;
use std::time::{Instant, SystemTime};

/// The recognised class of an object, as a learned-catalogue label. Identity
/// is the class name — the key the sorting `assignments` map to a bin. Classes
/// are learned independently of bins (see the object catalogue), so this is a
/// free-form name, not a fixed enum. `None` on a [`DetectedObject`] means
/// unrecognised: it is not sorted, and rides the belt to the end catch bin.
pub type ClassName = String;

#[derive(Debug, Clone)]
pub struct DetectedObject {
    /// Stable track id assigned by the tracker (0 before tracking).
    pub id: u64,
    /// Bounding box in pixel coordinates (camera frame).
    pub rect: Rect,
    /// Object center in robot coordinates (mm); z is always 0 (belt plane),
    /// the pick height comes from configuration.
    pub world_pos: Option<Position>,
    /// Recognised class name, or `None` when unrecognised. Drives sorting: an
    /// object is picked only if its class is assigned to a bin.
    pub class: Option<ClassName>,
    // Populated once the classifier leaves stub state (Phase B); carried for
    // the UI/logging. Not read yet.
    #[allow(dead_code)]
    pub confidence: f32,
    /// Wall-clock capture time — for logging only (can go backwards under
    /// NTP); staleness checks must use `seen_at`.
    #[allow(dead_code)]
    pub timestamp: SystemTime,
    /// Monotonic capture time, used for pick staleness checks.
    pub seen_at: Instant,
}
