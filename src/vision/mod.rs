pub mod calibration;
pub mod classifier;
pub mod detector;
pub mod tracker;

use crate::hardware::Position;
use opencv::core::Rect;
use std::time::SystemTime;

#[derive(Debug, Clone)]
pub enum ObjectClass {
    Unknown,
    Brick2x4,
    Brick2x2,
    Brick4x4,
    Plate2x4,
}

#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub id: u64,
    pub rect: Rect,                  // Bounding box in Pixel coords
    pub world_pos: Option<Position>, // Center in Robot coords
    pub class: ObjectClass,
    pub confidence: f32,
    pub timestamp: SystemTime,
}
