use super::DetectedObject;
use opencv::core::Rect;
use std::collections::HashMap;

pub struct TrackedObject {
    pub id: u64,
    pub last_rect: Rect,
    pub missed_frames: u32,
    pub detected: DetectedObject,
}

pub struct Tracker {
    next_id: u64,
    active_objects: HashMap<u64, TrackedObject>,
    max_missed_frames: u32,
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            active_objects: HashMap::new(),
            max_missed_frames: 5,
        }
    }

    pub fn update(&mut self, detections: Vec<DetectedObject>, _belt_shift_y: f32) -> Vec<TrackedObject> {
        // Simple IOU or distance matching would go here.
        // For Phase 2 (Basic), we just assume new detections are new objects if list is empty,
        // or clear list.
        // Real implementation: Match detections to active_objects predicted positions.
        
        let mut result = Vec::new();
        
        // Very lazy tracking: If we see objects, replace tracking list. 
        // Real tracking requires state persistence.
        self.active_objects.clear(); 
        
        for det in detections {
            let id = self.next_id;
            self.next_id += 1;
            
            let obj = TrackedObject {
                id,
                last_rect: det.rect,
                missed_frames: 0,
                detected: det,
            };
            self.active_objects.insert(id, obj);
        }

        for obj in self.active_objects.values() {
            result.push(TrackedObject {
                id: obj.id, 
                last_rect: obj.last_rect,
                missed_frames: obj.missed_frames,
                detected: obj.detected.clone()
            });
        }
        
        result
    }
}
