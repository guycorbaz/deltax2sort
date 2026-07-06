use super::DetectedObject;
use opencv::core::Rect;

/// One physical object being followed across frames. All geometry is in
/// pixel coordinates (camera frame).
pub struct TrackedObject {
    pub id: u64,
    pub last_rect: Rect,
    /// Predicted center in pixels. Refreshed from the detection on a match;
    /// advanced by the belt shift on a miss, so the prediction keeps
    /// following the belt across consecutive missed frames.
    pub center: (f32, f32),
    /// Consecutive frames without a matching detection.
    pub missed_frames: u32,
    /// Total frames in which this track was detected.
    pub frames_seen: u32,
    /// Set once the track has been emitted by `take_ready` — this is the
    /// one-object-one-Pick guarantee.
    pub reported: bool,
    pub detected: DetectedObject,
}

fn rect_center(r: &Rect) -> (f32, f32) {
    (
        r.x as f32 + r.width as f32 / 2.0,
        r.y as f32 + r.height as f32 / 2.0,
    )
}

/// Multi-frame object tracker: greedy nearest-neighbour matching of
/// detections to active tracks, with belt-motion prediction along pixel +y.
pub struct Tracker {
    next_id: u64,
    tracks: Vec<TrackedObject>,
    /// Max distance (px) between a detection center and a track's predicted
    /// center to still count as the same object.
    max_match_dist_px: f32,
    /// Tracks unmatched for more than this many consecutive frames are
    /// evicted.
    max_missed_frames: u32,
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            tracks: Vec::new(),
            // Tuning constants; move to config if field experience demands.
            max_match_dist_px: 60.0,
            max_missed_frames: 5,
        }
    }

    /// Ingest one frame's detections. `belt_shift_px` is how far the belt
    /// moved objects along pixel +y since the previous frame (signed).
    pub fn update(&mut self, detections: Vec<DetectedObject>, belt_shift_px: f32) {
        // Predicted center of each active track: its (already belt-advanced)
        // center moved by this frame's belt motion.
        let predicted: Vec<(f32, f32)> = self
            .tracks
            .iter()
            .map(|t| (t.center.0, t.center.1 + belt_shift_px))
            .collect();

        // Greedy global matching: consider all (track, detection) pairs in
        // order of increasing distance, take each track/detection once.
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, &(px, py)) in predicted.iter().enumerate() {
            for (di, det) in detections.iter().enumerate() {
                let (dx, dy) = rect_center(&det.rect);
                let dist = ((dx - px).powi(2) + (dy - py).powi(2)).sqrt();
                if dist <= self.max_match_dist_px {
                    pairs.push((dist, ti, di));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut track_matched = vec![false; self.tracks.len()];
        let mut det_matched = vec![false; detections.len()];
        let mut assignments: Vec<(usize, usize)> = Vec::new();
        for (_, ti, di) in pairs {
            if !track_matched[ti] && !det_matched[di] {
                track_matched[ti] = true;
                det_matched[di] = true;
                assignments.push((ti, di));
            }
        }

        let mut detections: Vec<Option<DetectedObject>> =
            detections.into_iter().map(Some).collect();

        // Matched tracks: refresh geometry/timestamps, keep id and reported.
        for (ti, di) in assignments {
            let track = &mut self.tracks[ti];
            if let Some(mut det) = detections[di].take() {
                det.id = track.id;
                track.last_rect = det.rect;
                track.center = rect_center(&det.rect);
                track.detected = det;
                track.missed_frames = 0;
                track.frames_seen += 1;
            }
        }

        // Unmatched tracks age and drift with the belt (so a track missed
        // for several frames is still predicted where the belt carried it);
        // evict the ones gone too long.
        for (ti, matched) in track_matched.iter().enumerate() {
            if !matched {
                self.tracks[ti].missed_frames += 1;
                self.tracks[ti].center.1 += belt_shift_px;
            }
        }
        let max_missed = self.max_missed_frames;
        self.tracks.retain(|t| t.missed_frames <= max_missed);

        // Unmatched detections become new tracks.
        for det in detections.into_iter().flatten() {
            let id = self.next_id;
            self.next_id += 1;
            let mut det = det;
            det.id = id;
            self.tracks.push(TrackedObject {
                id,
                last_rect: det.rect,
                center: rect_center(&det.rect),
                missed_frames: 0,
                frames_seen: 1,
                reported: false,
                detected: det,
            });
        }
    }

    /// Bounding box + class name of every track matched in the current frame
    /// (`missed_frames == 0`), for the live overlay. Only currently-visible
    /// tracks are returned, so a box is never drawn over empty belt. The class
    /// is `None` until the object is recognised.
    pub fn current_overlays(&self) -> Vec<(Rect, Option<super::ClassName>)> {
        self.tracks
            .iter()
            .filter(|t| t.missed_frames == 0)
            .map(|t| (t.last_rect, t.detected.class.clone()))
            .collect()
    }

    /// Return every track detected in at least `min_seen` frames that has
    /// not been reported yet, marking it reported. Each physical object is
    /// therefore returned exactly once over its lifetime.
    pub fn take_ready(&mut self, min_seen: u32) -> Vec<DetectedObject> {
        let mut ready = Vec::new();
        for track in &mut self.tracks {
            if !track.reported && track.frames_seen >= min_seen {
                track.reported = true;
                ready.push(track.detected.clone());
            }
        }
        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Instant, SystemTime};

    fn detection_at(x: i32, y: i32) -> DetectedObject {
        DetectedObject {
            id: 0,
            rect: Rect::new(x, y, 20, 20),
            world_pos: None,
            class: None,
            confidence: 0.0,
            timestamp: SystemTime::now(),
            seen_at: Instant::now(),
        }
    }

    #[test]
    fn moving_object_keeps_its_id_and_is_reported_once() {
        let mut tracker = Tracker::new();
        // Belt moves the object 30 px per frame along +y; the tracker
        // predicts that, so matching survives the motion.
        for frame in 0..10 {
            tracker.update(vec![detection_at(100, 100 + 30 * frame)], 30.0);
            let ready = tracker.take_ready(3);
            if frame == 2 {
                assert_eq!(ready.len(), 1, "ready on 3rd sighting");
                assert_eq!(ready[0].id, 1, "id stable across frames");
            } else {
                assert!(ready.is_empty(), "frame {frame}: no duplicate emission");
            }
        }
    }

    #[test]
    fn flickering_object_is_still_reported_once() {
        let mut tracker = Tracker::new();
        tracker.update(vec![detection_at(100, 100)], 0.0);
        tracker.update(vec![detection_at(100, 105)], 0.0);
        assert!(tracker.take_ready(3).is_empty());
        // Missed for 2 frames (within max_missed_frames = 5)...
        tracker.update(vec![], 0.0);
        tracker.update(vec![], 0.0);
        // ...then reappears near the predicted position: same track.
        tracker.update(vec![detection_at(102, 108)], 0.0);
        let ready = tracker.take_ready(3);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, 1);
        tracker.update(vec![detection_at(102, 110)], 0.0);
        assert!(tracker.take_ready(3).is_empty(), "already reported");
    }

    #[test]
    fn stale_track_is_evicted_and_new_object_gets_new_id() {
        let mut tracker = Tracker::new();
        for _ in 0..3 {
            tracker.update(vec![detection_at(100, 100)], 0.0);
        }
        assert_eq!(tracker.take_ready(3).len(), 1);
        // Gone for more than max_missed_frames (5): evicted.
        for _ in 0..6 {
            tracker.update(vec![], 0.0);
        }
        assert!(tracker.tracks.is_empty(), "track evicted after 6 misses");
        // A new object at the same place is a NEW track (new id, new Pick).
        for _ in 0..3 {
            tracker.update(vec![detection_at(100, 100)], 0.0);
        }
        let ready = tracker.take_ready(3);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, 2);
    }

    #[test]
    fn missed_track_prediction_accumulates_belt_shift() {
        let mut tracker = Tracker::new();
        // Belt moves 30 px/frame. Object seen once at y=100, then occluded
        // for 4 frames (belt carries it to y≈250 — far beyond the 60 px
        // match radius of its ORIGINAL position), then reappears there.
        tracker.update(vec![detection_at(100, 100)], 30.0);
        for _ in 0..4 {
            tracker.update(vec![], 30.0);
        }
        tracker.update(vec![detection_at(100, 250)], 30.0);
        assert_eq!(
            tracker.tracks.len(),
            1,
            "reappearing object must rejoin its belt-drifted track, not spawn a new one"
        );
        assert_eq!(tracker.tracks[0].id, 1);
    }

    #[test]
    fn current_overlays_shows_only_tracks_visible_this_frame() {
        let mut tracker = Tracker::new();
        // Two objects seen this frame → both overlaid.
        tracker.update(vec![detection_at(100, 100), detection_at(300, 300)], 0.0);
        assert_eq!(tracker.current_overlays().len(), 2);
        // Next frame only one is detected: the missed one must NOT be drawn
        // (no box over empty belt), so exactly one overlay remains.
        tracker.update(vec![detection_at(100, 100)], 0.0);
        let overlays = tracker.current_overlays();
        assert_eq!(overlays.len(), 1);
        assert_eq!(overlays[0].0, Rect::new(100, 100, 20, 20));
    }

    #[test]
    fn far_detection_is_a_separate_track() {
        let mut tracker = Tracker::new();
        tracker.update(vec![detection_at(100, 100)], 0.0);
        // 200 px away: beyond max_match_dist_px (60), must not steal the id.
        tracker.update(vec![detection_at(300, 100)], 0.0);
        assert_eq!(tracker.tracks.len(), 2);
        assert_ne!(tracker.tracks[0].id, tracker.tracks[1].id);
    }
}
