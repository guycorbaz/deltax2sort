use super::DetectedObject;
use log::debug;
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
    /// Set once recognition has been attempted (whatever the result), so each
    /// object is embedded at most once over its lifetime.
    pub classified: bool,
    pub detected: DetectedObject,
}

fn rect_center(r: &Rect) -> (f32, f32) {
    (
        r.x as f32 + r.width as f32 / 2.0,
        r.y as f32 + r.height as f32 / 2.0,
    )
}

fn rect_area(r: &Rect) -> f32 {
    (r.width.max(0) as f32) * (r.height.max(0) as f32)
}

/// True when two bounding-box areas are within `max_ratio` of each other
/// (larger / smaller ≤ max_ratio). A zero area is never compatible.
fn areas_compatible(a: f32, b: f32, max_ratio: f32) -> bool {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    lo > 0.0 && hi <= lo * max_ratio
}

fn rect_intersection_area(a: &Rect, b: &Rect) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    ((x2 - x1).max(0) as f32) * ((y2 - y1).max(0) as f32)
}

/// Overlap coefficient: intersection over the smaller box area (in [0, 1]).
/// High when one box sits largely inside the other (a split fragment); near
/// zero for boxes that merely touch (adjacent distinct parts).
fn overlap_coeff(a: &Rect, b: &Rect) -> f32 {
    let min_area = rect_area(a).min(rect_area(b));
    if min_area > 0.0 {
        rect_intersection_area(a, b) / min_area
    } else {
        0.0
    }
}

/// A recently evicted track that had already been reported (its Pick emitted).
/// Kept briefly, drifting with the belt, so the same object reappearing after a
/// long occlusion (e.g. the arm sweeping through view during a pick) is tracked
/// again WITHOUT emitting a second Pick.
struct Ghost {
    center: (f32, f32),
    area: f32,
    /// Frames left before this ghost is forgotten.
    ttl: u32,
}

/// Multi-frame object tracker: greedy nearest-neighbour matching of
/// detections to active tracks, with belt-motion prediction along pixel +y.
pub struct Tracker {
    next_id: u64,
    tracks: Vec<TrackedObject>,
    /// Recently evicted, already-reported tracks — see [`Ghost`].
    ghosts: Vec<Ghost>,
    /// How many frames a ghost is remembered (a bit longer than a pick, so an
    /// object occluded for the whole pick is still recognised on reappearance).
    ghost_frames: u32,
    /// Max distance (px) between a detection center and a track's predicted
    /// center to still count as the same object.
    max_match_dist_px: f32,
    /// Largest ratio between a track's and a detection's bounding-box areas for
    /// them to still be considered the same object. Stops a much bigger blob
    /// (e.g. a gripper edge) within the distance radius from stealing a small
    /// part's identity.
    max_area_ratio: f32,
    /// A new detection overlapping an already-tracked object by at least this
    /// coefficient is treated as a split-blob fragment and suppressed (no
    /// duplicate track), rather than a distinct adjacent part.
    min_birth_overlap: f32,
    /// Tracks unmatched for more than this many consecutive frames are
    /// evicted.
    max_missed_frames: u32,
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            tracks: Vec::new(),
            ghosts: Vec::new(),
            // Tuning constants; move to config if field experience demands.
            max_match_dist_px: 60.0,
            max_area_ratio: 3.0,
            min_birth_overlap: 0.5,
            ghost_frames: 30, // ~1 s at 30 fps
            max_missed_frames: 5,
        }
    }

    /// Ingest one frame's detections. `belt_shift_px` is how far the belt
    /// moved objects along pixel +y since the previous frame (signed).
    pub fn update(&mut self, detections: Vec<DetectedObject>, belt_shift_px: f32) {
        // Age ghosts of recently-picked objects: drift them with the belt and
        // forget the expired ones.
        for g in &mut self.ghosts {
            g.center.1 += belt_shift_px;
            g.ttl = g.ttl.saturating_sub(1);
        }
        self.ghosts.retain(|g| g.ttl > 0);

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
            let track_area = rect_area(&self.tracks[ti].last_rect);
            for (di, det) in detections.iter().enumerate() {
                let (dx, dy) = rect_center(&det.rect);
                let dist = ((dx - px).powi(2) + (dy - py).powi(2)).sqrt();
                // Gate on both proximity AND size similarity: a detection far
                // off in size is a different physical thing even if its center
                // lands within the distance radius.
                if dist <= self.max_match_dist_px
                    && areas_compatible(track_area, rect_area(&det.rect), self.max_area_ratio)
                {
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
        // Evict tracks missed too long. A track that was already reported
        // becomes a ghost, so its object is not re-picked if it reappears
        // (a long occlusion, e.g. the arm sweeping through view during a pick).
        let max_missed = self.max_missed_frames;
        let ghost_frames = self.ghost_frames;
        let mut new_ghosts: Vec<Ghost> = self
            .tracks
            .iter()
            .filter(|t| t.missed_frames > max_missed && t.reported)
            .map(|t| Ghost {
                center: t.center,
                area: rect_area(&t.last_rect),
                ttl: ghost_frames,
            })
            .collect();
        self.ghosts.append(&mut new_ghosts);
        self.tracks.retain(|t| t.missed_frames <= max_missed);

        // Unmatched detections become new tracks — unless they are a split of
        // an object already tracked this frame. A threshold split yields a
        // second blob heavily overlapping the first (which matched the track);
        // suppress it so it can't become a duplicate Pick. Adjacent distinct
        // parts barely overlap, so they are left to spawn their own tracks.
        for det in detections.into_iter().flatten() {
            let is_split = self.tracks.iter().any(|t| {
                t.missed_frames == 0 && overlap_coeff(&det.rect, &t.last_rect) >= self.min_birth_overlap
            });
            if is_split {
                debug!(
                    "Tracker: suppressing split-blob detection at {:?} (overlaps a tracked object)",
                    det.rect
                );
                continue;
            }
            // A detection matching a ghost is the same object reappearing after
            // a long occlusion: track it, but pre-mark it reported so it is not
            // picked a second time.
            let (dcx, dcy) = rect_center(&det.rect);
            let det_area = rect_area(&det.rect);
            let ghost_match = self.ghosts.iter().position(|g| {
                let dist = ((dcx - g.center.0).powi(2) + (dcy - g.center.1).powi(2)).sqrt();
                dist <= self.max_match_dist_px
                    && areas_compatible(g.area, det_area, self.max_area_ratio)
            });
            let reported = if let Some(i) = ghost_match {
                self.ghosts.swap_remove(i);
                debug!(
                    "Tracker: reappeared object at {:?} matches a recently-picked track — tracking without re-picking",
                    det.rect
                );
                true
            } else {
                false
            };
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
                reported,
                classified: false,
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

    /// Recognise each visible track that has reached `min_seen` frames and has
    /// not been classified yet, caching the class on the track (so an object is
    /// embedded once, and the overlay + the emitted pick both see the class).
    /// `classify(rect)` returns the recognised class name, or `None`.
    pub fn classify_ready_tracks<F>(&mut self, min_seen: u32, mut classify: F)
    where
        F: FnMut(&Rect) -> Option<super::ClassName>,
    {
        for t in &mut self.tracks {
            if !t.classified && t.missed_frames == 0 && t.frames_seen >= min_seen {
                t.detected.class = classify(&t.last_rect);
                t.classified = true;
            }
        }
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
        detection_sized(x, y, 20, 20)
    }

    fn detection_sized(x: i32, y: i32, w: i32, h: i32) -> DetectedObject {
        DetectedObject {
            id: 0,
            rect: Rect::new(x, y, w, h),
            world_pos: None,
            class: None,
            confidence: 0.0,
            timestamp: SystemTime::now(),
            seen_at: Instant::now(),
        }
    }

    #[test]
    fn a_much_larger_blob_does_not_steal_a_small_tracks_identity() {
        let mut tracker = Tracker::new();
        tracker.update(vec![detection_sized(100, 100, 20, 20)], 0.0); // small part, id 1
        // A large blob (70x70 = 4900 px² vs 400) whose center is only ~21 px
        // away — well inside the 60 px radius, so distance alone would match —
        // must NOT steal the small track; it becomes its own track instead.
        tracker.update(vec![detection_sized(90, 90, 70, 70)], 0.0);
        assert_eq!(
            tracker.tracks.len(),
            2,
            "the oversized blob must not steal the small track"
        );
        let small = tracker.tracks.iter().find(|t| t.id == 1).unwrap();
        assert_eq!(small.last_rect.width, 20, "small track kept its own box");
        assert_eq!(small.missed_frames, 1, "small track went unmatched this frame");
    }

    #[test]
    fn a_split_blob_does_not_spawn_a_duplicate_track() {
        let mut tracker = Tracker::new();
        // Frame 1: the object appears whole.
        tracker.update(vec![detection_sized(100, 100, 40, 40)], 0.0);
        // Frame 2: thresholding splits it into two overlapping blobs at the
        // same spot. One matches the track; the other overlaps it heavily and
        // must be suppressed, not spawn a second track (→ a duplicate Pick).
        tracker.update(
            vec![
                detection_sized(100, 100, 30, 40),
                detection_sized(115, 100, 25, 40),
            ],
            0.0,
        );
        assert_eq!(tracker.tracks.len(), 1, "a split blob must not duplicate the track");
    }

    #[test]
    fn a_split_at_birth_yields_one_track() {
        let mut tracker = Tracker::new();
        // Two heavily overlapping blobs the very first frame (a split object).
        tracker.update(
            vec![
                detection_sized(100, 100, 30, 30),
                detection_sized(110, 100, 30, 30),
            ],
            0.0,
        );
        assert_eq!(tracker.tracks.len(), 1, "two overlapping birth blobs = one object");
    }

    #[test]
    fn adjacent_distinct_parts_stay_separate() {
        let mut tracker = Tracker::new();
        // Two small parts side by side, not overlapping: legitimately distinct.
        tracker.update(
            vec![
                detection_sized(100, 100, 20, 20),
                detection_sized(140, 100, 20, 20),
            ],
            0.0,
        );
        assert_eq!(tracker.tracks.len(), 2, "distinct adjacent parts must stay separate");
    }

    #[test]
    fn a_moderate_size_change_still_matches_the_same_track() {
        let mut tracker = Tracker::new();
        tracker.update(vec![detection_sized(100, 100, 20, 20)], 0.0); // 400 px²
        // Grows to 30x30 = 900 px² (ratio 2.25 < 3): still the same object.
        tracker.update(vec![detection_sized(100, 100, 30, 30)], 0.0);
        assert_eq!(tracker.tracks.len(), 1, "moderate growth keeps one track");
        assert_eq!(tracker.tracks[0].id, 1);
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
    fn reappearance_after_long_occlusion_is_tracked_but_not_re_picked() {
        // #5: an object picked (reported) then occluded long enough to be
        // evicted must NOT be picked again when it reappears.
        let mut tracker = Tracker::new();
        for _ in 0..3 {
            tracker.update(vec![detection_at(100, 100)], 0.0);
        }
        assert_eq!(tracker.take_ready(3).len(), 1, "first Pick");
        // Gone for more than max_missed_frames (5): evicted → becomes a ghost.
        for _ in 0..6 {
            tracker.update(vec![], 0.0);
        }
        assert!(tracker.tracks.is_empty(), "track evicted after 6 misses");
        // Reappears at the same place (belt stopped): a fresh track (new id),
        // but pre-marked reported, so no duplicate Pick.
        for _ in 0..3 {
            tracker.update(vec![detection_at(100, 100)], 0.0);
        }
        assert!(
            tracker.take_ready(3).is_empty(),
            "the reappearing object must not be picked a second time"
        );
        assert_eq!(tracker.tracks.len(), 1, "it is still tracked (id 2)");
        assert_eq!(tracker.tracks[0].id, 2);
    }

    #[test]
    fn a_genuinely_new_object_after_the_ghost_expires_is_picked() {
        let mut tracker = Tracker::new();
        for _ in 0..3 {
            tracker.update(vec![detection_at(100, 100)], 0.0);
        }
        assert_eq!(tracker.take_ready(3).len(), 1);
        for _ in 0..6 {
            tracker.update(vec![], 0.0); // evicted → ghost
        }
        // Wait out the ghost window (ghost_frames = 30).
        for _ in 0..31 {
            tracker.update(vec![], 0.0);
        }
        // A new object at the same spot is now a genuine new Pick.
        for _ in 0..3 {
            tracker.update(vec![detection_at(100, 100)], 0.0);
        }
        assert_eq!(
            tracker.take_ready(3).len(),
            1,
            "after the ghost expires a new object is picked"
        );
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
    fn ready_tracks_are_classified_once_and_cached() {
        let mut tracker = Tracker::new();
        let mut calls = 0;
        // Below min_seen: not classified yet.
        tracker.update(vec![detection_at(100, 100)], 0.0);
        tracker.update(vec![detection_at(100, 100)], 0.0);
        tracker.classify_ready_tracks(3, |_| {
            calls += 1;
            Some("brick".to_string())
        });
        assert_eq!(calls, 0, "not classified before reaching min_seen");
        // Third sighting → ready → classified exactly once.
        tracker.update(vec![detection_at(100, 100)], 0.0);
        tracker.classify_ready_tracks(3, |_| {
            calls += 1;
            Some("brick".to_string())
        });
        assert_eq!(calls, 1);
        assert_eq!(tracker.current_overlays()[0].1.as_deref(), Some("brick"));
        // A later frame must NOT re-embed (cached).
        tracker.update(vec![detection_at(100, 100)], 0.0);
        tracker.classify_ready_tracks(3, |_| {
            calls += 1;
            Some("brick".to_string())
        });
        assert_eq!(calls, 1, "an object is classified at most once");
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
