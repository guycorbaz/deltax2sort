# Deferred Work

Findings surfaced by reviews but out of scope for their story. Each item is
migrated to a GitHub issue (see links) — GitHub is the tracker of record.

## From spec-wire-vision-loop review (2026-07-03)

1. **Stale picks are only checked at enqueue time** — the 7-command sequence
   holds frozen coordinates and can execute long after the TTL passed when
   picks queue up. Needs an expiry check when a pick sequence *starts*
   executing (drop whole group, never mid-sequence).
2. **Picking at last-seen position is incoherent with a moving belt** — even a
   fresh pick at 100 mm/s misses; PICK_TTL only trims the extreme tail.
   Blocked on robot `Position` feedback + intercept planning (existing TODO).
3. **Declined picks lose the object permanently** — `reported` is set before
   the orchestrator accepts; a pick dropped (paused/stale) is never re-emitted.
   Needs accept/decline feedback or re-arm semantics.
4. **Long occlusion still yields double picks** — belt-drift accumulation now
   survives ~5 missed frames, but an arm sweep longer than that evicts the
   track and the reappearing object re-emits.
5. **Blob split spawns duplicate tracks** — a detection born within an existing
   track's radius (thresholding split) becomes a second track → second pick.
6. **Blocking `VideoCapture::read` on the async executor, no deadline** —
   dedicated camera thread + channel (existing TODO), USB stall hangs a worker.
7. **Nothing can send `Resume`** — one command failure pauses the orchestrator
   forever and all picks are silently dropped; UI needs Resume + state display.
8. **Size-blind track matching** — a large blob near a track's predicted center
   steals its identity; factor area/shape into the match cost.
9. **Objects seen < min_seen frames vanish silently** — no counter/log when a
   fast object crosses the view with 1-2 sightings.
10. **Vision-loop death is invisible** — `JoinHandle` discarded in main;
    UI keeps showing "Connected" after a permanent vision stop.
11. **Scalar `mm_per_px` cannot express rotation/offset** — calibration wizard
    (existing TODO) must measure the full affine transform.
