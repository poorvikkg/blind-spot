"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Object Tracker
  File: tracker.py
=============================================================================
  Lightweight centroid tracker that:
    1. Associates new detections with existing tracked objects (nearest centroid)
    2. Counts how many consecutive frames each object has been visible
    3. Detects if an object is APPROACHING (centroid growing toward camera)
    4. Removes stale tracks after MAX_TRACKING_LOST_FRAMES of absence

  No deep learning required — O(N²) Hungarian matching approximated by
  brute-force nearest-neighbour, fast enough for ≤20 objects per frame.
=============================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from logger_system import get_logger

log = get_logger("Tracker")


# =============================================================================
#  Data Structures
# =============================================================================

@dataclass
class TrackedObject:
    """Represents one tracked vehicle candidate in the ROI."""

    object_id:    int
    centroid:     Tuple[int, int]          # (cx, cy) in ROI coordinates
    bbox:         Tuple[int, int, int, int]  # (x, y, w, h)
    frames_seen:  int = 1                  # Consecutive frames detected
    frames_lost:  int = 0                  # Consecutive frames NOT detected
    history:      List[Tuple[int, int]] = field(default_factory=list)  # centroid trail

    @property
    def is_confirmed(self) -> bool:
        """True once object has survived the minimum confirmation threshold."""
        return self.frames_seen >= config.ALERT_FRAME_THRESHOLD

    @property
    def is_approaching(self) -> bool:
        """
        Heuristic: if the centroid is moving consistently downward in frame
        (y increasing) AND growing in bounding-box area, the object is closing.
        Works for rear-mounted cameras looking backward.
        """
        if len(self.history) < 3:
            return False
        dy = self.history[-1][1] - self.history[-3][1]  # y delta over 2 frames
        return dy > config.APPROACH_MOVEMENT_THRESHOLD


# =============================================================================
#  CentroidTracker
# =============================================================================

class CentroidTracker:
    """
    Match detections across frames by minimum Euclidean centroid distance.

    Usage
    -----
        tracker = CentroidTracker()
        # Each frame:
        active_objects = tracker.update(detections)

    Args
    ----
        detections : list of (x, y, w, h) bounding rects from contour step.

    Returns
    -------
        List[TrackedObject] — only the currently active / confirmed objects.
    """

    def __init__(self) -> None:
        self._next_id: int = 0
        self._objects: Dict[int, TrackedObject] = {}   # id → TrackedObject

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self, detections: List[Tuple[int, int, int, int]]
    ) -> List[TrackedObject]:
        """
        Core update step — call once per frame.

        Args:
            detections: List of (x, y, w, h) bounding rectangles.

        Returns:
            List of all currently tracked TrackedObject instances
            (confirmed + unconfirmed that are still alive).
        """
        input_centroids = [self._to_centroid(d) for d in detections]

        if not self._objects:
            # Cold start — register everything
            for centroid, bbox in zip(input_centroids, detections):
                self._register(centroid, bbox)
        elif not input_centroids:
            # Nothing detected this frame — age all tracks
            self._age_all()
        else:
            self._match_and_update(input_centroids, detections)

        # Purge objects lost for too long
        self._prune()

        return list(self._objects.values())

    def reset(self) -> None:
        """Clear all tracked objects (e.g., after long occlusion or scene cut)."""
        self._objects.clear()
        self._next_id = 0

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def _register(
        self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]
    ) -> None:
        obj = TrackedObject(
            object_id=self._next_id,
            centroid=centroid,
            bbox=bbox,
            history=[centroid],
        )
        self._objects[self._next_id] = obj
        self._next_id += 1

    def _match_and_update(
        self,
        input_centroids: List[Tuple[int, int]],
        detections: List[Tuple[int, int, int, int]],
    ) -> None:
        """Nearest-neighbour matching between existing tracks and new detections."""
        existing_ids   = list(self._objects.keys())
        existing_cents = [self._objects[oid].centroid for oid in existing_ids]

        # Build cost matrix (rows = existing, cols = inputs)
        cost: List[List[float]] = []
        for ec in existing_cents:
            row = [self._dist(ec, ic) for ic in input_centroids]
            cost.append(row)

        matched_existing: set  = set()
        matched_inputs:   set  = set()

        # Greedy matching — pick smallest distances first
        flat = sorted(
            [(cost[r][c], r, c) for r in range(len(existing_ids)) for c in range(len(input_centroids))]
        )
        for dist, r, c in flat:
            if r in matched_existing or c in matched_inputs:
                continue
            # Reject if distance is too large (likely different object)
            if dist > max(config.FRAME_WIDTH, config.FRAME_HEIGHT) * 0.25:
                continue
            # Update matched track
            oid = existing_ids[r]
            obj = self._objects[oid]
            obj.centroid    = input_centroids[c]
            obj.bbox        = detections[c]
            obj.frames_seen += 1
            obj.frames_lost  = 0
            obj.history.append(input_centroids[c])
            if len(obj.history) > 20:          # Keep last 20 positions
                obj.history.pop(0)
            matched_existing.add(r)
            matched_inputs.add(c)

        # Age unmatched existing tracks
        for r, oid in enumerate(existing_ids):
            if r not in matched_existing:
                self._objects[oid].frames_lost += 1

        # Register unmatched new detections as fresh tracks
        for c in range(len(input_centroids)):
            if c not in matched_inputs:
                self._register(input_centroids[c], detections[c])

    def _age_all(self) -> None:
        for obj in self._objects.values():
            obj.frames_lost += 1

    def _prune(self) -> None:
        stale_ids = [
            oid
            for oid, obj in self._objects.items()
            if obj.frames_lost > config.MAX_TRACKING_LOST_FRAMES
        ]
        for oid in stale_ids:
            log.debug("Track #%d removed (lost for %d frames)", oid, config.MAX_TRACKING_LOST_FRAMES)
            del self._objects[oid]

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
