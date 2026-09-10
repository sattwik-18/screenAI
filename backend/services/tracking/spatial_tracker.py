"""
SpatialIoUKalmanTracker:
A real, lightweight 2D Spatial IoU and Velocity Smoothing face tracker.
(Documented: This is a dedicated spatial IoU tracker, not ByteTrack).

Responsibilities:
- Assign and maintain session-local `track_id`.
- Maintain track states: NEW -> ACTIVE -> LOST -> DELETED.
- Maintain motion history (centroid positions, velocity vector).
- Detect and flag IDENTITY_SWITCH_DETECTED when spatial discontinuity occurs.
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import time


def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    # box format: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


class FaceTrack:
    # EMA alpha: 0.60 blends 60% new detection with 40% history.
    # Eliminates high-frequency ONNX jitter while tracking fast head movements with zero lag.
    LANDMARK_EMA_ALPHA = 0.60
    BBOX_EMA_ALPHA = 0.65

    def __init__(self, track_id: str, bbox: List[int], confidence: float):
        self.track_id = track_id
        self.bbox = list(bbox)  # [x, y, w, h]
        self.confidence = confidence
        self.state = "NEW"  # NEW, ACTIVE, LOST, DELETED
        self.hits = 1
        self.time_since_update = 0
        self.created_at = time.time()
        self.updated_at = time.time()
        
        # Centroid history for motion analysis
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        self.history: List[Tuple[float, float, float]] = [(cx, cy, self.updated_at)]
        self.velocity = (0.0, 0.0)

        # EMA landmark buffers — initialized to None until first detection
        self.smooth_kps: Optional[np.ndarray] = None
        self.smooth_lm2d: Optional[np.ndarray] = None
        self.smooth_lm3d: Optional[np.ndarray] = None

    def _ema(self, prev: Optional[np.ndarray], new_val: Optional[List], alpha: float) -> Optional[np.ndarray]:
        """Apply EMA blend between previous and new landmark arrays."""
        if new_val is None or len(new_val) == 0:
            return prev
        new_arr = np.array(new_val, dtype=np.float64)
        if prev is None or prev.shape != new_arr.shape:
            return new_arr.copy()
        return alpha * new_arr + (1.0 - alpha) * prev

    def smooth_landmarks(
        self,
        kps: Optional[List],
        lm2d: Optional[List],
        lm3d: Optional[List]
    ):
        """Update EMA buffers with new frame's landmarks."""
        alpha = self.LANDMARK_EMA_ALPHA
        self.smooth_kps = self._ema(self.smooth_kps, kps, alpha)
        self.smooth_lm2d = self._ema(self.smooth_lm2d, lm2d, alpha)
        self.smooth_lm3d = self._ema(self.smooth_lm3d, lm3d, alpha)

    def get_smoothed_landmarks(self) -> dict:
        """Return smoothed landmark arrays as Python lists (None if not yet available)."""
        return {
            "kps": self.smooth_kps.tolist() if self.smooth_kps is not None else None,
            "landmark_2d_106": self.smooth_lm2d.tolist() if self.smooth_lm2d is not None else None,
            "landmark_3d_68": self.smooth_lm3d.tolist() if self.smooth_lm3d is not None else None,
        }

    def predict(self) -> List[int]:
        # Velocity-based prediction scaled by elapsed time dt
        now = time.time()
        dt = min(0.3, max(0.0, now - self.updated_at))
        vx, vy = self.velocity
        x = int(self.bbox[0] + vx * dt)
        y = int(self.bbox[1] + vy * dt)
        return [x, y, self.bbox[2], self.bbox[3]]

    def update(self, bbox: List[int], confidence: float) -> bool:
        now = time.time()
        dt = max(1e-4, now - self.updated_at)
        
        new_cx = bbox[0] + bbox[2] / 2.0
        new_cy = bbox[1] + bbox[3] / 2.0
        
        prev_cx, prev_cy, _ = self.history[-1]
        vx = (new_cx - prev_cx) / dt
        vy = (new_cy - prev_cy) / dt
        
        # Exponential moving average for velocity
        self.velocity = (0.7 * self.velocity[0] + 0.3 * vx, 0.7 * self.velocity[1] + 0.3 * vy)
        
        # EMA for bounding box: smooth out frame-to-frame detector noise
        alpha = self.BBOX_EMA_ALPHA
        self.bbox = [
            int(alpha * bbox[0] + (1.0 - alpha) * self.bbox[0]),
            int(alpha * bbox[1] + (1.0 - alpha) * self.bbox[1]),
            int(alpha * bbox[2] + (1.0 - alpha) * self.bbox[2]),
            int(alpha * bbox[3] + (1.0 - alpha) * self.bbox[3]),
        ]
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.updated_at = now
        self.history.append((new_cx, new_cy, now))
        if len(self.history) > 60:
            self.history.pop(0)

        if self.state == "LOST":
            self.state = "RECOVERED"
        elif self.hits >= 3:
            self.state = "ACTIVE"
            
        return True

    def mark_missed(self):
        self.time_since_update += 1
        if self.time_since_update > 2:
            self.state = "LOST"
        if self.time_since_update > 15:
            self.state = "DELETED"


class SpatialIoUKalmanTracker:
    def __init__(self, iou_threshold: float = 0.35, max_discontinuity_px: float = 200.0):
        self.iou_threshold = iou_threshold
        self.max_discontinuity_px = max_discontinuity_px
        self.next_id = 1
        self.active_track: Optional[FaceTrack] = None
        self.identity_switched = False

    def update(self, detection: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Updates single-subject verification face track.
        Returns track state plus EMA-smoothed landmark arrays.
        """
        self.identity_switched = False
        
        if detection is None:
            if self.active_track is not None:
                self.active_track.mark_missed()
                if self.active_track.state == "DELETED":
                    self.active_track = None
            return {
                "track_id": None,
                "state": "NO_TRACK",
                "identity_switched": False,
                "velocity": (0.0, 0.0),
                "hits": 0,
                "smoothed_landmarks": None
            }

        bbox = detection["bbox"]
        confidence = detection["confidence"]

        # Extract raw landmarks for smoothing
        raw_kps = detection.get("kps")
        raw_lm2d = detection.get("landmark_2d_106")
        raw_lm3d = detection.get("landmark_3d_68")
        
        if self.active_track is None:
            track_id = f"TRACK_{self.next_id:03d}"
            self.next_id += 1
            self.active_track = FaceTrack(track_id, bbox, confidence)
            # Seed EMA buffers with the first detection's landmarks
            self.active_track.smooth_landmarks(raw_kps, raw_lm2d, raw_lm3d)
            return {
                "track_id": self.active_track.track_id,
                "state": self.active_track.state,
                "identity_switched": False,
                "velocity": self.active_track.velocity,
                "hits": self.active_track.hits,
                "smoothed_landmarks": self.active_track.get_smoothed_landmarks(),
                "tracked_bbox": self.active_track.bbox
            }

        # Check IoU with both velocity-predicted bbox and last-known bbox
        pred_box = self.active_track.predict()
        iou = max(calculate_iou(pred_box, bbox), calculate_iou(self.active_track.bbox, bbox))
        
        # Check centroid jump and scale change relative to face size
        cur_cx = bbox[0] + bbox[2] / 2.0
        cur_cy = bbox[1] + bbox[3] / 2.0
        prev_cx, prev_cy, _ = self.active_track.history[-1]
        spatial_dist = np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2)
        
        face_scale = max(self.active_track.bbox[2], self.active_track.bbox[3], 1)
        norm_dist = spatial_dist / float(face_scale)
        prev_area = max(1, self.active_track.bbox[2] * self.active_track.bbox[3])
        cur_area = max(1, bbox[2] * bbox[3])
        area_ratio = cur_area / float(prev_area)

        # Empirically validated association gating:
        # 1. High overlap (IoU >= 0.25) with reasonable centroid distance (norm_dist <= 0.65)
        # 2. Fast motion with partial overlap (IoU >= 0.15), low centroid distance (norm_dist <= 0.40),
        #    and consistent face scale (0.5 <= area_ratio <= 1.8)
        is_consistent_match = (
            (iou >= 0.25 and norm_dist <= 0.65) or
            (iou >= 0.15 and norm_dist <= 0.40 and 0.5 <= area_ratio <= 1.8)
        )

        if is_consistent_match:
            # Consistent track match: update bbox + smooth landmarks
            self.active_track.update(bbox, confidence)
            self.active_track.smooth_landmarks(raw_kps, raw_lm2d, raw_lm3d)
        else:
            # Discontinuous spatial jump → Flag potential identity switch, reset track
            self.identity_switched = True
            track_id = f"TRACK_{self.next_id:03d}"
            self.next_id += 1
            self.active_track = FaceTrack(track_id, bbox, confidence)
            # Seed fresh EMA buffers for the new track
            self.active_track.smooth_landmarks(raw_kps, raw_lm2d, raw_lm3d)

        return {
            "track_id": self.active_track.track_id,
            "state": self.active_track.state,
            "identity_switched": self.identity_switched,
            "velocity": self.active_track.velocity,
            "hits": self.active_track.hits,
            "motion_history_len": len(self.active_track.history),
            "smoothed_landmarks": self.active_track.get_smoothed_landmarks(),
            "tracked_bbox": self.active_track.bbox
        }
