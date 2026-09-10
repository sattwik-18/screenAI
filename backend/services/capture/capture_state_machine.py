"""
Guided Multi-Angle Capture State Machine.
Manages:
- Direct 3-step sequence: FRONT -> LEFT (~20°) -> RIGHT (~20°).
- Hold-to-capture stabilization: Requires 3 consecutive frames in the target zone.
- Auto-capture: Captures crisp face screenshot thumbnail (base64 JPEG) and ArcFace embedding.
- Direction tracking guidance and yaw target windows.
- Persistent candidate buffer for multi-frame biometric verification matching.
"""

from typing import Dict, Any, List, Optional
import time
import base64
import cv2
import numpy as np


class BurstCandidate:
    def __init__(
        self,
        frame_bgr: np.ndarray,
        face_dict: Dict[str, Any],
        pose: Dict[str, Any],
        quality: Dict[str, Any],
        liveness: Dict[str, Any],
        embedding: np.ndarray,
        angle_name: str
    ):
        self.frame_bgr = frame_bgr
        self.face_dict = face_dict
        self.pose = pose
        self.quality = quality
        self.liveness = liveness
        self.embedding = embedding
        self.angle_name = angle_name
        self.timestamp = time.time()
        self.score = quality.get("composite_score", 0.0)


class CaptureStateMachine:
    def __init__(self):
        self.state = "CAPTURING_FRONT"  # WAITING_FOR_FACE, CAPTURING_FRONT, CAPTURING_LEFT, CAPTURING_RIGHT, CAPTURE_COMPLETE
        self.hold_count = 0
        self.required_hold_frames = 8  # ~0.5s at 16-18 FPS: gives user time to settle and eliminate motion blur
        
        # Captured snapshots per angle (contains base64 screenshot thumbnails)
        self.captured_snapshots: Dict[str, Dict[str, Any]] = {}

        # Best candidate frame buffer per angle collected during the hold period
        self.best_hold_candidate: Dict[str, Optional[Dict[str, Any]]] = {
            "FRONT": None,
            "LEFT": None,
            "RIGHT": None
        }

        # Burst candidate buffers per angle for evidence fusion & matching
        self.burst_candidates: Dict[str, List[BurstCandidate]] = {
            "FRONT": [],
            "LEFT": [],
            "RIGHT": []
        }
        self.max_candidates_per_angle = 3

        # Shutter trigger flag (active for single frame)
        self.just_captured: Optional[str] = None
        self.instruction = "Look straight at the camera (FRONT)"

    def reset(self):
        self.state = "CAPTURING_FRONT"
        self.hold_count = 0
        self.captured_snapshots = {}
        self.best_hold_candidate = {"FRONT": None, "LEFT": None, "RIGHT": None}
        self.burst_candidates = {"FRONT": [], "LEFT": [], "RIGHT": []}
        self.just_captured = None
        self.instruction = "Look straight at the camera (FRONT)"

    def _create_snapshot_thumbnail(self, frame_bgr: np.ndarray, face_dict: Optional[Dict[str, Any]]) -> Optional[str]:
        """Crops the upright face with margin and returns a clean base64 JPEG thumbnail."""
        try:
            h, w = frame_bgr.shape[:2]
            if face_dict and "bbox" in face_dict:
                bbox = face_dict["bbox"]
                bx, by, bw, bh = bbox
                margin_x = int(bw * 0.28)
                margin_y = int(bh * 0.28)
                x1 = max(0, int(bx - margin_x))
                y1 = max(0, int(by - margin_y))
                x2 = min(w, int(bx + bw + margin_x))
                y2 = min(h, int(by + bh + margin_y))
                crop = frame_bgr[y1:y2, x1:x2]
            else:
                crop = frame_bgr

            if crop is None or crop.size == 0:
                crop = frame_bgr

            # Resize to high-density crisp portrait thumbnail (240x240)
            thumb = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"
        except Exception:
            return None

    def step(
        self,
        detection_status: str,
        face_dict: Optional[Dict[str, Any]],
        pose: Dict[str, Any],
        quality: Dict[str, Any],
        liveness: Dict[str, Any],
        frame_bgr: Optional[np.ndarray],
        embedding: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        # Reset one-shot shutter flash flag
        self.just_captured = None

        if detection_status == "NO_FACE" or face_dict is None:
            self.instruction = "Position your face in the center of the frame"
            self.hold_count = max(0, self.hold_count - 1)
            return self._build_status(pose=pose)

        if detection_status == "MULTIPLE_SUBJECTS":
            self.instruction = "Multiple faces detected. Ensure only one person is visible."
            self.hold_count = 0
            return self._build_status(pose=pose)

        yaw = pose.get("yaw_deg", 0.0)
        pitch = pose.get("pitch_deg", 0.0)

        # Forgiving tolerances designed for natural human laptop/webcam posture
        # Yaw is the primary multi-angle discriminator; pitch allows natural desk tilt
        pitch_acceptable = abs(pitch) <= 35.0

        if self.state == "CAPTURING_FRONT":
            target_angle = "FRONT"
            in_zone = abs(yaw) <= 12.0 and pitch_acceptable
            if in_zone:
                self.instruction = "Position perfect. HOLD STILL..."
            elif yaw < -12.0:
                self.instruction = "Turn head slightly RIGHT toward center"
            elif yaw > 12.0:
                self.instruction = "Turn head slightly LEFT toward center"
            elif pitch < -35.0:
                self.instruction = "Raise head slightly UP"
            elif pitch > 35.0:
                self.instruction = "Lower head slightly DOWN"

        elif self.state == "CAPTURING_LEFT":
            target_angle = "LEFT"
            in_zone = (-38.0 <= yaw <= -8.0) and pitch_acceptable
            if in_zone:
                self.instruction = "Left angle held. HOLD STILL..."
            elif yaw > -8.0:
                self.instruction = "Turn head SLIGHTLY LEFT (~20°)"
            else:
                self.instruction = "Too far left. Turn slightly back towards center"

        elif self.state == "CAPTURING_RIGHT":
            target_angle = "RIGHT"
            in_zone = (8.0 <= yaw <= 38.0) and pitch_acceptable
            if in_zone:
                self.instruction = "Right angle held. HOLD STILL..."
            elif yaw < 8.0:
                self.instruction = "Turn head SLIGHTLY RIGHT (~20°)"
            else:
                self.instruction = "Too far right. Turn slightly back towards center"

        else:
            # CAPTURE_COMPLETE
            target_angle = "COMPLETE"
            in_zone = False
            self.instruction = "Biometric capture complete! Click EVALUATE VERIFICATION."
            return self._build_status(pose=pose, in_zone_override=False)

        # Hold accumulation & sharpest candidate frame selection
        if in_zone:
            self.hold_count += 1

            # Track the sharpest, highest-quality frame across the hold window
            # to guarantee motion-blur-free screenshot captures
            sharpness = float(quality.get("sharpness", 0.0))
            comp_score = float(quality.get("composite_score", 0.85))
            frame_score = (sharpness * 0.7) + (comp_score * 30.0)

            curr_best = self.best_hold_candidate.get(target_angle)
            if curr_best is None or frame_score > curr_best.get("score", -1.0):
                self.best_hold_candidate[target_angle] = {
                    "score": frame_score,
                    "frame_bgr": frame_bgr.copy() if frame_bgr is not None else None,
                    "face_dict": face_dict,
                    "pose": pose,
                    "quality": quality,
                    "liveness": liveness,
                    "embedding": embedding,
                    "yaw": yaw,
                    "pitch": pitch,
                    "composite": comp_score
                }

            if self.hold_count >= self.required_hold_frames:
                # AUTO CAPTURE TRIGGERED!
                # Commit the sharpest candidate acquired during the stable hold period
                best = self.best_hold_candidate.get(target_angle)
                best_frame = best["frame_bgr"] if best and best["frame_bgr"] is not None else frame_bgr
                best_face = best["face_dict"] if best else face_dict
                best_pose = best["pose"] if best else pose
                best_quality = best["quality"] if best else quality
                best_liveness = best["liveness"] if best else liveness
                best_embedding = best["embedding"] if best else embedding
                best_yaw = best["yaw"] if best else yaw
                best_pitch = best["pitch"] if best else pitch
                best_composite = best["composite"] if best else comp_score

                thumb_b64 = None
                if best_frame is not None:
                    thumb_b64 = self._create_snapshot_thumbnail(best_frame, best_face)

                # Fallback embedding if not yet computed
                if best_embedding is None:
                    best_embedding = np.zeros((512,), dtype=np.float32)

                # Store snapshot proof
                self.captured_snapshots[target_angle] = {
                    "angle": target_angle,
                    "thumbnail": thumb_b64,
                    "yaw": round(float(best_yaw), 1),
                    "pitch": round(float(best_pitch), 1),
                    "score": round(float(best_composite), 2),
                    "timestamp": time.time()
                }

                # Store burst candidate
                self._add_burst_candidate(
                    angle=target_angle,
                    frame_bgr=best_frame if best_frame is not None else np.zeros((10, 10, 3), dtype=np.uint8),
                    face_dict=best_face,
                    pose=best_pose,
                    quality=best_quality,
                    liveness=best_liveness,
                    embedding=best_embedding
                )

                self.just_captured = target_angle
                self.hold_count = 0
                self.best_hold_candidate[target_angle] = None

                # Advance state
                if self.state == "CAPTURING_FRONT":
                    self.state = "CAPTURING_LEFT"
                    self.instruction = "Front verified! Now turn head SLIGHTLY LEFT"
                elif self.state == "CAPTURING_LEFT":
                    self.state = "CAPTURING_RIGHT"
                    self.instruction = "Left profile verified! Now turn head SLIGHTLY RIGHT"
                elif self.state == "CAPTURING_RIGHT":
                    self.state = "CAPTURE_COMPLETE"
                    self.instruction = "All angles captured! Processing multi-angle verification..."
        else:
            # Decay gently so brief tracking jitter does not restart the counter from scratch
            self.hold_count = max(0, self.hold_count - 1)
            if self.hold_count == 0:
                self.best_hold_candidate[target_angle] = None

        return self._build_status(pose=pose, in_zone_override=in_zone)

    def _add_burst_candidate(
        self,
        angle: str,
        frame_bgr: np.ndarray,
        face_dict: Dict[str, Any],
        pose: Dict[str, Any],
        quality: Dict[str, Any],
        liveness: Dict[str, Any],
        embedding: np.ndarray
    ):
        candidate = BurstCandidate(
            frame_bgr=frame_bgr,
            face_dict=face_dict,
            pose=pose,
            quality=quality,
            liveness=liveness,
            embedding=embedding,
            angle_name=angle
        )
        self.burst_candidates[angle].append(candidate)
        self.burst_candidates[angle].sort(key=lambda c: c.score, reverse=True)
        if len(self.burst_candidates[angle]) > self.max_candidates_per_angle:
            self.burst_candidates[angle] = self.burst_candidates[angle][:self.max_candidates_per_angle]

    def _build_status(self, pose: Optional[Dict[str, Any]] = None, in_zone_override: Optional[bool] = None) -> Dict[str, Any]:
        yaw = (pose or {}).get("yaw_deg", 0.0)

        # Target specs according to current state
        if self.state == "CAPTURING_FRONT":
            target_angle = "FRONT"
            target_yaw = 0.0
            target_yaw_range = [-12.0, 12.0]
            in_target_zone = in_zone_override if in_zone_override is not None else abs(yaw) <= 12.0
            if in_target_zone:
                direction_hint = "HOLD_STILL"
            elif yaw < -12.0:
                direction_hint = "TURN_RIGHT"
            else:
                direction_hint = "TURN_LEFT"
        elif self.state == "CAPTURING_LEFT":
            target_angle = "LEFT"
            target_yaw = -20.0
            target_yaw_range = [-38.0, -8.0]
            in_target_zone = in_zone_override if in_zone_override is not None else (-38.0 <= yaw <= -8.0)
            if in_target_zone:
                direction_hint = "HOLD_STILL"
            elif yaw > -8.0:
                direction_hint = "TURN_LEFT"
            else:
                direction_hint = "TURN_RIGHT"
        elif self.state == "CAPTURING_RIGHT":
            target_angle = "RIGHT"
            target_yaw = 20.0
            target_yaw_range = [8.0, 38.0]
            in_target_zone = in_zone_override if in_zone_override is not None else (8.0 <= yaw <= 38.0)
            if in_target_zone:
                direction_hint = "HOLD_STILL"
            elif yaw < 8.0:
                direction_hint = "TURN_RIGHT"
            else:
                direction_hint = "TURN_LEFT"
        else:
            target_angle = "COMPLETE"
            target_yaw = 0.0
            target_yaw_range = [-12.0, 12.0]
            in_target_zone = False
            direction_hint = "COMPLETE"

        hold_progress = round(min(1.0, self.hold_count / max(1, self.required_hold_frames)), 2)

        return {
            "state": self.state,
            "target_angle": target_angle,
            "target_yaw": target_yaw,
            "target_yaw_range": target_yaw_range,
            "in_target_zone": in_target_zone,
            "direction_hint": direction_hint,
            "degrees_to_target": round(float(yaw - target_yaw), 1),
            "instruction": self.instruction,
            "hold_progress": hold_progress,
            "just_captured": self.just_captured,
            "front_captured": "FRONT" in self.captured_snapshots,
            "left_captured": "LEFT" in self.captured_snapshots,
            "right_captured": "RIGHT" in self.captured_snapshots,
            "front_count": len(self.burst_candidates.get("FRONT", [])),
            "left_count": len(self.burst_candidates.get("LEFT", [])),
            "right_count": len(self.burst_candidates.get("RIGHT", [])),
            "snapshots": self.captured_snapshots,
            "is_complete": self.state == "CAPTURE_COMPLETE"
        }
