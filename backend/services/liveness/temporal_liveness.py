"""
Temporal Motion Consistency Analyzer for Liveness Detection.
Evaluates micro-motion dynamics over a rolling temporal buffer.
Detects:
- Completely static frames (Photo attack: zero micro-motion).
- Natural physiological involuntary micro-motion (Live human: healthy micro-jitter).
- Extreme chaotic displacement (Video replay / loop artifact).
"""

from typing import Dict, Any, List
import numpy as np


class TemporalLivenessAnalyzer:
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.history: List[Dict[str, Any]] = []

    def push_frame(
        self,
        landmarks_3d: List[List[float]],
        pose: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Updates temporal buffer and computes motion consistency score.
        """
        if not landmarks_3d or len(landmarks_3d) < 5:
            return {
                "temporal_samples": len(self.history),
                "motion_consistency_score": 0.50,
                "motion_state": "BUFFERING",
                "variance": 0.0
            }

        # Store key landmark positions: nose tip (index 30 in 68-pt), left eye (36), right eye (45)
        nose = landmarks_3d[30] if len(landmarks_3d) > 30 else landmarks_3d[0]
        l_eye = landmarks_3d[36] if len(landmarks_3d) > 36 else landmarks_3d[1]
        
        self.history.append({
            "nose": nose,
            "l_eye": l_eye,
            "pitch": pose.get("pitch_deg", 0.0),
            "yaw": pose.get("yaw_deg", 0.0)
        })

        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < 6:
            return {
                "temporal_samples": len(self.history),
                "motion_consistency_score": 0.60,
                "motion_state": "BUFFERING",
                "variance": 0.0
            }

        # Calculate coordinate variance across the window
        nose_coords = np.array([h["nose"][:2] for h in self.history])
        variances = np.var(nose_coords, axis=0)
        total_var = float(np.sum(variances))

        # Check for static photo attack (total_var ~ 0.0)
        if total_var < 0.02:
            motion_state = "STATIC_PHOTO_SUSPECTED"
            score = 0.05
        elif 0.05 <= total_var <= 15.0:
            # Healthy micro-movement range
            motion_state = "PHYSIOLOGICAL_MOTION_DETECTED"
            score = 0.95
        elif 15.0 < total_var <= 35.0:
            # Active movement
            motion_state = "ACTIVE_MOVEMENT"
            score = 0.90
        else:
            # Very chaotic movement
            motion_state = "EXCESSIVE_JITTER"
            score = 0.65

        return {
            "temporal_samples": len(self.history),
            "motion_consistency_score": round(score, 3),
            "motion_state": motion_state,
            "variance": round(total_var, 3),
            "coordinate_variance": round(total_var, 3)
        }
