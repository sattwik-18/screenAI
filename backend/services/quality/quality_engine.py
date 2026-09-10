"""
Multidimensional Face Quality Assessment Engine.
Preserves individual raw quality signals and computes calibrated eligibility.

Signals:
- Sharpness (Laplacian variance)
- Exposure & Illumination (Mean & Std Dev of luminance)
- Face Size Ratio (Face area / Frame area)
- Pose Alignment Deviation (Deviation from ideal target angle)
- Contrast & Dynamic Range
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


class QualityEngine:
    def __init__(
        self,
        min_sharpness: float = 80.0,
        min_face_ratio: float = 0.05,
        min_illumination: float = 45.0,
        max_illumination: float = 215.0
    ):
        self.min_sharpness = min_sharpness
        self.min_face_ratio = min_face_ratio
        self.min_illumination = min_illumination
        self.max_illumination = max_illumination

    def assess(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int],  # [x, y, w, h]
        pose: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes multidimensional quality metrics on the detected face crop.
        """
        frame_h, frame_w = frame_bgr.shape[:2]
        x, y, w, h = face_bbox
        
        # Safe crop boundary
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)
        
        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return {
                "acceptable": False,
                "rejection_reason": "EMPTY_CROP",
                "composite_score": 0.0,
                "sharpness": 0.0,
                "illumination_mean": 0.0,
                "illumination_std": 0.0,
                "face_size_ratio": 0.0,
                "pose_penalty": 1.0
            }

        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness via Laplacian Variance
        laplacian = cv2.Laplacian(gray_crop, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # 2. Illumination and Exposure
        illum_mean = float(np.mean(gray_crop))
        illum_std = float(np.std(gray_crop))

        # 3. Face Size Ratio
        face_area = w * h
        frame_area = frame_w * frame_h
        size_ratio = float(face_area / max(1, frame_area))

        # 4. Pose Penalty (deviation from optimal)
        pitch = abs(pose.get("pitch_deg", 0.0))
        yaw = abs(pose.get("yaw_deg", 0.0))
        roll = abs(pose.get("roll_deg", 0.0))
        pose_penalty = min(1.0, (pitch * 0.02 + yaw * 0.015 + roll * 0.03))

        # Normalized Sub-Scores (0.0 to 1.0 scale)
        sharpness_norm = min(1.0, sharpness / 400.0)
        
        # Exposure score: bell curve centered at 128
        exposure_norm = max(0.0, 1.0 - (abs(illum_mean - 128.0) / 128.0))
        contrast_norm = min(1.0, illum_std / 50.0)
        size_norm = min(1.0, size_ratio / 0.20)
        pose_norm = max(0.0, 1.0 - pose_penalty)

        # Calibrated weighted composite score:
        # Sharpness: 35%, Exposure: 20%, Contrast: 15%, Size: 15%, Pose: 15%
        composite = (
            0.35 * sharpness_norm +
            0.20 * exposure_norm +
            0.15 * contrast_norm +
            0.15 * size_norm +
            0.15 * pose_norm
        )

        # Check hard floors
        acceptable = True
        rejection_reason = None

        if sharpness < self.min_sharpness:
            acceptable = False
            rejection_reason = f"BLUR_DETECTED (Sharpness {sharpness:.1f} < {self.min_sharpness})"
        elif illum_mean < self.min_illumination:
            acceptable = False
            rejection_reason = f"UNDER_EXPOSED (Mean light {illum_mean:.1f} < {self.min_illumination})"
        elif illum_mean > self.max_illumination:
            acceptable = False
            rejection_reason = f"OVER_EXPOSED (Mean light {illum_mean:.1f} > {self.max_illumination})"
        elif size_ratio < self.min_face_ratio:
            acceptable = False
            rejection_reason = f"SUBJECT_TOO_FAR (Size ratio {size_ratio:.2f} < {self.min_face_ratio})"

        return {
            "acceptable": acceptable,
            "rejection_reason": rejection_reason,
            "composite_score": round(composite, 3),
            "sharpness": round(sharpness, 1),
            "sharpness_norm": round(sharpness_norm, 3),
            "illumination_mean": round(illum_mean, 1),
            "illumination_std": round(illum_std, 1),
            "exposure_norm": round(exposure_norm, 3),
            "contrast_norm": round(contrast_norm, 3),
            "face_size_ratio": round(size_ratio, 4),
            "size_norm": round(size_norm, 3),
            "pose_penalty": round(pose_penalty, 3),
            "pose_norm": round(pose_norm, 3)
        }
