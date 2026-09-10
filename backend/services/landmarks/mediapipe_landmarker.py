"""
MediaPipe Face Landmarker Service.
Provides 478 dense 3D facial landmarks, iris tracking, transformation matrix pose, and EAR.
Operates on the full frame or SCRFD-guided regions.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import cv2
import base64
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python

# Canonical MediaPipe 468/478 index mapping to DLib 68 landmark standard:
# Jaw: 0-16, Right eyebrow: 17-21, Left eyebrow: 22-26, Nose bridge: 27-30,
# Nose base: 31-35, Right eye: 36-41, Left eye: 42-47, Outer mouth: 48-59, Inner mouth: 60-67
MEDIAPIPE_TO_68_INDICES = [
    # Jawline (0-16)
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
    # Right eyebrow (17-21)
    70, 63, 105, 66, 107,
    # Left eyebrow (22-26)
    336, 296, 334, 293, 300,
    # Nose bridge (27-30)
    168, 197, 5, 4,
    # Nose bottom (31-35)
    75, 97, 2, 326, 305,
    # Right eye (36-41)
    33, 160, 158, 133, 153, 144,
    # Left eye (42-47)
    362, 385, 387, 263, 373, 380,
    # Outer lip (48-59)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
    # Inner lip (60-67)
    78, 95, 88, 178, 87, 14, 317, 402
]


def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Decomposes a 3x3 rotation matrix into Euler angles (pitch, yaw, roll) in degrees.
    Convention: intrinsic rotations matching face tracking standard.
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0

    return (
        float(np.degrees(pitch)),
        float(np.degrees(yaw)),
        float(np.degrees(roll))
    )


class MediaPipeFaceLandmarkerService:
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "landmarks", "face_landmarker.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe face landmarker model not found at {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _validate_geometry(
        landmarks_478: List[List[float]],
        scrfd_bbox: Optional[List[int]],
        frame_w: int,
        frame_h: int
    ) -> Tuple[bool, str, List[str]]:
        """
        Validates that predicted landmarks are geometrically and anatomically consistent
        with the SCRFD detected face region.
        """
        if not landmarks_478 or len(landmarks_478) < 468:
            return False, "GEOMETRY_INVALID", ["Insufficient landmarks (< 468)"]

        if scrfd_bbox is None or len(scrfd_bbox) != 4:
            return True, "VALID", []

        bx, by, bw, bh = scrfd_bbox
        xs = [p[0] for p in landmarks_478]
        ys = [p[1] for p in landmarks_478]

        reasons = []

        # 1. Centroid containment within expanded SCRFD bbox
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        pad_x = bw * 0.35
        pad_y = bh * 0.35
        if not (bx - pad_x <= cx <= bx + bw + pad_x and by - pad_y <= cy <= by + bh + pad_y):
            reasons.append(f"Landmark centroid ({cx:.1f}, {cy:.1f}) outside SCRFD boundary [{bx}, {by}, {bw}, {bh}]")

        # 2. Inter-Ocular Distance (IOD) between irises (468: right iris, 473: left iris)
        r_iris = np.array(landmarks_478[468][:2])
        l_iris = np.array(landmarks_478[473][:2])
        iod = float(np.linalg.norm(r_iris - l_iris))
        if iod < (0.12 * bw) or iod > (0.85 * bw):
            reasons.append(f"Inter-ocular distance ({iod:.1f}px) outside plausible range [{0.12*bw:.1f}, {0.85*bw:.1f}]")

        # 3. Vertical anatomical alignment: Eyes Y < Mouth Y < Chin Y
        eyes_y = float((r_iris[1] + l_iris[1]) / 2.0)
        mouth_y = float((landmarks_478[13][1] + landmarks_478[14][1]) / 2.0)
        chin_y = float(landmarks_478[152][1])
        if not (eyes_y < mouth_y < chin_y):
            reasons.append(f"Anatomical vertical inversion: Eyes {eyes_y:.1f}, Mouth {mouth_y:.1f}, Chin {chin_y:.1f}")

        # 4. Outlier containment: fraction of landmarks outside expanded boundary
        outliers = 0
        for x, y, _ in landmarks_478:
            if not (bx - pad_x <= x <= bx + bw + pad_x and by - pad_y <= y <= by + bh + pad_y):
                outliers += 1
        outlier_ratio = outliers / len(landmarks_478)
        if outlier_ratio > 0.20:
            reasons.append(f"{outlier_ratio*100:.1f}% of landmarks outside expanded face boundary")

        if reasons:
            return False, "GEOMETRY_INVALID", reasons

        return True, "VALID", []

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        scrfd_bbox: Optional[List[int]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Processes BGR frame and returns dense 478 3D landmarks, 68-pt subset,
        iris centers, head pose Euler angles, EAR, and geometric validity.
        Inference is strictly bounded to the SCRFD face crop to prevent
        detecting or jumping to clothing graphics or background clutter.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]

        # Bind inference strictly to SCRFD face crop
        crop_origin_x = 0
        crop_origin_y = 0
        inference_w = w
        inference_h = h

        if scrfd_bbox is not None and len(scrfd_bbox) == 4:
            bx, by, bw, bh = scrfd_bbox
            # 25% margin around SCRFD bounding box to ensure chin and forehead are contained
            pad_x = int(bw * 0.25)
            pad_y = int(bh * 0.25)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0 and (x2 - x1) >= 40 and (y2 - y1) >= 40:
                crop_origin_x = x1
                crop_origin_y = y1
                inference_w = x2 - x1
                inference_h = y2 - y1
                input_frame = crop
            else:
                input_frame = frame_bgr
        else:
            input_frame = frame_bgr

        rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        raw_landmarks = result.face_landmarks[0]
        # Translate from crop-relative coordinates to full-frame pixel coordinates
        landmarks_478: List[List[float]] = []
        for lm in raw_landmarks:
            landmarks_478.append([
                round(float(crop_origin_x + lm.x * inference_w), 1),
                round(float(crop_origin_y + lm.y * inference_h), 1),
                round(float(lm.z * inference_w), 1)
            ])

        # Validate geometric consistency against SCRFD bounding box
        geom_valid, geom_status, geom_reasons = self._validate_geometry(
            landmarks_478, scrfd_bbox, w, h
        )

        # 68-point DLib subset in pixel space
        landmarks_68: List[List[float]] = []
        for idx in MEDIAPIPE_TO_68_INDICES:
            if idx < len(landmarks_478):
                landmarks_68.append(landmarks_478[idx])

        # Irises (468 = right iris center, 473 = left iris center)
        right_iris = landmarks_478[468] if len(landmarks_478) > 468 else landmarks_68[36]
        left_iris = landmarks_478[473] if len(landmarks_478) > 473 else landmarks_68[42]

        # Extract head pose Euler angles from the 4x4 facial transformation matrix
        pitch, yaw, roll = 0.0, 0.0, 0.0
        if result.facial_transformation_matrixes and len(result.facial_transformation_matrixes) > 0:
            mat = np.array(result.facial_transformation_matrixes[0])
            R = mat[:3, :3]
            pitch, yaw, roll = rotation_matrix_to_euler_angles(R)

        # Classify orientation
        orientation = "FRONT"
        if yaw < -14.0:
            orientation = "LOOKING_LEFT"
        elif yaw > 14.0:
            orientation = "LOOKING_RIGHT"
        elif pitch < -15.0:
            orientation = "LOOKING_DOWN"
        elif pitch > 15.0:
            orientation = "LOOKING_UP"

        # Calculate EAR (Eye Aspect Ratio) from landmarks
        def calc_ear(indices: List[int]) -> float:
            pts = [np.array(landmarks_478[i][:2]) for i in indices]
            v1 = np.linalg.norm(pts[1] - pts[5])
            v2 = np.linalg.norm(pts[2] - pts[4])
            h_dist = np.linalg.norm(pts[0] - pts[3])
            if h_dist < 1e-4:
                return 0.30
            return float((v1 + v2) / (2.0 * h_dist))

        # MediaPipe eye indices: Right eye [33, 160, 158, 133, 153, 144], Left eye [362, 385, 387, 263, 373, 380]
        ear_right = round(calc_ear([33, 160, 158, 133, 153, 144]), 3)
        ear_left = round(calc_ear([362, 385, 387, 263, 373, 380]), 3)

        # Periocular crops around irises
        bbox_w = scrfd_bbox[2] if scrfd_bbox else (w * 0.3)
        eye_box_size = int(max(40, bbox_w * 0.35))

        left_crop = [
            max(0, int(left_iris[0] - eye_box_size / 2)),
            max(0, int(left_iris[1] - eye_box_size / 2)),
            min(w, eye_box_size),
            min(h, eye_box_size)
        ]
        right_crop = [
            max(0, int(right_iris[0] - eye_box_size / 2)),
            max(0, int(right_iris[1] - eye_box_size / 2)),
            min(w, eye_box_size),
            min(h, eye_box_size)
        ]

        # Extract direct image crops for the UI periocular cards
        left_b64 = None
        right_b64 = None
        try:
            lx, ly, lw, lh = left_crop
            rx, ry, rw, rh = right_crop
            if lw > 10 and lh > 10 and ly + lh <= h and lx + lw <= w:
                crop_l = frame_bgr[ly:ly+lh, lx:lx+lw]
                if crop_l.size > 0:
                    r_l = cv2.resize(crop_l, (120, 90))
                    _, buf_l = cv2.imencode('.jpg', r_l, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    left_b64 = base64.b64encode(buf_l).decode('ascii')
            if rw > 10 and rh > 10 and ry + rh <= h and rx + rw <= w:
                crop_r = frame_bgr[ry:ry+rh, rx:rx+rw]
                if crop_r.size > 0:
                    r_r = cv2.resize(crop_r, (120, 90))
                    _, buf_r = cv2.imencode('.jpg', r_r, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    right_b64 = base64.b64encode(buf_r).decode('ascii')
        except Exception:
            pass

        return {
            "engine": "mediapipe_478",
            "geometry_valid": geom_valid,
            "geometry_status": geom_status,
            "geometry_reasons": geom_reasons,
            "landmarks_478": landmarks_478 if geom_valid else [],
            "landmark_3d_68": landmarks_68 if geom_valid else [],
            "left_iris": left_iris,
            "right_iris": right_iris,
            "pose": {
                "pitch_deg": round(pitch, 2),
                "yaw_deg": round(yaw, 2),
                "roll_deg": round(roll, 2),
                "orientation": orientation,
                "mesh_vertices": landmarks_478 if geom_valid else [],
                "mesh_vertices_68": landmarks_68 if geom_valid else [],
                "is_monocular_3d": True,
                "geometry_valid": geom_valid,
                "geometry_status": geom_status
            },
            "ear_left": ear_left,
            "ear_right": ear_right,
            "left_periocular_crop": left_crop,
            "right_periocular_crop": right_crop,
            "left_crop_b64": left_b64,
            "right_crop_b64": right_b64,
            "landmark_count": len(landmarks_478) if geom_valid else 0
        }
