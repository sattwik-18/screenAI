"""
Facial Landmark and Periocular Analysis Pipeline.
Uses MediaPipe Face Landmarker (478 3D points, iris tracking, metric pose) exclusively.
Generates 5-stage side-by-side debug telemetry:
SCRFD Bbox -> Face Crop (base64) -> Local Landmarks -> 3D Mesh & Pose
"""

from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import base64

from services.landmarks.mediapipe_landmarker import MediaPipeFaceLandmarkerService


class LandmarkPipeline:
    def __init__(self):
        self._mp_service: Optional[MediaPipeFaceLandmarkerService] = None

    def _get_mp_service(self) -> MediaPipeFaceLandmarkerService:
        if self._mp_service is None:
            self._mp_service = MediaPipeFaceLandmarkerService()
        return self._mp_service

    def process_landmarks(
        self,
        face_data: Dict[str, Any],
        frame_width: int,
        frame_height: int,
        frame_bgr: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Parses dense landmarks with MediaPipe Face Landmarker (478 3D points),
        computes periocular crops, eye aspect ratios (EAR), and debug crop telemetry.
        """
        bbox = face_data.get("bbox", [0, 0, 100, 100])

        # Execute MediaPipe Face Landmarker
        if frame_bgr is not None:
            try:
                mp_svc = self._get_mp_service()
                mp_res = mp_svc.process_frame(frame_bgr, scrfd_bbox=bbox)
                if mp_res is not None:
                    # Build debug crop with local landmark coordinates and aligned face
                    debug_info = self._extract_debug_crop(
                        frame_bgr,
                        bbox,
                        mp_res["landmarks_478"] if mp_res.get("geometry_valid", True) else [],
                        kps=face_data.get("kps")
                    )
                    debug_info["geometry_valid"] = mp_res.get("geometry_valid", True)
                    debug_info["geometry_status"] = mp_res.get("geometry_status", "VALID")
                    debug_info["geometry_reasons"] = mp_res.get("geometry_reasons", [])
                    
                    return {
                        "engine": "mediapipe_478",
                        "geometry_valid": mp_res.get("geometry_valid", True),
                        "geometry_status": mp_res.get("geometry_status", "VALID"),
                        "geometry_reasons": mp_res.get("geometry_reasons", []),
                        "landmark_count_2d": len(mp_res["landmarks_478"]),
                        "landmark_count_3d": len(mp_res["landmarks_478"]),
                        "landmark_2d_106": mp_res["landmark_3d_68"],  # DLib 68 subset for compatibility
                        "landmark_3d_68": mp_res["landmark_3d_68"],
                        "landmarks_478": mp_res["landmarks_478"],
                        "left_eye_center": mp_res["left_iris"][:2],
                        "right_eye_center": mp_res["right_iris"][:2],
                        "left_periocular_crop": mp_res["left_periocular_crop"],
                        "right_periocular_crop": mp_res["right_periocular_crop"],
                        "left_crop_b64": mp_res.get("left_crop_b64"),
                        "right_crop_b64": mp_res.get("right_crop_b64"),
                        "ear_left": mp_res["ear_left"],
                        "ear_right": mp_res["ear_right"],
                        "periocular_available": True,
                        "pose": mp_res["pose"],
                        "debug_pipeline": debug_info
                    }
            except Exception as e:
                print(f"[LandmarkPipeline] MediaPipe processing error: {e}")

        # Fallback when frame_bgr is not provided or face landmarker misses: use SCRFD 5 keypoints
        kps = face_data.get("kps", [])
        left_eye_center = None
        right_eye_center = None
        if len(kps) >= 2:
            left_eye_center = [float(kps[0][0]), float(kps[0][1])]
            right_eye_center = [float(kps[1][0]), float(kps[1][1])]

        eye_box_size = int(max(40, bbox[2] * 0.35))
        left_periocular_crop = None
        right_periocular_crop = None

        if left_eye_center is not None:
            lx, ly = left_eye_center
            left_periocular_crop = [
                max(0, int(lx - eye_box_size / 2)),
                max(0, int(ly - eye_box_size / 2)),
                min(frame_width, eye_box_size),
                min(frame_height, eye_box_size)
            ]

        if right_eye_center is not None:
            rx, ry = right_eye_center
            right_periocular_crop = [
                max(0, int(rx - eye_box_size / 2)),
                max(0, int(ry - eye_box_size / 2)),
                min(frame_width, eye_box_size),
                min(frame_height, eye_box_size)
            ]

        return {
            "engine": "mediapipe_478",
            "landmark_count_2d": 0,
            "landmark_count_3d": 0,
            "landmark_2d_106": [],
            "landmark_3d_68": [],
            "landmarks_478": [],
            "left_eye_center": left_eye_center,
            "right_eye_center": right_eye_center,
            "left_periocular_crop": left_periocular_crop,
            "right_periocular_crop": right_periocular_crop,
            "ear_left": 0.30,
            "ear_right": 0.30,
            "periocular_available": left_periocular_crop is not None and right_periocular_crop is not None,
            "debug_pipeline": None
        }

    def _extract_debug_crop(
        self,
        frame_bgr: np.ndarray,
        bbox: List[int],
        landmarks: List[List[float]],
        target_size: int = 180,
        kps: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Extracts face crop with 20% margin, transforms full-frame landmarks into local
        crop pixel coordinates, generates 112x112 canonical aligned ArcFace crop,
        and encodes both as base64 JPEG for the UI diagnostic view.
        """
        fh, fw = frame_bgr.shape[:2]
        bx, by, bw, bh = bbox

        # 20% margin around SCRFD bounding box
        pad_x = int(bw * 0.20)
        pad_y = int(bh * 0.20)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(fw, bx + bw + pad_x)
        y2 = min(fh, by + bh + pad_y)

        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return {"crop_base64": None, "aligned_face_b64": None, "local_landmarks": [], "crop_box": [x1, y1, crop_w, crop_h]}

        # Resize to fixed target size for display
        resized_crop = cv2.resize(crop, (target_size, target_size))

        # Project landmarks into local resized crop coordinates (0 to target_size)
        scale_x = target_size / float(crop_w)
        scale_y = target_size / float(crop_h)

        local_landmarks: List[List[float]] = []
        for pt in landmarks:
            lx = (pt[0] - x1) * scale_x
            ly = (pt[1] - y1) * scale_y
            local_landmarks.append([round(lx, 1), round(ly, 1)])

        # Encode crop as JPEG base64
        _, buf = cv2.imencode('.jpg', resized_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode('ascii')

        # Generate canonical 112x112 ArcFace aligned face thumbnail
        aligned_b64 = None
        if kps is not None and len(kps) == 5:
            try:
                from insightface.utils import face_align
                aligned = face_align.norm_crop(frame_bgr, landmark=np.array(kps), image_size=112)
                _, a_buf = cv2.imencode('.jpg', aligned, [cv2.IMWRITE_JPEG_QUALITY, 85])
                aligned_b64 = f"data:image/jpeg;base64,{base64.b64encode(a_buf).decode('ascii')}"
            except Exception:
                pass

        return {
            "crop_base64": f"data:image/jpeg;base64,{b64}",
            "aligned_face_b64": aligned_b64,
            "local_landmarks": local_landmarks,
            "crop_box": [x1, y1, crop_w, crop_h],
            "crop_size": target_size
        }
