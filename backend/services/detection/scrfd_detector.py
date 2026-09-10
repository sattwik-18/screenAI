"""
SCRFD Face Detection Service.
Adheres to strict single-subject verification policy:
- Exactly 1 face: Status EXACTLY_ONE -> returns bounding box, confidence, landmarks.
- 0 faces: Status NO_FACE.
- 2+ faces: Status MULTIPLE_SUBJECTS -> halts verification and instructs user.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis


class DetectionResult:
    def __init__(
        self,
        status: str,
        faces: List[Dict[str, Any]],
        message: str = ""
    ):
        self.status = status  # EXACTLY_ONE, NO_FACE, MULTIPLE_SUBJECTS, FACE_TOO_SMALL
        self.faces = faces
        self.message = message

    @property
    def primary_face(self) -> Optional[Dict[str, Any]]:
        if self.status == "EXACTLY_ONE" and len(self.faces) == 1:
            return self.faces[0]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "face_count": len(self.faces),
            "message": self.message,
            "faces": [
                {
                    "bbox": f["bbox"],
                    "confidence": float(f["confidence"]),
                    "kps": f.get("kps", []),
                    "width": f["width"],
                    "height": f["height"],
                }
                for f in self.faces
            ]
        }


class SCRFDDetector:
    def __init__(
        self,
        min_confidence: float = 0.65,
        min_face_size: int = 80,
        det_size: Tuple[int, int] = (640, 640)
    ):
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.det_size = det_size
        
        # Initialize InsightFace buffalo_s app (detection + ArcFace recognition only)
        self.app = FaceAnalysis(
            name="buffalo_s",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=self.det_size)

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        if image_bgr is None or image_bgr.size == 0:
            return DetectionResult(status="NO_FACE", faces=[], message="Input frame is empty")

        raw_faces = self.app.get(image_bgr)
        
        # Filter by confidence
        valid_faces = []
        for face in raw_faces:
            det_score = float(getattr(face, "det_score", 0.0))
            if det_score < self.min_confidence:
                continue
            
            bbox = face.bbox.astype(int).tolist()
            x1, y1, x2, y2 = bbox
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            
            kps = face.kps.tolist() if hasattr(face, "kps") and face.kps is not None else []
            landmark_2d = face.landmark_2d_106.tolist() if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None else []
            landmark_3d = face.landmark_3d_68.tolist() if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None else []
            pose = face.pose.tolist() if hasattr(face, "pose") and face.pose is not None else [0.0, 0.0, 0.0]
            embedding = face.embedding if hasattr(face, "embedding") and face.embedding is not None else None

            valid_faces.append({
                "bbox": [x1, y1, width, height],
                "bbox_raw": [x1, y1, x2, y2],
                "confidence": det_score,
                "width": width,
                "height": height,
                "kps": kps,
                "landmark_2d_106": landmark_2d,
                "landmark_3d_68": landmark_3d,
                "pose": pose,
                "embedding": embedding,
                "_raw_face": face
            })

        if len(valid_faces) == 0:
            return DetectionResult(
                status="NO_FACE",
                faces=[],
                message="No subject detected in camera viewport"
            )

        img_h = image_bgr.shape[0]
        # Prominence ranking: higher confidence, upper half of frame (faces sit above clothing)
        valid_faces.sort(
            key=lambda f: (f["confidence"] * 2.0 + (f["width"] * f["height"]) / (img_h * img_h) - (f["bbox"][1] / img_h)),
            reverse=True
        )

        # Suppress spurious clothing/torso graphics if an authentic head face exists above
        if len(valid_faces) > 1:
            primary = valid_faces[0]
            filtered = [primary]
            for other in valid_faces[1:]:
                is_torso_graphic = (other["bbox"][1] > img_h * 0.45) and (other["confidence"] < primary["confidence"] - 0.08)
                if not is_torso_graphic:
                    filtered.append(other)
            valid_faces = filtered

        if len(valid_faces) > 1:
            return DetectionResult(
                status="MULTIPLE_SUBJECTS",
                faces=valid_faces,
                message=f"Multiple faces ({len(valid_faces)}) detected. Ensure only one subject is in frame."
            )

        face = valid_faces[0]
        if face["width"] < self.min_face_size or face["height"] < self.min_face_size:
            return DetectionResult(
                status="FACE_TOO_SMALL",
                faces=valid_faces,
                message=f"Face size ({face['width']}x{face['height']}) is below minimum requirement ({self.min_face_size}px). Please move closer."
            )

        return DetectionResult(
            status="EXACTLY_ONE",
            faces=valid_faces,
            message="Single subject successfully acquired"
        )

    def detect_with_orientation(
        self, image_bgr: np.ndarray, check_rotations: bool = True
    ) -> Tuple[DetectionResult, int, np.ndarray]:
        """
        Detects face in image. If NO_FACE is found and check_rotations is True,
        checks 270°, 90°, and 180° rotations to auto-detect sideways/upside-down cameras.
        Returns: (DetectionResult, rotation_deg, oriented_image_bgr)
        where rotation_deg is 0, 90, 180, or 270.
        """
        res = self.detect(image_bgr)
        if res.status == "EXACTLY_ONE" or not check_rotations:
            return res, 0, image_bgr

        # Test rotations in order of likelihood for webcams (270°, 90°, 180°)
        rotation_checks = [
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
            (90, cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
        ]

        best_res = res
        best_angle = 0
        best_score = 0.0
        best_rotated = image_bgr

        for angle, rot_code in rotation_checks:
            rotated = cv2.rotate(image_bgr, rot_code)
            rot_res = self.detect(rotated)
            if rot_res.status == "EXACTLY_ONE":
                face = rot_res.faces[0]
                score = face["confidence"]
                roll = abs(face["pose"][2]) if face.get("pose") and len(face["pose"]) > 2 else 0.0

                # Face is genuinely upright when roll is close to 0 (< 45 degrees)
                if roll <= 45.0 and score >= self.min_confidence:
                    return rot_res, angle, rotated

                if roll <= 45.0 and score > best_score:
                    best_score = score
                    best_angle = angle
                    best_res = rot_res
                    best_rotated = rotated

        if best_angle != 0:
            return best_res, best_angle, best_rotated

        return res, 0, image_bgr

