"""
Verification Session Manager.
Maintains in-memory session pipelines, captured burst frames, and document embeddings.
"""

from typing import Dict, Any, Optional, List
import time
import uuid
import numpy as np

from services.detection.scrfd_detector import SCRFDDetector
from services.tracking.spatial_tracker import SpatialIoUKalmanTracker
from services.landmarks.landmark_pipeline import LandmarkPipeline
from services.geometry3d.pose_estimator import PoseAndGeometryEstimator
from services.quality.quality_engine import QualityEngine
from services.liveness.liveness_ensemble import LivenessEnsemble
from services.recognition.arcface_embedder import ArcFaceEmbedder
from services.capture.capture_state_machine import CaptureStateMachine
from services.document_face.document_extractor import DocumentFaceExtractor
from services.matching.multi_frame_matcher import MultiFrameMatcher
from services.decision.evidence_engine import EvidenceDecisionEngine


def unrotate_face_dict(face_dict: Optional[Dict[str, Any]], angle: int, orig_w: int, orig_h: int) -> Optional[Dict[str, Any]]:
    """Maps bbox and kps from rotated image coordinates back to original frame coordinates."""
    if angle == 0 or face_dict is None:
        return face_dict

    res = dict(face_dict)
    rx, ry, rw, rh = face_dict["bbox"]

    if angle == 270:
        x = max(0, orig_w - 1 - (ry + rh))
        y = rx
        w = rh
        h = rw
    elif angle == 90:
        x = ry
        y = max(0, orig_h - 1 - (rx + rw))
        w = rh
        h = rw
    elif angle == 180:
        x = max(0, orig_w - 1 - (rx + rw))
        y = max(0, orig_h - 1 - (ry + rh))
        w = rw
        h = rh
    else:
        return face_dict

    res["bbox"] = [int(x), int(y), int(w), int(h)]
    if "kps" in face_dict and face_dict["kps"]:
        orig_kps = []
        for pt in face_dict["kps"]:
            px, py = pt[0], pt[1]
            if angle == 270:
                orig_kps.append([orig_w - 1 - py, px])
            elif angle == 90:
                orig_kps.append([py, orig_h - 1 - px])
            elif angle == 180:
                orig_kps.append([orig_w - 1 - px, orig_h - 1 - py])
        res["kps"] = orig_kps

    return res


class VerificationSession:
    def __init__(self, session_id: str, shared_detector: SCRFDDetector, shared_liveness: LivenessEnsemble):
        self.session_id = session_id
        self.created_at = time.time()
        self.detector = shared_detector
        self.liveness_ensemble = shared_liveness
        self.tracker = SpatialIoUKalmanTracker()
        self.landmarks_pipe = LandmarkPipeline()
        self.pose_estimator = PoseAndGeometryEstimator()
        self.quality_engine = QualityEngine()
        rec_model = self.detector.app.models.get("recognition") if hasattr(self.detector, "app") and hasattr(self.detector.app, "models") else None
        self.embedder = ArcFaceEmbedder(recognition_model=rec_model)
        self.capture_machine = CaptureStateMachine()
        self.doc_extractor = DocumentFaceExtractor(self.detector, self.embedder)
        self.matcher = MultiFrameMatcher(embedder=self.embedder)
        self.decision_engine = EvidenceDecisionEngine()
        
        self.uploaded_documents: List[Dict[str, Any]] = []
        self.last_telemetry: Dict[str, Any] = {}
        self.session_integrity = {
            "identity_switched": False,
            "multiple_faces_detected": False
        }
        self.final_report: Optional[Dict[str, Any]] = None

        # Performance: frame counter and liveness cache
        self._frame_count: int = 0
        self._liveness_interval: int = 5  # run liveness ONNX every 5th frame to maintain 25+ FPS tracking
        self._cached_liveness: Dict[str, Any] = {
            "liveness_score": 0.0, "is_live": False,
            "sensors": {"depth_available": False, "depth_consistency": None,
                        "ir_available": False, "ir_response": None}
        }

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        t0 = time.time()
        self._frame_count += 1

        orig_h, orig_w = frame_bgr.shape[:2]

        # ── 1. Detection ────────────────────────────────────────────────────
        # Native orientation detect first (fast path, zero rotation overhead)
        det_res = self.detector.detect(frame_bgr)
        detected_angle = 0

        # Only sweep rotations if native frame has no face detected
        if det_res.status == "NO_FACE":
            det_res, detected_angle, _ = self.detector.detect_with_orientation(frame_bgr)

        face_dict = det_res.primary_face
        if detected_angle != 0 and face_dict is not None:
            face_dict = unrotate_face_dict(face_dict, detected_angle, orig_w, orig_h)

        # Disambiguate multiple detections: preserve active track continuity
        if len(det_res.faces) > 1 and self.tracker.active_track is not None:
            from services.tracking.spatial_tracker import calculate_iou
            active_box = self.tracker.active_track.bbox
            best_face = None
            best_iou = -1.0
            for f in det_res.faces:
                mapped = unrotate_face_dict(f, detected_angle, orig_w, orig_h) if detected_angle != 0 else f
                iou = calculate_iou(mapped["bbox"], active_box)
                if iou > best_iou:
                    best_iou = iou
                    best_face = mapped
            if best_face is not None and best_iou > 0.15:
                face_dict = best_face

        h, w = orig_h, orig_w
        oriented_bgr = frame_bgr

        if det_res.status == "MULTIPLE_SUBJECTS":
            self.session_integrity["multiple_faces_detected"] = True

        # ── 2. Tracking + EMA landmark smoothing ────────────────────────────
        track_res = self.tracker.update(face_dict)
        if track_res.get("identity_switched", False):
            self.session_integrity["identity_switched"] = True

        # Inject EMA-smoothed landmarks and tracked bbox back into face_dict so downstream
        # pipelines (pose, landmark, ArcFace) operate on temporally stable data.
        if face_dict is not None:
            if track_res.get("tracked_bbox") is not None:
                face_dict["bbox"] = track_res["tracked_bbox"]
            smoothed = track_res.get("smoothed_landmarks")
            if smoothed:
                if smoothed.get("kps") is not None:
                    face_dict["kps"] = smoothed["kps"]
                if smoothed.get("landmark_2d_106") is not None:
                    face_dict["landmark_2d_106"] = smoothed["landmark_2d_106"]
                if smoothed.get("landmark_3d_68") is not None:
                    face_dict["landmark_3d_68"] = smoothed["landmark_3d_68"]

        # Default outputs when no valid face
        pose_res = {"pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0, "orientation": "CENTER", "mesh_vertices": []}
        quality_res = {"acceptable": False, "composite_score": 0.0, "sharpness": 0.0}
        lm_res = {"left_periocular_crop": None, "right_periocular_crop": None, "landmark_count_2d": 0}
        embedding = None

        face_bbox = face_dict["bbox"] if face_dict else None
        tracking_confidence = round(face_dict["confidence"], 3) if face_dict else 0.0

        if face_dict is not None:
            # ── 3. Landmarks & Periocular (MediaPipe 478) ────────────────────
            # MediaPipe is strictly bounded to the SCRFD face crop.
            # We preserve the authentic SCRFD face bounding box and never overwrite it.
            lm_res = self.landmarks_pipe.process_landmarks(
                face_dict, w, h, frame_bgr=oriented_bgr
            )

            # ── 4. Pose & 3D Geometry ────────────────────────────────────────
            # Use exact metric pose if provided by landmark engine, else fallback to pose_estimator
            pose_res = lm_res.get("pose") or self.pose_estimator.estimate_pose(face_dict)
            if not lm_res.get("geometry_valid", True):
                pose_res["geometry_valid"] = False
                pose_res["geometry_status"] = lm_res.get("geometry_status", "GEOMETRY_INVALID")
                pose_res["mesh_vertices"] = []

            # ── 5. Quality ───────────────────────────────────────────────────
            quality_res = self.quality_engine.assess(oriented_bgr, face_dict["bbox"], pose_res)

            # ── 6. Liveness Ensemble (subsampled every N frames) ─────────────
            # Running ONNX liveness + texture every frame costs ~80-120ms.
            # Cache the result and refresh only every _liveness_interval frames.
            if self._frame_count % self._liveness_interval == 0:
                try:
                    self._cached_liveness = self.liveness_ensemble.evaluate_frame(
                        oriented_bgr,
                        face_dict["bbox"],
                        lm_res.get("landmark_3d_68", face_dict.get("landmark_3d_68", [])),
                        pose_res
                    )
                except Exception:
                    pass  # keep previous cached result on error
            liveness_res = self._cached_liveness

            # ── 7. ArcFace Embedding (conditional) ──────────────────────────
            capture_state = self.capture_machine.state
            need_embedding = (quality_res.get("acceptable", False) or quality_res.get("composite_score", 0.0) >= 0.35) and capture_state in (
                "CAPTURING_FRONT", "CAPTURING_LEFT", "CAPTURING_RIGHT"
            )
            if need_embedding:
                try:
                    embedding = self.embedder.extract_embedding(oriented_bgr, face_dict)
                except Exception:
                    embedding = None

        else:
            liveness_res = self._cached_liveness

        # ── 8. Capture State Machine ─────────────────────────────────────────
        capture_res = self.capture_machine.step(
            detection_status=det_res.status,
            face_dict=face_dict,
            pose=pose_res,
            quality=quality_res,
            liveness=liveness_res,
            frame_bgr=oriented_bgr if face_dict is not None else None,
            embedding=embedding
        )

        latency_ms = (time.time() - t0) * 1000.0

        instruction = capture_res.get("instruction", "Look at the camera")
        if detected_angle != 0 and instruction == "No face detected. Please face the camera.":
            instruction = f"Subject acquired at {detected_angle}°. Auto-aligning..."

        telemetry = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "latency_ms": round(latency_ms, 1),
            "detection_status": det_res.status,
            "message": det_res.message,
            "track_id": track_res.get("track_id"),
            "track_state": track_res.get("state"),
            "bbox": face_bbox,
            "confidence": tracking_confidence,
            "pose": pose_res,
            "quality": quality_res,
            "liveness": liveness_res,
            "capture": capture_res,
            "detected_rotation": detected_angle,
            "aspect_orientation": "VERTICAL" if (h > w or detected_angle in [90, 270]) else "HORIZONTAL",
            "frame_width": w,
            "frame_height": h,
            "periocular": {
                "left_crop": lm_res.get("left_periocular_crop"),
                "right_crop": lm_res.get("right_periocular_crop"),
                "left_b64": lm_res.get("left_crop_b64"),
                "right_b64": lm_res.get("right_crop_b64"),
                "ear_left": lm_res.get("ear_left"),
                "ear_right": lm_res.get("ear_right")
            },
            "instruction": instruction,
            "landmark_engine": lm_res.get("engine", "mediapipe_478"),
            "geometry_valid": lm_res.get("geometry_valid", True),
            "geometry_status": lm_res.get("geometry_status", "VALID"),
            "geometry_reasons": lm_res.get("geometry_reasons", []),
            "landmarks_478": lm_res.get("landmarks_478", []),
            "debug_pipeline": lm_res.get("debug_pipeline")
        }

        self.last_telemetry = telemetry
        return telemetry

    def generate_final_verdict(self) -> Dict[str, Any]:
        """
        Runs multi-frame matching between captured candidates and uploaded documents,
        then fuses all evidence to render final decision.
        """
        # Match live candidates vs documents
        matching_res = self.matcher.match_live_against_documents(
            live_candidates=self.capture_machine.burst_candidates,
            documents=self.uploaded_documents
        )

        # Collect best validated liveness and quality evidence across all captures and session history
        all_cands = []
        for angle in ("FRONT", "LEFT", "RIGHT"):
            all_cands.extend(self.capture_machine.burst_candidates.get(angle, []))

        # Best quality evidence
        if all_cands:
            best_q = max(all_cands, key=lambda c: float(c.quality.get("composite_score", 0.0)))
            quality_evidence = best_q.quality
        else:
            quality_evidence = self.last_telemetry.get("quality", {"composite_score": 0.0, "acceptable": False})

        # Best validated liveness evidence (pool candidates + live cached liveness)
        liveness_pool = [c.liveness for c in all_cands if c.liveness and c.liveness.get("liveness_score", 0.0) > 0]
        if self._cached_liveness and self._cached_liveness.get("liveness_score", 0.0) > 0:
            liveness_pool.append(self._cached_liveness)
        if self.last_telemetry.get("liveness", {}).get("liveness_score", 0.0) > 0:
            liveness_pool.append(self.last_telemetry["liveness"])

        if liveness_pool:
            liveness_evidence = max(liveness_pool, key=lambda l: (
                float(l.get("liveness_score", 0.0)) + (0.10 if l.get("is_live", False) else 0.0)
            ))
        else:
            liveness_evidence = self._cached_liveness or {"liveness_score": 0.0, "is_live": False}

        report = self.decision_engine.evaluate_session(
            session_id=self.session_id,
            liveness_data=liveness_evidence,
            quality_data=quality_evidence,
            matching_data=matching_res,
            session_integrity=self.session_integrity
        )

        report["models_audited"] = {
            "face_detector": "SCRFD-500M (buffalo_s)",
            "landmarks_2d": "2D-106Det (buffalo_s)",
            "landmarks_3d_pose": "1K-3D-68 (buffalo_s)",
            "anti_spoofing": "MiniFASNetV2 (Silent-Face-Anti-Spoofing)",
            "face_recognition": "ArcFace-MobileFaceNet 512-d",
            "decision_policy": self.decision_engine.policy_version
        }

        self.final_report = report
        return report


class SessionManager:
    def __init__(self):
        # Shared detector and liveness instances to conserve resources
        self.shared_detector = SCRFDDetector(min_confidence=0.60)
        self.shared_liveness = LivenessEnsemble(model_path="models/liveness/minifasnet_v2.onnx")
        self.sessions: Dict[str, VerificationSession] = {}

    def create_session(self, session_id: Optional[str] = None) -> VerificationSession:
        if not session_id:
            random_suffix = str(uuid.uuid4())[:6].upper()
            session_id = f"VRF-{random_suffix}"
        session = VerificationSession(
            session_id=session_id,
            shared_detector=self.shared_detector,
            shared_liveness=self.shared_liveness
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[VerificationSession]:
        return self.sessions.get(session_id)
