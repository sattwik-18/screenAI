"""
Unit test for MiniFASNet PAD and Liveness Ensemble.
"""

import cv2
from services.detection.scrfd_detector import SCRFDDetector
from services.liveness.liveness_ensemble import LivenessEnsemble


def test_liveness_ensemble_real_face():
    img = cv2.imread("videoframe_4589.png")
    detector = SCRFDDetector(min_confidence=0.60)
    det_res = detector.detect(img)
    face = det_res.primary_face
    assert face is not None

    ensemble = LivenessEnsemble(model_path="models/liveness/minifasnet_v2.onnx")
    res = ensemble.evaluate_frame(
        frame_bgr=img,
        face_bbox=face["bbox"],
        landmarks_3d=face["landmark_3d_68"],
        pose={"pitch_deg": 0.0, "yaw_deg": 0.0}
    )

    assert "liveness_score" in res
    assert "deep_pad" in res
    assert "temporal" in res
    assert "texture" in res
    assert res["sensors"]["depth_available"] is False
    assert res["sensors"]["depth_consistency"] is None  # Strict rule: must be None, not 1.0!
