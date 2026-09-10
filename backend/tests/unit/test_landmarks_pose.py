"""
Unit test for Landmark Pipeline and Pose Estimator.
"""

import cv2
from services.detection.scrfd_detector import SCRFDDetector
from services.landmarks.landmark_pipeline import LandmarkPipeline
from services.geometry3d.pose_estimator import PoseAndGeometryEstimator


def test_landmarks_and_pose_extraction():
    img = cv2.imread("videoframe_4589.png")
    detector = SCRFDDetector(min_confidence=0.60)
    det_res = detector.detect(img)
    face = det_res.primary_face
    assert face is not None

    h, w = img.shape[:2]
    landmark_pipe = LandmarkPipeline()
    lm_res = landmark_pipe.process_landmarks(face, w, h, frame_bgr=img)

    assert lm_res["engine"] == "mediapipe_478"
    assert lm_res["landmark_count_2d"] == 478
    assert lm_res["landmark_count_3d"] == 478
    assert len(lm_res["landmarks_478"]) == 478
    assert len(lm_res["landmark_3d_68"]) == 68
    assert lm_res["periocular_available"] is True
    assert lm_res["left_periocular_crop"] is not None
    assert lm_res["right_periocular_crop"] is not None

    pose_res = lm_res["pose"]
    assert "pitch_deg" in pose_res
    assert "yaw_deg" in pose_res
    assert "roll_deg" in pose_res
    assert pose_res["is_monocular_3d"] is True
    assert len(pose_res["mesh_vertices"]) == 478
