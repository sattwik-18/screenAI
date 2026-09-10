"""
Unit test for SCRFD face detection.
Tests:
- Single face detection on reference frame
- Format of bounding boxes and landmarks
- Enforcement of EXACTLY_ONE status
"""

import os
import cv2
import pytest
from services.detection.scrfd_detector import SCRFDDetector


def test_scrfd_detection_single_face():
    img_path = "videoframe_4589.png"
    assert os.path.exists(img_path), "Test image videoframe_4589.png missing"
    
    detector = SCRFDDetector(min_confidence=0.60)
    img = cv2.imread(img_path)
    res = detector.detect(img)
    
    assert res.status == "EXACTLY_ONE"
    assert len(res.faces) == 1
    
    face = res.primary_face
    assert face is not None
    assert face["confidence"] > 0.70
    assert len(face["bbox"]) == 4
    assert face["width"] > 100
    assert face["height"] > 100
    assert len(face["kps"]) == 5
    # InsightFace landmark models (1k3d68, 2d106det) are purged; landmarks are now handled by MediaPipe
    assert len(face.get("landmark_2d_106", [])) == 0
    assert len(face.get("landmark_3d_68", [])) == 0


def test_scrfd_empty_image():
    detector = SCRFDDetector()
    empty = cv2.imread("non_existent_path.jpg")
    res = detector.detect(empty)
    assert res.status == "NO_FACE"
