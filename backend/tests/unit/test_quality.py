"""
Unit tests for Quality Assessment Engine.
"""

import cv2
import numpy as np
from services.quality.quality_engine import QualityEngine


def test_quality_sharpness_and_illumination():
    engine = QualityEngine()
    
    # Sharp image
    img = cv2.imread("videoframe_4589.png")
    bbox = [160, 360, 340, 510]
    pose = {"pitch_deg": -5.0, "yaw_deg": 10.0, "roll_deg": 0.0}
    
    res = engine.assess(img, bbox, pose)
    assert res["acceptable"] is True
    assert res["sharpness"] > 60.0
    assert 0.0 <= res["composite_score"] <= 1.0

    # Artificially blurred image
    blurred = cv2.GaussianBlur(img, (35, 35), 0)
    res_blur = engine.assess(blurred, bbox, pose)
    assert res_blur["sharpness"] < res["sharpness"]
