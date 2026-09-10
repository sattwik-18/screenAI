"""
Unit tests for SpatialIoUKalmanTracker.
Tests:
- Initialization of TRACK_001
- Temporal association and update
- Discontinuous jump causing IDENTITY_SWITCH_DETECTED
"""

from services.tracking.spatial_tracker import SpatialIoUKalmanTracker


def test_spatial_tracker_lifecycle():
    tracker = SpatialIoUKalmanTracker(iou_threshold=0.30, max_discontinuity_px=100.0)
    
    # Frame 1
    det1 = {"bbox": [100, 100, 150, 150], "confidence": 0.95}
    t1 = tracker.update(det1)
    assert t1["track_id"] == "TRACK_001"
    assert t1["identity_switched"] is False
    
    # Frame 2: Smooth motion (slight shift)
    det2 = {"bbox": [104, 102, 150, 150], "confidence": 0.96}
    t2 = tracker.update(det2)
    assert t2["track_id"] == "TRACK_001"
    assert t2["identity_switched"] is False
    
    # Frame 3: Discontinuous teleportation (> 300px away)
    det3 = {"bbox": [550, 600, 150, 150], "confidence": 0.94}
    t3 = tracker.update(det3)
    assert t3["identity_switched"] is True
    assert t3["track_id"] == "TRACK_002"
