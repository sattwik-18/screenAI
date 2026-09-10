"""
Unit test for Evidence Fusion and Decision Engine.
Tests:
- Genuine case -> VERIFIED
- Liveness failure -> FAILED
- Borderline match -> MANUAL_REVIEW
- Session integrity violation -> FAILED
- Missing depth sensor -> recorded as None/null, does not grant passing credit
"""

from services.decision.evidence_engine import EvidenceDecisionEngine


def test_evidence_decision_scenarios():
    engine = EvidenceDecisionEngine()

    # Scenario 1: Genuine Verification
    genuine_liveness = {"liveness_score": 0.94, "is_live": True, "sensors": {"depth_consistency": None}}
    genuine_quality = {"composite_score": 0.88, "acceptable": True}
    genuine_match = {
        "aggregate_match_score": 0.85,
        "cross_angle_consistency": {"composite_consistency": 0.72, "consistency_pass": True},
        "document_matches": [{"doc_type": "passport", "robust_score": 0.85}]
    }
    genuine_integrity = {"identity_switched": False, "multiple_faces_detected": False}

    res_genuine = engine.evaluate_session("VRF-001", genuine_liveness, genuine_quality, genuine_match, genuine_integrity)
    assert res_genuine["decision"] == "VERIFIED"
    assert res_genuine["evidence_breakdown"]["liveness"]["depth_sensor"] is None

    # Scenario 2: Spoof Attack (Liveness Fail)
    spoof_liveness = {"liveness_score": 0.35, "is_live": False, "sensors": {"depth_consistency": None}}
    res_spoof = engine.evaluate_session("VRF-002", spoof_liveness, genuine_quality, genuine_match, genuine_integrity)
    assert res_spoof["decision"] == "FAILED"
    assert any("LIVENESS_FAILED" in r for r in res_spoof["rejection_reasons"])

    # Scenario 3: Borderline Match -> MANUAL_REVIEW
    borderline_match = {
        "aggregate_match_score": 0.65,
        "cross_angle_consistency": {"composite_consistency": 0.58, "consistency_pass": True},
        "document_matches": [{"doc_type": "passport", "robust_score": 0.65}]
    }
    res_borderline = engine.evaluate_session("VRF-003", genuine_liveness, genuine_quality, borderline_match, genuine_integrity)
    assert res_borderline["decision"] == "MANUAL_REVIEW"

    # Scenario 4: Identity Switch Detected -> FAILED
    compromised_integrity = {"identity_switched": True, "multiple_faces_detected": False}
    res_compromised = engine.evaluate_session("VRF-004", genuine_liveness, genuine_quality, genuine_match, compromised_integrity)
    assert res_compromised["decision"] == "FAILED"
    assert any("SESSION_INTEGRITY_VIOLATION" in r for r in res_compromised["rejection_reasons"])
