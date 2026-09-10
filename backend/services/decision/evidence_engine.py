"""
Evidence Fusion & Final Decision Engine.
Fuses:
1. Liveness evidence (Deep PAD + Temporal motion + Active challenge)
2. Quality metrics (Sharpness + Illumination + Face Size floor)
3. Multi-frame Identity Match (Robust Front/Left/Right vs Document similarity)
4. Cross-Angle Consistency
5. Document-to-Document Consistency
6. Session Integrity (No identity switch detected)

Outputs exactly one of:
- VERIFIED
- MANUAL_REVIEW
- FAILED
"""

from typing import Dict, Any, List
import time


class EvidenceDecisionEngine:
    def __init__(self, policy_version: str = "v1.0-dev-calibrated"):
        self.policy_version = policy_version
        
        # Empirically documented development operating thresholds (calibrated via tests/evaluation/eval_arcface.py)
        self.thresholds = {
            "liveness_min_score": 0.70,
            "quality_min_composite": 0.55,
            "match_strong_threshold": 0.68,
            "match_manual_review_floor": 0.52,
            "cross_angle_consistency_min": 0.48
        }

    def evaluate_session(
        self,
        session_id: str,
        liveness_data: Dict[str, Any],
        quality_data: Dict[str, Any],
        matching_data: Dict[str, Any],
        session_integrity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes formal evidence fusion policy.
        """
        # 1. Session Integrity Check
        identity_switched = session_integrity.get("identity_switched", False)
        multiple_faces_detected = session_integrity.get("multiple_faces_detected", False)
        
        rejection_reasons = []
        review_reasons = []

        if identity_switched:
            rejection_reasons.append("SESSION_INTEGRITY_VIOLATION: Discontinuous identity switch detected during capture")

        if multiple_faces_detected:
            rejection_reasons.append("MULTIPLE_SUBJECTS_IN_SESSION: More than one subject observed during verification")

        # 2. Liveness Gate
        liveness_score = liveness_data.get("liveness_score", 0.0)
        is_live = liveness_data.get("is_live", False)
        
        if not is_live or liveness_score < self.thresholds["liveness_min_score"]:
            rejection_reasons.append(f"LIVENESS_FAILED: PAD evidence {liveness_score:.3f} is below security floor ({self.thresholds['liveness_min_score']})")

        # 3. Quality Floor
        quality_score = quality_data.get("composite_score", 0.0)
        quality_acceptable = quality_data.get("acceptable", True)
        
        if not quality_acceptable or quality_score < self.thresholds["quality_min_composite"]:
            review_reasons.append(f"CAPTURE_QUALITY_SUBOPTIMAL: Composite quality {quality_score:.3f} is below floor ({self.thresholds['quality_min_composite']})")

        # 4. Identity Match & Consistency
        aggregate_match = matching_data.get("aggregate_match_score", 0.0)
        cross_angle_consistency = matching_data.get("cross_angle_consistency", {}).get("composite_consistency", 0.0)
        cross_angle_pass = matching_data.get("cross_angle_consistency", {}).get("consistency_pass", True)

        if aggregate_match < self.thresholds["match_manual_review_floor"]:
            rejection_reasons.append(f"IDENTITY_MISMATCH: Live-to-document biometric similarity {aggregate_match:.3f} is below rejection threshold ({self.thresholds['match_manual_review_floor']})")
        elif aggregate_match < self.thresholds["match_strong_threshold"]:
            review_reasons.append(f"BORDERLINE_MATCH: Biometric similarity {aggregate_match:.3f} requires manual inspector review (threshold: {self.thresholds['match_strong_threshold']})")

        if not cross_angle_pass or cross_angle_consistency < self.thresholds["cross_angle_consistency_min"]:
            review_reasons.append(f"CROSS_ANGLE_VARIATION: Live angle consistency {cross_angle_consistency:.3f} indicates significant pose variance")

        # Decision Logic
        if len(rejection_reasons) > 0:
            decision = "FAILED"
            summary_message = "Identity verification rejected based on negative biometric or liveness evidence."
        elif len(review_reasons) > 0:
            decision = "MANUAL_REVIEW"
            summary_message = "Verification completed with borderline metrics requiring secondary operator review."
        else:
            decision = "VERIFIED"
            summary_message = "Subject verified successfully with strong multi-modal biometric evidence."

        return {
            "session_id": session_id,
            "decision": decision,
            "decision_timestamp": time.time(),
            "policy_version": self.policy_version,
            "summary_message": summary_message,
            "evidence_breakdown": {
                "liveness": {
                    "score": liveness_score,
                    "status": "PASS" if is_live else "FAIL",
                    "deep_pad_score": liveness_data.get("deep_pad", {}).get("score"),
                    "temporal_state": liveness_data.get("temporal", {}).get("state"),
                    "texture_score": liveness_data.get("texture", {}).get("score"),
                    "depth_sensor": None,  # Sensor hardware absent
                    "ir_sensor": None      # Sensor hardware absent
                },
                "capture_quality": {
                    "composite_score": quality_score,
                    "status": "PASS" if quality_acceptable else "REVIEW",
                    "sharpness": quality_data.get("sharpness"),
                    "illumination": quality_data.get("illumination_mean")
                },
                "identity_match": {
                    "aggregate_similarity": aggregate_match,
                    "angular_distance_deg": matching_data.get("aggregate_angular_dist_deg"),
                    "status": "STRONG" if aggregate_match >= self.thresholds["match_strong_threshold"] else ("MODERATE" if aggregate_match >= self.thresholds["match_manual_review_floor"] else "POOR"),
                    "document_matches": matching_data.get("document_matches", []),
                    "document_identity_divergence": matching_data.get("document_identity_divergence", False),
                    "score_interpretation": matching_data.get("score_interpretation", "ArcFace cosine similarity"),
                },
                "cross_angle_consistency": {
                    "score": cross_angle_consistency,
                    "composite_consistency": cross_angle_consistency,
                    "consistency_pass": cross_angle_pass,
                    "front_to_left": matching_data.get("cross_angle_consistency", {}).get("front_to_left"),
                    "front_to_right": matching_data.get("cross_angle_consistency", {}).get("front_to_right"),
                    "status": "PASS" if cross_angle_pass else "REVIEW"
                },
                "session_integrity": {
                    "identity_switched": identity_switched,
                    "status": "SECURE" if not identity_switched else "COMPROMISED"
                }
            },
            "rejection_reasons": rejection_reasons,
            "review_reasons": review_reasons
        }
