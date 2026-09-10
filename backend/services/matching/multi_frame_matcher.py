"""
Multi-Frame Face Matcher & Cross-Angle Consistency Analyzer.

Implements a forensic-grade 1:1 biometric matching pipeline:

  MATCHING STRATEGY (published InsightFace/AdaFace best practices):
  ─────────────────────────────────────────────────────────────────
  1. Per angle: pick top-3 burst candidates by quality composite score.
     Compute quality-weighted cosine similarity (each frame weighted by its
     measured quality score — low-sharpness frames get diminished influence).

  2. Angular evidence fusion (forensic weighting):
     - FRONT (80%): direct 1:1 match against frontal document portrait.
       Frontal-to-frontal is the gold-standard forensic comparison.
     - LEFT + RIGHT (10% each): cross-pose corroboration. Contributes only
       when available; cross-pose ArcFace pairs typically score 0.55–0.72.

  3. Cross-angle consistency: validates subject identity across pose changes.
     Measures FRONT↔LEFT and FRONT↔RIGHT cosine similarity.
     Low consistency (<0.50) signals identity switch or spoofing attempt.

  4. Document quality weighting: higher-resolution, sharper document photos
     get higher fusion weight across multiple uploaded documents.

  5. Angular distance reporting (degrees): more interpretable than raw cosine
     for forensic operators. Genuine threshold ≈ 55° (cosine ≈ 0.57).

References:
  - InsightFace multi-frame aggregation: github.com/deepinsight/insightface
  - AdaFace quality-adaptive weighting: github.com/mk-minchul/AdaFace
  - ISO/IEC 19795-1: Biometric Performance Testing and Reporting
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from services.recognition.arcface_embedder import ArcFaceEmbedder


class MultiFrameMatcher:
    def __init__(self, embedder: Optional[ArcFaceEmbedder] = None):
        self.embedder = embedder if embedder is not None else ArcFaceEmbedder()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _best_pair_median(self, listA: list, listB: list) -> float:
        """
        Robust cross-set similarity: compute all pairwise cosines, return median.
        Median is more resistant to outlier frames than max or mean.
        """
        if not listA or not listB:
            return 0.0
        sims = [
            self.embedder.cosine_similarity(a, b)
            for a in listA for b in listB
        ]
        return float(np.median(sims)) if sims else 0.0

    def _angle_score(
        self,
        candidates: list,
        doc_emb: np.ndarray,
        sharpness_floor: float = 15.0,
        top_k: int = 3,
    ) -> Tuple[float, float, int]:
        """
        For a single angle's burst candidates, compute quality-weighted cosine
        similarity against the document embedding.

        Returns:
            (weighted_cosine, best_angular_distance_degrees, num_valid_candidates)
        """
        valid = [
            c for c in candidates
            if c.embedding is not None
            and float(c.quality.get("sharpness", 100.0)) >= sharpness_floor
        ]
        if not valid:
            return 0.0, 180.0, 0

        # Sort by composite quality score (best frames first)
        valid.sort(
            key=lambda c: float(c.quality.get("composite_score", 0.5)),
            reverse=True,
        )
        top = valid[:top_k]

        # Per-frame cosine + quality weight
        sims = np.array([
            self.embedder.cosine_similarity(c.embedding, doc_emb) for c in top
        ], dtype=np.float64)
        quals = np.array([
            max(0.2, float(c.quality.get("composite_score", 0.5))) for c in top
        ], dtype=np.float64)

        # Quality-weighted cosine
        w_cos = float(np.dot(sims, quals) / quals.sum())

        # Best (highest cosine → smallest angle) frame's angular distance
        best_cos = float(sims.max())
        best_cos_clamped = max(-1.0, min(1.0, best_cos))
        best_ang = float(np.degrees(np.arccos(best_cos_clamped)))

        return w_cos, best_ang, len(valid)

    # ── Main API ──────────────────────────────────────────────────────────────

    def match_live_against_documents(
        self,
        live_candidates: Dict[str, List[Any]],
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Forensic multi-frame 1:1 biometric matching.

        Compares quality-weighted live burst captures (FRONT, LEFT, RIGHT) against
        each uploaded identity document's extracted portrait embedding.

        Returns a structured evidence dictionary with cosine scores, angular
        distances, cross-angle consistency, and per-document breakdown.
        """
        if not documents:
            return {
                "matched": False,
                "error": "NO_DOCUMENTS_UPLOADED",
                "message": "At least one identity document is required for verification.",
            }

        # ── 1. Extract all available live embeddings per angle ────────────────
        front_embs = [
            c.embedding for c in live_candidates.get("FRONT", [])
            if c.embedding is not None
        ]
        left_embs = [
            c.embedding for c in live_candidates.get("LEFT", [])
            if c.embedding is not None
        ]
        right_embs = [
            c.embedding for c in live_candidates.get("RIGHT", [])
            if c.embedding is not None
        ]

        # ── 2. Cross-angle consistency (measures subject stability) ───────────
        front_left_sim  = self._best_pair_median(front_embs, left_embs)
        front_right_sim = self._best_pair_median(front_embs, right_embs)
        left_right_sim  = self._best_pair_median(left_embs, right_embs)

        # Composite: average of available frontal cross-pairs (FRONT↔L and FRONT↔R)
        frontal_cross_sims = [s for s in [front_left_sim, front_right_sim] if s > 0.0]
        cross_angle_composite = float(np.mean(frontal_cross_sims)) if frontal_cross_sims else 0.0

        # Angular distance equivalents for reporting
        def _ang(cos: float) -> float:
            return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))

        # ── 3. Per-document matching ──────────────────────────────────────────
        document_matches: List[Dict[str, Any]] = []
        doc_scores: List[float] = []
        doc_weights: List[float] = []

        for doc in documents:
            doc_id   = doc.get("doc_id", "unknown")
            doc_type = doc.get("doc_type", "document")
            raw_emb  = doc.get("embedding", [])
            doc_emb  = np.array(raw_emb, dtype=np.float32)

            if doc_emb.size != 512:
                continue

            # Normalize doc embedding (should already be, but defensive re-norm)
            doc_norm = np.linalg.norm(doc_emb)
            if doc_norm > 1e-6:
                doc_emb = (doc_emb / doc_norm).astype(np.float32)
            else:
                continue

            # Per-angle quality-weighted scores
            front_score, front_ang, front_n = self._angle_score(
                live_candidates.get("FRONT", []), doc_emb
            )
            left_score, left_ang, left_n = self._angle_score(
                live_candidates.get("LEFT", []), doc_emb
            )
            right_score, right_ang, right_n = self._angle_score(
                live_candidates.get("RIGHT", []), doc_emb
            )

            # ── Forensic Evidence Fusion ──────────────────────────────────────
            # FRONT: 80% weight — direct forensic 1:1 counterpart to document portrait
            # LEFT/RIGHT: 10% each — cross-pose corroboration, only if available
            # Weights are normalized so missing angles don't distort the score
            w_front = 0.80 if front_n > 0 else 0.0
            w_left  = 0.10 if left_n  > 0 else 0.0
            w_right = 0.10 if right_n > 0 else 0.0
            total_w = w_front + w_left + w_right

            if total_w > 0:
                combined = (
                    w_front * front_score
                    + w_left  * left_score
                    + w_right * right_score
                ) / total_w
            else:
                combined = 0.0

            # Combined angular distance (weighted average)
            if total_w > 0:
                combined_ang = (
                    w_front * front_ang
                    + w_left  * left_ang
                    + w_right * right_ang
                ) / total_w
            else:
                combined_ang = 180.0

            # ── Document Portrait Quality Weight ──────────────────────────────
            # Higher-res, sharper document photos are more reliable references.
            doc_w_px    = float(doc.get("width", 100))
            doc_h_px    = float(doc.get("height", 100))
            doc_sharp   = float(doc.get("sharpness", 100.0))
            size_factor  = float(np.clip(min(doc_w_px, doc_h_px) / 100.0, 0.3, 1.0))
            sharp_factor = float(np.clip(doc_sharp / 120.0, 0.4, 1.0))
            doc_quality_weight = size_factor * sharp_factor

            doc_scores.append(combined)
            doc_weights.append(doc_quality_weight)

            match_level = (
                "STRONG"   if combined >= 0.68 else
                "MODERATE" if combined >= 0.52 else
                "LOW"
            )

            document_matches.append({
                "doc_id":                 doc_id,
                "doc_type":               doc_type,
                "front_match_score":      round(front_score, 4),
                "front_angular_dist_deg": round(front_ang,   2),
                "left_match_score":       round(left_score,  4),
                "left_angular_dist_deg":  round(left_ang,    2),
                "right_match_score":      round(right_score, 4),
                "right_angular_dist_deg": round(right_ang,   2),
                "robust_score":           round(combined,    4),
                "angular_distance_deg":   round(combined_ang, 2),
                "portrait_quality_weight": round(doc_quality_weight, 3),
                "match_level":            match_level,
            })

        # ── 4. Document-to-Document consistency (if 2+ docs) ─────────────────
        doc_to_doc_sim      = None
        doc_divergence_warn = False
        if len(documents) >= 2:
            d1_emb = np.array(documents[0]["embedding"], dtype=np.float32)
            d2_emb = np.array(documents[1]["embedding"], dtype=np.float32)
            doc_to_doc_sim  = round(self.embedder.cosine_similarity(d1_emb, d2_emb), 4)
            doc_divergence_warn = doc_to_doc_sim < 0.45

        # ── 5. Final aggregate score across documents ─────────────────────────
        if doc_scores and sum(doc_weights) > 0:
            aggregate = float(
                np.sum(np.array(doc_scores) * np.array(doc_weights))
                / np.sum(doc_weights)
            )
        elif doc_scores:
            aggregate = float(np.mean(doc_scores))
        else:
            aggregate = 0.0

        aggregate_ang = _ang(aggregate)

        return {
            "matched":            True,
            "metric":             "cosine_similarity",
            "score_interpretation": (
                "Quality-adaptive ArcFace TTA cosine similarity. "
                "Genuine pairs: 0.55–0.85. Impostors: −0.1–0.30."
            ),
            "aggregate_match_score":     round(aggregate, 4),
            "aggregate_angular_dist_deg": round(aggregate_ang, 2),
            "cross_angle_consistency": {
                "front_to_left":          round(front_left_sim,      4),
                "front_to_right":         round(front_right_sim,     4),
                "left_to_right":          round(left_right_sim,      4),
                "composite_consistency":  round(cross_angle_composite, 4),
                "consistency_pass":       cross_angle_composite >= 0.48,
            },
            "document_matches":              document_matches,
            "document_to_document_consistency": doc_to_doc_sim,
            "document_identity_divergence":     doc_divergence_warn,
        }
