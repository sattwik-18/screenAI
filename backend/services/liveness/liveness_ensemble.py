"""
Unified Liveness Presentation Attack Detection (PAD) Ensemble.
Combines:
1. MiniFASNetV2 Deep PAD classifier.
2. Temporal micro-motion buffer.
3. High-frequency Fourier texture analysis.
4. Sensor abstraction (Depth/IR marked strictly as None when unavailable).
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from services.liveness.minifasnet_pad import MiniFASNetPAD
from services.liveness.temporal_liveness import TemporalLivenessAnalyzer
from services.liveness.texture_frequency import TextureFrequencyAnalyzer
from services.sensors.provider import SensorHub


class LivenessEnsemble:
    def __init__(self, model_path: Optional[str] = None):
        self.minifasnet = MiniFASNetPAD(model_path=model_path)
        self.temporal = TemporalLivenessAnalyzer(window_size=15)
        self.texture = TextureFrequencyAnalyzer()
        self.sensor_hub = SensorHub()

    def evaluate_frame(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks_3d: List[List[float]],
        pose: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Runs comprehensive liveness ensemble.
        """
        # 1. Deep PAD
        pad_res = self.minifasnet.evaluate(frame_bgr, face_bbox)
        
        # 2. Temporal micro-motion
        temp_res = self.temporal.push_frame(landmarks_3d, pose)
        
        # 3. Frequency & Moiré
        text_res = self.texture.analyze(frame_bgr, face_bbox)
        
        # 4. Sensor availability
        sensors = self.sensor_hub.get_sensor_status()

        # Fused liveness score calculation
        # Weights: Deep PAD: 50%, Temporal: 30%, Texture: 20%
        deep_score = pad_res["live_score"]
        temp_score = temp_res["motion_consistency_score"]
        text_score = text_res["texture_score"]

        fused_score = (0.50 * deep_score) + (0.30 * temp_score) + (0.20 * text_score)
        
        # Determine status
        is_live = fused_score >= 0.72 and not text_res["moiré_detected"] and temp_res["motion_state"] != "STATIC_PHOTO_SUSPECTED"

        return {
            "liveness_score": round(fused_score, 4),
            "is_live": is_live,
            "deep_pad": {
                "score": deep_score,
                "print_attack": pad_res["print_attack_score"],
                "replay_attack": pad_res["replay_attack_score"]
            },
            "temporal": {
                "score": temp_score,
                "state": temp_res.get("motion_state", "BUFFERING"),
                "variance": temp_res.get("variance", temp_res.get("coordinate_variance", 0.0))
            },
            "texture": {
                "score": text_score,
                "moiré_detected": text_res["moiré_detected"],
                "high_freq_ratio": text_res["high_freq_ratio"]
            },
            "sensors": {
                "depth_available": sensors["depth_available"],
                "depth_consistency": None,  # Explicitly null when hardware absent
                "ir_available": sensors["ir_available"],
                "ir_response": None        # Explicitly null when hardware absent
            }
        }
