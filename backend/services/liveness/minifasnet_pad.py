"""
MiniFASNetV2 Deep Presentation Attack Detection (PAD) Service.
Uses the ONNX model trained on Silent-Face-Anti-Spoofing.
Preprocesses:
- Crops face with 2.7x scale margin
- Resizes to 80x80 pixels
- Normalizes BGR tensor [1, 3, 80, 80]
- Softmax outputs 3 classes: [Class 0: Live, Class 1: Print Attack, Class 2: Replay Attack]
"""

import os
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort


class MiniFASNetPAD:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None or not os.path.exists(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidate = os.path.join(base_dir, "models", "liveness", "minifasnet_v2.onnx")
            if os.path.exists(candidate):
                model_path = candidate
            elif not model_path:
                model_path = "models/liveness/minifasnet_v2.onnx"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MiniFASNet model not found at {model_path}")
        
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.scale_factor = 2.7

    def _crop_scaled_face(
        self,
        img: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        h, w = img.shape[:2]
        x, y, box_w, box_h = bbox
        
        # Calculate center and scaled dimensions
        cx = x + box_w / 2.0
        cy = y + box_h / 2.0
        
        scaled_w = box_w * self.scale_factor
        scaled_h = box_h * self.scale_factor
        
        x1 = max(0, int(cx - scaled_w / 2.0))
        y1 = max(0, int(cy - scaled_h / 2.0))
        x2 = min(w, int(cx + scaled_w / 2.0))
        y2 = min(h, int(cy + scaled_h / 2.0))
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return cv2.resize(img, (80, 80))
        return cv2.resize(crop, (80, 80))

    def evaluate(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """
        Runs MiniFASNet inference on 2.7x scaled crop.
        Returns softmax probabilities for Live vs Spoof.
        """
        crop_80 = self._crop_scaled_face(frame_bgr, face_bbox)
        
        # Format tensor: BGR, HWC -> CHW, float32
        tensor = crop_80.astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        raw_logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        probs = exp_logits / np.sum(exp_logits)

        # Silent-Face-Anti-Spoofing MiniFASNet class mapping:
        # Class 1: Real / Live Human
        # Class 0: 2D Photo / Print Attack
        # Class 2: Screen / Replay Attack
        live_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
        print_attack = float(probs[0])
        replay_attack = float(probs[2]) if len(probs) > 2 else 0.0

        is_live = live_score >= 0.70

        return {
            "model": "MiniFASNetV2",
            "live_score": round(live_score, 4),
            "print_attack_score": round(print_attack, 4),
            "replay_attack_score": round(replay_attack, 4),
            "is_live": is_live,
            "raw_logits": [round(float(l), 3) for l in raw_logits]
        }
