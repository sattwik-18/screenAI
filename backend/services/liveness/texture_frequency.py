"""
High-Frequency Texture and Fourier Spectral Anti-Spoofing Analyzer.
Analyzes 2D Fourier power spectrum on cropped facial skin to detect:
- Display screen pixel grids (Moiré artifacts).
- Paper matte print textures.
- Natural specular skin reflectance.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


class TextureFrequencyAnalyzer:
    def analyze(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        bx, by, bw, bh = bbox
        
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w, bx + bw)
        y2 = min(h, by + bh)
        
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return {
                "texture_score": 0.50,
                "high_freq_ratio": 0.0,
                "moiré_detected": False
            }

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # 2D Fast Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

        # Segment spectrum into low-frequency core and high-frequency perimeter
        rows, cols = 128, 128
        crow, ccol = rows // 2, cols // 2
        r_low = 20
        r_high = 50

        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)

        low_freq_mask = dist_from_center <= r_low
        high_freq_mask = dist_from_center > r_high

        low_energy = np.mean(magnitude_spectrum[low_freq_mask])
        high_energy = np.mean(magnitude_spectrum[high_freq_mask])

        ratio = float(high_energy / max(1e-4, low_energy))
        
        # Real skin has balanced mid-to-high frequency decay (~0.35 to 0.65).
        # Phone screen display replay often has sharp periodic peaks or compressed high-freq drops.
        moiré_detected = ratio > 0.82 or ratio < 0.22
        
        if moiré_detected:
            texture_score = 0.35
        else:
            texture_score = 0.92

        return {
            "texture_score": round(texture_score, 3),
            "high_freq_ratio": round(ratio, 4),
            "moiré_detected": moiré_detected
        }
