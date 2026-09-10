"""
Sensor Provider Abstraction for Biometric Verification.
Supports RGB, Depth, and IR sensors.
Enforces the strict rule: Missing sensors return None (null), NEVER pass or 1.0.
"""

from typing import Optional, Dict, Any
import numpy as np


class BaseSensorProvider:
    def is_available(self) -> bool:
        raise NotImplementedError


class RGBSensorProvider(BaseSensorProvider):
    def is_available(self) -> bool:
        return True

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr


class DepthSensorProvider(BaseSensorProvider):
    """
    Physical depth sensor provider (e.g. RealSense, TrueDepth, Time-of-Flight).
    Returns None if physical depth hardware is not attached.
    """
    def __init__(self, hardware_present: bool = False):
        self._hardware_present = hardware_present

    def is_available(self) -> bool:
        return self._hardware_present

    def get_depth_map(self) -> Optional[np.ndarray]:
        if not self._hardware_present:
            return None
        # Placeholder for actual hardware driver read
        return None


class IRSensorProvider(BaseSensorProvider):
    """
    Infrared camera sensor provider.
    Returns None if IR hardware is not attached.
    """
    def __init__(self, hardware_present: bool = False):
        self._hardware_present = hardware_present

    def is_available(self) -> bool:
        return self._hardware_present

    def get_ir_frame(self) -> Optional[np.ndarray]:
        if not self._hardware_present:
            return None
        return None


class SensorHub:
    def __init__(self):
        self.rgb = RGBSensorProvider()
        self.depth = DepthSensorProvider(hardware_present=False)
        self.ir = IRSensorProvider(hardware_present=False)

    def get_sensor_status(self) -> Dict[str, Any]:
        return {
            "rgb_available": self.rgb.is_available(),
            "depth_available": self.depth.is_available(),
            "ir_available": self.ir.is_available(),
        }
