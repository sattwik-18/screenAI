"""
Monocular 3D Face Geometry and Pose Estimator.
Explicitly documented: Monocular 3D Face Geometry (not physical depth).

Calculates:
- Head pose Euler angles: Pitch, Yaw, Roll.
- Monocular 3D mesh wireframe vertices.
- 3D coordinate axes projection vectors.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class PoseAndGeometryEstimator:
    def estimate_pose(self, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts Euler angles (pitch, yaw, roll) from face model output.
        """
        raw_pose = face_data.get("pose", [0.0, 0.0, 0.0])
        
        # Raw pose from 1k3d68 is [pitch, yaw, roll] in degrees
        pitch = float(raw_pose[0]) if len(raw_pose) > 0 else 0.0
        yaw = float(raw_pose[1]) if len(raw_pose) > 1 else 0.0
        roll = float(raw_pose[2]) if len(raw_pose) > 2 else 0.0
        
        # Classify current orientation
        orientation = "FRONT"
        if yaw < -14.0:
            orientation = "LOOKING_LEFT"
        elif yaw > 14.0:
            orientation = "LOOKING_RIGHT"
        elif pitch < -15.0:
            orientation = "LOOKING_DOWN"
        elif pitch > 15.0:
            orientation = "LOOKING_UP"

        # Extract 3D vertices for wireframe mesh.
        # InsightFace's 1k3d68 model applies the inverse affine transform internally
        # before returning results, so these coordinates are already in full-frame
        # pixel space — no further remapping is needed.
        raw_3d = face_data.get("landmark_3d_68", [])
        mesh_vertices = []
        if len(raw_3d) >= 68:
            mesh_vertices = [
                [round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1)]
                for p in raw_3d
            ]


        return {
            "pitch_deg": round(pitch, 2),
            "yaw_deg": round(yaw, 2),
            "roll_deg": round(roll, 2),
            "orientation": orientation,
            "is_monocular_3d": True,
            "mesh_vertices_count": len(mesh_vertices),
            "mesh_vertices": mesh_vertices,
            "pose_vector": [pitch, yaw, roll]
        }
