"""
Pydantic Schemas for Verification Sessions, Document Ingestion, and Evidence Dossiers.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str
    created_at: float
    model_policy_version: str


class DocumentUploadResponse(BaseModel):
    success: bool
    doc_id: str
    doc_type: str
    face_detected: bool
    confidence: float
    sharpness: float
    face_crop_url: Optional[str] = None
    message: str = ""


class TelemetryFrameMessage(BaseModel):
    type: str = "frame_telemetry"
    timestamp: float
    fps: float
    latency_ms: float
    detection_status: str  # EXACTLY_ONE, NO_FACE, MULTIPLE_SUBJECTS, FACE_TOO_SMALL
    track_id: Optional[str] = None
    bbox: Optional[List[int]] = None  # [x, y, w, h]
    pose: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)
    liveness: Dict[str, Any] = Field(default_factory=dict)
    capture: Dict[str, Any] = Field(default_factory=dict)
    periocular: Dict[str, Any] = Field(default_factory=dict)
    instruction: str = ""


class FinalVerificationReport(BaseModel):
    session_id: str
    decision: str  # VERIFIED, MANUAL_REVIEW, FAILED
    decision_timestamp: float
    policy_version: str
    summary_message: str
    evidence_breakdown: Dict[str, Any]
    rejection_reasons: List[str] = Field(default_factory=list)
    review_reasons: List[str] = Field(default_factory=list)
    models_audited: Dict[str, str] = Field(default_factory=dict)
