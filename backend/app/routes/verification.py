"""
Verification REST Endpoints.
Handles session creation, document upload, current analysis retrieval, and final report.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
import cv2
import numpy as np
import base64

from app.schemas.session import (
    CreateSessionResponse,
    DocumentUploadResponse,
    FinalVerificationReport
)
from app.services.session_manager import SessionManager, VerificationSession

router = APIRouter(prefix="/api/v1/verification", tags=["verification"])

# Global session manager instance
session_manager = SessionManager()


def get_session_or_404(session_id: str) -> VerificationSession:
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session():
    """
    Creates a new biometric verification session.
    """
    session = session_manager.create_session()
    return CreateSessionResponse(
        session_id=session.session_id,
        status="SESSION_INITIALIZED",
        created_at=session.created_at,
        model_policy_version=session.decision_engine.policy_version
    )


@router.post("/{session_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
    doc_type: str = Form("passport"),  # passport, national_id, visa
    doc_id: str = Form("doc_001")
):
    """
    Uploads an identity document (Passport, National ID, Visa).
    Extracts the portrait face, calculates quality, and extracts ArcFace embedding.
    Does NOT require OCR.
    """
    session = get_session_or_404(session_id)
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    extraction_result = session.doc_extractor.extract_document_face(
        document_bgr=img_bgr,
        doc_type=doc_type,
        doc_id=doc_id
    )

    if not extraction_result.get("success", False):
        return DocumentUploadResponse(
            success=False,
            doc_id=doc_id,
            doc_type=doc_type,
            face_detected=False,
            confidence=0.0,
            sharpness=0.0,
            face_crop_url=None,
            message=extraction_result.get("message", "Document face extraction failed")
        )

    # Store in session document pool
    session.uploaded_documents.append(extraction_result)

    return DocumentUploadResponse(
        success=True,
        doc_id=doc_id,
        doc_type=doc_type,
        face_detected=True,
        confidence=extraction_result["confidence"],
        sharpness=extraction_result["sharpness"],
        face_crop_url=extraction_result["face_crop_url"],
        message="Document face portrait successfully extracted and biometrically encoded"
    )


@router.delete("/{session_id}/documents/{doc_id}")
async def delete_document(session_id: str, doc_id: str):
    """
    Deletes an uploaded document from the active session's biometric pool.
    """
    session = get_session_or_404(session_id)
    initial_len = len(session.uploaded_documents)
    session.uploaded_documents = [
        d for d in session.uploaded_documents if d.get("doc_id") != doc_id
    ]
    if len(session.uploaded_documents) == initial_len:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found in session")
    return {
        "success": True,
        "deleted_doc_id": doc_id,
        "remaining_documents": len(session.uploaded_documents)
    }


@router.get("/{session_id}/analysis")
async def get_analysis(session_id: str):
    """
    Returns the latest frame analysis telemetry.
    """
    session = get_session_or_404(session_id)
    return {
        "session_id": session_id,
        "telemetry": session.last_telemetry,
        "documents_count": len(session.uploaded_documents),
        "capture_state": session.capture_machine.state
    }


@router.get("/{session_id}/result", response_model=FinalVerificationReport)
async def get_verification_result(session_id: str):
    """
    Generates and returns the final biometric verification report.
    Fuses liveness, multi-angle live captures, document matching, and session integrity.
    """
    session = get_session_or_404(session_id)
    report = session.generate_final_verdict()
    return report


@router.post("/{session_id}/reset")
@router.post("/{session_id}/reset-capture")
async def reset_capture(session_id: str, angle: str = None):
    """
    Resets the guided capture sequence or selectively retakes a single angle.
    Preserves uploaded identity documents so the user can re-verify multiple times.
    """
    session = get_session_or_404(session_id)
    if angle and angle in ["FRONT", "LEFT", "RIGHT"]:
        session.capture_machine.captured_snapshots.pop(angle, None)
        session.capture_machine.burst_candidates[angle] = []
        if angle == "FRONT":
            session.capture_machine.state = "CAPTURING_FRONT"
            session.capture_machine.instruction = "Look straight at the camera (FRONT)"
        elif angle == "LEFT":
            session.capture_machine.state = "CAPTURING_LEFT"
            session.capture_machine.instruction = "Turn head SLIGHTLY LEFT (~20°)"
        elif angle == "RIGHT":
            session.capture_machine.state = "CAPTURING_RIGHT"
            session.capture_machine.instruction = "Turn head SLIGHTLY RIGHT (~20°)"
    else:
        session.capture_machine.reset()
    return {"status": "CAPTURE_RESET", "session_id": session_id, "retake_angle": angle}

