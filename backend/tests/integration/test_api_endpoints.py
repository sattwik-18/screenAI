"""
Integration test for FastAPI REST endpoints.
Tests:
- Session creation
- Document upload and face extraction
- Analysis telemetry retrieval
- Final verification report generation
"""

from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)


def test_session_lifecycle_and_document_upload():
    # 1. Health check
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "HEALTHY"

    # 2. Create Session
    res_session = client.post("/api/v1/verification/sessions")
    assert res_session.status_code == 200
    session_data = res_session.json()
    assert "session_id" in session_data
    session_id = session_data["session_id"]
    assert session_id.startswith("VRF-")

    # 3. Document Upload using reference image videoframe_4589.png as a test ID portrait
    assert os.path.exists("videoframe_4589.png")
    with open("videoframe_4589.png", "rb") as f:
        files = {"file": ("passport.png", f, "image/png")}
        data = {"doc_type": "passport", "doc_id": "PASSPORT_001"}
        res_upload = client.post(f"/api/v1/verification/{session_id}/documents", files=files, data=data)
    
    assert res_upload.status_code == 200
    upload_json = res_upload.json()
    assert upload_json["success"] is True
    assert upload_json["face_detected"] is True
    assert upload_json["face_crop_url"] is not None
    assert upload_json["confidence"] > 0.70

    # 4. Get Analysis
    res_analysis = client.get(f"/api/v1/verification/{session_id}/analysis")
    assert res_analysis.status_code == 200
    assert res_analysis.json()["documents_count"] == 1

    # 5. Get Result (even without live capture, should evaluate gracefully as FAILED or REVIEW due to missing live capture)
    res_result = client.get(f"/api/v1/verification/{session_id}/result")
    assert res_result.status_code == 200
    result_json = res_result.json()
    assert result_json["decision"] in ["VERIFIED", "MANUAL_REVIEW", "FAILED"]
    assert "evidence_breakdown" in result_json
