"""
FastAPI Application Entry Point for Cert8fy Biometric Engine.
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.routes.verification import router as verification_router
from app.websocket.stream_handler import handle_verification_stream

app = FastAPI(
    title="Cert8fy Advanced Facial Identity Verification Engine",
    description="Real-Time Multimodal Facial Biometric Verification and Forensic Evidence Engine",
    version="1.0.0"
)

# Enable CORS for local web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST routes
app.include_router(verification_router)


# Register WebSocket stream route
@app.websocket("/api/v1/verification/{session_id}/stream")
async def websocket_stream(websocket: WebSocket, session_id: str):
    await handle_verification_stream(websocket, session_id)


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "HEALTHY",
        "engine": "Cert8fy Biometric Core",
        "hardware_acceleration": "ONNX Runtime + MediaPipe XNNPACK",
        "models_loaded": {
            "detector": "SCRFD-500M",
            "landmarks_3d_pose": "MediaPipe Face Landmarker (478 Dense)",
            "liveness_pad": "MiniFASNetV2",
            "recognition": "ArcFace-MobileFaceNet 512-d"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
