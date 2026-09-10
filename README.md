# Cert8fy — Advanced Facial Identity Verification Engine

Real-time multimodal biometric identity verification, presentation attack detection (PAD), and forensic document matching platform.

---

## Directory Structure

```text
screenAI/
├── backend/                       # Python Computer Vision & Biometric Backend
│   ├── app/                       # FastAPI web application (routes, schemas, websocket)
│   ├── services/                  # Computer vision modular services
│   │   ├── detection/             # SCRFD-500M face detector & multi-angle sweep
│   │   ├── landmarks/             # MediaPipe Face Landmarker (478 3D dense points)
│   │   ├── tracking/              # Spatial IoU & Kalman tracker
│   │   ├── recognition/           # ArcFace MobileFaceNet 512-d feature extraction
│   │   ├── liveness/              # MiniFASNetV2 Deep PAD anti-spoofing
│   │   ├── geometry3d/            # 3D pose Euler angles & mesh projection
│   │   ├── quality/               # Sharpness, illumination & face area scoring
│   │   ├── capture/               # 3-angle guided burst capture state machine
│   │   ├── document_face/         # ID document face extraction
│   │   ├── matching/              # Multi-frame cosine similarity matching
│   │   └── decision/              # Forensic evidence decision fusion engine
│   ├── models/                    # Neural network models & registry
│   ├── tests/                     # Automated unit and integration test suite
│   ├── requirements.txt           # Python dependencies
│   └── run_backend.bat            # Backend startup batch script
│
├── frontend/                      # React / Vite / TypeScript Web Application
│   ├── src/
│   │   ├── components/            # Camera viewport, HUD, Document deck, Dossier
│   │   ├── App.tsx                # Main workstation shell
│   │   └── index.css              # Dark forensic design system (Vanilla CSS)
│   ├── package.json
│   ├── vite.config.ts
│   └── run_frontend.bat           # Frontend startup batch script
│
├── docs/                          # Product and technical specifications
│   ├── PRD.md
│   └── TECH.md
│
└── start.bat                      # One-click launcher for both Backend & Frontend
```

---

## Running the Application

### Option A: One-Click Launch (Windows)
Double-click **`start.bat`** in the root directory. It will open two separate terminal windows:
1. **Backend Server** on `http://localhost:8000`
2. **Frontend Workstation** on `http://localhost:3000`

---

### Option B: Terminal Commands

#### Terminal 1 — Backend
```powershell
cd d:\screenAI\backend
$env:PYTHONPATH = "d:\screenAI\backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

*Or in Command Prompt (CMD):*
```cmd
cd /d d:\screenAI\backend
set PYTHONPATH=d:\screenAI\backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 — Frontend
```powershell
cd d:\screenAI\frontend
npm run dev
```

---

## Verifying the System

### Run Automated Tests
From the `backend/` directory:
```powershell
cd d:\screenAI\backend
$env:PYTHONPATH = "d:\screenAI\backend"
python -m pytest tests/
```
All 9 unit and integration tests will execute.

### Production Build
From the `frontend/` directory:
```powershell
cd d:\screenAI\frontend
npm run build
```
Builds the static bundle with zero TypeScript errors.

---

## Key Endpoints
- **Web Application**: `http://localhost:3000/`
- **Health Check**: `http://localhost:8000/api/v1/health`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **WebSocket Stream**: `ws://127.0.0.1:8000/api/v1/verification/{session_id}/stream`
