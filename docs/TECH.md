# Cert8fy Advanced Facial Verification --- TECH.md

**Architecture:** Real-time multimodal facial verification\
**Current scope:** Face + liveness + 3D geometry + multi-angle capture +
document-face matching\
**Explicitly excluded for now:** OCR and document text extraction

------------------------------------------------------------------------

# 1. Recommended Technology Stack

  -----------------------------------------------------------------------
  Layer                               Technology
  ----------------------------------- -----------------------------------
  Frontend                            Next.js + React + TypeScript

  Camera                              Browser MediaDevices / WebRTC

  Real-time transport                 WebSocket

  Visual overlay                      Canvas/WebGL

  3D visualization                    Three.js only where useful

  Backend API                         FastAPI + Python

  Face detection                      InsightFace SCRFD

  Face recognition                    InsightFace ArcFace-family model,
                                      subject to model licensing

  Facial landmarks                    MediaPipe Face Landmarker

  Dense 3D alignment                  3DDFA-V2

  Anti-spoofing                       MiniFASNet /
                                      Silent-Face-Anti-Spoofing family

  Computer vision                     OpenCV

  Tracking                            ByteTrack

  Model format                        ONNX

  GPU inference                       NVIDIA TensorRT

  Production model serving            NVIDIA Triton Inference Server

  GPU video pipeline                  NVIDIA DeepStream, if NVIDIA
                                      hardware is used

  Database                            PostgreSQL

  Vector search                       Qdrant

  Cache / coordination                Redis

  Object storage                      S3 / MinIO / Cloudflare R2

  Containers                          Docker

  Observability                       Prometheus + Grafana

  Reverse proxy                       Nginx / Caddy

  Testing                             Pytest + Playwright
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 2. Core Open-Source Projects

## InsightFace

Repository:

https://github.com/deepinsight/insightface

Use for:

-   face detection
-   face recognition
-   alignment
-   embedding generation
-   supporting face-analysis components

Important licensing note: the repository code is MIT, but the project's
current README states that released pretrained models/training data have
separate non-commercial research restrictions and that some recognition
models require licensing contact. **Do not assume the code license gives
you commercial rights to every pretrained weight. Audit the exact
model/weight you deploy before commercial use.**

------------------------------------------------------------------------

## MediaPipe

Repository:

https://github.com/google-ai-edge/mediapipe

Use Face Landmarker for:

-   dense landmarks
-   face tracking
-   blendshape coefficients
-   facial transformation matrices
-   browser-side or server-side visual analysis

The current Face Landmarker implementation exposes blendshapes and
facial transformation matrices, making it useful for the live
visualization and geometry layer.

------------------------------------------------------------------------

## 3DDFA-V2

Repository:

https://github.com/cleardusk/3DDFA_V2

Use for:

-   dense 3D face alignment
-   head pose
-   3DMM parameters
-   3D face rendering
-   geometry analysis

The project supports webcam/video use and ONNX Runtime acceleration.

Use it as a geometry/pose subsystem, not as the sole liveness detector.

------------------------------------------------------------------------

## Face Anti-Spoofing / MiniFASNet

Reference implementation:

https://github.com/yakhyo/face-anti-spoofing

Use as a starting point for passive RGB presentation-attack detection.

It provides MiniFASNetV1SE/V2 inference and ONNX support.

Do not treat a single anti-spoofing model as sufficient for a "most
advanced" system. Build an evidence ensemble around it.

------------------------------------------------------------------------

## ByteTrack

Repository:

https://github.com/FoundationVision/ByteTrack

Use for stable face track IDs across frames.

This enables:

``` text
Face #17
  |
  +-- frame 001
  +-- frame 002
  +-- frame 003
  +-- frame 004
```

rather than detecting every frame as an unrelated face.

------------------------------------------------------------------------

## NVIDIA TensorRT

Repository:

https://github.com/NVIDIA/TensorRT

Use after the prototype is working.

Pipeline:

``` text
PyTorch model
      |
      v
ONNX
      |
      v
TensorRT engine
      |
      v
GPU inference
```

Do not optimize prematurely. First establish correctness, then
benchmark, then optimize.

------------------------------------------------------------------------

## NVIDIA DeepStream

Repository:

https://github.com/NVIDIA/DeepStream

Use when the system moves from prototype to high-performance video
processing.

Potential pipeline:

``` text
Camera
  |
GStreamer
  |
DeepStream
  |
GPU decode
  |
SCRFD
  |
ByteTrack
  |
Face crops
  |
+----------+-----------+-----------+
|          |           |           |
3D       Liveness   Recognition  Quality
```

------------------------------------------------------------------------

## Qdrant

Repository:

https://github.com/qdrant/qdrant

Use for scalable face-embedding similarity search.

For a single user verification against 1--5 document faces, Qdrant is
not strictly necessary; direct vector comparison is simpler. Add Qdrant
when you need enrollment search, duplicate detection, or a larger
identity database.

------------------------------------------------------------------------

# 3. System Architecture

``` text
                         NEXT.JS CLIENT
                              |
                    Camera / WebRTC / UI
                              |
                         WebSocket
                              |
                              v
                    +-------------------+
                    |     FastAPI       |
                    | Verification API  |
                    +---------+---------+
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
        Session Manager   Frame Service    Document Service
             |                |                |
             |                v                v
             |          Face Pipeline     Face Extraction
             |                |                |
             |       +--------+--------+       |
             |       |        |        |       |
             |       v        v        v       |
             |     SCRFD   MediaPipe  3DDFA   |
             |       |        |        |       |
             |       +--------+--------+       |
             |                |                |
             |                v                |
             |           Liveness             |
             |                |                |
             |                v                |
             |          ArcFace Embedding <---+
             |                |
             +----------------+
                              |
                              v
                       Evidence Engine
                              |
                 +------------+-------------+
                 |            |             |
                 v            v             v
             PostgreSQL    Qdrant       Object Store
                              |
                              v
                       Decision Engine
                              |
                              v
                   VERIFIED / REVIEW / FAILED
```

------------------------------------------------------------------------

# 4. Live Video Pipeline

## Step 1 --- Camera acquisition

Frontend obtains a camera stream.

Prefer:

-   high resolution
-   front-facing camera when appropriate
-   autofocus
-   stable frame rate
-   permission handling

Do not upload every raw frame to the server.

Use a hybrid model:

``` text
Browser
  |
  +--> display camera locally
  |
  +--> send selected/necessary frames
  |
  +--> send compact metadata through WebSocket
```

------------------------------------------------------------------------

# 5. Step 2 --- Face Detection

SCRFD detects the face.

Output:

``` json
{
  "track_id": 17,
  "bbox": [x, y, w, h],
  "confidence": 0.99
}
```

Reject/ask for correction when:

-   no face
-   multiple faces
-   face too small
-   severe occlusion

For identity verification, default to exactly one subject.

------------------------------------------------------------------------

# 6. Step 3 --- Tracking

ByteTrack maintains temporal identity.

``` text
Frame 1 -> Track 17
Frame 2 -> Track 17
Frame 3 -> Track 17
Frame 4 -> Track 17
```

This is critical for:

-   liveness
-   pose sequence
-   motion consistency
-   best-frame selection

------------------------------------------------------------------------

# 7. Step 4 --- Face Landmarks

Run MediaPipe Face Landmarker.

Use:

-   landmark positions
-   blendshapes
-   transformation matrix
-   face geometry

The frontend can render a subtle mesh from these results.

Important:

The mesh is a **visualization and geometry signal**, not the final
liveness decision.

------------------------------------------------------------------------

# 8. Step 5 --- 3D Face Analysis

Run 3DDFA-V2 on selected frames or at a lower frequency than the
lightweight tracker.

Extract:

``` text
3DMM parameters
head pose
dense alignment
3D vertices
geometry stability
```

Use the 3D output for:

-   pose
-   geometry consistency
-   capture guidance
-   visualization
-   supporting liveness

Do not interpret monocular 3D reconstruction as physical depth proof.

------------------------------------------------------------------------

# 9. Step 6 --- Liveness

Build a liveness ensemble.

``` text
                    LIVE VIDEO
                        |
       +----------------+----------------+
       |                |                |
       v                v                v
   MiniFASNet       Temporal        Geometry
      RGB             Model          signals
       |                |                |
       +----------------+----------------+
                        |
               Optional depth/IR
                        |
                        v
                Evidence Fusion
                        |
                        v
                  LIVENESS SCORE
```

If depth hardware exists:

``` text
RGB face
   +
Depth face
   +
Temporal motion
   +
Active challenge
```

should be substantially stronger than RGB-only analysis.

------------------------------------------------------------------------

# 10. Step 7 --- Active Challenge

After passive analysis, ask for a short randomized motion sequence when
necessary.

Example:

``` text
"Look slightly left"
       |
       v
Track yaw trajectory
       |
       v
"Look slightly right"
       |
       v
Track opposite yaw
```

Verify the expected motion from the temporal landmarks/pose stream.

Do not rely on a single blink challenge.

------------------------------------------------------------------------

# 11. Step 8 --- Guided Multi-Angle Capture

Capture:

``` text
FRONT
LEFT
RIGHT
OPTIONAL UP
OPTIONAL DOWN
```

For each pose:

``` text
WAIT
  |
Detect target pose
  |
Check quality
  |
Check liveness
  |
Capture burst
  |
Rank frames
  |
Store best candidates
```

Recommended stored set:

``` text
front_01
left_01
right_01
up_01
down_01
```

plus a small number of backup frames when useful.

------------------------------------------------------------------------

# 12. Step 9 --- Best Frame Selection

For each candidate:

``` text
quality =
    sharpness
    pose_quality
    illumination
    face_size
    occlusion
    landmark_stability
    liveness
```

The weighting must be calibrated against validation data.

Do not invent a universal 0--100 score and call it probability.

------------------------------------------------------------------------

# 13. Step 10 --- Recognition Embeddings

For every selected live frame:

``` text
RGB frame
   |
face crop
   |
alignment
   |
recognition model
   |
embedding
```

For every document:

``` text
document image
   |
face detector
   |
face crop
   |
alignment
   |
recognition model
   |
embedding
```

------------------------------------------------------------------------

# 14. Step 11 --- Document Face Extraction

No OCR is needed.

Input:

``` text
passport.jpg
national_id.jpg
visa.jpg
```

Pipeline:

``` text
Document image
      |
Face detector
      |
Face region
      |
Quality check
      |
Alignment
      |
Embedding
```

Store:

``` json
{
  "document_id": "doc_001",
  "document_type": "passport",
  "face_embedding_ref": "..."
}
```

The system does not need to know the person's name or document number
for this phase.

------------------------------------------------------------------------

# 15. Step 12 --- Multi-Frame Face Matching

Do not perform:

``` text
one_live_image vs one_document
```

Instead:

``` text
               LIVE EMBEDDINGS
                    |
      +-------------+-------------+
      |             |             |
    FRONT         LEFT          RIGHT
      |             |             |
      +-------------+-------------+
                    |
                    v
              MATCH ENGINE
                    |
          +---------+---------+
          |         |         |
       Passport    ID        Visa
          |         |         |
          v         v         v
        score     score     score
```

Use cosine similarity or the distance metric appropriate to the selected
recognition model.

------------------------------------------------------------------------

# 16. Step 13 --- Cross-Angle Consistency

Measure whether the live frames themselves are mutually consistent.

``` text
front ↔ left
front ↔ right
left ↔ right
```

This helps detect unstable/poor-quality captures and prevents one
anomalous frame from dominating the decision.

------------------------------------------------------------------------

# 17. Step 14 --- Evidence Fusion

Create one structured evidence object:

``` json
{
  "session_id": "VRF-001",
  "liveness": {
    "rgb_pad": 0.96,
    "temporal": 0.94,
    "depth": null,
    "active_challenge": 1.0
  },
  "quality": {
    "front": 0.94,
    "left": 0.91,
    "right": 0.93
  },
  "face_matches": {
    "passport": 0.96,
    "national_id": 0.95,
    "visa": 0.94
  },
  "cross_angle_consistency": 0.95
}
```

Then the decision engine produces:

``` text
VERIFIED
MANUAL_REVIEW
FAILED
```

------------------------------------------------------------------------

# 18. Threshold Calibration

This is one of the most important engineering tasks.

Never start with:

``` python
if similarity > 0.7:
    verified = True
```

Instead create a validation set containing:

``` text
same-person pairs
different-person pairs
different lighting
different pose
different cameras
blur
occlusion
spoof attempts
document photo variation
```

Measure:

-   ROC
-   TAR
-   FAR
-   FRR
-   EER
-   threshold stability
-   performance by capture condition

Then choose operating thresholds according to the risk of the
application.

------------------------------------------------------------------------

# 19. Recommended Repository Structure

``` text
cert8fy-face-engine/
│
├── apps/
│   ├── web/
│   │   ├── components/
│   │   ├── camera/
│   │   ├── analysis/
│   │   ├── capture/
│   │   └── verification/
│   │
│   └── api/
│       ├── routes/
│       ├── websocket/
│       ├── services/
│       └── schemas/
│
├── services/
│   ├── detection/
│   ├── tracking/
│   ├── landmarks/
│   ├── geometry3d/
│   ├── liveness/
│   ├── quality/
│   ├── recognition/
│   ├── document_face/
│   ├── matching/
│   └── decision/
│
├── models/
│   ├── detection/
│   ├── recognition/
│   ├── liveness/
│   └── geometry/
│
├── infrastructure/
│   ├── docker/
│   ├── postgres/
│   ├── qdrant/
│   ├── redis/
│   └── monitoring/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── model/
│   └── e2e/
│
├── scripts/
│   ├── benchmark/
│   ├── calibration/
│   └── export/
│
└── docs/
```

------------------------------------------------------------------------

# 20. API Design

## Start verification

``` http
POST /api/v1/verification/sessions
```

Response:

``` json
{
  "session_id": "VRF-829341",
  "status": "WAITING_FOR_CAMERA"
}
```

## Upload document image

``` http
POST /api/v1/verification/{session_id}/documents
```

No OCR required.

## Live stream

``` text
WS /api/v1/verification/{session_id}/stream
```

## Start guided capture

``` http
POST /api/v1/verification/{session_id}/capture/start
```

## Get live analysis

``` http
GET /api/v1/verification/{session_id}/analysis
```

## Final result

``` http
GET /api/v1/verification/{session_id}/result
```

------------------------------------------------------------------------

# 21. WebSocket Event Model

### Server -\> client

``` json
{
  "type": "face_analysis",
  "track_id": 17,
  "bbox": [100, 80, 320, 420],
  "quality": 0.94,
  "liveness": 0.97,
  "yaw": -4.2,
  "pitch": 1.8,
  "roll": 0.7
}
```

### Capture event

``` json
{
  "type": "capture_progress",
  "pose": "LEFT",
  "state": "CAPTURED"
}
```

### Completion

``` json
{
  "type": "verification_state",
  "state": "ANALYZING"
}
```

------------------------------------------------------------------------

# 22. GPU Deployment

Prototype:

``` text
Python
PyTorch
ONNX Runtime
OpenCV
```

Optimized:

``` text
ONNX
  |
TensorRT
  |
NVIDIA GPU
```

Large-scale:

``` text
Camera streams
      |
DeepStream
      |
Triton
      |
+-----------+-----------+-----------+
| SCRFD     | ArcFace   | Liveness |
+-----------+-----------+-----------+
```

Do not introduce DeepStream/Triton on day one unless the project already
requires multi-stream GPU scale. Build correctness first.

------------------------------------------------------------------------

# 23. Development Order

## Phase 0 --- Environment

Install:

-   Python
-   Node.js
-   CUDA if NVIDIA GPU is available
-   Docker
-   PostgreSQL
-   Redis

------------------------------------------------------------------------

## Phase 1 --- Face Detection

Implement:

``` text
camera
  |
SCRFD
  |
bbox
```

Success:

-   stable face detection
-   no false multiple-face behavior
-   good performance

------------------------------------------------------------------------

## Phase 2 --- Face Tracking

Add ByteTrack.

Success:

-   stable track ID
-   face follows across frames
-   temporary disappearance/reappearance handled correctly

------------------------------------------------------------------------

## Phase 3 --- Face Landmarks

Add MediaPipe Face Landmarker.

Success:

-   stable landmarks
-   face pose
-   browser overlay
-   subtle technical visualization

------------------------------------------------------------------------

## Phase 4 --- 3D Geometry

Add 3DDFA-V2.

Success:

-   stable 3D alignment
-   yaw/pitch/roll
-   3D mesh visualization
-   geometry remains stable during movement

------------------------------------------------------------------------

## Phase 5 --- Liveness

Add MiniFASNet.

Then add:

-   temporal evidence
-   pose trajectory
-   challenge-response
-   geometry consistency

Only after this should the system claim a liveness result.

------------------------------------------------------------------------

## Phase 6 --- Recognition

Add the chosen ArcFace-family model.

Build:

``` text
image -> face -> alignment -> embedding
```

Then benchmark same-person vs different-person pairs.

------------------------------------------------------------------------

## Phase 7 --- Multi-Angle Capture

Implement the guided capture state machine.

``` text
FRONT
  ↓
LEFT
  ↓
RIGHT
  ↓
OPTIONAL UP
  ↓
OPTIONAL DOWN
```

Add quality-based frame selection.

------------------------------------------------------------------------

## Phase 8 --- Document Face Matching

Upload passport/ID images.

Extract only their face regions.

Generate embeddings.

Compare:

``` text
live_front ↔ passport
live_left  ↔ passport
live_right ↔ passport

live_front ↔ ID
...
```

------------------------------------------------------------------------

## Phase 9 --- Evidence Engine

Implement:

``` text
liveness
+
quality
+
multi-angle consistency
+
document matches
=
decision
```

Return:

``` text
VERIFIED
MANUAL_REVIEW
FAILED
```

------------------------------------------------------------------------

## Phase 10 --- Calibration

Build a test dataset and tune thresholds.

This phase is mandatory before claiming that the system is accurate.

------------------------------------------------------------------------

## Phase 11 --- Performance

Only now add:

-   ONNX optimization
-   TensorRT
-   GPU batching
-   DeepStream
-   Triton
-   multi-session concurrency

------------------------------------------------------------------------

# 24. Hardware Recommendation

## Minimum prototype

-   1080p webcam
-   modern CPU
-   16 GB RAM
-   NVIDIA GPU preferred

## Strong prototype

-   1080p/4K camera
-   RTX-class NVIDIA GPU
-   32 GB RAM
-   controlled lighting

## Advanced hardware

Use a camera system with:

``` text
RGB
+
IR
+
Depth
```

Examples can include depth-capable RGB-D cameras or devices with
appropriate IR/depth sensors.

If you need genuine retinal/iris acquisition, use dedicated hardware
rather than pretending a normal webcam can do it.

------------------------------------------------------------------------

# 25. Final Technical Architecture

``` text
                         CAMERA
                           |
                           v
                     VIDEO INGEST
                           |
                           v
                       SCRFD
                           |
                           v
                       TRACKER
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
         LANDMARKS      3D FACE       QUALITY
         MediaPipe      3DDFA-V2      ENGINE
             |             |             |
             +-------------+-------------+
                           |
                           v
                    LIVENESS ENSEMBLE
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
          RGB PAD       TEMPORAL      DEPTH/IR
             |             |             |
             +-------------+-------------+
                           |
                           v
                  ACTIVE CHALLENGE
                           |
                           v
                  GUIDED MULTI-ANGLE
                       CAPTURE
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
           FRONT         LEFT          RIGHT
             |             |             |
             +-------------+-------------+
                           |
                           v
                    BEST FRAME SELECT
                           |
                           v
                    FACE EMBEDDINGS
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      PASSPORT           ID CARD           VISA
      FACE EMB           FACE EMB          FACE EMB
          |                |                |
          +----------------+----------------+
                           |
                           v
                   MULTI-FRAME MATCHER
                           |
                           v
                  CROSS-ANGLE CONSISTENCY
                           |
                           v
                    EVIDENCE FUSION
                           |
                +----------+----------+
                |          |          |
                v          v          v
             VERIFIED   REVIEW     FAILED
```

------------------------------------------------------------------------

# 26. What I Would Build First

Do NOT start with the beautiful UI.

Build this exact vertical slice first:

``` text
WEBCAM
  ↓
SCRFD
  ↓
TRACKING
  ↓
MEDIAPIPE LANDMARKS
  ↓
3D FACE GEOMETRY
  ↓
MINIFASNET
  ↓
QUALITY
  ↓
GUIDED FRONT/LEFT/RIGHT CAPTURE
  ↓
ARCFAce EMBEDDINGS
  ↓
UPLOAD PASSPORT IMAGE
  ↓
EXTRACT DOCUMENT FACE
  ↓
EMBED DOCUMENT FACE
  ↓
COMPARE
  ↓
FINAL EVIDENCE RESULT
```

Once this works reliably, build the forensic-style interface around the
real pipeline.

That order prevents the project from becoming a beautiful dashboard
wrapped around a weak face-recognition demo.
