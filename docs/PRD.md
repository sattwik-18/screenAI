# Cert8fy Advanced Facial Identity Verification --- PRD

**Document:** Product Requirements Document\
**Version:** 1.0\
**Status:** Engineering Blueprint\
**Scope:** Advanced facial identity verification only --- no
OCR/document-text extraction in this phase.

------------------------------------------------------------------------

## 1. Product Vision

Build a high-end, real-time biometric verification system that
determines whether a person standing in front of a camera is a live
human and whether that live person matches the face contained in
identity documents uploaded by the user.

The system should feel like a professional biometric/forensic analysis
workstation rather than a generic "AI dashboard."

The product has five core capabilities:

1.  Real-time face detection and tracking.
2.  Continuous facial geometry, landmark, pose, quality and liveness
    analysis.
3.  Optional depth/IR-assisted 3D analysis when compatible hardware is
    available.
4.  Guided multi-angle capture of the live person's best facial frames.
5.  Face-to-document-image matching followed by an evidence-based final
    verification score.

### Important scope clarification

A normal RGB webcam cannot literally inspect the human retina. The
system may analyze the **iris/periocular region** if camera resolution
and optics are sufficient, but actual retinal imaging requires
specialized ophthalmic hardware. "3D face mapping" also should not be
treated as proof of liveness by itself; liveness must combine multiple
independent signals.

------------------------------------------------------------------------

# 2. Primary User Journey

``` text
OPEN VERIFICATION SESSION
        |
        v
CAMERA INITIALIZATION
        |
        v
FACE ACQUIRED
        |
        v
QUALITY / POSE / TRACKING
        |
        v
PASSIVE LIVENESS ANALYSIS
        |
        +---- FAIL ----> Ask user to reposition / retry
        |
        v
GUIDED ACTIVE CAPTURE
        |
        +--> FRONT
        +--> LEFT
        +--> RIGHT
        +--> OPTIONAL UP/DOWN
        |
        v
BEST-FRAME SELECTION
        |
        v
3D / LANDMARK / PERIOCULAR ANALYSIS
        |
        v
EXTRACT FACE FROM EACH UPLOADED DOCUMENT IMAGE
        |
        v
GENERATE EMBEDDINGS
        |
        v
LIVE MULTI-FRAME ↔ DOCUMENT FACE MATCHING
        |
        v
CROSS-FRAME CONSISTENCY
        |
        v
EVIDENCE FUSION
        |
        v
FINAL VERIFICATION RESULT
        |
        +--> VERIFIED
        +--> MANUAL REVIEW
        +--> FAILED
```

------------------------------------------------------------------------

# 3. What the System Must NOT Do

-   Do not rely on a single selfie.
-   Do not rely on a single frame for liveness.
-   Do not treat a face mesh overlay as proof that a person is real.
-   Do not use a simple `similarity > 0.5` threshold.
-   Do not claim "retina scan" when using an ordinary RGB camera.
-   Do not infer unnecessary demographic attributes such as age, gender
    or emotion for identity verification.
-   Do not store raw biometric images indefinitely.
-   Do not put raw face images, embeddings or biometric templates on a
    public blockchain.

------------------------------------------------------------------------

# 4. Functional Requirements

## FR-01 --- Live Camera Session

The application shall open a live camera stream and maintain a stable
tracked face.

The live interface shall show:

-   camera feed
-   face bounding region
-   subtle landmark/mesh visualization
-   tracking state
-   capture state
-   camera FPS
-   resolution
-   face quality
-   pose
-   liveness state
-   depth state when available

The interface should remain visually restrained. Technical information
should appear contextually rather than as dozens of simultaneous
floating cards.

------------------------------------------------------------------------

## FR-02 --- Face Detection

The system shall detect a face in real time and assign a temporary
session-level tracking ID.

Output:

``` json
{
  "track_id": "face_001",
  "bbox": [x, y, width, height],
  "confidence": 0.98,
  "landmarks_available": true
}
```

The detector must remain stable under:

-   moderate head rotation
-   moderate movement
-   different lighting
-   partial occlusion
-   camera distance changes

------------------------------------------------------------------------

## FR-03 --- Facial Landmarks and 3D Geometry

The system shall maintain dense facial landmarks and a 3D facial
representation.

Signals may include:

-   eye landmarks
-   eyelid geometry
-   iris/periocular landmarks where supported
-   nose landmarks
-   mouth landmarks
-   jaw contour
-   facial pose
-   facial transformation matrix
-   dense 3D face alignment

The 3D representation is used for:

-   visualization
-   pose estimation
-   geometry consistency
-   frame quality
-   liveness evidence

It is NOT a standalone liveness detector.

------------------------------------------------------------------------

## FR-04 --- Liveness / Presentation Attack Detection

The system shall determine whether the camera is observing a live person
rather than a presentation attack.

The first version shall combine:

### Passive signals

-   RGB anti-spoofing model
-   temporal consistency
-   natural facial motion
-   landmark stability
-   pose consistency
-   texture/appearance cues
-   frame-to-frame behavior

### Active signals

When passive evidence is insufficient, the system shall issue a short
randomized challenge such as:

-   turn slightly left
-   turn slightly right
-   look toward the camera
-   move closer/farther

The system must verify that the requested motion occurred in the live
tracked face.

### Optional depth/IR signals

If compatible hardware exists:

-   depth geometry
-   RGB/depth consistency
-   IR response
-   depth motion consistency

The result should be an evidence vector rather than a single magical
boolean:

``` json
{
  "rgb_pad": 0.96,
  "temporal_consistency": 0.94,
  "depth_consistency": 0.98,
  "active_challenge": 1.0,
  "liveness_decision": "PASS"
}
```

------------------------------------------------------------------------

## FR-05 --- Quality Assessment

Every candidate frame shall receive a quality score.

Measure at least:

-   face size
-   sharpness
-   motion blur
-   exposure
-   illumination
-   occlusion
-   pose
-   landmark stability
-   camera focus
-   compression quality

Frames below the configured quality floor should not be used for
identity matching.

------------------------------------------------------------------------

## FR-06 --- Guided Multi-Angle Capture

After liveness is established, the system shall automatically capture
several high-quality facial frames.

Recommended initial sequence:

``` text
1. Neutral frontal
2. Slight left
3. Slight right
4. Optional slight upward
5. Optional slight downward
```

The user should not need to manually click a shutter for every frame.

The system should wait for a high-quality frame during each pose window.

Example:

``` text
FRONT      ✓ CAPTURED
LEFT       ✓ CAPTURED
RIGHT      ✓ CAPTURED
UP         ✓ CAPTURED
DOWN       ✓ CAPTURED
```

The system should retain multiple strong frames rather than only one.

------------------------------------------------------------------------

## FR-07 --- Best Frame Selection

For each pose, rank candidate frames using a quality function.

Conceptually:

``` text
FrameScore =
    quality
  + sharpness
  + pose_alignment
  + landmark_stability
  + liveness_confidence
```

The exact weighting must be calibrated using validation data.

------------------------------------------------------------------------

## FR-08 --- Document Face Extraction

For each uploaded identity document image:

1.  detect the document face region
2.  detect the face inside the document
3.  reject unusable/too-small faces
4.  align the face
5.  generate the identity embedding
6.  associate the embedding with that specific document

Example:

``` text
passport.jpg
    |
    +--> face_01
          |
          +--> embedding

national_id.jpg
    |
    +--> face_01
          |
          +--> embedding
```

No OCR is required in this phase.

------------------------------------------------------------------------

## FR-09 --- Face Recognition

The system shall use a modern face-recognition embedding model.

For every live frame:

``` text
LIVE FRAME
    |
Face Detection
    |
Alignment
    |
Recognition Model
    |
Embedding
```

For every document face:

``` text
DOCUMENT IMAGE
    |
Face Detection
    |
Alignment
    |
Recognition Model
    |
Embedding
```

The matcher shall compare live embeddings against document embeddings.

------------------------------------------------------------------------

## FR-10 --- Multi-Frame Matching

Do not make the decision from one live image.

For each document:

``` text
Live Front Embedding  ─┐
Live Left Embedding   ─┼──> Multi-frame matcher
Live Right Embedding  ─┤
Live Up Embedding     ─┤
Live Down Embedding   ─┘
                           |
                           v
                    Document match score
```

The system should evaluate both:

-   best-match evidence
-   consistency across multiple live frames

A person should not pass merely because one poor or unusual frame
happens to produce a high similarity.

------------------------------------------------------------------------

# 5. Final Decision Model

The final result should be an evidence-fusion decision rather than a raw
face-similarity number.

Recommended evidence groups:

``` text
A. Liveness
B. Face quality
C. Live-to-document face similarity
D. Cross-angle consistency
E. Document-to-document face consistency
F. Capture/session integrity
```

Example internal result:

``` json
{
  "liveness": 0.97,
  "face_quality": 0.93,
  "passport_match": 0.96,
  "national_id_match": 0.95,
  "cross_angle_consistency": 0.94,
  "document_face_consistency": 0.97,
  "decision": "VERIFIED"
}
```

Do not expose arbitrary percentages as absolute probabilities unless the
scores have been statistically calibrated.

------------------------------------------------------------------------

# 6. Decision States

The engine shall produce exactly one of:

### VERIFIED

Strong liveness + acceptable quality + sufficiently strong identity
evidence.

### MANUAL_REVIEW

Evidence is ambiguous, conflicting, or below automatic acceptance but
not clearly fraudulent.

### FAILED

Liveness failure, insufficient quality after retries, or identity
evidence below the configured rejection threshold.

------------------------------------------------------------------------

# 7. UI Requirements

The main interface should feel like professional biometric/forensic
software.

## Main layout

``` text
+-------------------------------------------------------------+
| Session / Case                         Camera Status        |
+---------------+---------------------------------------------+
|               |                                             |
| Navigation    |                 LIVE CAMERA                 |
|               |                                             |
| Session       |           face + subtle tracking            |
| Documents     |                                             |
| Analysis      |                                             |
| History       |---------------------------------------------|
|               | Liveness | Quality | Pose | Depth | Capture |
|               |                                             |
+---------------+---------------------------------------------+
```

Avoid:

-   excessive glowing cards
-   neon gradients
-   fake holographic HUDs
-   giant "98.74%" marketing numbers
-   unnecessary demographic analysis
-   every metric being visible simultaneously

The UI should expose deeper analysis when the operator selects a subject
or analysis category.

------------------------------------------------------------------------

# 8. Verification Result

The final screen should show:

``` text
VERIFICATION COMPLETE

LIVE PERSON                 PASS
FACE IDENTITY               MATCH
PASSPORT FACE               MATCH
NATIONAL ID FACE            MATCH
MULTI-ANGLE CONSISTENCY     PASS

STATUS                      VERIFIED

Verification ID: VRF-XXXXXX
```

Also provide an expandable technical evidence view.

------------------------------------------------------------------------

# 9. Data Requirements

The system shall maintain:

### Verification session

-   session ID
-   creation time
-   camera metadata
-   model versions
-   session state

### Live capture

-   selected frames
-   frame timestamps
-   pose metadata
-   quality metadata
-   liveness evidence

### Document face records

-   document ID
-   face crop reference
-   embedding reference
-   quality metadata

### Verification evidence

-   individual model outputs
-   match scores
-   decision
-   decision-policy version
-   model versions

Raw biometric data must have explicit retention controls.

------------------------------------------------------------------------

# 10. Non-Functional Requirements

## Performance

Initial target:

-   camera preview: 30 FPS
-   face tracking: near real time
-   liveness feedback: \<150 ms perceived latency where hardware allows
-   identity embedding: \<100 ms per selected frame on a modern GPU
-   final verification: ideally \<3 seconds after capture completion

These are engineering targets, not guaranteed hardware-independent
numbers.

## Reliability

The system should gracefully handle:

-   camera disconnect
-   low light
-   no face
-   multiple faces
-   face leaving frame
-   network interruption
-   GPU failure
-   invalid document image
-   insufficient face size

------------------------------------------------------------------------

# 11. Security / Privacy Requirements

-   Encrypt biometric data at rest.
-   Use TLS in transit.
-   Enforce authentication and role-based authorization.
-   Log access to biometric records.
-   Minimize raw image retention.
-   Separate identity metadata from biometric templates where practical.
-   Never expose biometric embeddings through client APIs unnecessarily.
-   Do not place biometric images or embeddings on-chain.
-   Provide deletion/retention controls.
-   Require appropriate consent and authorization for biometric
    processing.

------------------------------------------------------------------------

# 12. Success Criteria

The first production-quality prototype is successful when:

1.  A user can open the live camera.
2.  The system tracks a face continuously.
3.  Dense landmarks/3D geometry are rendered in real time.
4.  Liveness combines multiple signals.
5.  The user completes a guided multi-angle capture.
6.  The system automatically selects high-quality frames.
7.  Faces are extracted from uploaded passport/ID images.
8.  Live embeddings are compared with document embeddings.
9.  Multiple frames contribute to the identity decision.
10. The final result contains an explainable evidence breakdown.
11. The system can return VERIFIED / MANUAL REVIEW / FAILED.
12. The UI looks like professional operational software rather than an
    AI-generated dashboard.

------------------------------------------------------------------------

# 13. Out of Scope for V1

-   OCR
-   MRZ extraction
-   document authenticity verification
-   name/DOB consistency
-   passport chip/NFC verification
-   government database verification
-   voice biometrics
-   fingerprint biometrics
-   retina imaging
-   emotion recognition
-   demographic profiling

These can be added later without changing the core facial pipeline.
