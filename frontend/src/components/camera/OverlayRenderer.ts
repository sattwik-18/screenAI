/**
 * Forensic HUD Canvas Overlay Renderer.
 * Draws real model-derived telemetry:
 * - Tracking bounding box with corner brackets & Track ID
 * - Subtle facial landmarks & contour
 * - Optional monocular 3D wireframe mesh
 * - Periocular / Eye targeting crosshairs
 * - 3D head pose orientation compass
 */

export interface TelemetryData {
  detection_status?: string;
  track_id?: string;
  bbox?: [number, number, number, number]; // [x, y, w, h]
  confidence?: number;
  geometry_valid?: boolean;
  geometry_status?: string;
  geometry_reasons?: string[];
  pose?: {
    pitch_deg: number;
    yaw_deg: number;
    roll_deg: number;
    orientation: string;
    mesh_vertices?: number[][];
  };
  quality?: {
    acceptable: boolean;
    sharpness: number;
    composite_score: number;
  };
  liveness?: {
    is_live: boolean;
    liveness_score: number;
  };
  periocular?: {
    left_crop?: [number, number, number, number];
    right_crop?: [number, number, number, number];
    left_b64?: string;
    right_b64?: string;
    ear_left?: number;
    ear_right?: number;
  };
  latency_ms?: number;
  fps?: number;
  instruction?: string;
  frame_width?: number;
  frame_height?: number;
  landmark_engine?: string;
  landmarks_478?: number[][];
  debug_pipeline?: {
    crop_base64?: string;
    aligned_face_b64?: string;
    local_landmarks?: number[][];
    crop_box?: number[];
    crop_size?: number;
    geometry_valid?: boolean;
    geometry_status?: string;
    geometry_reasons?: string[];
  };
}

// Standard DLib 68-point facial landmark edge connectivity.
// Connects only anatomically adjacent landmarks — avoids the garbage
// lines produced by naive sequential i→i+1 connections.
const FACE_EDGES_68: [number, number][] = [
  // Jawline
  [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
  // Right eyebrow
  [17,18],[18,19],[19,20],[20,21],
  // Left eyebrow
  [22,23],[23,24],[24,25],[25,26],
  // Nose bridge
  [27,28],[28,29],[29,30],
  // Nose bottom
  [30,31],[31,32],[32,33],[33,34],[34,35],
  // Right eye
  [36,37],[37,38],[38,39],[39,40],[40,41],[41,36],
  // Left eye
  [42,43],[43,44],[44,45],[45,46],[46,47],[47,42],
  // Outer lip
  [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,48],
  // Inner lip
  [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60],
];

export class OverlayRenderer {
  public showMesh: boolean = true;
  public showLandmarks: boolean = true;
  public showPeriocular: boolean = true;

  // Real-time smoothed state (runs at 60 FPS in requestAnimationFrame)
  private _currentBbox: [number, number, number, number] | null = null;
  private _targetBbox: [number, number, number, number] | null = null;
  private _rawBbox: [number, number, number, number] | null = null;

  // Velocity tracking for predictive latency compensation
  private _prevCentroid: [number, number] | null = null;
  private _lastPacketTime: number = 0;
  private _velocity: [number, number] = [0, 0];

  public render(
    ctx: CanvasRenderingContext2D,
    canvasWidth: number,
    canvasHeight: number,
    videoWidth: number,
    videoHeight: number,
    telemetry: TelemetryData | null
  ) {
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    if (!telemetry || !telemetry.bbox || telemetry.detection_status !== 'EXACTLY_ONE') {
      // Reset tracking smoother when face is lost
      this._currentBbox = null;
      this._targetBbox = null;
      this._rawBbox = null;
      this._prevCentroid = null;
      this._velocity = [0, 0];
      this.drawAcquisitionReticle(ctx, canvasWidth, canvasHeight, telemetry?.detection_status);
      return;
    }

    // Mathematical coordinate transform accounting for object-fit: cover letterbox/crop offsets
    const inW = videoWidth || canvasWidth || 1;
    const inH = videoHeight || canvasHeight || 1;
    const scale = Math.max(canvasWidth / inW, canvasHeight / inH);
    const renderedW = inW * scale;
    const renderedH = inH * scale;
    const offsetX = (canvasWidth - renderedW) / 2;
    const offsetY = (canvasHeight - renderedH) / 2;

    const incoming = telemetry.bbox;
    const now = performance.now();

    // ── 1. Update Target & Compute Predictive Forward Lead ──────────────────
    if (
      this._rawBbox === null ||
      incoming[0] !== this._rawBbox[0] ||
      incoming[1] !== this._rawBbox[1] ||
      incoming[2] !== this._rawBbox[2] ||
      incoming[3] !== this._rawBbox[3]
    ) {
      const cx = incoming[0] + incoming[2] / 2;
      const cy = incoming[1] + incoming[3] / 2;

      if (this._prevCentroid && this._lastPacketTime > 0) {
        const dt = Math.max(0.016, (now - this._lastPacketTime) / 1000.0);
        const instVx = (cx - this._prevCentroid[0]) / dt;
        const instVy = (cy - this._prevCentroid[1]) / dt;
        // EMA filter on velocity to reject measurement noise
        this._velocity[0] = 0.60 * this._velocity[0] + 0.40 * instVx;
        this._velocity[1] = 0.60 * this._velocity[1] + 0.40 * instVy;
      }
      this._prevCentroid = [cx, cy];
      this._lastPacketTime = now;
      this._rawBbox = incoming;

      // Latency compensation: project target ahead by the measured roundtrip latency (~30-65ms)
      // Clamped to 75ms to guarantee stability and prevent overshooting
      const leadTimeSec = Math.min(0.075, Math.max(0.020, (telemetry.latency_ms || 40) / 1000.0));
      const leadX = this._velocity[0] * leadTimeSec;
      const leadY = this._velocity[1] * leadTimeSec;

      this._targetBbox = [
        incoming[0] + leadX,
        incoming[1] + leadY,
        incoming[2],
        incoming[3],
      ];

      if (!this._currentBbox) {
        this._currentBbox = [...this._targetBbox];
      }
    }

    // ── 2. Adaptive 60 FPS Convergence ─────────────────────────────────────
    if (!this._currentBbox || !this._targetBbox) {
      this._currentBbox = [...incoming];
      this._targetBbox = [...incoming];
    }

    const cur = this._currentBbox;
    const tgt = this._targetBbox;
    const dist = Math.hypot(tgt[0] - cur[0], tgt[1] - cur[1]);

    // Adaptive alpha:
    // Still face (dist < 0.8px): alpha = 0.18 -> deadband filters out 100% of micro-jitter.
    // Dynamic movement (dist > 15px): alpha = 0.75 -> instantaneous snap, zero lag behind video.
    let alpha = 0.38;
    if (dist < 0.8) {
      alpha = 0.18;
    } else if (dist > 15) {
      alpha = 0.75;
    }

    cur[0] += (tgt[0] - cur[0]) * alpha;
    cur[1] += (tgt[1] - cur[1]) * alpha;
    cur[2] += (tgt[2] - cur[2]) * alpha;
    cur[3] += (tgt[3] - cur[3]) * alpha;

    // ── 3. Unified Synchronized Displacement ─────────────────────────────────
    // shiftX and shiftY allow landmarks, 3D mesh, and eyes to glide in 100% lockstep
    // with the smoothed and predicted face position at 60 FPS without separating.
    const rawBox = this._rawBbox || incoming;
    const shiftX = cur[0] - rawBox[0];
    const shiftY = cur[1] - rawBox[1];

    const [bx, by, bw, bh] = cur;
    const x = offsetX + bx * scale;
    const y = offsetY + by * scale;
    const w = bw * scale;
    const h = bh * scale;

    const isGeometryValid = telemetry.geometry_valid !== false && telemetry.geometry_status !== 'GEOMETRY_INVALID';

    // 1. Draw Tracking Bounding Bracket
    this.drawTrackingBox(ctx, x, y, w, h, telemetry, isGeometryValid);

    // 2. Draw Periocular Targets (Eyes) — strictly synchronized with face motion
    if (this.showPeriocular && isGeometryValid && telemetry.periocular) {
      this.drawPeriocularTargets(ctx, telemetry.periocular, scale, offsetX, offsetY, shiftX, shiftY);
    }

    // 3. Draw 3D Wireframe Mesh — strictly synchronized with face motion
    const vertices = telemetry.landmarks_478 || telemetry.pose?.mesh_vertices;
    if (this.showMesh && isGeometryValid && vertices && vertices.length > 0) {
      this.drawMesh(ctx, vertices, scale, offsetX, offsetY, shiftX, shiftY);
    }

    // 4. Draw 3D Pose Compass
    if (telemetry.pose) {
      this.drawPoseCompass(ctx, x + w + 12, y + 8, telemetry.pose);
    }
  }

  private drawTrackingBox(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    telemetry: TelemetryData,
    isGeometryValid: boolean = true
  ) {
    const isLive = telemetry.liveness?.is_live ?? false;
    let color = isLive ? "#10B981" : "#00D8F6";
    if (!isGeometryValid) {
      color = "#EF4444"; // Red border on invalid geometry
    }
    const bracketLen = Math.min(24, w * 0.2);

    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(x, y + bracketLen);
    ctx.lineTo(x, y);
    ctx.lineTo(x + bracketLen, y);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(x + w - bracketLen, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + bracketLen);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(x, y + h - bracketLen);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + bracketLen, y + h);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(x + w - bracketLen, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - bracketLen);
    ctx.stroke();

    // Tag header
    const confPct = Math.round((telemetry.confidence || 0) * 100);
    let modeLabel = "DET";
    if (!isGeometryValid) {
      modeLabel = "GEOM INVALID";
    } else if (Boolean(telemetry.landmarks_478 && telemetry.landmarks_478.length >= 468)) {
      modeLabel = "LOCK";
    }
    const tagWidth = !isGeometryValid ? 200 : 175;

    ctx.fillStyle = "rgba(6, 8, 13, 0.90)";
    ctx.fillRect(x, y - 22, tagWidth, 20);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y - 22, tagWidth, 20);

    ctx.fillStyle = color;
    ctx.font = "11px 'JetBrains Mono', monospace";
    const trackText = `${telemetry.track_id || "TRACK_001"} | ${confPct}% ${modeLabel}`;
    ctx.fillText(trackText, x + 6, y - 8);

    ctx.restore();
  }

  private drawPeriocularTargets(
    ctx: CanvasRenderingContext2D,
    periocular: NonNullable<TelemetryData["periocular"]>,
    scale: number,
    offsetX: number,
    offsetY: number,
    shiftX: number = 0,
    shiftY: number = 0
  ) {
    const drawEyeBox = (crop: [number, number, number, number], label: string) => {
      const ex = offsetX + (crop[0] + shiftX) * scale;
      const ey = offsetY + (crop[1] + shiftY) * scale;
      const ew = crop[2] * scale;
      const eh = crop[3] * scale;

      ctx.save();
      ctx.strokeStyle = "rgba(0, 216, 246, 0.45)";
      ctx.lineWidth = 1;
      ctx.strokeRect(ex, ey, ew, eh);

      // Center crosshair
      const cx = ex + ew / 2;
      const cy = ey + eh / 2;
      ctx.beginPath();
      ctx.moveTo(cx - 5, cy);
      ctx.lineTo(cx + 5, cy);
      ctx.moveTo(cx, cy - 5);
      ctx.lineTo(cx, cy + 5);
      ctx.stroke();

      ctx.fillStyle = "rgba(0, 216, 246, 0.85)";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillText(label, ex + 2, ey - 3);

      ctx.restore();
    };

    if (periocular.left_crop) drawEyeBox(periocular.left_crop, "EYE_L");
    if (periocular.right_crop) drawEyeBox(periocular.right_crop, "EYE_R");
  }

  private drawMesh(
    ctx: CanvasRenderingContext2D,
    vertices: number[][],
    scale: number,
    offsetX: number,
    offsetY: number,
    shiftX: number = 0,
    shiftY: number = 0
  ) {
    if (vertices.length < 68) return;
    ctx.save();

    const isDense478 = vertices.length >= 468;

    if (isDense478) {
      // 1. Draw 478 dense landmark points
      ctx.fillStyle = 'rgba(0, 216, 246, 0.45)';
      for (let i = 0; i < vertices.length; i++) {
        const px = offsetX + (vertices[i][0] + shiftX) * scale;
        const py = offsetY + (vertices[i][1] + shiftY) * scale;
        ctx.fillRect(px - 1, py - 1, 2, 2);
      }

      // 2. Draw prominent Iris Targeting Reticles (468 = Right Iris, 473 = Left Iris)
      [468, 473].forEach((irisIdx) => {
        if (irisIdx < vertices.length) {
          const ix = offsetX + (vertices[irisIdx][0] + shiftX) * scale;
          const iy = offsetY + (vertices[irisIdx][1] + shiftY) * scale;
          ctx.strokeStyle = '#10B981';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(ix, iy, 4, 0, 2 * Math.PI);
          ctx.stroke();

          ctx.fillStyle = '#10B981';
          ctx.beginPath();
          ctx.arc(ix, iy, 1.5, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    } else {
      // Legacy 68-point sparse landmark dots
      ctx.fillStyle = 'rgba(245, 158, 11, 0.65)';
      for (let i = 0; i < Math.min(68, vertices.length); i++) {
        const px = offsetX + (vertices[i][0] + shiftX) * scale;
        const py = offsetY + (vertices[i][1] + shiftY) * scale;
        ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
      }

      // Draw 68-point anatomical edges
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
      ctx.lineWidth = 0.8;
      for (const [a, b] of FACE_EDGES_68) {
        if (a >= vertices.length || b >= vertices.length) continue;
        ctx.beginPath();
        ctx.moveTo(offsetX + (vertices[a][0] + shiftX) * scale, offsetY + (vertices[a][1] + shiftY) * scale);
        ctx.lineTo(offsetX + (vertices[b][0] + shiftX) * scale, offsetY + (vertices[b][1] + shiftY) * scale);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  private drawPoseCompass(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    pose: NonNullable<TelemetryData["pose"]>
  ) {
    ctx.save();
    ctx.fillStyle = "rgba(9, 12, 18, 0.90)";
    ctx.strokeStyle = "#17202E";
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, 125, 75);
    ctx.strokeRect(x, y, 125, 75);

    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#54657E";
    ctx.fillText("HEAD POSE (DEG)", x + 8, y + 14);

    ctx.fillStyle = "#E6EDF5";
    ctx.fillText(`YAW   : ${pose.yaw_deg > 0 ? "+" : ""}${pose.yaw_deg}°`, x + 8, y + 32);
    ctx.fillText(`PITCH : ${pose.pitch_deg > 0 ? "+" : ""}${pose.pitch_deg}°`, x + 8, y + 48);
    ctx.fillText(`ROLL  : ${pose.roll_deg > 0 ? "+" : ""}${pose.roll_deg}°`, x + 8, y + 64);

    ctx.restore();
  }

  private drawAcquisitionReticle(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    status?: string
  ) {
    ctx.save();
    ctx.strokeStyle = status === "MULTIPLE_SUBJECTS" ? "rgba(239, 68, 68, 0.7)" : "rgba(84, 101, 126, 0.35)";
    ctx.lineWidth = 1.5;

    const cx = w / 2;
    const cy = h / 2;
    const rw = Math.min(240, w * 0.5);
    const rh = Math.min(300, h * 0.6);
    const bLen = 20;

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(cx - rw / 2, cy - rh / 2 + bLen);
    ctx.lineTo(cx - rw / 2, cy - rh / 2);
    ctx.lineTo(cx - rw / 2 + bLen, cy - rh / 2);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(cx + rw / 2 - bLen, cy - rh / 2);
    ctx.lineTo(cx + rw / 2, cy - rh / 2);
    ctx.lineTo(cx + rw / 2, cy - rh / 2 + bLen);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(cx - rw / 2, cy + rh / 2 - bLen);
    ctx.lineTo(cx - rw / 2, cy + rh / 2);
    ctx.lineTo(cx - rw / 2 + bLen, cy + rh / 2);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(cx + rw / 2 - bLen, cy + rh / 2);
    ctx.lineTo(cx + rw / 2, cy + rh / 2);
    ctx.lineTo(cx + rw / 2, cy + rh / 2 - bLen);
    ctx.stroke();

    ctx.fillStyle = status === "MULTIPLE_SUBJECTS" ? "#EF4444" : "#54657E";
    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(
      status === "MULTIPLE_SUBJECTS" ? "MULTIPLE SUBJECTS DETECTED" : "AWAITING BIOMETRIC ACQUISITION",
      cx,
      cy + rh / 2 + 24
    );

    ctx.restore();
  }
}
