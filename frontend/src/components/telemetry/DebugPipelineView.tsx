import React, { useRef, useEffect } from 'react';
import { X } from 'lucide-react';
import { TelemetryData } from '../camera/OverlayRenderer';

interface DebugPipelineViewProps {
  telemetry: TelemetryData | null;
  isOpen: boolean;
  onClose: () => void;
}

export const DebugPipelineView: React.FC<DebugPipelineViewProps> = ({
  telemetry,
  isOpen,
  onClose
}) => {
  const cropCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const alignedCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const localLmCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mesh3dCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const debug = telemetry?.debug_pipeline;
  const isGeomValid = telemetry?.geometry_valid !== false && telemetry?.geometry_status !== 'GEOMETRY_INVALID';

  // Render Stage 2 (Face Crop), Stage 3 (Aligned Face), & Stage 4 (Local Landmarks)
  useEffect(() => {
    if (!isOpen) return;

    if (debug?.crop_base64) {
      const img = new Image();
      img.onload = () => {
        // 1. Draw raw crop on Stage 2 canvas
        if (cropCanvasRef.current) {
          const ctx = cropCanvasRef.current.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, 180, 180);
            ctx.drawImage(img, 0, 0, 180, 180);
          }
        }

        // 2. Draw crop + local landmarks on Stage 4 canvas
        if (localLmCanvasRef.current) {
          const ctx = localLmCanvasRef.current.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, 180, 180);
            ctx.drawImage(img, 0, 0, 180, 180);

            const lms = debug.local_landmarks || [];
            if (isGeomValid) {
              ctx.fillStyle = '#00D8F6';
              for (let i = 0; i < lms.length; i++) {
                const [lx, ly] = lms[i];
                ctx.beginPath();
                ctx.arc(lx, ly, 1.8, 0, 2 * Math.PI);
                ctx.fill();
              }

              // Highlight eye / iris centers (MediaPipe irises: 468 & 473)
              if (lms.length > 473) {
                ctx.fillStyle = '#10B981';
                ctx.strokeStyle = '#10B981';
                ctx.lineWidth = 1.5;
                [468, 473].forEach((idx) => {
                  if (lms[idx]) {
                    const [ix, iy] = lms[idx];
                    ctx.beginPath();
                    ctx.arc(ix, iy, 4, 0, 2 * Math.PI);
                    ctx.stroke();
                  }
                });
              }
            } else {
              // Draw invalid geometry watermark
              ctx.fillStyle = 'rgba(239, 68, 68, 0.4)';
              ctx.fillRect(0, 0, 180, 180);
              ctx.fillStyle = '#EF4444';
              ctx.font = '10px "JetBrains Mono", monospace';
              ctx.fillText('GEOMETRY INVALID', 24, 90);
            }
          }
        }
      };
      img.src = debug.crop_base64;
    }

    // 3. Draw ArcFace 112x112 canonical aligned face
    if (debug?.aligned_face_b64 && alignedCanvasRef.current) {
      const aImg = new Image();
      aImg.onload = () => {
        const ctx = alignedCanvasRef.current?.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, 180, 180);
          ctx.drawImage(aImg, 0, 0, 180, 180);
        }
      };
      aImg.src = debug.aligned_face_b64;
    }
  }, [isOpen, debug, isGeomValid]);

  // Render Stage 5 (3D Mesh Wireframe in local coordinate space)
  useEffect(() => {
    if (!isOpen || !mesh3dCanvasRef.current) return;
    const canvas = mesh3dCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, 180, 180);
    ctx.fillStyle = '#090C12';
    ctx.fillRect(0, 0, 180, 180);

    // Subtle coordinate grid
    ctx.strokeStyle = '#17202E';
    ctx.lineWidth = 0.5;
    for (let p = 20; p < 180; p += 30) {
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, 180);
      ctx.moveTo(0, p);
      ctx.lineTo(180, p);
      ctx.stroke();
    }

    const lms = debug?.local_landmarks || [];
    if (isGeomValid && lms.length > 0) {
      ctx.fillStyle = '#00D8F6';
      ctx.strokeStyle = 'rgba(0, 216, 246, 0.4)';
      ctx.lineWidth = 0.8;

      // Draw wireframe dots
      for (const [lx, ly] of lms) {
        ctx.fillRect(lx - 1, ly - 1, 2, 2);
      }

      // Draw pose compass overlay
      if (telemetry?.pose) {
        ctx.fillStyle = '#E6EDF5';
        ctx.font = '8px "JetBrains Mono", monospace';
        ctx.fillText(`Y: ${telemetry.pose.yaw_deg}°`, 8, 160);
        ctx.fillText(`P: ${telemetry.pose.pitch_deg}°`, 65, 160);
        ctx.fillText(`R: ${telemetry.pose.roll_deg}°`, 122, 160);
      }
    } else {
      ctx.fillStyle = isGeomValid ? '#54657E' : '#EF4444';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(isGeomValid ? 'NO 3D MESH DATA' : 'MESH SUPPRESSED (INVALID)', 15, 95);
    }
  }, [isOpen, debug, telemetry?.pose, isGeomValid]);

  if (!isOpen) return null;

  const bbox = telemetry?.bbox || [0, 0, 0, 0];
  const conf = telemetry?.confidence ?? 0;

  return (
    <div className="debug-modal-backdrop" onClick={onClose}>
      <div className="debug-modal-container" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '1240px', width: '96vw' }}>
        <div className="debug-modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#00D8F6', letterSpacing: '0.08em' }}>
              CV PIPELINE REAL-TIME DIAGNOSTIC VIEW
            </span>
            <span style={{ fontSize: '10px', color: '#54657E' }}>
              SCRFD BBOX → FACE CROP → ALIGNED FACE → LANDMARKS → 3D MESH → LIVE OVERLAY
            </span>
          </div>

          <button
            type="button"
            onClick={onClose}
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-medium)',
              color: '#8A99AD',
              padding: '4px 8px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '10px',
              fontFamily: 'var(--font-mono)',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <span style={{ background: 'rgba(0,0,0,0.5)', padding: '1px 4px', borderRadius: '2px', border: '1px solid var(--border-subtle)' }}>ESC</span>
            <X size={13} />
          </button>
        </div>

        {/* 6-Stage Side-by-Side Pipeline Strip */}
        <div className="debug-pipeline-grid" style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
          {/* Stage 1: SCRFD Bounding Box */}
          <div className="debug-stage-card">
            <div className="debug-stage-header">
              <span className="stage-step">STAGE 1</span>
              <span className="stage-title">SCRFD BBOX</span>
            </div>
            <div className="debug-stage-content" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '180px', background: '#090C12' }}>
              <div style={{ border: '2px dashed #00D8F6', padding: '14px 16px', borderRadius: '4px', textAlign: 'center' }}>
                <div style={{ color: '#00D8F6', fontSize: '11px', fontWeight: 600, fontFamily: '"JetBrains Mono", monospace' }}>
                  SCRFD-500M
                </div>
                <div style={{ color: '#E6EDF5', fontSize: '10px', marginTop: '6px', fontFamily: '"JetBrains Mono", monospace' }}>
                  [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]
                </div>
                <div style={{ color: '#10B981', fontSize: '10px', marginTop: '4px', fontFamily: '"JetBrains Mono", monospace' }}>
                  CONF: {(conf * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ fontSize: '9px', color: '#54657E', marginTop: '8px' }}>
                Single-Subject Anchor
              </div>
            </div>
            <div className="debug-stage-footer">
              Status: {telemetry?.detection_status || 'SEARCHING'}
            </div>
          </div>

          {/* Stage 2: Face Crop */}
          <div className="debug-stage-card">
            <div className="debug-stage-header">
              <span className="stage-step">STAGE 2</span>
              <span className="stage-title">FACE CROP (+25%)</span>
            </div>
            <div className="debug-stage-content" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#090C12' }}>
              <canvas ref={cropCanvasRef} width={180} height={180} style={{ width: '180px', height: '180px', objectFit: 'contain' }} />
            </div>
            <div className="debug-stage-footer">
              Crop: [{debug?.crop_box ? debug.crop_box.join(', ') : '--'}]
            </div>
          </div>

          {/* Stage 3: Aligned Face (112x112 Canonical ArcFace) */}
          <div className="debug-stage-card">
            <div className="debug-stage-header">
              <span className="stage-step">STAGE 3</span>
              <span className="stage-title">ALIGNED 112x112</span>
            </div>
            <div className="debug-stage-content" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#090C12' }}>
              <canvas ref={alignedCanvasRef} width={180} height={180} style={{ width: '180px', height: '180px', objectFit: 'contain' }} />
            </div>
            <div className="debug-stage-footer">
              ArcFace Canonical Crop
            </div>
          </div>

          {/* Stage 4: Local Landmarks */}
          <div className="debug-stage-card">
            <div className="debug-stage-header">
              <span className="stage-step">STAGE 4</span>
              <span className="stage-title">LOCAL LANDMARKS</span>
            </div>
            <div className="debug-stage-content" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#090C12' }}>
              <canvas ref={localLmCanvasRef} width={180} height={180} style={{ width: '180px', height: '180px', objectFit: 'contain' }} />
            </div>
            <div className="debug-stage-footer" style={{ color: isGeomValid ? '#10B981' : '#EF4444' }}>
              {isGeomValid ? `Valid (${debug?.local_landmarks?.length || 0} pts)` : `GEOM INVALID`}
            </div>
          </div>

          {/* Stage 5: 3D Wireframe Mesh */}
          <div className="debug-stage-card">
            <div className="debug-stage-header">
              <span className="stage-step">STAGE 5</span>
              <span className="stage-title">3D MESH & POSE</span>
            </div>
            <div className="debug-stage-content" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#090C12' }}>
              <canvas ref={mesh3dCanvasRef} width={180} height={180} style={{ width: '180px', height: '180px', objectFit: 'contain' }} />
            </div>
            <div className="debug-stage-footer">
              Pose: {telemetry?.pose?.orientation || 'CENTER'}
            </div>
          </div>

          {/* Stage 6: Live Composite Overlay */}
          <div className="debug-stage-card">
            <div className="debug-stage-header">
              <span className="stage-step">STAGE 6</span>
              <span className="stage-title">LIVE COMPOSITE</span>
            </div>
            <div className="debug-stage-content" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '12px', height: '180px', background: '#090C12' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ color: '#54657E', fontSize: '9px' }}>ENGINE</span>
                <span style={{ color: '#00D8F6', fontWeight: 600, fontSize: '9px' }}>
                  MediaPipe 478
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ color: '#54657E', fontSize: '9px' }}>LATENCY</span>
                <span style={{ color: '#10B981', fontWeight: 600, fontSize: '9px' }}>
                  {telemetry?.latency_ms ? `${telemetry.latency_ms} ms` : '--'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ color: '#54657E', fontSize: '9px' }}>GEOMETRY</span>
                <span style={{ color: isGeomValid ? '#10B981' : '#EF4444', fontWeight: 600, fontSize: '9px' }}>
                  {isGeomValid ? 'VALID' : 'INVALID'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ color: '#54657E', fontSize: '9px' }}>EAR (L/R)</span>
                <span style={{ color: '#00D8F6', fontSize: '9px' }}>
                  {telemetry?.periocular?.ear_left ?? '--'} / {telemetry?.periocular?.ear_right ?? '--'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#54657E', fontSize: '9px' }}>TRACK ID</span>
                <span style={{ color: '#E6EDF5', fontSize: '9px' }}>
                  {telemetry?.track_id || 'NONE'}
                </span>
              </div>
            </div>
            <div className="debug-stage-footer">
              Status: <span style={{ color: isGeomValid ? '#10B981' : '#EF4444', fontWeight: 600 }}>{isGeomValid ? 'LOCKED' : 'UNANCHORED'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
