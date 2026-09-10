import React, { useRef, useEffect } from 'react';
import { TelemetryData } from './OverlayRenderer';

interface PeriocularCardsProps {
  telemetry: TelemetryData | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export const PeriocularCards: React.FC<PeriocularCardsProps> = ({ telemetry, videoRef }) => {
  const leftEyeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rightEyeCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!telemetry?.periocular) return;
    const peri = telemetry.periocular;

    const drawCrop = (
      canvas: HTMLCanvasElement | null,
      b64: string | undefined,
      label: string
    ) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = 120;
      canvas.height = 90;
      ctx.clearRect(0, 0, 120, 90);

      const renderReticle = () => {
        // Technical targeting crosshairs
        ctx.strokeStyle = 'rgba(0, 216, 246, 0.7)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(60, 45, 18, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(60, 20);
        ctx.lineTo(60, 70);
        ctx.moveTo(35, 45);
        ctx.lineTo(85, 45);
        ctx.stroke();

        ctx.fillStyle = '#00D8F6';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillText(label, 6, 12);
      };

      if (b64) {
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0, 120, 90);
          renderReticle();
        };
        img.src = `data:image/jpeg;base64,${b64}`;
      } else {
        renderReticle();
      }
    };

    drawCrop(leftEyeCanvasRef.current, peri.left_b64, 'EYE_L');
    drawCrop(rightEyeCanvasRef.current, peri.right_b64, 'EYE_R');
  }, [telemetry]);

  const earL = telemetry?.periocular?.ear_left;
  const earR = telemetry?.periocular?.ear_right;
  const hasPeriocular = Boolean(
    (telemetry?.periocular?.left_b64 && telemetry?.periocular?.right_b64) ||
    (telemetry?.periocular?.left_crop && telemetry?.periocular?.right_crop)
  );

  return (
    <div className="periocular-card">
      <div className="periocular-header">
        <span>PERIOCULAR GEOMETRY</span>
        <span style={{ color: '#00D8F6' }}>RGB SENSOR</span>
      </div>

      {hasPeriocular ? (
        <div className="periocular-grid">
          <div className="periocular-box">
            <canvas ref={leftEyeCanvasRef} className="periocular-canvas" />
            <div className="periocular-label">
              <span>EAR_L</span>
              <span style={{ color: '#00D8F6' }}>{earL !== undefined ? earL.toFixed(2) : '--'}</span>
            </div>
          </div>

          <div className="periocular-box">
            <canvas ref={rightEyeCanvasRef} className="periocular-canvas" />
            <div className="periocular-label">
              <span>EAR_R</span>
              <span style={{ color: '#00D8F6' }}>{earR !== undefined ? earR.toFixed(2) : '--'}</span>
            </div>
          </div>
        </div>
      ) : (
        <div style={{ padding: '12px 6px', textAlign: 'center', color: '#54657E', fontSize: '9px' }}>
          PERIOCULAR DATA UNAVAILABLE
        </div>
      )}

      <div className="periocular-footer">
        Optical Periocular Tracking • Not Retinal Imaging
      </div>
    </div>
  );
};
