import React, { useState, useEffect, useRef } from 'react';
import { CheckCircle2, Loader2, Circle, ShieldCheck } from 'lucide-react';

interface ForensicEvaluationScannerProps {
  /** Live face captures: { FRONT?: string, LEFT?: string, RIGHT?: string } (base64 data-urls) */
  liveCaptures: Record<string, string | undefined>;
  /** Document portraits: array of { doc_id, doc_type, face_crop_url } */
  documents: Array<{ doc_id: string; doc_type: string; face_crop_url?: string }>;
  /** Called when the scanner finishes and the dossier should be shown */
  onComplete: () => void;
}

const STAGES = [
  {
    id: 1,
    label: 'STAGE 1: AFFINE NORMALIZATION & 106-PT FIDUCIAL MAPPING',
    duration: 600,
    detail: 'Aligning canonical 112×112 crop via 5-point affine transform…',
  },
  {
    id: 2,
    label: 'STAGE 2: DUAL-PASS 512-D ARCFACE TTA EMBEDDING EXTRACTION',
    duration: 800,
    detail: 'Extracting L2-normalized identity vectors + horizontal flip fusion…',
  },
  {
    id: 3,
    label: 'STAGE 3: CROSS-POSE PHOTOMETRIC & PERIOCULAR CORRELATION',
    duration: 700,
    detail: 'Correlating Front (80%) × Left (10%) × Right (10%) pose evidence…',
  },
  {
    id: 4,
    label: 'STAGE 4: ISO/IEC 19795 FORENSIC EVIDENCE FUSION',
    duration: 600,
    detail: 'Quality-weighted document fusion and decision policy evaluation…',
  },
];

const TOTAL_DURATION = STAGES.reduce((s, st) => s + st.duration, 0);

export const ForensicEvaluationScanner: React.FC<ForensicEvaluationScannerProps> = ({
  liveCaptures,
  documents,
  onComplete,
}) => {
  const [completedStages, setCompletedStages] = useState<number[]>([]);
  const [activeStage, setActiveStage] = useState<number>(0); // 1-indexed, 0 = none
  const [progress, setProgress] = useState(0);
  const [scanLine, setScanLine] = useState(0);
  const [done, setDone] = useState(false);
  const progressRef = useRef(0);
  const startTimeRef = useRef<number>(Date.now());
  const rafRef = useRef<number>(0);

  // Pick best live capture thumbnail
  const liveThumbnail =
    liveCaptures['FRONT'] ?? liveCaptures['LEFT'] ?? liveCaptures['RIGHT'] ?? null;
  const docThumbnail = documents[0]?.face_crop_url ?? null;

  // Animated progress bar + scan line
  useEffect(() => {
    startTimeRef.current = Date.now();

    const tick = () => {
      const elapsed = Date.now() - startTimeRef.current;
      const pct = Math.min(1, elapsed / TOTAL_DURATION);
      progressRef.current = pct;
      setProgress(pct);
      setScanLine((pct * 100) % 100);

      // Stage completion logic
      let cumulativeMs = 0;
      const newCompleted: number[] = [];
      let currentActive = 0;
      for (const st of STAGES) {
        if (elapsed > cumulativeMs) {
          currentActive = st.id;
        }
        if (elapsed > cumulativeMs + st.duration) {
          newCompleted.push(st.id);
        }
        cumulativeMs += st.duration;
      }
      setActiveStage(currentActive);
      setCompletedStages(newCompleted);

      if (pct < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        setDone(true);
        setActiveStage(0);
        // Small pause before calling onComplete so user sees 100%
        setTimeout(() => {
          onComplete();
        }, 420);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [onComplete]);

  const progressPct = Math.round(progress * 100);

  return (
    <div className="forensic-scanner-overlay">
      <div className="forensic-scanner-modal">
        {/* Header */}
        <div className="fscan-header">
          <div className="fscan-header-left">
            <span className="fscan-blink-dot" />
            <span className="fscan-title">FORENSIC 1:1 BIOMETRIC EVALUATION</span>
          </div>
          <div className="fscan-header-right">
            <span className="fscan-badge">ISO/IEC 19795 · ARCFACE-MFN · TTA-512D</span>
          </div>
        </div>

        {/* Side-by-Side Face Comparison */}
        <div className="fscan-comparison">
          {/* Document Portrait */}
          <div className="fscan-face-panel">
            <div className="fscan-face-label">DOCUMENT PORTRAIT</div>
            <div className="fscan-face-frame doc">
              {docThumbnail ? (
                <img src={docThumbnail} alt="Document face" className="fscan-face-img" />
              ) : (
                <div className="fscan-face-placeholder">
                  <span>NO DOC</span>
                </div>
              )}
              <div className="fscan-corner tl" />
              <div className="fscan-corner tr" />
              <div className="fscan-corner bl" />
              <div className="fscan-corner br" />
              {/* Scan line overlay */}
              <div
                className="fscan-scan-line"
                style={{ top: `${scanLine}%` }}
              />
            </div>
            <div className="fscan-face-meta">
              {documents[0]?.doc_type?.toUpperCase() ?? 'IDENTITY DOCUMENT'}
            </div>
          </div>

          {/* VS indicator */}
          <div className="fscan-vs-block">
            <div className="fscan-vs-line" />
            <div className="fscan-vs-label">1:1</div>
            <div className="fscan-vs-line" />
            {/* Animated pulse ring */}
            <div className="fscan-pulse-ring" />
          </div>

          {/* Live Capture */}
          <div className="fscan-face-panel">
            <div className="fscan-face-label">LIVE BIOMETRIC CAPTURE</div>
            <div className="fscan-face-frame live">
              {liveThumbnail ? (
                <img src={liveThumbnail} alt="Live face" className="fscan-face-img" />
              ) : (
                <div className="fscan-face-placeholder">
                  <span>NO CAPTURE</span>
                </div>
              )}
              <div className="fscan-corner tl" />
              <div className="fscan-corner tr" />
              <div className="fscan-corner bl" />
              <div className="fscan-corner br" />
              <div
                className="fscan-scan-line"
                style={{ top: `${(scanLine + 50) % 100}%` }}
              />
            </div>
            <div className="fscan-face-meta">
              FRONT · {liveCaptures['LEFT'] ? 'LEFT ·' : ''}{liveCaptures['RIGHT'] ? ' RIGHT' : ''}
              {' '}CAPTURED
            </div>
          </div>
        </div>

        {/* Stage Checklist */}
        <div className="fscan-stages">
          {STAGES.map((st) => {
            const isComplete = completedStages.includes(st.id);
            const isActive = activeStage === st.id;
            return (
              <div
                key={st.id}
                className={`fscan-stage-row ${isComplete ? 'complete' : isActive ? 'active' : 'pending'}`}
              >
                <div className="fscan-stage-check">
                  {isComplete ? (
                    <CheckCircle2 size={13} className="fscan-icon-complete" />
                  ) : isActive ? (
                    <Loader2 size={13} className="fscan-icon-spin" />
                  ) : (
                    <Circle size={13} className="fscan-icon-pending" />
                  )}
                </div>
                <div className="fscan-stage-text">
                  <span className="fscan-stage-label">{st.label}</span>
                  {isActive && (
                    <span className="fscan-stage-detail">{st.detail}</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Progress Bar */}
        <div className="fscan-progress-section">
          <div className="fscan-progress-header">
            <span className="fscan-progress-label">BIOMETRIC ANALYSIS PROGRESS</span>
            <span className="fscan-progress-pct">{progressPct}%</span>
          </div>
          <div className="fscan-progress-track">
            <div
              className="fscan-progress-fill"
              style={{ width: `${progressPct}%` }}
            />
            <div
              className="fscan-progress-glow"
              style={{ left: `${progressPct}%` }}
            />
          </div>
        </div>

        {done && (
          <div className="fscan-done-banner">
            <ShieldCheck size={14} className="fscan-done-icon" style={{ display: 'inline-flex', verticalAlign: 'middle', marginRight: '6px' }} />
            <span>EVIDENCE FUSION COMPLETE — RENDERING VERDICT</span>
          </div>
        )}
      </div>
    </div>
  );
};
