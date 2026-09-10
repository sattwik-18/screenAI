import React from 'react';
import { Check, CheckCircle2, ArrowRight, Loader2 } from 'lucide-react';

interface AngleProgressHUDProps {
  captureState?: {
    state?: string;
    sub_step?: string;
    instruction?: string;
    target_angle?: string;
    front_captured?: boolean;
    left_captured?: boolean;
    right_captured?: boolean;
    front_count?: number;
    left_count?: number;
    right_count?: number;
    is_complete?: boolean;
  };
  onEvaluate: () => void;
  evaluating: boolean;
}

export const AngleProgressHUD: React.FC<AngleProgressHUDProps> = ({
  captureState,
  onEvaluate,
  evaluating
}) => {
  const frontCaptured = captureState?.front_captured || (captureState?.front_count ?? 0) >= 1;
  const leftCaptured = captureState?.left_captured || (captureState?.left_count ?? 0) >= 1;
  const rightCaptured = captureState?.right_captured || (captureState?.right_count ?? 0) >= 1;
  const currentState = captureState?.state ?? 'CAPTURING_FRONT';
  const isComplete = captureState?.is_complete || (frontCaptured && leftCaptured && rightCaptured);

  return (
    <div className="lower-control-hud">
      <div className="angle-progress-group">
        <span className="hud-title">CAPTURE SEQUENCE:</span>

        {/* FRONT */}
        <div className={`angle-badge ${
          frontCaptured ? 'completed' : currentState === 'CAPTURING_FRONT' ? 'active' : ''
        }`}>
          <span>1. FRONT (0°)</span>
          {frontCaptured ? (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
              <Check size={10} /> CAPTURED
            </span>
          ) : (
            <span>PENDING</span>
          )}
        </div>

        {/* LEFT */}
        <div className={`angle-badge ${
          leftCaptured ? 'completed' : currentState === 'CAPTURING_LEFT' ? 'active' : ''
        }`}>
          <span>2. LEFT (~20°)</span>
          {leftCaptured ? (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
              <Check size={10} /> CAPTURED
            </span>
          ) : (
            <span>PENDING</span>
          )}
        </div>

        {/* RIGHT */}
        <div className={`angle-badge ${
          rightCaptured ? 'completed' : currentState === 'CAPTURING_RIGHT' ? 'active' : ''
        }`}>
          <span>3. RIGHT (~20°)</span>
          {rightCaptured ? (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
              <Check size={10} /> CAPTURED
            </span>
          ) : (
            <span>PENDING</span>
          )}
        </div>
      </div>

      <div>
        <button
          onClick={onEvaluate}
          disabled={evaluating}
          className={`btn-primary ${isComplete ? 'btn-ready-pulse' : ''}`}
          style={{ whiteSpace: 'nowrap' }}
        >
          {evaluating ? (
            <>
              <Loader2 size={12} className="fscan-icon-spin" />
              <span>FUSING EVIDENCE...</span>
            </>
          ) : isComplete ? (
            <>
              <CheckCircle2 size={12} />
              <span>EVALUATE VERIFICATION</span>
              <ArrowRight size={12} />
            </>
          ) : (
            <>
              <span>EVALUATE VERIFICATION</span>
              <ArrowRight size={12} />
            </>
          )}
        </button>
      </div>
    </div>
  );
};

