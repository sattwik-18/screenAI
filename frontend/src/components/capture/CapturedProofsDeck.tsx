import React, { useState } from 'react';
import { Camera, RotateCcw, CheckCircle2, Check, User, ArrowLeft, ArrowRight, X } from 'lucide-react';

export interface CapturedSnapshot {
  angle: string;
  thumbnail?: string; // base64 data URI
  yaw?: number;
  pitch?: number;
  score?: number;
  timestamp?: number;
}

interface CapturedProofsDeckProps {
  captureState?: {
    state?: string;
    target_angle?: string;
    in_target_zone?: boolean;
    hold_progress?: number;
    direction_hint?: string;
    front_captured?: boolean;
    left_captured?: boolean;
    right_captured?: boolean;
    snapshots?: Record<string, CapturedSnapshot>;
    is_complete?: boolean;
  };
  onRetakeAngle?: (angle?: string) => void;
}

export const CapturedProofsDeck: React.FC<CapturedProofsDeckProps> = ({ captureState, onRetakeAngle }) => {
  const [selectedProof, setSelectedProof] = useState<CapturedSnapshot | null>(null);

  const snapshots = captureState?.snapshots || {};
  const targetAngle = captureState?.target_angle || 'FRONT';
  const isComplete = captureState?.is_complete || false;

  const capturedCount = [
    captureState?.front_captured,
    captureState?.left_captured,
    captureState?.right_captured
  ].filter(Boolean).length;

  const proofSlots = [
    {
      key: 'FRONT',
      title: 'FRONTAL POSE',
      angleLabel: '0° CENTER',
      targetDesc: '|Yaw| ≤ 12°',
      snapshot: snapshots['FRONT'],
      isTarget: targetAngle === 'FRONT' && !captureState?.front_captured
    },
    {
      key: 'LEFT',
      title: 'LEFT PROFILE',
      angleLabel: '~20° LEFT',
      targetDesc: '-38° ≤ Yaw ≤ -8°',
      snapshot: snapshots['LEFT'],
      isTarget: targetAngle === 'LEFT' && !captureState?.left_captured
    },
    {
      key: 'RIGHT',
      title: 'RIGHT PROFILE',
      angleLabel: '~20° RIGHT',
      targetDesc: '+8° ≤ Yaw ≤ +38°',
      snapshot: snapshots['RIGHT'],
      isTarget: targetAngle === 'RIGHT' && !captureState?.right_captured
    }
  ];

  return (
    <div className="proofs-deck-card">
      <div className="proofs-deck-header">
        <div className="deck-title-group">
          <Camera size={14} style={{ color: 'var(--accent-cyan)' }} />
          <div>
            <div className="deck-title">CAPTURED BIOMETRIC PROOFS</div>
            <div className="deck-subtitle">MULTI-ANGLE IDENTITY SCREENSHOTS</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {capturedCount > 0 && onRetakeAngle && (
            <button
              onClick={() => onRetakeAngle()}
              className="btn-secondary"
              style={{ padding: '3px 8px', fontSize: '9px', height: '22px', display: 'inline-flex', alignItems: 'center', gap: '4px' }}
              title="Reset all 3 captures and re-capture from FRONT"
            >
              <RotateCcw size={10} />
              <span>RETAKE ALL</span>
            </button>
          )}
          <div className={`proofs-counter-badge ${isComplete ? 'complete' : ''}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <span>{isComplete ? '3/3 SECURED' : `${capturedCount}/3 CAPTURED`}</span>
            {isComplete && <CheckCircle2 size={11} />}
          </div>
        </div>
      </div>

      <div className="proofs-grid">
        {proofSlots.map((slot) => {
          const hasCaptured = !!slot.snapshot?.thumbnail;
          const snap = slot.snapshot;

          return (
            <div
              key={slot.key}
              className={`proof-card ${hasCaptured ? 'captured' : ''} ${slot.isTarget ? 'active-target' : ''}`}
              onClick={() => hasCaptured && snap && setSelectedProof(snap)}
              title={hasCaptured ? 'Click to inspect full forensic crop' : undefined}
            >
              {/* Card Status Indicator */}
              <div className="proof-card-header">
                <span className="proof-angle-name">{slot.title}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  {hasCaptured && onRetakeAngle && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onRetakeAngle(slot.key);
                      }}
                      className="proof-retake-mini-btn"
                      title={`Retake ${slot.title}`}
                      style={{
                        background: 'rgba(255, 255, 255, 0.08)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        color: 'var(--accent-cyan)',
                        borderRadius: '2px',
                        fontSize: '8px',
                        padding: '1px 5px',
                        cursor: 'pointer',
                        fontFamily: 'var(--font-mono)',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '3px'
                      }}
                    >
                      <RotateCcw size={8} />
                      <span>Retake</span>
                    </button>
                  )}
                  <span className={`proof-status-tag ${hasCaptured ? 'tag-captured' : slot.isTarget ? 'tag-target' : 'tag-pending'}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                    {hasCaptured ? (
                      <>
                        <Check size={9} />
                        <span>SAVED</span>
                      </>
                    ) : slot.isTarget ? (
                      'TARGET'
                    ) : (
                      'PENDING'
                    )}
                  </span>
                </div>
              </div>

              {/* Card Body: Thumbnail or Awaiting Silhouette */}
              <div className="proof-media-box">
                {hasCaptured && snap?.thumbnail ? (
                  <div className="proof-thumbnail-wrap">
                    <img
                      src={snap.thumbnail}
                      alt={`${slot.title} biometric proof`}
                      className="proof-img"
                    />
                    <div className="proof-lens-glare" />
                    <div className="proof-crosshair-tag">YAW {snap.yaw}°</div>
                  </div>
                ) : (
                  <div className="proof-placeholder-wrap">
                    <div className="proof-wireframe-icon">
                      {slot.key === 'FRONT' ? (
                        <User size={22} style={{ color: 'var(--accent-cyan)', opacity: 0.8 }} />
                      ) : slot.key === 'LEFT' ? (
                        <ArrowLeft size={22} style={{ color: 'var(--accent-cyan)', opacity: 0.8 }} />
                      ) : (
                        <ArrowRight size={22} style={{ color: 'var(--accent-cyan)', opacity: 0.8 }} />
                      )}
                    </div>
                    <div className="proof-angle-target">{slot.angleLabel}</div>
                    <div className="proof-angle-desc">{slot.targetDesc}</div>
                    {slot.isTarget && (
                      <div className="proof-target-indicator">
                        <span className="pulsing-beacon" />
                        <span>ALIGNING SENSOR</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Card Footer Metric */}
              <div className="proof-card-footer">
                {hasCaptured ? (
                  <div className="proof-metric-row">
                    <span className="metric-label">QUALITY</span>
                    <span className="metric-val">{Math.round((snap?.score ?? 0.85) * 100)}%</span>
                  </div>
                ) : (
                  <div className="proof-metric-row">
                    <span className="metric-label">STATUS</span>
                    <span className="metric-val-pending">
                      {slot.isTarget ? 'AWAITING HOLD' : 'IN QUEUE'}
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Forensic Proof Modal Preview */}
      {selectedProof && (
        <div className="proof-modal-overlay" onClick={() => setSelectedProof(null)}>
          <div className="proof-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="proof-modal-header">
              <span className="modal-title">FORENSIC PROOF SNAPSHOT - {selectedProof.angle}</span>
              <button className="modal-close-btn" onClick={() => setSelectedProof(null)} title="Close Preview">
                <X size={15} />
              </button>
            </div>
            <div className="proof-modal-body">
              {selectedProof.thumbnail && (
                <img
                  src={selectedProof.thumbnail}
                  alt={`${selectedProof.angle} Proof Large`}
                  className="proof-modal-img"
                />
              )}
              <div className="proof-modal-details">
                <div className="detail-row">
                  <span>ANGLE VECTOR:</span>
                  <strong>{selectedProof.angle} POSE</strong>
                </div>
                <div className="detail-row">
                  <span>HEAD YAW:</span>
                  <strong>{selectedProof.yaw}°</strong>
                </div>
                <div className="detail-row">
                  <span>HEAD PITCH:</span>
                  <strong>{selectedProof.pitch}°</strong>
                </div>
                <div className="detail-row">
                  <span>FORENSIC SCORE:</span>
                  <strong>{Math.round((selectedProof.score ?? 0.85) * 100)}%</strong>
                </div>
                <div className="detail-row">
                  <span>ACQUISITION:</span>
                  <strong>AUTO-CAPTURE ON HOLD</strong>
                </div>

                {onRetakeAngle && (
                  <button
                    onClick={() => {
                      onRetakeAngle(selectedProof.angle);
                      setSelectedProof(null);
                    }}
                    className="btn-primary"
                    style={{
                      marginTop: '10px',
                      width: '100%',
                      padding: '8px',
                      fontSize: '11px',
                      letterSpacing: '0.06em',
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px'
                    }}
                  >
                    <RotateCcw size={12} />
                    <span>RETAKE THIS ANGLE SNAPSHOT</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
