import React from 'react';
import { CheckCircle2, ChevronsLeft, ChevronsRight } from 'lucide-react';

interface DirectionTrackerHUDProps {
  telemetry: any;
  cameraActive: boolean;
}

export const DirectionTrackerHUD: React.FC<DirectionTrackerHUDProps> = ({
  telemetry,
  cameraActive
}) => {
  if (!cameraActive || !telemetry) return null;

  const capture = telemetry?.capture;
  const pose = telemetry?.pose;
  const currentYaw = pose?.yaw_deg ?? 0;
  const targetAngle = capture?.target_angle ?? 'FRONT';
  const inZone = capture?.in_target_zone ?? false;
  const holdProgress = capture?.hold_progress ?? 0;
  const directionHint = capture?.direction_hint ?? 'HOLD_STILL';
  const isComplete = capture?.is_complete ?? false;

  // Clamp yaw to display range [-45, +45]
  const clampedYaw = Math.max(-45, Math.min(45, currentYaw));
  // Map -45..+45 to 0%..100%
  const needlePercent = ((clampedYaw + 45) / 90) * 100;

  // Calculate target zone range %
  const [minYaw, maxYaw] = capture?.target_yaw_range ?? [-10, 10];
  const zoneLeftPercent = ((Math.max(-45, minYaw) + 45) / 90) * 100;
  const zoneRightPercent = ((Math.min(45, maxYaw) + 45) / 90) * 100;
  const zoneWidthPercent = Math.max(8, zoneRightPercent - zoneLeftPercent);

  // SVG circular progress calculation for hold
  const radius = 18;
  const circumference = 2 * Math.PI * radius;
  const strokeOffset = circumference - (holdProgress * circumference);

  return (
    <div className="direction-tracker-container">
      {/* Visual Yaw Compass Gauge */}
      <div className="yaw-gauge-card">
        <div className="yaw-gauge-header">
          <div className="yaw-target-badge">
            <span className="badge-dot" style={{ background: inZone ? 'var(--accent-emerald)' : 'var(--accent-cyan)' }} />
            <span>TARGET: {targetAngle === 'COMPLETE' ? 'ALL CAPTURED' : `${targetAngle} POSE`}</span>
          </div>

          <div className="yaw-live-numeric">
            <span>YAW:</span>
            <strong className={inZone ? 'text-emerald' : 'text-cyan'}>
              {currentYaw > 0 ? `+${currentYaw.toFixed(1)}°` : `${currentYaw.toFixed(1)}°`}
            </strong>
          </div>
        </div>

        {/* Gauge Track */}
        <div className="yaw-track-wrap">
          {/* Target Zone highlighted band */}
          {!isComplete && (
            <div
              className={`yaw-target-band ${inZone ? 'in-zone' : ''}`}
              style={{
                left: `${zoneLeftPercent}%`,
                width: `${zoneWidthPercent}%`
              }}
            >
              <span className="zone-label">{targetAngle}</span>
            </div>
          )}

          {/* Calibrated Ticks */}
          <div className="yaw-ticks">
            <div className="tick-mark left-limit"><span className="tick-label">-30°</span></div>
            <div className="tick-mark left-zone"><span className="tick-label">-20°</span></div>
            <div className="tick-mark center-zero"><span className="tick-label">0°</span></div>
            <div className="tick-mark right-zone"><span className="tick-label">+20°</span></div>
            <div className="tick-mark right-limit"><span className="tick-label">+30°</span></div>
          </div>

          {/* Dynamic User Yaw Needle / Reticle */}
          <div
            className={`yaw-needle ${inZone ? 'needle-locked' : ''}`}
            style={{ left: `${needlePercent}%` }}
          >
            <div className="needle-head" />
            <div className="needle-stem" />
          </div>
        </div>

        {/* Direction Advice & Hold Circular Progress Bar */}
        <div className="yaw-guidance-footer">
          {isComplete ? (
            <div className="guidance-chip complete">
              <CheckCircle2 size={13} className="check-icon" />
              <span>BIOMETRIC PROOFS ACQUIRED • READY FOR EVALUATION</span>
            </div>
          ) : inZone ? (
            <div className="guidance-chip hold-steady">
              <div className="radial-hold-indicator">
                <svg width="40" height="40" viewBox="0 0 44 44">
                  <circle
                    cx="22"
                    cy="22"
                    r={radius}
                    className="radial-bg"
                  />
                  <circle
                    cx="22"
                    cy="22"
                    r={radius}
                    className="radial-progress"
                    style={{
                      strokeDasharray: circumference,
                      strokeDashoffset: strokeOffset
                    }}
                  />
                </svg>
                <span className="radial-percent">{Math.round(holdProgress * 100)}%</span>
              </div>
              <div className="hold-text-group">
                <span className="hold-title">HOLD STILL • AUTO-CAPTURING</span>
                <span className="hold-sub">Stabilizing forensic biometric capture...</span>
              </div>
            </div>
          ) : directionHint === 'TURN_LEFT' ? (
            <div className="guidance-chip turn-left">
              <ChevronsLeft size={14} className="chevron-arrow left-pulse" />
              <span>TURN HEAD SLIGHTLY LEFT</span>
            </div>
          ) : directionHint === 'TURN_RIGHT' ? (
            <div className="guidance-chip turn-right">
              <span>TURN HEAD SLIGHTLY RIGHT</span>
              <ChevronsRight size={14} className="chevron-arrow right-pulse" />
            </div>
          ) : (
            <div className="guidance-chip center-align">
              <span>LOOK STRAIGHT AT CAMERA</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
