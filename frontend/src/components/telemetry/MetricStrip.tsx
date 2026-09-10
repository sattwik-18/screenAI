import React from 'react';
import { TelemetryData } from '../camera/OverlayRenderer';

interface MetricStripProps {
  telemetry: TelemetryData | null;
  cameraActive: boolean;
  sessionId: string;
  captureState?: any;
}

export const MetricStrip: React.FC<MetricStripProps> = ({
  telemetry,
  cameraActive,
  sessionId,
  captureState
}) => {
  const fps = telemetry?.fps;
  const latency = telemetry?.latency_ms;
  const trackId = telemetry?.track_id;
  const livenessScore = telemetry?.liveness?.liveness_score;
  const isLive = telemetry?.liveness?.is_live;
  const qualityScore = telemetry?.quality?.composite_score;
  const pose = telemetry?.pose;
  const stateName = captureState?.state || 'AWAITING_SENSOR';

  return (
    <header className="header-strip">
      <div className="header-left">
        <div className="header-brand">
          <span className={`brand-dot ${cameraActive ? 'live' : ''}`} />
          <span>CERT8FY</span>
          <span style={{ color: '#54657E', fontWeight: 400 }}>BIOMETRIC CORE</span>
        </div>

        <div className="header-meta">
          <span>SESSION: <strong style={{ color: '#E6EDF5' }}>{sessionId || 'INITIALIZING'}</strong></span>
          <span style={{ color: '#222E42' }}>|</span>
          <span>STATE: <strong style={{ color: '#00D8F6' }}>{stateName}</strong></span>
        </div>
      </div>

      <div className="header-metrics">
        <div className="metric-item">
          <span className="metric-label">FPS:</span>
          <span className="metric-val">{fps !== undefined ? fps.toFixed(1) : '--'}</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">LATENCY:</span>
          <span className="metric-val">{latency !== undefined ? `${latency.toFixed(0)}ms` : '--'}</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">TRACK:</span>
          <span className="metric-val cyan">{trackId || 'SEARCHING'}</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">LIVENESS:</span>
          <span className={`metric-val ${isLive ? 'live-pass' : livenessScore ? 'live-hold' : ''}`}>
            {livenessScore !== undefined ? `${(livenessScore * 100).toFixed(1)}%` : '--'}
            {isLive !== undefined ? (isLive ? ' (PASS)' : ' (HOLD)') : ''}
          </span>
        </div>

        <div className="metric-item">
          <span className="metric-label">QUALITY:</span>
          <span className="metric-val">
            {qualityScore !== undefined ? `${(qualityScore * 100).toFixed(0)}%` : '--'}
          </span>
        </div>

        {pose && (
          <div className="metric-item">
            <span className="metric-label">YAW:</span>
            <span className="metric-val">
              {pose.yaw_deg > 0 ? `+${pose.yaw_deg}` : pose.yaw_deg}°
            </span>
          </div>
        )}
      </div>
    </header>
  );
};
