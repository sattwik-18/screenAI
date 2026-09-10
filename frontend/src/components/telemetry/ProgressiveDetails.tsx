import React, { useState } from 'react';
import { TelemetryData } from '../camera/OverlayRenderer';

interface ProgressiveDetailsProps {
  telemetry: TelemetryData | null;
  showMesh: boolean;
  onToggleMesh: (enabled: boolean) => void;
}

export const ProgressiveDetails: React.FC<ProgressiveDetailsProps> = ({
  telemetry,
  showMesh,
  onToggleMesh
}) => {
  const [activeTab, setActiveTab] = useState<'liveness' | 'quality' | '3d' | 'audit'>('liveness');

  const liveness = (telemetry as any)?.liveness;
  const quality = (telemetry as any)?.quality;
  const pose = telemetry?.pose;

  return (
    <div className="telemetry-deck-card">
      <div className="tab-nav">
        <button
          onClick={() => setActiveTab('liveness')}
          className={`tab-btn ${activeTab === 'liveness' ? 'active' : ''}`}
        >
          LIVENESS
        </button>
        <button
          onClick={() => setActiveTab('quality')}
          className={`tab-btn ${activeTab === 'quality' ? 'active' : ''}`}
        >
          QUALITY
        </button>
        <button
          onClick={() => setActiveTab('3d')}
          className={`tab-btn ${activeTab === '3d' ? 'active' : ''}`}
        >
          3D POSE
        </button>
        <button
          onClick={() => setActiveTab('audit')}
          className={`tab-btn ${activeTab === 'audit' ? 'active' : ''}`}
        >
          AUDIT
        </button>
      </div>

      <div className="tab-body">
        {activeTab === 'liveness' && (
          <div>
            <div className="signal-grid">
              <div className="signal-box">
                <div className="signal-label">MiniFASNet Deep PAD</div>
                <div className="signal-value" style={{ color: '#00D8F6' }}>
                  {liveness?.deep_pad?.score !== undefined ? `${(liveness.deep_pad.score * 100).toFixed(1)}%` : '--'}
                </div>
                <div className="signal-sub">
                  Replay Risk: {liveness?.deep_pad?.replay_attack ? `${(liveness.deep_pad.replay_attack * 100).toFixed(1)}%` : '0%'}
                </div>
              </div>

              <div className="signal-box">
                <div className="signal-label">Temporal Motion</div>
                <div className="signal-value" style={{ color: '#10B981', fontSize: '11px' }}>
                  {liveness?.temporal?.state || 'MONITORING'}
                </div>
                <div className="signal-sub">
                  Variance: {liveness?.temporal?.variance ?? '--'}
                </div>
              </div>
            </div>

            <div className="signal-row">
              <span style={{ color: '#54657E' }}>Fourier Spectral Ratio:</span>
              <span style={{ color: '#E6EDF5', fontWeight: 600 }}>{liveness?.texture?.high_freq_ratio ?? '--'}</span>
            </div>

            <div className="signal-row">
              <span style={{ color: '#54657E' }}>Texture Spoof Artifact:</span>
              <span style={{ color: liveness?.texture?.moir\u00e9_detected ? '#EF4444' : '#10B981', fontWeight: 700 }}>
                {liveness?.texture?.moir\u00e9_detected ? 'DETECTED' : 'CLEAR (NATURAL)'}
              </span>
            </div>

            <div className="signal-row">
              <span style={{ color: '#54657E' }}>Hardware Sensors:</span>
              <span style={{ color: '#94A3B8' }}>DEPTH: N/A • IR: N/A</span>
            </div>
          </div>
        )}

        {activeTab === 'quality' && (
          <div>
            <div className="signal-grid">
              <div className="signal-box">
                <div className="signal-label">Laplacian Sharpness</div>
                <div className="signal-value">{quality?.sharpness ?? '--'}</div>
                <div className="signal-sub">Floor: 80.0</div>
              </div>

              <div className="signal-box">
                <div className="signal-label">Mean Illumination</div>
                <div className="signal-value">{quality?.illumination_mean ?? '--'}</div>
                <div className="signal-sub">Optimal: 128.0</div>
              </div>
            </div>

            <div className="signal-row">
              <span style={{ color: '#54657E' }}>Face Size Ratio:</span>
              <span style={{ color: '#E6EDF5' }}>{quality?.face_size_ratio ?? '--'}</span>
            </div>

            <div className="signal-row">
              <span style={{ color: '#54657E' }}>Enrollment Eligibility:</span>
              <span style={{ color: quality?.acceptable ? '#10B981' : '#F59E0B', fontWeight: 700 }}>
                {quality?.acceptable ? 'ACCEPTABLE' : 'BELOW FLOOR'}
              </span>
            </div>
          </div>
        )}

        {activeTab === '3d' && (
          <div>
            <div className="signal-row" style={{ marginBottom: '10px' }}>
              <div>
                <span style={{ fontWeight: 600, color: '#E6EDF5', display: 'block' }}>3D Face Mesh</span>
                <span style={{ fontSize: '9px', color: '#54657E' }}>68 Dense Vertices</span>
              </div>
              <button
                onClick={() => onToggleMesh(!showMesh)}
                className="btn-secondary"
                style={{ padding: '4px 8px', fontSize: '9px' }}
              >
                {showMesh ? 'MESH: ACTIVE' : 'MESH: OFF'}
              </button>
            </div>

            <div className="signal-row">
              <span style={{ color: '#54657E' }}>Pitch Angle:</span>
              <span style={{ color: '#E6EDF5' }}>{pose?.pitch_deg}°</span>
            </div>

            <div className="signal-row">
              <span style={{ color: '#54657E' }}>Yaw Angle:</span>
              <span style={{ color: '#E6EDF5' }}>{pose?.yaw_deg}°</span>
            </div>

            <div className="signal-row">
              <span style={{ color: '#54657E' }}>Roll Angle:</span>
              <span style={{ color: '#E6EDF5' }}>{pose?.roll_deg}°</span>
            </div>
          </div>
        )}

        {activeTab === 'audit' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
            <div className="signal-row">
              <span style={{ color: '#54657E' }}>DETECTOR:</span>
              <span style={{ color: '#E6EDF5' }}>SCRFD-500M (ONNX)</span>
            </div>
            <div className="signal-row">
              <span style={{ color: '#54657E' }}>LANDMARKS 2D:</span>
              <span style={{ color: '#E6EDF5' }}>2D-106Det</span>
            </div>
            <div className="signal-row">
              <span style={{ color: '#54657E' }}>LANDMARKS 3D:</span>
              <span style={{ color: '#E6EDF5' }}>1K-3D-68</span>
            </div>
            <div className="signal-row">
              <span style={{ color: '#54657E' }}>ANTI-SPOOFING:</span>
              <span style={{ color: '#E6EDF5' }}>MiniFASNetV2</span>
            </div>
            <div className="signal-row">
              <span style={{ color: '#54657E' }}>RECOGNITION:</span>
              <span style={{ color: '#E6EDF5' }}>ArcFace-MobileFaceNet 512-d</span>
            </div>
            <div className="signal-row" style={{ borderTop: '1px solid #17202E', marginTop: '4px', paddingTop: '6px' }}>
              <span style={{ color: '#54657E' }}>DECISION POLICY:</span>
              <span style={{ color: '#00D8F6' }}>v1.0-dev-calibrated</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
