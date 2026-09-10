import React, { useState, useEffect } from 'react';
import { X, AlertTriangle, RotateCcw } from 'lucide-react';

interface VerificationDossierProps {
  report: any;
  onClose: () => void;
  onReset: () => void;
  onReVerify: () => void;
}

export const VerificationDossier: React.FC<VerificationDossierProps> = ({
  report,
  onClose,
  onReset,
  onReVerify
}) => {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  // Close modal on Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!report) return null;

  const decision = report.decision || 'FAILED';
  const evidence = report.evidence_breakdown || {};
  const isVerified = decision === 'VERIFIED';
  const isReview = decision === 'MANUAL_REVIEW';

  const badgeClass = isVerified
    ? 'verified'
    : isReview
    ? 'review'
    : 'failed';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="dossier-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="dossier-header">
          <div className="dossier-id-block">
            <span className="dossier-super">FORENSIC BIOMETRIC VERIFICATION DOSSIER</span>
            <span className="dossier-id">{report.session_id}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div className={`decision-badge ${badgeClass}`}>
              {decision}
            </div>
            <button
              onClick={onClose}
              className="dossier-close-btn"
              title="Close Dossier (Esc)"
              style={{
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-muted)',
                borderRadius: '4px',
                width: '28px',
                height: '28px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer'
              }}
            >
              <X size={15} />
            </button>
          </div>
        </div>

        {/* Summary Box */}
        <div className="dossier-summary-box">
          {report.summary_message}
        </div>

        {/* 4-Stat Evidence Grid */}
        <div className="dossier-grid">
          {/* Liveness */}
          <div className="dossier-stat-box">
            <div className="dossier-stat-info">
              <span className="dossier-stat-label">Live Person (PAD)</span>
              <span className="dossier-stat-num">
                {evidence?.liveness?.score !== undefined ? `${(evidence.liveness.score * 100).toFixed(1)}%` : '--'}
              </span>
            </div>
            <span className={`status-tag ${evidence?.liveness?.status === 'PASS' ? 'pass' : 'fail'}`}>
              {evidence?.liveness?.status || 'FAIL'}
            </span>
          </div>

          {/* Identity Match */}
          <div className="dossier-stat-box">
            <div className="dossier-stat-info">
              <span className="dossier-stat-label">Biometric Similarity (ArcFace TTA)</span>
              <span className="dossier-stat-num">
                {evidence?.identity_match?.aggregate_similarity !== undefined
                  ? `${Number(evidence.identity_match.aggregate_similarity).toFixed(3)}`
                  : '--'}
              </span>
              <span style={{ fontSize: '9px', color: '#54657E' }}>
                {evidence?.identity_match?.angular_distance_deg !== undefined
                  ? `∠ ${Number(evidence.identity_match.angular_distance_deg).toFixed(1)}°  ·  Threshold: ≥0.68 (≤55°)`
                  : 'Threshold: ≥ 0.680 [Dev]'}
              </span>
            </div>
            <span className={`status-tag ${evidence?.identity_match?.status === 'STRONG' ? 'pass' : 'hold'}`}>
              {evidence?.identity_match?.status || 'LOW'}
            </span>
          </div>

          {/* Cross-Angle Consistency */}
          <div className="dossier-stat-box">
            <div className="dossier-stat-info">
              <span className="dossier-stat-label">Cross-Angle Consistency</span>
              <span className="dossier-stat-num">
                {evidence?.cross_angle_consistency?.composite_consistency !== undefined
                  ? `${Number(evidence.cross_angle_consistency.composite_consistency).toFixed(3)}`
                  : evidence?.cross_angle_consistency?.score !== undefined
                  ? `${Number(evidence.cross_angle_consistency.score).toFixed(3)}`
                  : '--'}
              </span>
              <span style={{ fontSize: '9px', color: '#54657E' }}>Consistency floor: ≥ 0.480</span>
            </div>
            <span className={`status-tag ${
              (evidence?.cross_angle_consistency?.consistency_pass ?? true) ? 'pass' : 'hold'
            }`}>
              {evidence?.cross_angle_consistency?.status ||
               ((evidence?.cross_angle_consistency?.consistency_pass ?? true) ? 'PASS' : 'REVIEW')}
            </span>
          </div>

          {/* Session Integrity */}
          <div className="dossier-stat-box">
            <div className="dossier-stat-info">
              <span className="dossier-stat-label">Session Integrity</span>
              <span className="dossier-stat-num">SINGLE SUBJECT</span>
              <span style={{ fontSize: '9px', color: '#10B981' }}>Hardware Depth/IR: N/A</span>
            </div>
            <span className="status-tag pass">
              {evidence?.session_integrity?.status || 'SECURE'}
            </span>
          </div>
        </div>

        {/* Divergence Warning if multiple documents have mismatching identities */}
        {evidence?.identity_match?.document_identity_divergence && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid #EF4444',
            borderRadius: '4px',
            padding: '8px 12px',
            marginBottom: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <AlertTriangle size={15} style={{ color: '#EF4444', flexShrink: 0 }} />
            <div>
              <strong>DOCUMENT IDENTITY DIVERGENCE DETECTED:</strong> Uploaded documents represent conflicting identities (similarity &lt; 0.450). Operator review required.
            </div>
          </div>
        )}

        {/* Document Matches Breakdown */}
        {evidence?.identity_match?.document_matches?.length > 0 && (
          <div className="dossier-doc-list">
            <span style={{ fontSize: '10px', color: '#54657E', textTransform: 'uppercase', fontWeight: 700 }}>
              Evaluated Document Faces (Quality-Weighted Fusion):
            </span>
          {evidence.identity_match.document_matches.map((doc: any, i: number) => (
              <div key={i} className="dossier-doc-row">
                <span style={{ color: '#00D8F6', fontWeight: 700 }}>{doc.doc_type?.toUpperCase()} ({doc.doc_id})</span>
                <div style={{ display: 'flex', gap: '10px', color: '#94A3B8', flexWrap: 'wrap' }}>
                  <span>F: {Number(doc.front_match_score).toFixed(3)}{doc.front_angular_dist_deg !== undefined ? ` (${Number(doc.front_angular_dist_deg).toFixed(1)}°)` : ''}</span>
                  <span>L: {Number(doc.left_match_score).toFixed(3)}{doc.left_angular_dist_deg !== undefined ? ` (${Number(doc.left_angular_dist_deg).toFixed(1)}°)` : ''}</span>
                  <span>R: {Number(doc.right_match_score).toFixed(3)}{doc.right_angular_dist_deg !== undefined ? ` (${Number(doc.right_angular_dist_deg).toFixed(1)}°)` : ''}</span>
                  <span style={{ color: '#00D8F6' }}>Wt: {doc.portrait_quality_weight ?? 1.0}</span>
                  <span style={{ color: '#10B981', fontWeight: 700 }}>
                    ∑ {Number(doc.robust_score).toFixed(3)}
                    {doc.angular_distance_deg !== undefined ? ` · ∠${Number(doc.angular_distance_deg).toFixed(1)}°` : ''}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Technical Details Toggle */}
        <div>
          <button
            onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
            style={{
              background: 'none',
              border: 'none',
              color: '#00D8F6',
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              cursor: 'pointer',
              textDecoration: 'underline'
            }}
          >
            {showTechnicalDetails ? '[-] HIDE AUDIT TRAIL & TRIGGERS' : '[+] VIEW AUDIT TRAIL & TRIGGERS'}
          </button>

          {showTechnicalDetails && (
            <div style={{
              marginTop: '10px',
              padding: '10px',
              background: 'var(--bg-input)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '4px',
              fontSize: '10px',
              maxHeight: '140px',
              overflowY: 'auto'
            }}>
              {report.rejection_reasons?.length > 0 && (
                <div style={{ marginBottom: '8px' }}>
                  <span style={{ color: '#EF4444', fontWeight: 700 }}>REJECTION TRIGGERS:</span>
                  {report.rejection_reasons.map((r: string, idx: number) => (
                    <div key={idx} style={{ color: '#94A3B8' }}>• {r}</div>
                  ))}
                </div>
              )}

              {report.review_reasons?.length > 0 && (
                <div style={{ marginBottom: '8px' }}>
                  <span style={{ color: '#F59E0B', fontWeight: 700 }}>MANUAL REVIEW TRIGGERS:</span>
                  {report.review_reasons.map((r: string, idx: number) => (
                    <div key={idx} style={{ color: '#94A3B8' }}>• {r}</div>
                  ))}
                </div>
              )}

              <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '6px' }}>
                <span style={{ color: '#54657E', fontWeight: 700 }}>MODEL MANIFEST:</span>
                {Object.entries(report.models_audited || {}).map(([k, v]: any) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', color: '#94A3B8' }}>
                    <span>{k}:</span>
                    <span style={{ color: '#E6EDF5' }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="dossier-actions" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <button onClick={onClose} className="btn-secondary" title="Close Dossier and return to live workstation">
            CLOSE DOSSIER
          </button>
          <button
            onClick={onReVerify}
            className="btn-primary"
            style={{ background: 'linear-gradient(135deg, var(--accent-cyan) 0%, #0088cc 100%)' }}
            title="Reset capture sequence to re-capture angles and re-evaluate without losing uploaded documents"
          >
            <RotateCcw size={12} />
            <span>RE-CAPTURE & RE-VERIFY</span>
          </button>
          <button onClick={onReset} className="btn-secondary" title="Start a completely new blank verification session">
            START NEW SESSION
          </button>
        </div>
      </div>
    </div>
  );
};
