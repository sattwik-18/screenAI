import React, { useState, useEffect, useCallback, useRef } from 'react';
import { CameraViewport } from './components/camera/CameraViewport';
import { MetricStrip } from './components/telemetry/MetricStrip';
import { AngleProgressHUD } from './components/capture/AngleProgressHUD';
import { CapturedProofsDeck } from './components/capture/CapturedProofsDeck';
import { DocumentDeck, DocumentRecord } from './components/documents/DocumentDeck';
import { ProgressiveDetails } from './components/telemetry/ProgressiveDetails';
import { VerificationDossier } from './components/dossier/VerificationDossier';
import { DebugPipelineView } from './components/telemetry/DebugPipelineView';
import { ForensicEvaluationScanner } from './components/evaluation/ForensicEvaluationScanner';
import { TelemetryData } from './components/camera/OverlayRenderer';

export const App: React.FC = () => {
  const [sessionId, setSessionId] = useState<string>('');
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [showMesh, setShowMesh] = useState<boolean>(true);
  const [debugOpen, setDebugOpen] = useState<boolean>(false);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [finalReport, setFinalReport] = useState<any | null>(null);
  const [evaluating, setEvaluating] = useState<boolean>(false);
  const [scannerOpen, setScannerOpen] = useState<boolean>(false);
  // Holds pending report while scanner animation is playing
  const pendingReportRef = useRef<any | null>(null);
  const scannerDoneRef = useRef<boolean>(false);

  // Initialize Session
  const initSession = useCallback(async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/verification/sessions', {
        method: 'POST'
      });
      const data = await res.json();
      setSessionId(data.session_id);
      setFinalReport(null);
      setDocuments([]);
    } catch (err) {
      console.error('Failed to initialize verification session:', err);
    }
  }, []);

  useEffect(() => {
    initSession();
  }, [initSession]);

  const handleDocumentUploaded = (doc: DocumentRecord) => {
    setDocuments((prev) => [...prev, doc]);
  };

  const handleDocumentDeleted = (docId: string) => {
    setDocuments((prev) => prev.filter((d) => d.doc_id !== docId));
  };

  const handleEvaluateResult = async () => {
    if (!sessionId) return;
    setEvaluating(true);
    pendingReportRef.current = null;
    scannerDoneRef.current = false;

    // 1. Open scanner immediately — it shows faces + 4-stage animation
    setScannerOpen(true);

    // 2. Concurrently fetch the actual backend result
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/v1/verification/${sessionId}/result`);
      const data = await res.json();
      pendingReportRef.current = data;
    } catch (err) {
      console.error('Error fetching verification report:', err);
      pendingReportRef.current = { error: true };
    }

    // 3. If scanner animation already finished, reveal dossier now
    if (scannerDoneRef.current) {
      setScannerOpen(false);
      setFinalReport(pendingReportRef.current);
      setEvaluating(false);
    }
    // Otherwise the handleScannerComplete callback below will fire when animation ends
  };

  // Called by the scanner when its animation finishes
  const handleScannerComplete = useCallback(() => {
    scannerDoneRef.current = true;
    // If backend already returned, reveal immediately; else wait for it
    if (pendingReportRef.current !== null) {
      setScannerOpen(false);
      setFinalReport(pendingReportRef.current);
      setEvaluating(false);
    }
    // else: when fetch resolves it will see scannerDoneRef.current === true and close
  }, []);

  const handleReset = async () => {
    await initSession();
  };

  const handleReVerify = async () => {
    setFinalReport(null);
    if (sessionId) {
      try {
        await fetch(`http://127.0.0.1:8000/api/v1/verification/${sessionId}/reset`, {
          method: 'POST'
        });
      } catch (e) {
        console.error('Failed to reset capture sequence:', e);
      }
    }
  };

  const handleRetakeAngle = async (angle?: string) => {
    if (!sessionId) return;
    try {
      const url = angle
        ? `http://127.0.0.1:8000/api/v1/verification/${sessionId}/reset?angle=${encodeURIComponent(angle)}`
        : `http://127.0.0.1:8000/api/v1/verification/${sessionId}/reset`;
      await fetch(url, { method: 'POST' });
    } catch (e) {
      console.error('Failed to retake angle:', e);
    }
  };

  return (
    <div className="workstation-container">
      {/* Top Telemetry Header */}
      <MetricStrip
        telemetry={telemetry}
        cameraActive={cameraActive}
        sessionId={sessionId}
        captureState={(telemetry as any)?.capture}
      />

      {/* Main Workstation Layout */}
      <div className="workstation-body">
        {/* Left Section: Dominant Camera Viewport & Capture HUD */}
        <div className="viewport-section">
          {sessionId ? (
            <CameraViewport
              sessionId={sessionId}
              onTelemetryUpdate={setTelemetry}
              showMesh={showMesh}
              onCameraActiveChange={setCameraActive}
              onOpenDebugView={() => setDebugOpen(true)}
            />
          ) : (
            <div className="camera-viewport-card" style={{ color: '#54657E', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
              INITIALIZING ENCRYPTED VERIFICATION SESSION...
            </div>
          )}

          {/* Guided Capture Progress HUD */}
          <AngleProgressHUD
            captureState={(telemetry as any)?.capture}
            onEvaluate={handleEvaluateResult}
            evaluating={evaluating}
          />
        </div>

        {/* Right Section: Document Deck & Progressive Disclosure Telemetry */}
        <div className="sidebar-section">
          {/* Multi-Angle Biometric Proofs Deck */}
          <CapturedProofsDeck
            captureState={(telemetry as any)?.capture}
            onRetakeAngle={handleRetakeAngle}
          />

          {/* Identity Document Deck */}
          <DocumentDeck
            sessionId={sessionId}
            documents={documents}
            onDocumentUploaded={handleDocumentUploaded}
            onDocumentDeleted={handleDocumentDeleted}
          />

          {/* Progressive Telemetry Details */}
          <ProgressiveDetails
            telemetry={telemetry}
            showMesh={showMesh}
            onToggleMesh={setShowMesh}
          />
        </div>
      </div>

      {/* 5-Stage Side-by-Side Debug Pipeline View */}
      <DebugPipelineView
        telemetry={telemetry}
        isOpen={debugOpen}
        onClose={() => setDebugOpen(false)}
      />

      {/* Forensic 1:1 Evaluation Scanner — plays during backend processing */}
      {scannerOpen && (
        <ForensicEvaluationScanner
          liveCaptures={{
            FRONT: (telemetry as any)?.capture?.captured_snapshots?.FRONT?.thumbnail,
            LEFT: (telemetry as any)?.capture?.captured_snapshots?.LEFT?.thumbnail,
            RIGHT: (telemetry as any)?.capture?.captured_snapshots?.RIGHT?.thumbnail,
          }}
          documents={documents.map((d) => ({
            doc_id: d.doc_id,
            doc_type: d.doc_type,
            face_crop_url: d.face_crop_url,
          }))}
          onComplete={handleScannerComplete}
        />
      )}

      {/* Verification Result Dossier Modal */}
      {finalReport && (
        <VerificationDossier
          report={finalReport}
          onClose={() => setFinalReport(null)}
          onReset={handleReset}
          onReVerify={handleReVerify}
        />
      )}
    </div>
  );
};
