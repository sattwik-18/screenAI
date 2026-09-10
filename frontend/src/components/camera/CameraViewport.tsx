import React, { useRef, useEffect, useState, useCallback } from 'react';
import { RotateCw, Camera, Activity } from 'lucide-react';
import { OverlayRenderer, TelemetryData } from './OverlayRenderer';
import { PeriocularCards } from './PeriocularCards';
import { DirectionTrackerHUD } from './DirectionTrackerHUD';

interface CameraViewportProps {
  sessionId: string;
  onTelemetryUpdate: (telemetry: TelemetryData) => void;
  showMesh: boolean;
  onCameraActiveChange: (active: boolean) => void;
  onOpenDebugView?: () => void;
}

export const CameraViewport: React.FC<CameraViewportProps> = ({
  sessionId,
  onTelemetryUpdate,
  showMesh,
  onCameraActiveChange,
  onOpenDebugView
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const rendererRef = useRef<OverlayRenderer>(new OverlayRenderer());
  const activeStreamRef = useRef<MediaStream | null>(null);
  const isStartingRef = useRef<boolean>(false);
  const inFlightRef = useRef<boolean>(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'CONNECTING' | 'STREAMING' | 'DISCONNECTED'>('CONNECTING');
  const [availableDevices, setAvailableDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  // Orientation & Auto-Rotation State
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [rotationAngle, setRotationAngle] = useState<number>(() => {
    const saved = localStorage.getItem('cert8fy_camera_rotation');
    return saved ? parseInt(saved, 10) : 0;
  });
  const [mirrorFeed, setMirrorFeed] = useState<boolean>(() => {
    const saved = localStorage.getItem('cert8fy_camera_mirror');
    return saved ? saved === 'true' : false;
  });
  const [stageSize, setStageSize] = useState<{ width: number; height: number }>({ width: 640, height: 480 });
  const [shutterFlash, setShutterFlash] = useState<boolean>(false);

  // Trigger shutter flash on auto-capture
  const justCaptured = (telemetry as any)?.capture?.just_captured;
  const prevCapturedRef = useRef<string | null>(null);

  useEffect(() => {
    if (justCaptured && justCaptured !== prevCapturedRef.current) {
      prevCapturedRef.current = justCaptured;
      setShutterFlash(true);
      const timer = setTimeout(() => setShutterFlash(false), 250);
      return () => clearTimeout(timer);
    }
  }, [justCaptured]);

  const toggleMirrorFeed = useCallback(() => {
    setMirrorFeed((prev) => {
      const next = !prev;
      localStorage.setItem('cert8fy_camera_mirror', next ? 'true' : 'false');
      return next;
    });
  }, []);

  const autoRotateRef = useRef<boolean>(autoRotate);
  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  const updateRotationAngle = useCallback((angleOrUpdater: number | ((prev: number) => number)) => {
    setRotationAngle((prev) => {
      const next = typeof angleOrUpdater === 'function' ? angleOrUpdater(prev) : angleOrUpdater;
      localStorage.setItem('cert8fy_camera_rotation', next.toString());
      return next;
    });
  }, []);

  const isVertical = rotationAngle === 90 || rotationAngle === 270;

  // Update mesh toggle on renderer
  useEffect(() => {
    rendererRef.current.showMesh = showMesh;
  }, [showMesh]);

  // Enumerate video devices safely
  const refreshDevices = useCallback(async () => {
    try {
      if (navigator.mediaDevices?.enumerateDevices) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter((d) => d.kind === 'videoinput');
        setAvailableDevices(videoInputs);
        if (videoInputs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoInputs[0].deviceId);
        }
      }
    } catch (e) {
      console.warn('Device enumeration warning:', e);
    }
  }, [selectedDeviceId]);

  // WebSocket Connection with Auto-reconnect & Orientation listener
  const connectWebSocket = useCallback(() => {
    if (!sessionId) return;

    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const wsUrl = `ws://127.0.0.1:8000/api/v1/verification/${sessionId}/stream`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      if (wsRef.current === ws) {
        setConnectionStatus('STREAMING');
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
      }
    };

    ws.onmessage = (event) => {
      if (wsRef.current !== ws) return;
      inFlightRef.current = false;
      try {
        const data = JSON.parse(event.data);
        if (data && !data.error) {
          setTelemetry(data);
          onTelemetryUpdate(data);

          // Auto-orientation detection from AI model: absolute assignment to avoid relative spin
          if (autoRotateRef.current && typeof data.detected_rotation === 'number' && data.detected_rotation !== 0) {
            console.log(`[AutoRotate] Server detected face orientation at ${data.detected_rotation}°. Setting orientation.`);
            updateRotationAngle(data.detected_rotation);
          }
        }
      } catch (e) {
        // Ignored
      }
    };

    ws.onclose = () => {
      inFlightRef.current = false;
      if (wsRef.current === ws) {
        setConnectionStatus('DISCONNECTED');
        if (!reconnectTimerRef.current) {
          reconnectTimerRef.current = setTimeout(() => {
            reconnectTimerRef.current = null;
            connectWebSocket();
          }, 2000);
        }
      }
    };

    ws.onerror = () => {
      inFlightRef.current = false;
      if (wsRef.current === ws) {
        setConnectionStatus('DISCONNECTED');
      }
    };
  }, [sessionId, onTelemetryUpdate, updateRotationAngle]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        const socket = wsRef.current;
        socket.onopen = null;
        socket.onmessage = null;
        socket.onclose = null;
        socket.onerror = null;
        socket.close();
        wsRef.current = null;
      }
    };
  }, [connectWebSocket]);

  // Start Camera Function with multi-tier fallback
  const startCamera = useCallback(async (forcedDeviceId?: string) => {
    if (isStartingRef.current) {
      return;
    }

    isStartingRef.current = true;
    setIsInitializing(true);
    setCameraError(null);
    setPermissionDenied(false);

    if (activeStreamRef.current) {
      try {
        activeStreamRef.current.getTracks().forEach((track) => track.stop());
      } catch (e) {
        console.warn('Error stopping previous stream tracks:', e);
      }
      activeStreamRef.current = null;
    }

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Webcam media devices API not supported on this browser context.');
      }

      const devId = forcedDeviceId || selectedDeviceId;
      let stream: MediaStream | null = null;
      let lastErr: any = null;

      try {
        const tier1Constraints: MediaStreamConstraints = {
          video: devId
            ? { deviceId: { exact: devId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false
        };
        stream = await navigator.mediaDevices.getUserMedia(tier1Constraints);
      } catch (err1: any) {
        lastErr = err1;
      }

      if (!stream) {
        try {
          const tier2Constraints: MediaStreamConstraints = {
            video: devId
              ? { deviceId: { exact: devId }, width: { ideal: 640 }, height: { ideal: 480 } }
              : { width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
          };
          stream = await navigator.mediaDevices.getUserMedia(tier2Constraints);
        } catch (err2: any) {
          lastErr = err2;
        }
      }

      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } catch (err3: any) {
          lastErr = err3;
        }
      }

      if (!stream) {
        throw lastErr || new Error('Unable to obtain webcam video stream.');
      }

      activeStreamRef.current = stream;

      const video = videoRef.current;
      if (video) {
        video.muted = true;
        video.playsInline = true;
        video.srcObject = stream;

        const handleCanPlay = async () => {
          try {
            await video.play();
            setCameraActive(true);
            onCameraActiveChange(true);
            setIsInitializing(false);
            isStartingRef.current = false;
          } catch (playErr: any) {
            if (playErr.name !== 'AbortError') {
              console.warn('video.play() rejected:', playErr);
              setCameraError(`Video playback was interrupted: ${playErr.message}`);
            }
            setIsInitializing(false);
            isStartingRef.current = false;
          }
        };

        if (video.readyState >= 2) {
          await handleCanPlay();
        } else {
          video.onloadeddata = handleCanPlay;
        }
      } else {
        setCameraActive(true);
        onCameraActiveChange(true);
        setIsInitializing(false);
        isStartingRef.current = false;
      }

      await refreshDevices();

    } catch (err: any) {
      console.error('Camera acquisition error:', err);
      setCameraActive(false);
      onCameraActiveChange(false);
      setIsInitializing(false);
      isStartingRef.current = false;

      let msg = err.message || 'Camera access denied or unavailable.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setPermissionDenied(true);
        msg = 'Camera permission was denied in your browser.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No physical webcam device detected. Please connect a webcam.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        msg = 'Camera is locked by another program (e.g. Zoom, Teams, Chrome tab). Close conflicting apps and retry.';
      }
      setCameraError(msg);
    }
  }, [selectedDeviceId, onCameraActiveChange, refreshDevices]);

  useEffect(() => {
    startCamera();
    return () => {
      if (activeStreamRef.current) {
        activeStreamRef.current.getTracks().forEach((t) => t.stop());
        activeStreamRef.current = null;
      }
    };
  }, []);

  const handleDeviceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newId = e.target.value;
    setSelectedDeviceId(newId);
    startCamera(newId);
  };

  const handleManualRotate = () => {
    updateRotationAngle((prev) => (prev + 90) % 360);
  };

  // Sync Canvas dimensions to Stage container size
  useEffect(() => {
    const targetElement = stageRef.current || containerRef.current;
    if (!targetElement || !canvasRef.current) return;

    const updateCanvasSize = () => {
      if (targetElement && canvasRef.current) {
        const rect = targetElement.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          canvasRef.current.width = rect.width;
          canvasRef.current.height = rect.height;
          setStageSize({ width: rect.width, height: rect.height });
        }
      }
    };

    const observer = new ResizeObserver(updateCanvasSize);
    observer.observe(targetElement);
    updateCanvasSize();

    return () => observer.disconnect();
  }, [isVertical]);

  // Frame Ingestion Loop: captures and pre-rotates frame at ~25 FPS to send over WebSocket
  // sentFrameDimsRef stores the actual (w, h) of the frame we sent — used for correct overlay scaling
  const sentFrameDimsRef = useRef<{ w: number; h: number }>({ w: 640, h: 360 });

  useEffect(() => {
    const offscreenCanvas = document.createElement('canvas');
    const offscreenCtx = offscreenCanvas.getContext('2d');
    let animationId: number;
    let lastSendTime = 0;

    const streamLoop = (time: number) => {
      // Release inFlight if stuck for > 100ms
      if (inFlightRef.current && time - lastSendTime >= 100) {
        inFlightRef.current = false;
      }

      // Stream frames at ~30 FPS (33ms intervals)
      if (!inFlightRef.current && time - lastSendTime >= 33) {
        lastSendTime = time;

        const ws = wsRef.current;
        const video = videoRef.current;

        if (
          ws &&
          ws.readyState === WebSocket.OPEN &&
          cameraActive &&
          video &&
          video.readyState >= 2 &&
          video.videoWidth > 0
        ) {
          const vw = video.videoWidth;
          const vh = video.videoHeight;
          const isRotated = rotationAngle === 90 || rotationAngle === 270;

          if (!isRotated) {
            // Horizontal (0° or 180°)
            const targetW = 640;
            const targetH = Math.round(640 * (vh / vw)) || 360;
            offscreenCanvas.width = targetW;
            offscreenCanvas.height = targetH;

            if (offscreenCtx) {
              offscreenCtx.save();
              if (mirrorFeed) {
                offscreenCtx.translate(targetW, 0);
                offscreenCtx.scale(-1, 1);
              }
              offscreenCtx.translate(targetW / 2, targetH / 2);
              if (rotationAngle !== 0) {
                offscreenCtx.rotate((rotationAngle * Math.PI) / 180);
              }
              offscreenCtx.drawImage(video, -targetW / 2, -targetH / 2, targetW, targetH);
              offscreenCtx.restore();
            }
            sentFrameDimsRef.current = { w: targetW, h: targetH };
          } else {
            // Vertical (90° or 270°)
            const targetH = 640;
            const targetW = Math.round(640 * (vh / vw)) || 360;
            offscreenCanvas.width = targetW;
            offscreenCanvas.height = targetH;

            if (offscreenCtx) {
              offscreenCtx.save();
              if (mirrorFeed) {
                offscreenCtx.translate(targetW, 0);
                offscreenCtx.scale(-1, 1);
              }
              offscreenCtx.translate(targetW / 2, targetH / 2);
              offscreenCtx.rotate((rotationAngle * Math.PI) / 180);
              offscreenCtx.drawImage(video, -targetH / 2, -targetW / 2, targetH, targetW);
              offscreenCtx.restore();
            }
            sentFrameDimsRef.current = { w: targetW, h: targetH };
          }

          inFlightRef.current = true;
          offscreenCanvas.toBlob(
            (blob) => {
              if (blob && ws.readyState === WebSocket.OPEN) {
                blob.arrayBuffer().then((buf) => {
                  ws.send(buf);
                  // Release immediately to maintain 30 FPS; backend Queue(maxsize=1) discards stale queued frames
                  inFlightRef.current = false;
                }).catch(() => {
                  inFlightRef.current = false;
                });
              } else {
                inFlightRef.current = false;
              }
            },
            'image/jpeg',
            0.72
          );
        }
      }

      // Render Overlay on main canvas
      // BUG-FIX: Use sentFrameDimsRef (the actual frame dimensions we sent to the server)
      // instead of telemetry.frame_width/frame_height (which are the server's post-rotation dims
      // and may be width/height swapped relative to what we sent, causing inverted scaleX/scaleY).
      if (canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const { w: sentW, h: sentH } = sentFrameDimsRef.current;
          rendererRef.current.render(ctx, canvas.width, canvas.height, sentW, sentH, telemetry);
        }
      }

      animationId = requestAnimationFrame(streamLoop);
    };

    animationId = requestAnimationFrame(streamLoop);
    return () => cancelAnimationFrame(animationId);
  }, [cameraActive, telemetry, rotationAngle, isVertical, mirrorFeed]);

  const vidW = videoRef.current?.videoWidth || 640;
  const vidH = videoRef.current?.videoHeight || 480;
  const stageAspectRatio = isVertical ? `${vidH} / ${vidW}` : `${vidW} / ${vidH}`;

  return (
    <div ref={containerRef} className="camera-viewport-card">
      {/* Dynamic Camera Stage (Vertical or Horizontal naturally matching webcam aspect) */}
      <div
        ref={stageRef}
        className={`camera-preview-stage ${isVertical ? 'vertical' : 'horizontal'}`}
        style={{ aspectRatio: stageAspectRatio }}
      >
        {/* 
          HTML5 Video Element:
          Positioned and scaled according to rotationAngle and aspect mode.
        */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="camera-video"
          style={
            !isVertical
              ? {
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  transform: `${mirrorFeed ? 'scaleX(-1)' : ''} rotate(${rotationAngle}deg)`,
                  opacity: cameraActive ? 1 : 0,
                  pointerEvents: 'none',
                  zIndex: 1,
                  transition: 'opacity 0.25s ease-in-out, transform 0.3s ease-out'
                }
              : {
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  width: `${stageSize.height}px`,
                  height: `${stageSize.width}px`,
                  transform: `translate(-50%, -50%) ${mirrorFeed ? 'scaleX(-1)' : ''} rotate(${rotationAngle}deg)`,
                  objectFit: 'cover',
                  opacity: cameraActive ? 1 : 0,
                  pointerEvents: 'none',
                  zIndex: 1,
                  transition: 'opacity 0.25s ease-in-out, transform 0.3s ease-out'
                }
          }
        />

        {/* Overlay Canvas */}
        <canvas
          ref={canvasRef}
          className="camera-canvas"
          style={{ zIndex: 10, pointerEvents: 'none' }}
        />
      </div>

      {/* Top Right Orientation Controls Toolbar */}
      <div className="viewport-orientation-controls">
        <div className="hud-chip">
          ASPECT: {isVertical ? 'VERTICAL (9:16)' : 'HORIZONTAL (16:9)'}
        </div>
        <button
          className={`hud-btn ${mirrorFeed ? 'active' : ''}`}
          onClick={toggleMirrorFeed}
          title="Toggle mirror reflection (selfie mirror vs natural camera)"
        >
          <span
            className="dot-indicator"
            style={{
              background: mirrorFeed ? 'var(--accent-cyan)' : 'var(--text-muted)'
            }}
          />
          MIRROR: {mirrorFeed ? 'ON' : 'OFF'}
        </button>
        <button
          className={`hud-btn ${autoRotate ? 'active' : ''}`}
          onClick={() => setAutoRotate((prev) => !prev)}
          title="Toggle automatic sensor orientation detection"
        >
          <span
            className="dot-indicator"
            style={{
              background: autoRotate ? 'var(--accent-cyan)' : 'var(--text-muted)'
            }}
          />
          AUTO-ROTATE: {autoRotate ? 'ON' : 'OFF'}
        </button>
        <button
          className="hud-btn"
          onClick={handleManualRotate}
          title="Manually cycle camera orientation by 90°"
        >
          <RotateCw size={11} />
          <span>{rotationAngle}°</span>
        </button>
      </div>

      {/* When camera is inactive or initializing, show explicit actionable UI */}
      {!cameraActive && (
        <div className="camera-unavailable-state" style={{ position: 'absolute', inset: 0, zIndex: 30, background: '#000' }}>
          <div className="camera-unavailable-icon">
            {isInitializing ? (
              <span className="loading-spinner-ring" />
            ) : (
              <Camera size={28} style={{ color: 'var(--text-muted)' }} />
            )}
          </div>

          <div className="camera-unavailable-title">
            {isInitializing ? 'INITIALIZING SENSOR FEED...' : 'CAMERA STREAM UNAVAILABLE'}
          </div>

          <div className="camera-unavailable-sub">
            {isInitializing
              ? 'Requesting webcam sensor hardware and initializing video stream pipeline...'
              : cameraError || 'No active RGB camera feed detected. Please allow camera permissions or connect a webcam.'}
          </div>

          {permissionDenied && (
            <div className="permission-troubleshoot-box">
              <div className="troubleshoot-header">HOW TO ENABLE CAMERA IN BROWSER:</div>
              <ol className="troubleshoot-steps">
                <li>Look at the top-left of your browser URL bar (next to <code>localhost:3000</code>).</li>
                <li>Click the <strong>Tune / Sliders</strong> or <strong>Lock</strong> icon.</li>
                <li>Toggle <strong>Camera</strong> permissions to <strong>Allow</strong>.</li>
                <li>Click <strong>CONNECT CAMERA SENSOR</strong> below to resume.</li>
              </ol>
            </div>
          )}

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
            {availableDevices.length > 1 && (
              <select
                value={selectedDeviceId}
                onChange={handleDeviceChange}
                className="doc-type-select"
                title="Select active camera input device"
              >
                {availableDevices.map((dev, idx) => (
                  <option key={dev.deviceId || idx} value={dev.deviceId}>
                    {dev.label || `Camera ${idx + 1}`}
                  </option>
                ))}
              </select>
            )}

            <button
              onClick={() => startCamera()}
              disabled={isInitializing}
              className="btn-primary"
              style={{ minWidth: '180px' }}
            >
              {isInitializing ? 'CONNECTING SENSOR...' : 'CONNECT CAMERA SENSOR'}
            </button>
          </div>
        </div>
      )}

      {/* Viewport Top Left Status Chips & Diagnostic Pipeline Trigger */}
      <div className="viewport-top-badge" style={{ zIndex: 28 }}>
        <div className={`hud-chip ${cameraActive ? 'active' : ''}`}>
          SENSOR: {cameraActive ? 'RGB LIVE' : isInitializing ? 'INITIALIZING' : 'DISCONNECTED'}
        </div>
        <div className={`hud-chip ${connectionStatus === 'STREAMING' ? 'active' : ''}`}>
          FEED: {connectionStatus}
        </div>
        {onOpenDebugView && (
          <button
            type="button"
            className="hud-btn"
            onClick={onOpenDebugView}
            style={{
              borderColor: 'var(--accent-cyan)',
              color: 'var(--accent-cyan)',
              background: 'rgba(0, 216, 246, 0.15)',
              fontWeight: 700,
              boxShadow: '0 0 10px rgba(0, 216, 246, 0.25)',
              cursor: 'pointer'
            }}
            title="Open 6-Stage Pipeline Real-Time Diagnostic View"
          >
            <Activity size={12} style={{ color: 'var(--accent-cyan)' }} />
            <span>PIPELINE AUDIT</span>
          </button>
        )}
      </div>

      {/* Upper-Right Periocular Biometric Cards */}
      <div className="viewport-periocular" style={{ zIndex: 25, top: '48px' }}>
        <PeriocularCards telemetry={telemetry} videoRef={videoRef} />
      </div>

      {/* Visual Direction Tracker HUD (Top Center) */}
      <DirectionTrackerHUD
        telemetry={telemetry}
        cameraActive={cameraActive}
      />

      {/* Camera Shutter Flash Effect on Auto-Capture */}
      {shutterFlash && (
        <div className="camera-shutter-flash" />
      )}

      {/* Bottom Center Guidance Pill */}
      <div className="viewport-guidance-bar" style={{ zIndex: 25 }}>
        {telemetry?.instruction ||
          (cameraActive
            ? 'AWAITING SUBJECT ACQUISITION'
            : isInitializing
            ? 'INITIALIZING SENSOR HARDWARE...'
            : 'AWAITING CAMERA SENSOR')}
      </div>
    </div>
  );
};
