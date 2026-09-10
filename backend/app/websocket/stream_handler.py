"""
Real-Time Video Stream WebSocket Handler.
Handles streaming video frames over WebSocket:
- Accepts binary image bytes (JPEG / WebP) or JSON frame messages.
- Decodes image buffer.
- Runs asynchronous pipeline through VerificationSession.
- Sends back real-time telemetry: Bbox, Pose, Landmarks, Quality, Liveness, Capture guidance.
"""

from fastapi import WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import json
import base64
import time

from app.routes.verification import session_manager


import asyncio

async def handle_verification_stream(websocket: WebSocket, session_id: str):
    print(f"WS: incoming connection for session {session_id}", flush=True)
    await websocket.accept()
    session = session_manager.get_session(session_id)
    if session is None:
        print(f"WS: auto-creating session {session_id}", flush=True)
        session = session_manager.create_session(session_id=session_id)
    print(f"WS: session ready for {session_id}", flush=True)

    # Latest-frame queue with maxsize=1 drops stale queued frames so the pipeline
    # always executes on the newest physical webcam frame with negligible queue delay.
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def receive_loop():
        try:
            while True:
                message = await websocket.receive()
                frame_bgr = None
                if "bytes" in message and message["bytes"]:
                    nparr = np.frombuffer(message["bytes"], np.uint8)
                    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                elif "text" in message and message["text"]:
                    try:
                        data = json.loads(message["text"])
                        if "image" in data:
                            raw_b64 = data["image"]
                            if "," in raw_b64:
                                raw_b64 = raw_b64.split(",")[1]
                            img_bytes = base64.b64decode(raw_b64)
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    except Exception:
                        continue

                if frame_bgr is not None:
                    # Drop stale frame if worker hasn't consumed it yet
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_queue.put(frame_bgr)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            print(f"WS receive_loop error: {e}", flush=True)

    async def process_loop():
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 30.0

        try:
            while True:
                frame_bgr = await frame_queue.get()
                fps_counter += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = round(fps_counter / elapsed, 1)
                    fps_counter = 0
                    fps_start_time = time.time()

                try:
                    telemetry = session.process_frame(frame_bgr)
                    telemetry["fps"] = current_fps
                    if fps_counter % 15 == 0:
                        print(f"[STREAM] det={telemetry.get('detection_status')} ms={telemetry.get('latency_ms')} fps={current_fps}", flush=True)
                    await websocket.send_json(telemetry)
                except Exception as frame_err:
                    import traceback
                    print(f"WS frame processing error: {frame_err}\n{traceback.format_exc()}", flush=True)
                    continue
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            print(f"WS process_loop error: {e}", flush=True)

    recv_task = asyncio.create_task(receive_loop())
    proc_task = asyncio.create_task(process_loop())

    try:
        done, pending = await asyncio.wait(
            [recv_task, proc_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    finally:
        recv_task.cancel()
        proc_task.cancel()
        print(f"WS: stream connection closed for session {session_id}", flush=True)
