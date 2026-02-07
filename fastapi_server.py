"""
FastAPI Server for Fall Detection System
Real-time video streaming with WebSockets

COMPONENT:
1. FastAPI app initialization
2. API endpoints:
3. WebSocket endpoints:
4. Video processing loop
5. HTML frontend

KEY FEATURES:
- Real-time video streaming via WebSocket (/ws/video)
- Live dashboard updates via WebSocket (/ws/dashboard)
- RESTful API endpoints for control and status
- Interactive web interface with video feed, metrics, and alerts
- Smart image saving with per-person cooldown to prevent duplicates
- Edge-optimized inference using TFLite model

"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
import time
import base64
import threading
import asyncio
import os
from fall_detector_core import (
    FallDetectorConfig,
    load_tflite_model,
    load_yolo_model,
    load_mediapipe_pose,
    process_frame_for_display,
    FallDetector,
)

app = FastAPI(title="Fall Detection System")

# Configuration
config = FallDetectorConfig()

print("Loading models...")
interpreter, input_details, output_details = load_tflite_model(config.tflite_path)
yolo, device = load_yolo_model(config.yolo_pt)
pose_detector, mp_pose = load_mediapipe_pose(config)
detector = FallDetector(config)
print("Models loaded successfully")

# Global state for video processing
video_state = {
    "running": False,
    "cap": None,
    "frame_count": 0,
    "start_time": None,
    "fps_start_time": None,
    "current_fps": config.target_fps,
    "last_detections": None,
    "alerts": [],
    "total_frames": 0,
    "total_falls_detected": 0,
    "last_fall_state": False,
    "session_start_time": None,
    "latest_frame": None,
    "processing_thread": None,
    "latest_metrics": {
        "fps": 0.0,
        "fused_prob": 0.0,
        "cnn_prob": 0.0,
        "psi_val": 0.0,
        "fall_flag": False,
    },
}

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_index():
    """Serve the main HTML page"""
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return HTMLResponse(
            """
        <html>
            <body>
                <h1>Fall Detection System</h1>
                <p>Please create static/index.html file</p>
            </body>
        </html>
        """
        )


@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return {
        "running": video_state["running"],
        "fps": video_state["latest_metrics"]["fps"],
        "total_frames": video_state["total_frames"],
        "total_falls": video_state["total_falls_detected"],
        "alerts_count": len(video_state["alerts"]),
    }


@app.get("/api/alerts")
async def get_alerts():
    """Get recent alerts"""
    return {"alerts": video_state["alerts"][-10:]}


@app.get("/api/captures")
async def get_captures():
    """Get list of saved captures"""
    if not os.path.exists(config.save_dir):
        return {"captures": []}

    captures = sorted(
        [f for f in os.listdir(config.save_dir) if f.endswith(".jpg")],
        reverse=True,
    )[:20]

    return {"captures": captures}


@app.get("/api/image/{filename:path}")
async def get_image(filename: str):
    """Serve saved fall detection images"""
    # Extract just the filename
    filename_only = os.path.basename(filename)
    # Remove any directory components for security
    filename_only = filename_only.replace("/", "").replace("\\", "")

    image_path = os.path.join(config.save_dir, filename_only)

    image_path = os.path.normpath(image_path)
    save_dir_norm = os.path.normpath(config.save_dir)
    if not image_path.startswith(save_dir_norm):
        return {"error": "Invalid path"}, 403

    if os.path.exists(image_path) and filename_only.endswith((".jpg", ".jpeg", ".png")):
        return FileResponse(image_path)
    else:
        return {"error": "Image not found"}, 404


class StartRequest(BaseModel):
    webcam_index: int = 0


@app.post("/api/start")
async def start_detection(request: StartRequest):
    """Start video detection"""
    if video_state["running"]:
        return {"status": "already_running"}

    webcam_index = request.webcam_index

    # Stop any existing thread
    if (
        video_state.get("processing_thread") is not None
        and video_state["processing_thread"].is_alive()
    ):
        video_state["running"] = False
        video_state["processing_thread"].join(timeout=1.0)

    # Open webcam
    print(f"Opening webcam {webcam_index}...")
    video_state["cap"] = cv2.VideoCapture(int(webcam_index))

    # Set camera properties for better performance
    video_state["cap"].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_state["cap"].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_state["cap"].set(cv2.CAP_PROP_FPS, 30)

    if not video_state["cap"].isOpened():
        return {"status": "error", "message": f"Could not open webcam {webcam_index}"}

    print(f"Webcam {webcam_index} opened successfully")

    # Initialize state
    video_state["running"] = True
    video_state["start_time"] = time.time()
    video_state["fps_start_time"] = time.time()
    video_state["session_start_time"] = time.time()
    video_state["frame_count"] = 0
    video_state["last_detections"] = None
    video_state["total_frames"] = 0
    video_state["total_falls_detected"] = 0
    video_state["last_fall_state"] = False
    video_state["latest_frame"] = None

    # Start processing thread immediately
    print("Starting video processing thread...")
    video_state["processing_thread"] = threading.Thread(
        target=process_video_loop, name="video_processor", daemon=True
    )
    video_state["processing_thread"].start()

    print(f"Detection started with webcam {webcam_index}")
    return {"status": "started", "webcam_index": webcam_index}


@app.post("/api/stop")
async def stop_detection():
    """Stop video detection"""
    video_state["running"] = False

    # Wait for thread to finish
    if (
        video_state["processing_thread"] is not None
        and video_state["processing_thread"].is_alive()
    ):
        video_state["processing_thread"].join(timeout=2.0)

    if video_state["cap"] is not None:
        video_state["cap"].release()
        video_state["cap"] = None

    video_state["processing_thread"] = None

    # Reset dashboard values and stats to zero
    video_state["latest_metrics"] = {
        "fps": 0.0,
        "fused_prob": 0.0,
        "cnn_prob": 0.0,
        "psi_val": 0.0,
        "fall_flag": False,
    }
    video_state["total_frames"] = 0
    video_state["total_falls_detected"] = 0
    video_state["frame_count"] = 0
    video_state["alerts"] = []
    video_state["last_fall_state"] = False
    video_state["current_fps"] = 0.0
    video_state["latest_frame"] = None

    print("Detection stopped and stats reset")
    return {"status": "stopped"}


@app.post("/api/reset")
async def reset_statistics():
    """Reset statistics"""
    video_state["total_falls_detected"] = 0
    video_state["last_fall_state"] = False
    video_state["alerts"] = []
    video_state["session_start_time"] = time.time()
    video_state["frame_count"] = 0
    video_state["start_time"] = time.time()
    video_state["total_frames"] = 0

    return {"status": "reset"}


def process_video_loop():
    """Background thread for processing video frames"""
    print("Video processing thread started")
    frame_count = 0

    while video_state["running"]:
        if video_state["cap"] is None or not video_state["cap"].isOpened():
            print("❌ Camera not available, stopping thread")
            video_state["running"] = False
            break

        ret, frame = video_state["cap"].read()
        if not ret:
            print("❌ Failed to read frame, stopping thread")
            video_state["running"] = False
            break

        frame_count += 1

        # Resize frame for consistent display
        frame = cv2.resize(frame, (1280, 720))

        timestamp = time.time() - video_state["start_time"]
        video_state["frame_count"] += 1
        video_state["total_frames"] += 1

        # Calculate FPS
        if video_state["frame_count"] % 30 == 0:
            elapsed = time.time() - video_state["fps_start_time"]
            if elapsed > 0:
                video_state["current_fps"] = 30.0 / elapsed
            video_state["fps_start_time"] = time.time()
        else:
            elapsed = time.time() - video_state["fps_start_time"]
            if elapsed > 0:
                video_state["current_fps"] = video_state["frame_count"] / elapsed

        # Process frame
        result = process_frame_for_display(
            frame,
            timestamp,
            interpreter,
            input_details,
            output_details,
            yolo,
            pose_detector,
            mp_pose,
            detector,
            config,
            fps=video_state["current_fps"],
            frame_counter=video_state["frame_count"],
            last_detections=video_state["last_detections"],
        )

        (
            annotated_frame,
            fused_prob,
            psi_val,
            cnn_prob,
            fall_flag,
            new_detections,
            saved_file,
        ) = result

        # Update state
        video_state["last_detections"] = new_detections

        # Track falls - Event-based counting
        # Only increment on new fall eventf
        current_fall_state = fall_flag
        if current_fall_state and not video_state["last_fall_state"]:
            video_state["total_falls_detected"] += 1
            print(f"Fall event #{video_state['total_falls_detected']} detected")
        video_state["last_fall_state"] = current_fall_state

        # Handle saved file (alert)
        if saved_file:
            # Extract the filename for alert
            filename_only = os.path.basename(saved_file)
            video_state["alerts"].append(
                {
                    "timestamp": time.time(),
                    "filename": filename_only,
                    "fused_prob": fused_prob,
                    "psi_score": psi_val,
                    "cnn_prob": cnn_prob,
                }
            )
            print(f"ALERT: Fall detected and saved to {saved_file}")

        # Convert RGB to BGR for encoding
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Update latest frame and metrics
        video_state["latest_frame"] = annotated_frame_bgr
        video_state["latest_metrics"] = {
            "fps": video_state["current_fps"],
            "fused_prob": fused_prob,
            "cnn_prob": cnn_prob,
            "psi_val": psi_val,
            "fall_flag": fall_flag,
        }

        # Control FPS
        time.sleep(1.0 / config.target_fps)


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    await websocket.accept()
    print("Video WebSocket client connected")

    try:
        while True:
            if not video_state["running"]:
                # Send black frame when not running
                black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    black_frame,
                    "Click 'Start Detection' to begin",
                    (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                _, buffer = cv2.imencode(
                    ".jpg", black_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                frame_bytes = base64.b64encode(buffer).decode("utf-8")
                await websocket.send_json(
                    {
                        "type": "frame",
                        "data": frame_bytes,
                    }
                )
                await asyncio.sleep(0.1)
                continue

            # Send latest frame
            if video_state["latest_frame"] is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode(
                    ".jpg", video_state["latest_frame"], [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                frame_bytes = base64.b64encode(buffer).decode("utf-8")

                await websocket.send_json(
                    {
                        "type": "frame",
                        "data": frame_bytes,
                    }
                )
            else:
                # Send placeholder while waiting for first frame
                placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Initializing camera...",
                    (450, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                _, buffer = cv2.imencode(
                    ".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                frame_bytes = base64.b64encode(buffer).decode("utf-8")
                await websocket.send_json(
                    {
                        "type": "frame",
                        "data": frame_bytes,
                    }
                )

            await asyncio.sleep(0.033)  # ~30 FPS

    except WebSocketDisconnect:
        print("Video WebSocket client disconnected")
    except Exception as e:
        print(f"Video WebSocket error: {e}")
        import traceback

        traceback.print_exc()


@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for dashboard updates"""
    await websocket.accept()
    print("Dashboard WebSocket client connected")

    try:
        while True:
            # Send dashboard data
            await websocket.send_json(
                {
                    "type": "dashboard",
                    "metrics": video_state["latest_metrics"],
                    "stats": {
                        "total_frames": video_state["total_frames"],
                        "total_falls": video_state["total_falls_detected"],
                        "alerts_count": len(video_state["alerts"]),
                        "session_time": (
                            time.time() - video_state["session_start_time"]
                            if video_state["session_start_time"]
                            else 0
                        ),
                    },
                    "alerts": video_state["alerts"][-10:],
                }
            )

            await asyncio.sleep(0.1)  # Update 10 times per second

    except WebSocketDisconnect:
        print("Dashboard WebSocket client disconnected")
    except Exception as e:
        print(f"Dashboard WebSocket error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Fall Detection System - FastAPI Server")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("Open http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop")

    uvicorn.run(app, host="0.0.0.0", port=8000)
