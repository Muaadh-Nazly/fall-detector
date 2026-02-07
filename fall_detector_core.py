"""
Fall Detector Core Module
Contains all the core functions and classes for fall detection
Separated from fastapi for better organization and efficiency

COMPONENT:
1. Model Loading Functions: Load TFLite, YOLO, and MediaPipe models
2. Core Processing Functions: Image preprocessing and CNN inference
3. PSI Calculation: Pose Stability Index computation for multi-modal fusion
4. Detector Classes: Temporal smoothing, person tracking, and fall detection logic
5. Frame Processing: Complete frame-by-frame processing pipeline

KEY FEATURES:
- TFLite Inference: Optimized int8 quantized model for edge deployment
- Multi-modal Fusion: Combines CNN predictions with PSI (Pose Stability Index)
- Temporal Smoothing: Exponential Moving Average (EMA) for robust detection
- Person Tracking: IoU-based tracking for consistent person IDs across frames
- Image Capture: Per-person cooldown to prevent duplicate captures
"""

import cv2
import numpy as np
import time
from PIL import Image
import torch
import mediapipe as mp
import tensorflow as tf
import os
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from urllib.error import URLError
from ssl import SSLError


# CONFIGURATION CLASS


@dataclass
class FallDetectorConfig:
    """Configuration class for fall detector parameters"""

    tflite_path: str = "fall-detector-lite.tflite"
    yolo_pt: str = "yolov5s.pt"

    # Processing parameters
    frame_pref: Tuple[int, int] = (128, 128)
    yolo_conf_thr: float = 0.4
    yolo_input_size: int = 416

    # Fusion weights
    psi_weight: float = 0.35
    cnn_weight: float = 0.65

    # Temporal smoothing
    ema_alpha: float = 0.35

    # Detection thresholds
    frame_fall_threshold: float = 0.5
    required_consecutive_seconds: int = 3

    # MediaPipe settings
    mp_model_complexity: int = 1
    mp_min_detection_confidence: float = 0.5
    mp_min_tracking_confidence: float = 0.5

    # System settings
    save_dir: str = "fall_captures"
    webcam_index: int = 0
    target_fps: float = 40.0
    max_history_seconds: int = 10

    # Performance optimization settings
    yolo_skip_frames: int = 2
    pose_skip_frames: int = 1

    # PSI calculation parameters
    velocity_scale: float = 10.0
    head_hip_scale: float = 2.0
    fps_normalization: bool = True

    def __post_init__(self):
        """Validate and adjust parameters"""
        assert 0.0 <= self.psi_weight <= 1.0, "PSI weight must be in [0, 1]"
        assert 0.0 <= self.cnn_weight <= 1.0, "CNN weight must be in [0, 1]"
        assert (
            abs(self.psi_weight + self.cnn_weight - 1.0) < 1e-6
        ), "Weights must sum to 1.0"
        assert 0.0 < self.ema_alpha <= 1.0, "EMA alpha must be in (0, 1]"
        assert 0.0 <= self.frame_fall_threshold <= 1.0, "Threshold must be in [0, 1]"
        os.makedirs(self.save_dir, exist_ok=True)


# MODEL LOADING FUNCTIONS


def load_tflite_model(model_path: str):
    """Load and validate TFLite model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        assert len(input_details) > 0, "Model has no input"
        assert len(output_details) > 0, "Model has no output"

        return interpreter, input_details, output_details
    except Exception as e:
        raise RuntimeError(f"Failed to load TFLite model: {e}")


def load_yolo_model(model_path: str):
    """Load YOLOv5 model"""
    if not os.path.exists(model_path):
        print(
            f"Warning: YOLO model not found at {model_path}, attempting to download..."
        )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yolo = torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_path, force_reload=False
        ).to(device)
        yolo.classes = [0]  # person only
        return yolo, device
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv5 model: {e}")


def load_mediapipe_pose(config: FallDetectorConfig):
    """Load MediaPipe Pose detector"""
    mp_pose = mp.solutions.pose

    try:
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.mp_model_complexity,
            enable_segmentation=False,
            min_detection_confidence=config.mp_min_detection_confidence,
            min_tracking_confidence=config.mp_min_tracking_confidence,
        )
        return pose_detector, mp_pose
    except (URLError, SSLError) as e:
        # Fallback to complexity 1
        print("⚠️ SSL Certificate error. Falling back to model_complexity=1")
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=config.mp_min_detection_confidence,
            min_tracking_confidence=config.mp_min_tracking_confidence,
        )
        return pose_detector, mp_pose


# CORE PROCESSING FUNCTIONS


def classify_crop_tflite_np(
    crop_bgr: np.ndarray,
    interpreter,
    input_details,
    output_details,
    config: FallDetectorConfig,
    preserve_aspect: bool = True,
) -> Tuple[float, np.ndarray]:
    """Classify crop using TFLite model"""
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        if preserve_aspect:
            h, w = crop_rgb.shape[:2]
            target_h, target_w = config.frame_pref
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            resized = cv2.copyMakeBorder(
                resized,
                pad_h,
                target_h - new_h - pad_h,
                pad_w,
                target_w - new_w - pad_w,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
        else:
            resized = cv2.resize(crop_rgb, config.frame_pref)

        normalized = resized.astype(np.float32) / 255.0
        input_scale, input_zero_point = input_details[0]["quantization"]
        output_scale, output_zero_point = output_details[0]["quantization"]

        if input_scale != 0:
            quantized = normalized / input_scale + input_zero_point
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
        else:
            quantized = normalized

        input_data = np.expand_dims(quantized, axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        if output_scale != 0:
            output_data = (
                output_data.astype(np.float32) - output_zero_point
            ) * output_scale

        probs = tf.nn.softmax(output_data, axis=-1).numpy().ravel()
        fall_prob = float(probs[0])
        return fall_prob, probs
    except Exception as e:
        print(f"Error in classification: {e}")
        return 0.0, np.array([0.5, 0.5])


def calculate_psi(
    landmarks,
    image_shape,
    config: FallDetectorConfig,
    prev_state=None,
    fps: float = 25.0,
) -> Tuple[float, Dict]:
    """Compute Pose Stability Index (PSI) with FPS-normalized velocity"""
    h, w = image_shape[:2]

    def _get_xy(i):
        lm = landmarks[i]
        return np.array([lm.x * w, lm.y * h])

    try:
        ls = _get_xy(11)
        rs = _get_xy(12)
        lh = _get_xy(23)
        rh = _get_xy(24)
        nose = _get_xy(0)
    except (IndexError, AttributeError):
        return 0.0, prev_state if prev_state else {}

    shoulders_mid = (ls + rs) / 2.0
    hips_mid = (lh + rh) / 2.0

    torso_vec = shoulders_mid - hips_mid
    vertical = np.array([0.0, -1.0])
    torso_norm = np.linalg.norm(torso_vec) + 1e-6
    torso_unit = torso_vec / torso_norm
    cos_angle = np.dot(torso_unit, vertical)
    torso_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    torso_angle_deg = np.degrees(torso_angle)

    head_hip_dist = np.linalg.norm(nose - hips_mid) / max(h, w)
    hip_y = hips_mid[1] / h

    hip_vel = 0.0
    v_score = 0.0
    if prev_state is not None and "hip_y" in prev_state:
        delta_y = hip_y - prev_state["hip_y"]
        if config.fps_normalization:
            hip_vel = delta_y * fps
            v_score = np.clip(hip_vel * (config.velocity_scale / fps), 0.0, 1.0)
        else:
            hip_vel = delta_y
            v_score = np.clip(hip_vel * config.velocity_scale, 0.0, 1.0)

    t_score = np.clip(torso_angle_deg / 90.0, 0.0, 1.0)
    hh_score = 1.0 - np.clip(head_hip_dist * config.head_hip_scale, 0.0, 1.0)

    psi_raw = 0.5 * t_score + 0.35 * v_score + 0.15 * hh_score
    psi = float(np.clip(psi_raw, 0.0, 1.0))

    state = {
        "hip_y": hip_y,
        "torso_angle_deg": torso_angle_deg,
        "timestamp": time.time(),
    }

    return psi, state


# DETECTOR CLASSES


class TemporalSmoother:
    """Exponential Moving Average smoother"""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema = None

    def update(self, new_score: float) -> float:
        if self.ema is None:
            self.ema = new_score
        else:
            self.ema = self.alpha * new_score + (1 - self.alpha) * self.ema
        return self.ema

    def reset(self):
        self.ema = None


class SimplePersonTracker:
    """Simple person tracker using bounding box IoU"""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["last_seen"] += 1
                if self.tracks[track_id]["last_seen"] > self.max_age:
                    del self.tracks[track_id]
            return []

        matched = set()
        for det in detections:
            best_iou = 0
            best_track_id = None

            for track_id, track in self.tracks.items():
                if track_id in matched:
                    continue
                iou = self._calculate_iou(det["bbox"], track["bbox"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                det["track_id"] = best_track_id
                self.tracks[best_track_id]["bbox"] = det["bbox"]
                self.tracks[best_track_id]["last_seen"] = 0
                matched.add(best_track_id)
            else:
                track_id = self.next_id
                self.next_id += 1
                det["track_id"] = track_id
                self.tracks[track_id] = {
                    "bbox": det["bbox"],
                    "last_seen": 0,
                    "state": {},
                }

        for track_id in list(self.tracks.keys()):
            if track_id not in matched:
                self.tracks[track_id]["last_seen"] += 1
                if self.tracks[track_id]["last_seen"] > self.max_age:
                    del self.tracks[track_id]

        return detections

    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class FallDetector:
    def __init__(self, config: FallDetectorConfig):
        self.config = config
        self.cnn_w = config.cnn_weight
        self.psi_w = config.psi_weight
        self.required_consecutive = config.required_consecutive_seconds
        self.fps = config.target_fps
        self.frame_fall_threshold = config.frame_fall_threshold

        self.person_smoothers = {}
        self.person_states = {}
        self.person_fall_seconds = {}
        self.fall_seconds = deque(maxlen=config.max_history_seconds)
        self.last_saved_second = -100
        self.tracker = SimplePersonTracker()

        # For frame skipping
        self.person_pose_counters = {}
        self.last_psi_scores = {}
        self.cached_landmarks = {}

        # Track per-person image saves
        self.person_save_count = {}  # track_id -> count of saved images
        self.person_last_save_time = {}  # track_id -> timestamp of last save
        self.max_saves_per_person = 3  # Maximum images to save per person
        self.cooldown_seconds = 12 * 60  # 12 minutes cooldown

    def frame_decision(
        self, track_id: int, cnn_prob: float, psi_score: float, timestamp_seconds: float
    ) -> Dict:

        if track_id not in self.person_smoothers:
            self.person_smoothers[track_id] = TemporalSmoother(self.config.ema_alpha)
            self.person_fall_seconds[track_id] = deque(
                maxlen=self.config.max_history_seconds
            )

        fused_prob = cnn_prob * self.cnn_w + psi_score * self.psi_w
        smoothed = self.person_smoothers[track_id].update(fused_prob)
        frame_fall = smoothed >= self.frame_fall_threshold

        sec = int(timestamp_seconds)

        if (
            len(self.person_fall_seconds[track_id]) == 0
            or self.person_fall_seconds[track_id][-1][0] != sec
        ):
            self.person_fall_seconds[track_id].append((sec, frame_fall))

        if len(self.fall_seconds) == 0 or self.fall_seconds[-1][0] != sec:
            self.fall_seconds.append((sec, frame_fall))
        else:
            self.fall_seconds[-1] = (sec, self.fall_seconds[-1][1] or frame_fall)

        sec_ok = False
        if len(self.fall_seconds) >= self.required_consecutive:
            recent_seconds = list(self.fall_seconds)[-self.required_consecutive :]
            if all(sec_data[1] for sec_data in recent_seconds):
                if (sec - self.last_saved_second) >= self.required_consecutive:
                    sec_ok = True
                    self.last_saved_second = sec

        return {
            "fused_prob": float(fused_prob),
            "smoothed_prob": float(smoothed),
            "frame_fall": bool(frame_fall),
            "confirmed_by_seconds": sec_ok,
        }

    def should_save_for_person(self, track_id: int, current_time: float) -> bool:
        """
        Check if we should save an image for this person.
        Returns True if:
        - We haven't saved max_saves_per_person images yet, OR
        - Cooldown period has passed since last save
        """
        if track_id not in self.person_save_count:
            self.person_save_count[track_id] = 0
            self.person_last_save_time[track_id] = 0.0

        if self.person_save_count[track_id] < self.max_saves_per_person:
            return True

        # If cooldown has passed, reset counter and allow saving
        time_since_last = current_time - self.person_last_save_time[track_id]
        if time_since_last >= self.cooldown_seconds:
            self.person_save_count[track_id] = 0
            return True

        return False

    def record_person_save(self, track_id: int, current_time: float):
        """Record that we saved an image for this person"""
        if track_id not in self.person_save_count:
            self.person_save_count[track_id] = 0
        self.person_save_count[track_id] += 1
        self.person_last_save_time[track_id] = current_time

    def cleanup_old_tracks(self, current_second: int):
        max_age_seconds = 10
        tracks_to_remove = []
        for track_id in self.person_smoothers.keys():
            if (
                track_id in self.person_fall_seconds
                and len(self.person_fall_seconds[track_id]) > 0
            ):
                last_sec = self.person_fall_seconds[track_id][-1][0]
                if current_second - last_sec > max_age_seconds:
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.person_smoothers[track_id]
            del self.person_fall_seconds[track_id]
            if track_id in self.person_states:
                del self.person_states[track_id]
            # Also cleanup save tracking for very old tracks
            if track_id in self.person_last_save_time:
                current_time_float = float(current_second)
                time_since_last = (
                    current_time_float - self.person_last_save_time[track_id]
                )
                if time_since_last > self.cooldown_seconds * 2:
                    if track_id in self.person_save_count:
                        del self.person_save_count[track_id]
                    del self.person_last_save_time[track_id]


# FRAME PROCESSING FUNCTION


def process_frame_for_display(
    frame_bgr: np.ndarray,
    timestamp_seconds: float,
    interpreter,
    input_details,
    output_details,
    yolo,
    pose_detector,
    mp_pose,
    detector: FallDetector,
    config: FallDetectorConfig,
    fps: float = None,
    frame_counter: int = 0,
    last_detections: pd.DataFrame = None,
) -> Tuple:
    """Process frame with fall detection"""

    if fps is None:
        fps = config.target_fps

    if last_detections is None:
        last_detections = pd.DataFrame(
            columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
        )

    try:
        h, w = frame_bgr.shape[:2]

        # Frame skipping for YOLO
        run_yolo = frame_counter % (config.yolo_skip_frames + 1) == 0

        if run_yolo:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            results = yolo(pil, size=config.yolo_input_size)
            df = results.pandas().xyxy[0]
            last_detections = (
                df
                if len(df) > 0
                else pd.DataFrame(
                    columns=[
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "confidence",
                        "class",
                        "name",
                    ]
                )
            )
        else:
            df = (
                last_detections
                if isinstance(last_detections, pd.DataFrame)
                else pd.DataFrame(
                    columns=[
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "confidence",
                        "class",
                        "name",
                    ]
                )
            )

        annotated = frame_bgr.copy()
        highest_fused_prob = 0.0
        fall_flag_any = False
        psi_val_display = 0.0
        cnn_prob_display = 0.0

        detections = []
        for _, row in df.iterrows():
            if row["confidence"] < config.yolo_conf_thr:
                continue
            x1, y1, x2, y2 = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )
            detections.append({"bbox": (x1, y1, x2, y2), "row": row})

        tracked_detections = detector.tracker.update(detections)

        for det in tracked_detections:
            track_id = det["track_id"]
            row = det["row"]
            x1, y1, x2, y2 = det["bbox"]

            crop = frame_bgr[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            cnn_prob, _ = classify_crop_tflite_np(
                crop,
                interpreter,
                input_details,
                output_details,
                config,
                preserve_aspect=True,
            )
            cnn_prob_display = max(cnn_prob_display, cnn_prob)

            # Pose skipping
            if track_id not in detector.person_pose_counters:
                detector.person_pose_counters[track_id] = 0
            detector.person_pose_counters[track_id] += 1
            run_pose = (
                detector.person_pose_counters[track_id] % (config.pose_skip_frames + 1)
                == 0
            )

            padding = 20
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(w, x2 + padding)
            crop_y2 = min(h, y2 + padding)

            person_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

            psi_score = 0.0
            landmarks_pixel = None

            if person_crop.size > 0 and run_pose:
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                mp_res = pose_detector.process(person_crop_rgb)

                if mp_res.pose_landmarks:
                    crop_h, crop_w = person_crop.shape[:2]
                    landmarks_full_frame = []
                    landmarks_pixel = []

                    for lm in mp_res.pose_landmarks.landmark:
                        x_crop = lm.x * crop_w
                        y_crop = lm.y * crop_h
                        x_full = x_crop + crop_x1
                        y_full = y_crop + crop_y1

                        class Landmark:
                            def __init__(self, x, y, z, visibility):
                                self.x = x
                                self.y = y
                                self.z = z
                                self.visibility = visibility

                        lm_full = Landmark(
                            x=x_full / w, y=y_full / h, z=lm.z, visibility=lm.visibility
                        )
                        landmarks_full_frame.append(lm_full)
                        landmarks_pixel.append(
                            (int(x_full), int(y_full), lm.visibility)
                        )

                    prev_state = detector.person_states.get(track_id)
                    psi_score, new_state = calculate_psi(
                        landmarks_full_frame,
                        frame_bgr.shape,
                        config,
                        prev_state,
                        fps=fps,
                    )
                    detector.person_states[track_id] = new_state
                    psi_val_display = max(psi_val_display, psi_score)

                    # Draw pose
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        if (
                            start_idx < len(landmarks_pixel)
                            and end_idx < len(landmarks_pixel)
                            and landmarks_pixel[start_idx][2] > 0.5
                            and landmarks_pixel[end_idx][2] > 0.5
                        ):
                            start_point = (
                                landmarks_pixel[start_idx][0],
                                landmarks_pixel[start_idx][1],
                            )
                            end_point = (
                                landmarks_pixel[end_idx][0],
                                landmarks_pixel[end_idx][1],
                            )
                            cv2.line(annotated, start_point, end_point, (0, 200, 0), 1)

                    for x, y, visibility in landmarks_pixel:
                        if visibility > 0.5:
                            cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

                    detector.last_psi_scores[track_id] = psi_score
                    detector.cached_landmarks[track_id] = landmarks_pixel

            elif not run_pose and track_id in detector.person_states:
                psi_score = detector.last_psi_scores.get(track_id, 0.0)
                psi_val_display = max(psi_val_display, psi_score)
                cached_landmarks = detector.cached_landmarks.get(track_id)
                if cached_landmarks:
                    landmarks_pixel = cached_landmarks
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        if (
                            start_idx < len(landmarks_pixel)
                            and end_idx < len(landmarks_pixel)
                            and landmarks_pixel[start_idx][2] > 0.5
                            and landmarks_pixel[end_idx][2] > 0.5
                        ):
                            start_point = (
                                landmarks_pixel[start_idx][0],
                                landmarks_pixel[start_idx][1],
                            )
                            end_point = (
                                landmarks_pixel[end_idx][0],
                                landmarks_pixel[end_idx][1],
                            )
                            cv2.line(annotated, start_point, end_point, (0, 200, 0), 1)
                    for x, y, visibility in landmarks_pixel:
                        if visibility > 0.5:
                            cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

            out = detector.frame_decision(
                track_id, cnn_prob, psi_score, timestamp_seconds
            )
            fused = out["fused_prob"]
            smoothed = out["smoothed_prob"]
            frame_fall = out["frame_fall"]
            confirmed_seconds = out["confirmed_by_seconds"]

            highest_fused_prob = max(highest_fused_prob, fused)
            if frame_fall:
                fall_flag_any = True

            color = (0, 0, 255) if frame_fall else (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} F:{cnn_prob:.2f} P:{psi_score:.2f} S:{smoothed:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(10, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

            if confirmed_seconds:
                # Check whether to save for this person
                current_time = time.time()
                if detector.should_save_for_person(track_id, current_time):
                    try:
                        tstr = time.strftime("%m-%d_%H-%M-%S", time.localtime())
                        fname = os.path.join(
                            config.save_dir, f"fall_capture_{tstr}_person{track_id}.jpg"
                        )
                        cv2.imwrite(fname, annotated)
                        detector.record_person_save(track_id, current_time)
                        saves_remaining = (
                            detector.max_saves_per_person
                            - detector.person_save_count[track_id]
                        )
                        print(
                            f"💾 Saved fall capture for person {track_id}: {fname} ({saves_remaining} saves remaining before cooldown)"
                        )
                        return (
                            annotated,
                            highest_fused_prob,
                            psi_val_display,
                            cnn_prob_display,
                            fall_flag_any,
                            last_detections,
                            fname,
                        )
                    except Exception as e:
                        print(f"Error saving capture: {e}")
                else:
                    # Person is in cooldown period
                    time_until_cooldown = detector.cooldown_seconds - (
                        current_time - detector.person_last_save_time[track_id]
                    )
                    minutes_left = int(time_until_cooldown / 60)
                    if minutes_left > 0:
                        print(
                            f"⏸️  Person {track_id} in cooldown ({minutes_left}m remaining). Skipping save."
                        )

        detector.cleanup_old_tracks(int(timestamp_seconds))

        panel_text = f"Fused:{highest_fused_prob:.2f} | CNN:{cnn_prob_display:.2f} | PSI:{psi_val_display:.2f} | Fall:{fall_flag_any}"
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            panel_text,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return (
            annotated_rgb,
            highest_fused_prob,
            psi_val_display,
            cnn_prob_display,
            fall_flag_any,
            last_detections,
            None,
        )

    except Exception as e:
        print(f"Error in process_frame_for_display: {e}")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            f"Error: {str(e)}",
            (10, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        empty_df = pd.DataFrame(
            columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
        )
        return (
            cv2.cvtColor(blank, cv2.COLOR_BGR2RGB),
            0.0,
            0.0,
            0.0,
            False,
            empty_df,
            None,
        )
