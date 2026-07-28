#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Iris Recognition System web app for Jetson Nano.

Python 3.6 compatible.
"""

import argparse
import atexit
import logging
import math
import os
import signal
import socket
import subprocess
import threading
import time
import traceback
import uuid
import glob

try:
    from flask import Flask, request, jsonify, render_template, Response, send_from_directory
except Exception as e:
    print("ERROR: Flask is not installed.")
    print("Install it on Jetson with:")
    print("  sudo apt update")
    print("  sudo apt install python3-flask -y")
    raise e

import database
from iris_backend import IrisBackend, meta_to_json


APP_TITLE = "Iris Recognition System"
DEFAULT_DB = os.path.join("data", "iris_users_web.json")
DEFAULT_THRESHOLD = 0.45
DEFAULT_MARGIN = 0.05
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
# Demo-friendly static assets: avoid stale CSS/JS after quick Jetson updates.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

ARGS = None
BACKEND = None
CAMERA = None
CAMERA_ANALYZER = None
CAMERA_CAPTURES = {"photo": None, "query": None, "left": None, "right": None}


def shutdown_camera():
    try:
        if CAMERA is not None:
            CAMERA.release()
    except Exception:
        pass


def handle_shutdown(signum, frame):
    shutdown_camera()
    raise SystemExit(0)


def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def elapsed_ms(start):
    return int((time.time() - start) * 1000.0)


def get_local_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.getcwd(), path))


def upload_dir():
    path = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def safe_filename(name):
    name = os.path.basename(name or "upload.jpg")
    out = []
    for ch in name:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out)
    return name or "upload.jpg"


def save_upload(file_obj, prefix):
    if file_obj is None:
        raise RuntimeError("No file uploaded")
    filename = prefix + "_" + uuid.uuid4().hex[:12] + "_" + safe_filename(file_obj.filename)
    path = os.path.join(upload_dir(), filename)
    file_obj.save(path)
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        raise RuntimeError("Uploaded file is empty")
    return path


def jetson_csi_pipeline(sensor_id, width, height, framerate):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1"
    ) % (int(sensor_id), int(width), int(height), int(framerate))


class CameraManager(object):
    def __init__(self):
        self.source = "csi"
        self.width = 1280
        self.height = 720
        self.framerate = 21
        self.preview_width = 1280
        self.preview_height = 720
        self.preview_fps = 21
        self.preview_sensor_fps = 21
        self.preview_quality = 70
        self.preview_dir = os.path.join("/tmp", "iris_web_preview")
        self.preview_proc = None
        self.preview_devnull = None
        self.cap = None
        self.lock = threading.Lock()
        self.last_error = ""
        self.frame_bytes = None
        self.frame_timestamp = 0.0
        self.frame_sequence = 0
        self.frame_path = None
        self.frame_reads = 0
        self.frame_served = 0
        self.frame_races = 0
        self.frame_failures = 0
        self.frame_times = []

    def configure(self, source, width, height, framerate,
                  preview_width=1280, preview_height=720, preview_fps=21,
                  preview_sensor_fps=21):
        self.source = str(source or "csi")
        self.width = int(width or 1280)
        self.height = int(height or 720)
        self.framerate = int(framerate or 21)
        self.preview_width = int(preview_width or 1280)
        self.preview_height = int(preview_height or 720)
        self.preview_fps = int(preview_fps or 21)
        self.preview_sensor_fps = int(preview_sensor_fps or 21)

    def source_label(self):
        return self.source

    def is_csi_source(self):
        low = self.source.strip().lower()
        return low in ("csi", "csi0", "imx219", "nvargus") or low.startswith("csi:")

    def csi_sensor_id(self):
        low = self.source.strip().lower()
        if low.startswith("csi:"):
            try:
                return int(low.split(":", 1)[1])
            except Exception:
                return 0
        return 0

    def cv2(self):
        import cv2
        return cv2

    def source_value(self):
        source = self.source.strip()
        low = source.lower()
        if low in ("csi", "csi0", "imx219", "nvargus"):
            return jetson_csi_pipeline(0, self.width, self.height, self.framerate)
        if low.startswith("csi:"):
            try:
                sensor_id = int(low.split(":", 1)[1])
            except Exception:
                sensor_id = 0
            return jetson_csi_pipeline(sensor_id, self.width, self.height, self.framerate)
        if source.isdigit():
            return int(source)
        return source

    def open_locked(self):
        if self.is_csi_source():
            return

        cv2 = self.cv2()
        if self.cap is not None and self.cap.isOpened():
            return

        source = self.source_value()
        if isinstance(source, str) and "!" in source:
            cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(source)

        if not (cap is not None and cap.isOpened()):
            self.last_error = "Could not open camera source " + self.source
            try:
                cap.release()
            except Exception:
                pass
            raise RuntimeError(self.last_error)

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap = cap
        self.last_error = ""

    def release(self):
        with self.lock:
            self.release_locked()

    def release_locked(self):
        self.stop_preview_locked()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.frame_bytes = None
        self.frame_timestamp = 0.0
        self.frame_path = None

    def cleanup_preview_files_locked(self):
        if not os.path.exists(self.preview_dir):
            os.makedirs(self.preview_dir)
        for path in glob.glob(os.path.join(self.preview_dir, "frame_*.jpg")):
            try:
                os.remove(path)
            except Exception:
                pass

    def preview_command(self):
        source_caps = (
            "video/x-raw(memory:NVMM), width=%d, height=%d, "
            "framerate=%d/1, format=NV12"
        ) % (int(self.preview_width), int(self.preview_height), int(self.preview_sensor_fps))
        out_caps = "video/x-raw, framerate=%d/1" % int(self.preview_fps)
        return [
            "gst-launch-1.0",
            "-q",
            "nvarguscamerasrc",
            "sensor-id=%d" % int(self.csi_sensor_id()),
            "!",
            source_caps,
            "!",
            "nvvidconv",
            "!",
            "video/x-raw, format=I420",
            "!",
            "videorate",
            "!",
            out_caps,
            "!",
            "jpegenc",
            "quality=%d" % int(self.preview_quality),
            "!",
            "multifilesink",
            "location=" + os.path.join(self.preview_dir, "frame_%05d.jpg"),
            "max-files=20"
        ]

    def start_preview_locked(self):
        if not self.is_csi_source():
            return
        if self.preview_proc is not None and self.preview_proc.poll() is None:
            return

        self.stop_preview_locked()
        self.cleanup_preview_files_locked()
        self.preview_devnull = open(os.devnull, "w")
        try:
            self.preview_proc = subprocess.Popen(
                self.preview_command(),
                stdout=self.preview_devnull,
                stderr=self.preview_devnull
            )
        except Exception as e:
            self.stop_preview_locked()
            self.last_error = "Could not start CSI preview: " + str(e)
            raise RuntimeError(self.last_error)

    def stop_preview_locked(self):
        proc = self.preview_proc
        self.preview_proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self.preview_devnull is not None:
            try:
                self.preview_devnull.close()
            except Exception:
                pass
        self.preview_devnull = None

    def latest_preview_file_locked(self):
        deadline = time.time() + 3.0
        while time.time() < deadline:
            files = [
                path for path in glob.glob(os.path.join(self.preview_dir, "frame_*.jpg"))
                if os.path.exists(path) and os.path.getsize(path) > 0
            ]
            if files:
                files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                now = time.time()
                for path in files:
                    if now - os.path.getmtime(path) >= 0.06:
                        return path
                return files[0]
            if self.preview_proc is not None and self.preview_proc.poll() is not None:
                self.last_error = "CSI preview exited before producing a frame"
                raise RuntimeError(self.last_error)
            time.sleep(0.05)
        self.last_error = "Timed out waiting for CSI preview frame"
        raise RuntimeError(self.last_error)

    def remember_frame_locked(self, path, data, modified_at):
        if path != self.frame_path:
            now = time.time()
            self.frame_sequence += 1
            self.frame_path = path
            self.frame_timestamp = float(modified_at or now)
            self.frame_times.append(now)
            cutoff = now - 3.0
            self.frame_times = [value for value in self.frame_times if value >= cutoff]
        self.frame_bytes = data
        self.frame_reads += 1
        self.last_error = ""

    def read_preview_frame_locked(self):
        """Read a completed rolling JPEG without exposing multifilesink races.

        GStreamer's multifilesink deletes old frame files while the web server is
        serving requests. A filename returned by glob can therefore disappear
        before open(). Keep the last verified JPEG in memory and retry the small
        race window instead of returning a broken image to the browser.
        """
        startup = self.frame_bytes is None
        deadline = time.time() + (3.0 if startup else 0.18)
        last_exception = None

        while time.time() < deadline:
            files = glob.glob(os.path.join(self.preview_dir, "frame_*.jpg"))
            candidates = []
            for path in files:
                try:
                    modified_at = os.path.getmtime(path)
                    size = os.path.getsize(path)
                except OSError as e:
                    self.frame_races += 1
                    last_exception = e
                    continue
                if size > 1024:
                    candidates.append((modified_at, path))
            candidates.sort(reverse=True)

            # Skip the actively-written newest file when another completed file
            # exists. This also prevents partial JPEG decodes on slower storage.
            now = time.time()
            for modified_at, path in candidates:
                if now - modified_at < 0.025 and len(candidates) > 1:
                    continue
                try:
                    with open(path, "rb") as frame_file:
                        data = frame_file.read()
                except OSError as e:
                    self.frame_races += 1
                    last_exception = e
                    continue
                if len(data) < 1024 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                    last_exception = RuntimeError("CSI preview produced an incomplete JPEG")
                    continue
                self.remember_frame_locked(path, data, modified_at)
                return data

            if self.preview_proc is not None and self.preview_proc.poll() is not None:
                self.frame_failures += 1
                self.last_error = "CSI preview process stopped unexpectedly"
                raise RuntimeError(self.last_error)
            if self.frame_bytes is not None:
                # Serving the last verified frame is preferable to a visible
                # flash while a rolling file is replaced.
                return self.frame_bytes
            time.sleep(0.015)

        if self.frame_bytes is not None:
            return self.frame_bytes
        self.frame_failures += 1
        if last_exception is not None:
            self.last_error = "Could not read CSI preview frame: " + str(last_exception)
        else:
            self.last_error = "Timed out waiting for CSI preview frame"
        raise RuntimeError(self.last_error)

    def csi_jpeg_command(self, path):
        caps = (
            "video/x-raw(memory:NVMM), width=%d, height=%d, "
            "framerate=%d/1, format=NV12"
        ) % (int(self.width), int(self.height), int(self.framerate))
        return [
            "gst-launch-1.0",
            "-q",
            "nvarguscamerasrc",
            "sensor-id=%d" % int(self.csi_sensor_id()),
            "num-buffers=1",
            "!",
            caps,
            "!",
            "nvvidconv",
            "!",
            "video/x-raw(memory:NVMM), format=NV12",
            "!",
            "nvjpegenc",
            "!",
            "filesink",
            "location=" + path
        ]

    def save_csi_snapshot_locked(self, prefix):
        self.stop_preview_locked()
        time.sleep(0.15)
        filename = prefix + "_" + uuid.uuid4().hex[:12] + ".jpg"
        path = os.path.join(upload_dir(), filename)
        try:
            with open(os.devnull, "w") as devnull:
                code = subprocess.call(
                    self.csi_jpeg_command(path),
                    stdout=devnull,
                    stderr=devnull,
                    timeout=10
                )
        except TypeError:
            with open(os.devnull, "w") as devnull:
                proc = subprocess.Popen(
                    self.csi_jpeg_command(path),
                    stdout=devnull,
                    stderr=devnull
                )
                proc.wait()
                code = proc.returncode
        except Exception as e:
            self.last_error = "CSI capture failed: " + str(e)
            raise RuntimeError(self.last_error)

        if code != 0 or not os.path.exists(path) or os.path.getsize(path) <= 0:
            self.last_error = "CSI capture failed for source " + self.source
            raise RuntimeError(self.last_error)

        self.last_error = ""
        return path, (int(self.height), int(self.width), 3)

    def save_csi_snapshot(self, prefix):
        with self.lock:
            return self.save_csi_snapshot_locked(prefix)

    def snapshot(self):
        with self.lock:
            self.open_locked()
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok = False
                frame = None
            if not ok or frame is None:
                self.release_locked()
                self.open_locked()
                try:
                    ok, frame = self.cap.read()
                except Exception:
                    ok = False
                    frame = None
            if not ok or frame is None or getattr(frame, "size", 0) <= 0:
                self.last_error = "Could not read frame from camera"
                raise RuntimeError(self.last_error)
            return frame

    def frame_jpeg(self):
        if self.is_csi_source():
            with self.lock:
                self.start_preview_locked()
                data = self.read_preview_frame_locked()
                self.frame_served += 1
            return data, (int(self.preview_height), int(self.preview_width), 3)

        cv2 = self.cv2()
        frame = self.snapshot()
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok:
            raise RuntimeError("Could not encode camera frame")
        return encoded.tobytes(), tuple(frame.shape)

    def current_frame(self):
        import cv2
        import numpy as np

        data, _ = self.frame_jpeg()
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or getattr(frame, "size", 0) <= 0:
            raise RuntimeError("Could not decode current camera frame")
        return frame, data

    def save_current_frame(self, prefix, rotation=0):
        import cv2

        frame, data = self.current_frame()
        rotation = int(rotation or 0) % 360
        if rotation:
            frame = rotate_frame(frame, rotation)
            data = None

        filename = prefix + "_" + uuid.uuid4().hex[:12] + ".jpg"
        path = os.path.join(upload_dir(), filename)
        if data is not None:
            with open(path, "wb") as output:
                output.write(data)
        else:
            ok = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                raise RuntimeError("Could not save current camera frame")
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            raise RuntimeError("Saved camera frame is empty")
        return path, tuple(frame.shape)

    def save_snapshot(self, prefix):
        if self.is_csi_source():
            return self.save_csi_snapshot(prefix)

        cv2 = self.cv2()
        frame = self.snapshot()
        filename = prefix + "_" + uuid.uuid4().hex[:12] + ".jpg"
        path = os.path.join(upload_dir(), filename)
        ok = cv2.imwrite(path, frame)
        if not ok or not os.path.exists(path):
            raise RuntimeError("Could not save camera frame")
        return path, tuple(frame.shape)

    def status(self):
        opened = False
        try:
            opened = self.cap is not None and self.cap.isOpened()
        except Exception:
            opened = False
        preview_running = False
        try:
            preview_running = self.preview_proc is not None and self.preview_proc.poll() is None
        except Exception:
            preview_running = False
        measured_fps = 0.0
        if len(self.frame_times) >= 2:
            duration = self.frame_times[-1] - self.frame_times[0]
            if duration > 0:
                measured_fps = float(len(self.frame_times) - 1) / duration
        frame_age_ms = None
        if self.frame_timestamp:
            frame_age_ms = max(0, int((time.time() - self.frame_timestamp) * 1000.0))
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "framerate": self.framerate,
            "preview_width": self.preview_width,
            "preview_height": self.preview_height,
            "preview_fps": self.preview_fps,
            "preview_running": preview_running,
            "mode": "persistent-gst-preview" if self.is_csi_source() else "opencv",
            "opened": opened,
            "last_error": self.last_error,
            "frame_sequence": self.frame_sequence,
            "frame_age_ms": frame_age_ms,
            "measured_fps": round(measured_fps, 2),
            "frames_read": self.frame_reads,
            "frames_served": self.frame_served,
            "file_races_recovered": self.frame_races,
            "frame_failures": self.frame_failures
        }


def rotate_frame(frame, rotation):
    """Rotate an OpenCV frame using operations available in OpenCV 3.2."""
    import cv2

    rotation = int(rotation or 0) % 360
    if rotation == 90:
        return cv2.flip(cv2.transpose(frame), 1)
    if rotation == 180:
        return cv2.flip(frame, -1)
    if rotation == 270:
        return cv2.flip(cv2.transpose(frame), 0)
    return frame


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


class CameraQualityAnalyzer(object):
    """Low-cost capture guidance using the OpenCV data already on the Jetson."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending_event = threading.Event()
        self.pending_analysis = None
        self.worker_thread = None
        self.face_detector = None
        self.eye_detector = None
        self.detector_error = ""
        self.preferred_rotation = 0
        self.last_orientation_search = 0.0
        self.motion_reference = None
        self.last_result = None
        self.last_result_time = 0.0

    def ensure_worker(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        self.worker_thread = threading.Thread(target=self.worker_loop, name="iris-quality-worker")
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def request_analysis(self, frame, requested_rotation="auto", quality_threshold=72.0):
        self.ensure_worker()
        with self.pending_lock:
            # Keep only the newest frame; stale analysis is not useful for motion
            # guidance and must never build an unbounded work queue.
            self.pending_analysis = (frame.copy(), requested_rotation, quality_threshold)
        self.pending_event.set()
        return self.last_result

    def request_jpeg_analysis(self, jpeg_data, requested_rotation="auto", quality_threshold=72.0):
        self.ensure_worker()
        with self.pending_lock:
            self.pending_analysis = (bytes(jpeg_data), requested_rotation, quality_threshold)
        self.pending_event.set()
        return self.last_result

    def worker_loop(self):
        while True:
            self.pending_event.wait()
            self.pending_event.clear()
            with self.pending_lock:
                pending = self.pending_analysis
                self.pending_analysis = None
            if pending is None:
                continue
            frame, requested_rotation, quality_threshold = pending
            try:
                if isinstance(frame, bytes):
                    import cv2
                    import numpy as np
                    frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None or getattr(frame, "size", 0) <= 0:
                        raise RuntimeError("Could not decode camera frame for quality analysis")
                self.analyze(frame, requested_rotation, quality_threshold)
            except Exception as e:
                self.detector_error = str(e)

    def load_detectors(self):
        if self.face_detector is not None and self.eye_detector is not None:
            return
        try:
            import cv2
            roots = [
                "/usr/share/opencv4/haarcascades",
                "/usr/share/opencv/haarcascades",
                "/usr/local/share/opencv4/haarcascades"
            ]
            root = None
            for candidate in roots:
                if os.path.exists(os.path.join(candidate, "haarcascade_frontalface_default.xml")):
                    root = candidate
                    break
            if root is None:
                raise RuntimeError("OpenCV Haar cascade data was not found")
            self.face_detector = cv2.CascadeClassifier(
                os.path.join(root, "haarcascade_frontalface_default.xml")
            )
            eye_path = os.path.join(root, "haarcascade_eye_tree_eyeglasses.xml")
            if not os.path.exists(eye_path):
                eye_path = os.path.join(root, "haarcascade_eye.xml")
            self.eye_detector = cv2.CascadeClassifier(eye_path)
            if self.face_detector.empty() or self.eye_detector.empty():
                raise RuntimeError("OpenCV could not load Haar cascade data")
            self.detector_error = ""
        except Exception as e:
            self.detector_error = str(e)
            self.face_detector = None
            self.eye_detector = None

    def detect_face_and_eyes(self, gray):
        import cv2

        height, width = gray.shape[:2]
        scale = min(1.0, 640.0 / float(max(width, height)))
        if scale < 1.0:
            small = cv2.resize(gray, (int(round(width * scale)), int(round(height * scale))))
        else:
            small = gray
        equalized = cv2.equalizeHist(small)
        inv = 1.0 / scale

        faces = []
        eyes = []
        if self.face_detector is not None:
            detected = self.face_detector.detectMultiScale(
                equalized,
                scaleFactor=1.12,
                minNeighbors=4,
                minSize=(70, 70)
            )
            faces = [tuple(int(round(value * inv)) for value in item) for item in detected]

        face = max(faces, key=lambda item: item[2] * item[3]) if faces else None
        if self.eye_detector is not None:
            if face is not None:
                fx, fy, fw, fh = face
                top_height = max(1, int(fh * 0.62))
                roi = gray[fy:fy + top_height, fx:fx + fw]
                roi_scale = min(1.0, 480.0 / float(max(fw, top_height)))
                if roi_scale < 1.0:
                    roi_small = cv2.resize(
                        roi,
                        (int(round(fw * roi_scale)), int(round(top_height * roi_scale)))
                    )
                else:
                    roi_small = roi
                detected_eyes = self.eye_detector.detectMultiScale(
                    cv2.equalizeHist(roi_small),
                    scaleFactor=1.10,
                    minNeighbors=4,
                    minSize=(25, 16)
                )
                eye_inv = 1.0 / roi_scale
                for ex, ey, ew, eh in detected_eyes:
                    eyes.append((
                        fx + int(round(ex * eye_inv)),
                        fy + int(round(ey * eye_inv)),
                        int(round(ew * eye_inv)),
                        int(round(eh * eye_inv))
                    ))
            else:
                detected_eyes = self.eye_detector.detectMultiScale(
                    equalized,
                    scaleFactor=1.10,
                    minNeighbors=5,
                    minSize=(35, 22)
                )
                eyes = [tuple(int(round(value * inv)) for value in item) for item in detected_eyes]

        # Reject implausible boxes and prefer the largest eye near the guide.
        plausible = []
        for eye in eyes:
            ex, ey, ew, eh = eye
            aspect = float(ew) / float(max(1, eh))
            if 0.75 <= aspect <= 4.2 and ew >= 28 and eh >= 16:
                center_dx = ((ex + ew / 2.0) - width / 2.0) / float(max(1, width))
                center_dy = ((ey + eh / 2.0) - height / 2.0) / float(max(1, height))
                centrality = math.sqrt(center_dx * center_dx + center_dy * center_dy)
                rank = (ew * eh) * (1.2 - min(0.7, centrality))
                plausible.append((rank, eye))
        plausible.sort(reverse=True)
        selected_eye = plausible[0][1] if plausible else None
        return face, [item[1] for item in plausible], selected_eye

    def iris_evidence(self, eye_gray):
        import cv2
        import numpy as np

        if eye_gray is None or getattr(eye_gray, "size", 0) <= 0:
            return 0.0, None
        height, width = eye_gray.shape[:2]
        if width < 24 or height < 16:
            return 0.0, None
        blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
        best_confidence = 0.0
        best_circle = None

        try:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(8, height // 3),
                param1=70,
                param2=max(12, int(min(width, height) * 0.16)),
                minRadius=max(3, int(height * 0.10)),
                maxRadius=max(5, int(height * 0.48))
            )
            if circles is not None:
                for cx, cy, radius in circles[0]:
                    centrality = math.sqrt(
                        ((float(cx) - width / 2.0) / max(1.0, width / 2.0)) ** 2 +
                        ((float(cy) - height / 2.0) / max(1.0, height / 2.0)) ** 2
                    )
                    confidence = 0.86 - min(0.45, centrality * 0.35)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_circle = (int(round(cx)), int(round(cy)), int(round(radius)))
        except Exception:
            pass

        # A dark, circular pupil is useful evidence when Hough is conservative.
        try:
            threshold = int(np.percentile(blurred, 20))
            _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
            contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contour_result[-2]
            eye_area = float(width * height)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < eye_area * 0.004 or area > eye_area * 0.35:
                    continue
                perimeter = float(cv2.arcLength(contour, True))
                if perimeter <= 0:
                    continue
                circularity = clamp01((4.0 * math.pi * area) / (perimeter * perimeter))
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                edge_margin = min(cx, cy, width - cx, height - cy)
                if edge_margin < radius * 0.55:
                    continue
                confidence = 0.24 + 0.48 * circularity
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_circle = (int(round(cx)), int(round(cy)), int(round(radius)))
        except Exception:
            pass
        return clamp01(best_confidence), best_circle

    def orientation_candidates(self, requested_rotation):
        if str(requested_rotation).lower() != "auto":
            try:
                return [int(requested_rotation) % 360]
            except Exception:
                return [0]
        now = time.time()
        if self.last_result is not None and now - self.last_orientation_search < 4.0:
            return [self.preferred_rotation]
        self.last_orientation_search = now
        values = [self.preferred_rotation, 0, 90, 270, 180]
        output = []
        for value in values:
            if value not in output:
                output.append(value)
        return output

    def analyze(self, frame, requested_rotation="auto", quality_threshold=72.0):
        import cv2

        with self.lock:
            self.load_detectors()
            best = None
            for rotation in self.orientation_candidates(requested_rotation):
                oriented = rotate_frame(frame, rotation)
                gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
                face, eyes, eye = self.detect_face_and_eyes(gray)
                rank = (1000000 if eye is not None else 0) + (100000 if face is not None else 0)
                if eye is not None:
                    rank += eye[2] * eye[3]
                candidate = (rank, rotation, oriented, gray, face, eyes, eye)
                if best is None or candidate[0] > best[0]:
                    best = candidate
                if eye is not None and face is not None:
                    break

            _, rotation, oriented, gray, face, eyes, eye = best
            if eye is not None or face is not None:
                self.preferred_rotation = rotation
            height, width = gray.shape[:2]

            brightness_region = gray
            eye_box = None
            crop_box = None
            iris_circle = None
            iris_confidence = 0.0
            eye_ratio = 0.0
            center_score = 0.0
            if eye is not None:
                ex, ey, ew, eh = eye
                ex = max(0, min(int(ex), width - 1))
                ey = max(0, min(int(ey), height - 1))
                ew = max(1, min(int(ew), width - ex))
                eh = max(1, min(int(eh), height - ey))
                eye_box = {"x": ex, "y": ey, "width": ew, "height": eh}
                brightness_region = gray[ey:ey + eh, ex:ex + ew]
                iris_confidence, local_circle = self.iris_evidence(brightness_region)
                if local_circle is not None:
                    iris_circle = {
                        "x": ex + local_circle[0],
                        "y": ey + local_circle[1],
                        "radius": local_circle[2]
                    }
                eye_ratio = float(ew) / float(max(1, width))
                dx = ((ex + ew / 2.0) - width / 2.0) / float(max(1, width / 2.0))
                dy = ((ey + eh / 2.0) - height / 2.0) / float(max(1, height / 2.0))
                center_score = clamp01(1.0 - math.sqrt(dx * dx + dy * dy))
                pad_x = int(round(ew * 0.38))
                pad_y = int(round(eh * 0.55))
                cx = max(0, ex - pad_x)
                cy = max(0, ey - pad_y)
                cw = min(width - cx, ew + pad_x * 2)
                ch = min(height - cy, eh + pad_y * 2)
                crop_box = {"x": cx, "y": cy, "width": cw, "height": ch}

            brightness = float(brightness_region.mean())
            contrast = float(brightness_region.std())
            sharpness = float(cv2.Laplacian(brightness_region, cv2.CV_64F).var())
            motion_frame = cv2.resize(gray, (160, 100))
            if self.motion_reference is None or self.motion_reference.shape != motion_frame.shape:
                stability = 0.50
            else:
                difference = float(cv2.absdiff(motion_frame, self.motion_reference).mean())
                stability = clamp01(1.0 - difference / 22.0)
            self.motion_reference = motion_frame

            if brightness < 55.0:
                brightness_score = clamp01(brightness / 55.0)
            elif brightness > 205.0:
                brightness_score = clamp01((255.0 - brightness) / 50.0)
            else:
                brightness_score = 1.0
            # Laplacian variance depends strongly on sensor resolution. Values
            # around 18-25 are already usable for this NoIR module, so avoid a
            # desktop-camera scale that would make the quality score unreachable.
            sharpness_score = clamp01((sharpness - 5.0) / 35.0)
            contrast_score = clamp01((contrast - 5.0) / 25.0)
            size_score = clamp01(eye_ratio / 0.18)
            if eye_ratio > 0.62:
                size_score = clamp01((0.82 - eye_ratio) / 0.20)
            detection_score = 0.0
            if face is not None:
                detection_score += 0.12
            if eye is not None:
                detection_score += 0.55
            detection_score += 0.33 * iris_confidence

            score = 100.0 * (
                0.22 * sharpness_score +
                0.15 * brightness_score +
                0.08 * contrast_score +
                0.25 * detection_score +
                0.15 * size_score +
                0.10 * stability +
                0.05 * center_score
            )
            score = round(max(0.0, min(100.0, score)), 1)

            ready = bool(
                eye is not None and
                iris_confidence >= 0.22 and
                eye_ratio >= 0.075 and
                brightness >= 32.0 and brightness <= 232.0 and
                sharpness >= 18.0 and
                stability >= 0.62 and
                score >= float(quality_threshold)
            )
            if eye is None:
                guidance = "Center one open eye inside the guide"
                state = "searching"
            elif eye_ratio < 0.075:
                guidance = "Move closer until the iris fills the guide"
                state = "adjust"
            elif eye_ratio > 0.68:
                guidance = "Move slightly farther away"
                state = "adjust"
            elif center_score < 0.48:
                guidance = "Move the eye into the center guide"
                state = "adjust"
            elif brightness < 32.0:
                guidance = "Too dark — add soft light near the camera"
                state = "warning"
            elif brightness > 232.0:
                guidance = "Too bright — reduce direct light"
                state = "warning"
            elif sharpness < 18.0:
                guidance = "Image is blurry — adjust distance and hold still"
                state = "warning"
            elif iris_confidence < 0.22:
                guidance = "Open the eye wider and reduce glare"
                state = "adjust"
            elif stability < 0.62:
                guidance = "Hold still for automatic capture"
                state = "steady"
            elif score < float(quality_threshold):
                guidance = "Almost ready — keep the eye centered and still"
                state = "steady"
            else:
                guidance = "Good capture — hold still"
                state = "good"

            result = {
                "score": score,
                "ready": ready,
                "state": state,
                "guidance": guidance,
                "rotation": rotation,
                "frame": {"width": width, "height": height},
                "face_box": (
                    {"x": face[0], "y": face[1], "width": face[2], "height": face[3]}
                    if face is not None else None
                ),
                "eye_box": eye_box,
                "crop_box": crop_box,
                "iris_circle": iris_circle,
                "eyes_detected": len(eyes),
                "detector_error": self.detector_error or None,
                "metrics": {
                    "brightness": round(brightness, 1),
                    "sharpness": round(sharpness, 1),
                    "contrast": round(contrast, 1),
                    "iris_confidence": round(iris_confidence, 3),
                    "eye_size": round(eye_ratio, 3),
                    "motion_stability": round(stability, 3),
                    "center_score": round(center_score, 3)
                },
                "quality_threshold": float(quality_threshold),
                "captured_at": now_text()
            }
            self.last_result = result
            self.last_result_time = time.time()
            return result, oriented


def parse_bool(value, default=False):
    if value is None:
        return bool(default)
    low = str(value).strip().lower()
    if low in ("1", "true", "yes", "on", "y"):
        return True
    if low in ("0", "false", "no", "off", "n"):
        return False
    return bool(default)


def parse_threshold(value):
    try:
        threshold = float(value)
    except Exception:
        threshold = float(ARGS.threshold)
    if threshold < 0.0:
        threshold = 0.0
    if threshold > 1.0:
        threshold = 1.0
    return threshold


def parse_margin(value):
    try:
        margin = float(value)
    except Exception:
        margin = DEFAULT_MARGIN
    if margin < 0.0:
        margin = 0.0
    if margin > 0.50:
        margin = 0.50
    return margin


def parse_eye(value):
    low = str(value or "").strip().lower()
    if low in ("l", "left"):
        return "left"
    if low in ("r", "right"):
        return "right"
    return None


def parse_camera_target(value):
    low = str(value or "query").strip().lower()
    if low in ("left", "right", "query"):
        return low
    return "query"


def parse_quality_threshold(value):
    try:
        threshold = float(value)
    except Exception:
        threshold = 72.0
    return max(50.0, min(95.0, threshold))


def recognition_options(default_eye=None):
    query_eye = parse_eye(request.form.get("query_eye") or default_eye)
    same_side_only = parse_bool(request.form.get("same_side_only"), False)
    margin_enabled = parse_bool(request.form.get("margin_enabled"), False)
    margin = parse_margin(request.form.get("margin"))
    eye_filter = query_eye if same_side_only else None
    return query_eye, same_side_only, margin_enabled, margin, eye_filter


def image_stats(path):
    stats = {}
    try:
        import cv2
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return stats
        stats["brightness"] = float(gray.mean())
        stats["contrast"] = float(gray.std())
        stats["focus"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception as e:
        stats["stats_error"] = str(e)
    return stats


def assess_camera_capture(path):
    meta = BACKEND.assess_image(path)
    stats = image_stats(path)
    warnings = []
    brightness = stats.get("brightness")
    focus = stats.get("focus")
    if brightness is not None and brightness < 12.0:
        warnings.append("Frame is very dark")
    if brightness is not None and brightness > 245.0:
        warnings.append("Frame is overexposed")
    if focus is not None and focus < 3.0:
        warnings.append("Frame may be blurry")
    return meta, stats, warnings


def capture_record(path, meta, stats, warnings):
    return {
        "path": path,
        "filename": os.path.basename(path),
        "captured_at": now_text(),
        "meta": meta,
        "stats": stats,
        "warnings": warnings
    }


def photo_record(path, shape):
    height = int(shape[0]) if len(shape) > 0 else 0
    width = int(shape[1]) if len(shape) > 1 else 0
    return {
        "path": path,
        "filename": os.path.basename(path),
        "captured_at": now_text(),
        "width": width,
        "height": height
    }


def image_url(filename):
    return "/api/camera/image/" + safe_filename(filename)


def write_camera_image(prefix, image, quality=92):
    import cv2

    filename = prefix + "_" + uuid.uuid4().hex[:12] + ".jpg"
    path = os.path.join(upload_dir(), filename)
    ok = cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok or not os.path.exists(path) or os.path.getsize(path) <= 0:
        raise RuntimeError("Could not save camera image")
    return path


def create_guided_capture(target, requested_rotation="auto", quality_threshold=72.0):
    """Quality-gate and save an eye crop while retaining the unprocessed frame."""
    import cv2

    frame, raw_jpeg = CAMERA.current_frame()
    analysis, oriented = CAMERA_ANALYZER.analyze(
        frame,
        requested_rotation=requested_rotation,
        quality_threshold=quality_threshold
    )
    if not analysis.get("ready"):
        return analysis, None

    crop_box = analysis.get("crop_box")
    if not crop_box:
        analysis["ready"] = False
        analysis["guidance"] = "Iris crop was not detected — try again"
        return analysis, None

    raw_path = None
    crop_path = None
    processed_path = None
    try:
        raw_filename = "camera_raw_" + target + "_" + uuid.uuid4().hex[:12] + ".jpg"
        raw_path = os.path.join(upload_dir(), raw_filename)
        with open(raw_path, "wb") as output:
            output.write(raw_jpeg)

        x = int(crop_box["x"])
        y = int(crop_box["y"])
        w = int(crop_box["width"])
        h = int(crop_box["height"])
        crop = oriented[y:y + h, x:x + w]
        if crop is None or getattr(crop, "size", 0) <= 0:
            raise RuntimeError("Detected iris crop was empty")
        crop_path = write_camera_image("camera_" + target, crop, 94)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed_path = write_camera_image("camera_processed_" + target, enhanced, 94)

        try:
            meta, stats, warnings = assess_camera_capture(crop_path)
        except Exception as assess_error:
            analysis["ready"] = False
            analysis["guidance"] = "Recognition check failed: " + str(assess_error)
            for path in (raw_path, crop_path, processed_path):
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            return analysis, None

        rec = capture_record(crop_path, meta, stats, warnings)
        rec["quality"] = analysis
        rec["raw_filename"] = os.path.basename(raw_path)
        rec["processed_filename"] = os.path.basename(processed_path)
        CAMERA_CAPTURES[target] = rec
        return analysis, rec
    except Exception:
        for path in (raw_path, crop_path, processed_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        raise


def crop_camera_photo(photo, crop_x, crop_y, crop_w, crop_h, target):
    import cv2

    image = cv2.imread(photo["path"], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not read captured photo")

    height, width = image.shape[:2]
    x = int(round(float(crop_x)))
    y = int(round(float(crop_y)))
    w = int(round(float(crop_w)))
    h = int(round(float(crop_h)))

    if w < 40 or h < 40:
        raise RuntimeError("Crop box is too small")

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        raise RuntimeError("Crop is empty")

    filename = "camera_crop_" + target + "_" + uuid.uuid4().hex[:12] + ".jpg"
    path = os.path.join(upload_dir(), filename)
    ok = cv2.imwrite(path, crop)
    if not ok or not os.path.exists(path):
        raise RuntimeError("Could not save crop")

    center_x = x + (w / 2.0)
    auto_eye = "left" if center_x < (width / 2.0) else "right"
    return path, (h, w, 3), auto_eye, {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "photo_width": width,
        "photo_height": height,
        "center_x": center_x
    }


@app.route("/", methods=["GET"])
def index():
    summary = database.db_summary(ARGS.db)
    ip = get_local_ip()
    return render_template(
        "index.html",
        app_title=APP_TITLE,
        default_threshold="{0:.2f}".format(float(ARGS.threshold)),
        default_margin="{0:.2f}".format(float(DEFAULT_MARGIN)),
        server_port=ARGS.port,
        initial_status="Online",
        initial_user_count=summary["count"],
        initial_server="%s:%d" % (ip, ARGS.port)
    )


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return Response(status=204)


@app.route("/api/health", methods=["GET"])
def api_health():
    summary = database.db_summary(ARGS.db)
    return jsonify({
        "ok": True,
        "app": APP_TITLE,
        "time": now_text(),
        "ip": get_local_ip(),
        "port": ARGS.port,
        "db_count": summary["count"],
        "db_users": summary["users"],
        "engine": ARGS.engine,
        "meta": ARGS.meta,
        "db": ARGS.db,
        "threshold": float(ARGS.threshold),
        "camera": CAMERA.status() if CAMERA is not None else None
    })


@app.route("/api/users", methods=["GET"])
def api_users():
    users = database.list_users(ARGS.db)
    return jsonify({"ok": True, "count": len(users), "users": users})


@app.route("/api/delete_all", methods=["POST"])
def api_delete_all():
    database.reset_db(ARGS.db)
    return jsonify({"ok": True, "message": "Database cleared"})


@app.route("/api/register", methods=["POST"])
def api_register():
    start = time.time()
    try:
        name = (request.form.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Name is required"}), 400

        left_file = request.files.get("left")
        right_file = request.files.get("right")
        if left_file is None:
            return jsonify({"ok": False, "error": "Left eye image is required"}), 400
        if right_file is None:
            return jsonify({"ok": False, "error": "Right eye image is required"}), 400

        left_path = save_upload(left_file, "left")
        right_path = save_upload(right_file, "right")

        t = time.time()
        left_emb, left_meta = BACKEND.make_embedding(left_path)
        left_ms = elapsed_ms(t)

        t = time.time()
        right_emb, right_meta = BACKEND.make_embedding(right_path)
        right_ms = elapsed_ms(t)

        summary = database.register_both(ARGS.db, name, left_emb, right_emb)

        return jsonify({
            "ok": True,
            "message": "Enrolled both eyes for " + name,
            "name": name,
            "elapsed_ms": elapsed_ms(start),
            "left_ms": left_ms,
            "right_ms": right_ms,
            "left_meta": meta_to_json(left_meta),
            "right_meta": meta_to_json(right_meta),
            "db": summary
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    start = time.time()
    try:
        threshold = parse_threshold(request.form.get("threshold") or ARGS.threshold)
        img_file = request.files.get("image")
        if img_file is None:
            return jsonify({"ok": False, "error": "Iris image is required"}), 400

        query_eye, same_side_only, margin_enabled, margin, eye_filter = recognition_options()
        img_path = save_upload(img_file, "query")
        t = time.time()
        emb, meta = BACKEND.make_embedding(img_path)
        infer_ms = elapsed_ms(t)
        result = database.recognize_with_options(
            ARGS.db,
            emb,
            threshold,
            8,
            eye_filter=eye_filter,
            margin=margin if margin_enabled else 0.0
        )

        return jsonify({
            "ok": True,
            "matched": result["matched"],
            "name": result["name"],
            "eye": result["eye"],
            "score": result["score"],
            "threshold": threshold,
            "query_eye": query_eye,
            "same_side_only": same_side_only,
            "margin_enabled": margin_enabled,
            "required_margin": margin if margin_enabled else 0.0,
            "score_margin": result["score_margin"],
            "second_best_different_identity_score": result["second_best_different_identity_score"],
            "threshold_pass": result["threshold_pass"],
            "margin_pass": result["margin_pass"],
            "top_scores": result["top_scores"],
            "elapsed_ms": elapsed_ms(start),
            "inference_ms": infer_ms,
            "meta": meta_to_json(meta)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


@app.route("/api/camera/status", methods=["GET"])
def api_camera_status():
    return jsonify({
        "ok": True,
        "camera": CAMERA.status() if CAMERA is not None else None,
        "captures": {
            key: (
                {
                    "filename": rec["filename"],
                    "captured_at": rec["captured_at"],
                    "image_url": image_url(rec["filename"])
                } if rec else None
            )
            for key, rec in CAMERA_CAPTURES.items()
        }
    })


@app.route("/api/camera/quality", methods=["GET"])
def api_camera_quality():
    start = time.time()
    try:
        rotation = request.args.get("rotation") or "auto"
        quality_threshold = parse_quality_threshold(request.args.get("threshold"))
        jpeg_data, _ = CAMERA.frame_jpeg()
        analysis = CAMERA_ANALYZER.request_jpeg_analysis(
            jpeg_data,
            requested_rotation=rotation,
            quality_threshold=quality_threshold
        )
        if analysis is None:
            return jsonify({
                "ok": True,
                "analyzing": True,
                "analysis": None,
                "camera": CAMERA.status(),
                "elapsed_ms": elapsed_ms(start)
            })
        return jsonify({
            "ok": True,
            "analysis": analysis,
            "analyzing": False,
            "analysis_age_ms": max(0, int((time.time() - CAMERA_ANALYZER.last_result_time) * 1000.0)),
            "camera": CAMERA.status(),
            "elapsed_ms": elapsed_ms(start)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 503


@app.route("/api/camera/frame", methods=["GET"])
def api_camera_frame():
    try:
        data, _ = CAMERA.frame_jpeg()
        response = Response(data, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        camera_status = CAMERA.status()
        response.headers["X-Frame-Sequence"] = str(camera_status.get("frame_sequence", 0))
        response.headers["X-Camera-FPS"] = str(camera_status.get("measured_fps", 0.0))
        response.headers["X-Frame-Age-Ms"] = str(camera_status.get("frame_age_ms") or 0)
        return response
    except Exception as e:
        response = jsonify({"ok": False, "error": str(e)})
        response.status_code = 503
        return response


@app.route("/api/camera/photo", methods=["POST"])
def api_camera_photo():
    start = time.time()
    try:
        frame, _ = CAMERA.current_frame()
        analysis, oriented = CAMERA_ANALYZER.analyze(frame, requested_rotation="auto", quality_threshold=72.0)
        path = write_camera_image("camera_photo", oriented, 94)
        shape = tuple(oriented.shape)
        rec = photo_record(path, shape)
        rec["quality"] = analysis
        CAMERA_CAPTURES["photo"] = rec
        return jsonify({
            "ok": True,
            "message": "Photo captured",
            "filename": rec["filename"],
            "image_url": image_url(rec["filename"]),
            "captured_at": rec["captured_at"],
            "width": rec["width"],
            "height": rec["height"],
            "analysis": analysis,
            "elapsed_ms": elapsed_ms(start)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


@app.route("/api/camera/auto-capture", methods=["POST"])
def api_camera_auto_capture():
    start = time.time()
    try:
        target = parse_camera_target(request.form.get("target") or "query")
        rotation = request.form.get("rotation") or "auto"
        quality_threshold = parse_quality_threshold(request.form.get("threshold"))
        analysis, rec = create_guided_capture(
            target,
            requested_rotation=rotation,
            quality_threshold=quality_threshold
        )
        if rec is None:
            return jsonify({
                "ok": True,
                "captured": False,
                "ready": False,
                "target": target,
                "message": analysis.get("guidance") or "Capture quality is not ready",
                "analysis": analysis,
                "elapsed_ms": elapsed_ms(start)
            })
        return jsonify({
            "ok": True,
            "captured": True,
            "ready": True,
            "target": target,
            "message": "Quality-checked iris captured",
            "filename": rec["filename"],
            "image_url": image_url(rec["filename"]),
            "raw_image_url": image_url(rec["raw_filename"]),
            "processed_image_url": image_url(rec["processed_filename"]),
            "captured_at": rec["captured_at"],
            "analysis": analysis,
            "stats": rec["stats"],
            "warnings": rec["warnings"],
            "elapsed_ms": elapsed_ms(start)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


@app.route("/api/camera/crop", methods=["POST"])
def api_camera_crop():
    start = time.time()
    crop_path = None
    try:
        photo = CAMERA_CAPTURES.get("photo")
        if photo is None:
            return jsonify({"ok": False, "error": "Capture a photo first"}), 400

        purpose = str(request.form.get("purpose") or "query").strip().lower()
        if purpose not in ("query", "enroll"):
            purpose = "query"

        crop_path, shape, auto_eye, crop_box = crop_camera_photo(
            photo,
            request.form.get("x"),
            request.form.get("y"),
            request.form.get("width"),
            request.form.get("height"),
            purpose
        )

        ready = True
        meta = {}
        stats = image_stats(crop_path)
        warnings = []
        message = "Iris crop ready"
        try:
            meta, stats, warnings = assess_camera_capture(crop_path)
        except Exception as assess_error:
            ready = False
            message = str(assess_error)

        if not ready:
            try:
                os.remove(crop_path)
            except Exception:
                pass
            return jsonify({
                "ok": True,
                "ready": False,
                "message": message,
                "purpose": purpose,
                "auto_eye": auto_eye,
                "crop_box": crop_box,
                "stats": stats,
                "elapsed_ms": elapsed_ms(start)
            })

        requested_target = parse_camera_target(request.form.get("target"))
        if purpose == "query":
            target = "query"
        elif requested_target in ("left", "right"):
            target = requested_target
            auto_eye = requested_target
        else:
            target = auto_eye
        rec = capture_record(crop_path, meta, stats, warnings)
        rec["eye"] = auto_eye
        rec["crop_box"] = crop_box
        rec["source_photo"] = photo["filename"]
        CAMERA_CAPTURES[target] = rec

        return jsonify({
            "ok": True,
            "ready": True,
            "message": message,
            "purpose": purpose,
            "target": target,
            "auto_eye": auto_eye,
            "filename": rec["filename"],
            "image_url": image_url(rec["filename"]),
            "captured_at": rec["captured_at"],
            "frame_shape": list(shape),
            "crop_box": crop_box,
            "meta": meta_to_json(meta),
            "stats": stats,
            "warnings": warnings,
            "elapsed_ms": elapsed_ms(start)
        })
    except Exception as e:
        if crop_path:
            try:
                os.remove(crop_path)
            except Exception:
                pass
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


@app.route("/api/camera/image/<filename>", methods=["GET"])
def api_camera_image(filename):
    response = send_from_directory(upload_dir(), safe_filename(filename))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/api/camera/release", methods=["POST"])
def api_camera_release():
    try:
        CAMERA.release()
        if CAMERA_ANALYZER is not None:
            CAMERA_ANALYZER.motion_reference = None
        return jsonify({"ok": True, "camera": CAMERA.status()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/camera/capture", methods=["POST"])
def api_camera_capture():
    start = time.time()
    path = None
    try:
        target = parse_camera_target(request.form.get("target"))
        require_ready = parse_bool(request.form.get("require_ready"), True)
        # Use the persistent preview frame. Reopening nvarguscamerasrc for a
        # still image interrupts the live feed and was another source of flash.
        path, shape = CAMERA.save_current_frame(
            "camera_" + target,
            rotation=CAMERA_ANALYZER.preferred_rotation
        )

        ready = True
        meta = {}
        stats = image_stats(path)
        warnings = []
        message = "Captured frame"

        try:
            meta, stats, warnings = assess_camera_capture(path)
            message = "Iris ready"
        except Exception as assess_error:
            ready = False
            message = str(assess_error)

        if require_ready and not ready:
            try:
                os.remove(path)
            except Exception:
                pass
            return jsonify({
                "ok": True,
                "ready": False,
                "target": target,
                "message": message,
                "stats": stats,
                "elapsed_ms": elapsed_ms(start)
            })

        rec = capture_record(path, meta, stats, warnings)
        CAMERA_CAPTURES[target] = rec
        return jsonify({
            "ok": True,
            "ready": ready,
            "target": target,
            "message": message,
            "filename": rec["filename"],
            "image_url": image_url(rec["filename"]),
            "captured_at": rec["captured_at"],
            "frame_shape": list(shape),
            "meta": meta_to_json(meta),
            "stats": stats,
            "warnings": warnings,
            "elapsed_ms": elapsed_ms(start)
        })
    except Exception as e:
        if path:
            try:
                os.remove(path)
            except Exception:
                pass
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


@app.route("/api/camera/recognize", methods=["POST"])
def api_camera_recognize():
    start = time.time()
    try:
        target = parse_camera_target(request.form.get("target") or "query")
        rec = CAMERA_CAPTURES.get(target)
        if rec is None:
            return jsonify({"ok": False, "error": "No captured " + target + " image yet"}), 400

        default_eye = rec.get("eye") or (target if target in ("left", "right") else None)
        query_eye, same_side_only, margin_enabled, margin, eye_filter = recognition_options(default_eye)
        threshold = parse_threshold(request.form.get("threshold") or ARGS.threshold)

        t = time.time()
        emb, meta = BACKEND.make_embedding(rec["path"])
        infer_ms = elapsed_ms(t)
        result = database.recognize_with_options(
            ARGS.db,
            emb,
            threshold,
            8,
            eye_filter=eye_filter,
            margin=margin if margin_enabled else 0.0
        )

        return jsonify({
            "ok": True,
            "matched": result["matched"],
            "name": result["name"],
            "eye": result["eye"],
            "score": result["score"],
            "threshold": threshold,
            "target": target,
            "query_eye": query_eye,
            "same_side_only": same_side_only,
            "margin_enabled": margin_enabled,
            "required_margin": margin if margin_enabled else 0.0,
            "score_margin": result["score_margin"],
            "second_best_different_identity_score": result["second_best_different_identity_score"],
            "threshold_pass": result["threshold_pass"],
            "margin_pass": result["margin_pass"],
            "top_scores": result["top_scores"],
            "elapsed_ms": elapsed_ms(start),
            "inference_ms": infer_ms,
            "meta": meta_to_json(meta),
            "image_url": image_url(rec["filename"])
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


@app.route("/api/camera/register", methods=["POST"])
def api_camera_register():
    start = time.time()
    try:
        name = (request.form.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Name is required"}), 400
        left_rec = CAMERA_CAPTURES.get("left")
        right_rec = CAMERA_CAPTURES.get("right")
        if left_rec is None:
            return jsonify({"ok": False, "error": "Capture a suitable left eye first"}), 400
        if right_rec is None:
            return jsonify({"ok": False, "error": "Capture a suitable right eye first"}), 400

        t = time.time()
        left_emb, left_meta = BACKEND.make_embedding(left_rec["path"])
        left_ms = elapsed_ms(t)

        t = time.time()
        right_emb, right_meta = BACKEND.make_embedding(right_rec["path"])
        right_ms = elapsed_ms(t)

        summary = database.register_both(ARGS.db, name, left_emb, right_emb)
        return jsonify({
            "ok": True,
            "message": "Enrolled camera captures for " + name,
            "name": name,
            "elapsed_ms": elapsed_ms(start),
            "left_ms": left_ms,
            "right_ms": right_ms,
            "left_meta": meta_to_json(left_meta),
            "right_meta": meta_to_json(right_meta),
            "left_image_url": image_url(left_rec["filename"]),
            "right_image_url": image_url(right_rec["filename"]),
            "db": summary
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="iris_model_fp16.engine", help="TensorRT engine path")
    parser.add_argument("--meta", default="iris_model.metadata.json", help="metadata JSON path")
    parser.add_argument("--db", default=DEFAULT_DB, help="local JSON enrollment database")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="default cosine threshold")
    parser.add_argument("--host", default="0.0.0.0", help="host address")
    parser.add_argument("--port", type=int, default=8000, help="web server port")
    parser.add_argument("--camera-source", default="csi", help="camera source: 0, /dev/video0, csi, csi:1, or GStreamer pipeline")
    parser.add_argument("--camera-width", type=int, default=1280, help="camera capture width")
    parser.add_argument("--camera-height", type=int, default=720, help="camera capture height")
    parser.add_argument("--camera-fps", type=int, default=21, help="camera capture framerate")
    parser.add_argument("--preview-width", type=int, default=1280, help="CSI preview width")
    parser.add_argument("--preview-height", type=int, default=720, help="CSI preview height")
    parser.add_argument("--preview-fps", type=int, default=21, help="CSI preview output framerate")
    parser.add_argument("--preview-sensor-fps", type=int, default=21, help="CSI sensor framerate for preview mode")
    return parser


def main():
    global ARGS, BACKEND, CAMERA, CAMERA_ANALYZER
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    atexit.register(shutdown_camera)
    try:
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
    except Exception:
        pass

    parser = build_parser()
    ARGS = parser.parse_args()
    ARGS.engine = resolve_path(ARGS.engine)
    ARGS.meta = resolve_path(ARGS.meta)
    ARGS.db = resolve_path(ARGS.db)
    CAMERA = CameraManager()
    CAMERA_ANALYZER = CameraQualityAnalyzer()
    CAMERA.configure(
        ARGS.camera_source,
        ARGS.camera_width,
        ARGS.camera_height,
        ARGS.camera_fps,
        ARGS.preview_width,
        ARGS.preview_height,
        ARGS.preview_fps,
        ARGS.preview_sensor_fps
    )

    print("")
    print("=" * 72)
    print(APP_TITLE)
    print("=" * 72)
    print("Loading metadata:", ARGS.meta)
    print("Loading TensorRT engine:", ARGS.engine)
    BACKEND = IrisBackend(ARGS.engine, ARGS.meta)
    BACKEND.load()

    ip = get_local_ip()
    print("")
    print("Server ready.")
    print("Open on Jetson:        http://127.0.0.1:%d" % ARGS.port)
    print("Open on local network: http://%s:%d" % (ip, ARGS.port))
    print("Database:", ARGS.db)
    print("Default threshold: %.2f" % float(ARGS.threshold))
    print("Camera source:", ARGS.camera_source)
    print("Capture mode: %dx%d@%d" % (ARGS.camera_width, ARGS.camera_height, ARGS.camera_fps))
    print("Preview mode: %dx%d@%d" % (ARGS.preview_width, ARGS.preview_height, ARGS.preview_fps))
    print("HTTP request logging: warning only")
    print("=" * 72)
    print("")
    app.run(host=ARGS.host, port=ARGS.port, threaded=False, debug=False)


if __name__ == "__main__":
    main()
