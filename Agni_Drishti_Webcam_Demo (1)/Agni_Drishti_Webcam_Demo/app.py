"""Offline Vision-X local prototype. Run with: python app.py"""
from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import logging
import math
import os
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("offline_vision")
sys.path.insert(0, str(ROOT / "python"))
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def config(name: str) -> dict:
    candidates = [ROOT / name, ROOT / "config" / name]
    for p in candidates:
        try:
            if p.exists():
                text = p.read_text(encoding="utf-8")
                return json.loads(text)
        except Exception:
            pass
    return {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

def discover_lan_ipv4_addresses():
    addresses = set()
    route_address = None
    probe = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("192.0.2.1", 9))
        route_address = probe.getsockname()[0]
        addresses.add(route_address)
    except OSError:
        pass
    finally:
        if probe is not None:
            probe.close()
    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            addresses.add(result[4][0])
    except OSError:
        pass
    private_addresses = {
        address for address in addresses
        if ipaddress.ip_address(address).is_private
        and not ipaddress.ip_address(address).is_loopback
        and not ipaddress.ip_address(address).is_link_local
    }
    ordered = []
    if route_address in private_addresses:
        ordered.append(route_address)
    ordered.extend(sorted(private_addresses - set(ordered)))
    return ordered


def iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / max(1, aw*ah + bw*bh - inter)


class Tracker:
    """Session-local motion/appearance tracker with bounded short-term ReID."""

    def __init__(self, fps: float = 15.0):
        self.fps = fps
        self.tracks = {}
        self.next_id = 1
        cfg = config("tracker.yaml")
        self.max_lost = int(cfg.get("max_lost_frames", 12))
        self.min_iou = float(cfg.get("minimum_iou", 0.15))
        self.max_lost_seconds = float(cfg.get("max_lost_seconds", 2.5))
        self.reid_window_seconds = float(cfg.get("reid_window_seconds", 8.0))
        self.reid_threshold = float(cfg.get("reid_similarity_threshold", 0.75))
        self.min_detection_confidence = float(cfg.get("min_detection_confidence", 0.25))
        self.max_center_distance = float(cfg.get("max_center_distance", 1.5))
        self.min_association_score = float(cfg.get("minimum_association_score", 0.18))
        if not 0.0 <= self.min_association_score <= 1.0:
            raise ValueError("tracker minimum_association_score must be between 0 and 1")
        self.recent_lost = {}

    @staticmethod
    def _descriptor_for_detection(detection, frame=None):
        if frame is None or cv2 is None:
            return None
        x, y, width, height = [int(v) for v in detection.get("bbox", (0, 0, 0, 0))]
        x0 = max(0, x + int(width * 0.15))
        x1 = min(frame.shape[1], x + max(1, int(width * 0.85)))
        y0 = max(0, y + int(height * 0.10))
        y1 = min(frame.shape[0], y + max(1, int(height * 0.75)))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame[y0:y1, x0:x1]
        if not crop.size:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        middle = max(1, hsv.shape[0] // 2)
        histograms = []
        for section in (hsv[:middle], hsv[middle:]):
            if not section.size:
                section = hsv
            hist = cv2.calcHist([section], [0, 1], None, [8, 4], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
            histograms.extend(float(value) for value in hist.flatten())
        aspect = max(0.0, min(1.0, float(width) / max(1.0, float(height))))
        return histograms + [aspect]

    @staticmethod
    def _similarity(lhs, rhs):
        if not lhs or not rhs:
            return 0.0
        if len(lhs) != 65 or len(rhs) != 65:
            return 0.0
        upper_score = sum(min(float(a), float(b)) for a, b in zip(lhs[:32], rhs[:32]))
        lower_score = sum(min(float(a), float(b)) for a, b in zip(lhs[32:64], rhs[32:64]))
        aspect_score = max(0.0, 1.0 - abs(float(lhs[64]) - float(rhs[64])))
        return max(0.0, min(1.0, upper_score * 0.4 + lower_score * 0.4 + aspect_score * 0.2))

    @staticmethod
    def _center(bbox):
        return bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0

    def _predicted_bbox(self, track, timestamp):
        x, y, width, height = track["bbox"]
        elapsed = max(0.0, timestamp - track["last_seen"])
        vx, vy = track.get("velocity_px_s", (0.0, 0.0))
        diagonal = math.hypot(width, height)
        dx = max(-diagonal, min(diagonal, vx * elapsed))
        dy = max(-diagonal, min(diagonal, vy * elapsed))
        return [x + dx, y + dy, width, height]

    def _association_score(self, track, detection, timestamp, appearance):
        predicted = self._predicted_bbox(track, timestamp)
        overlap = iou(predicted, detection["bbox"])
        old_center = self._center(predicted)
        new_center = self._center(detection["bbox"])
        scale = max(1.0, math.hypot(predicted[2], predicted[3]))
        distance = math.hypot(new_center[0] - old_center[0], new_center[1] - old_center[1]) / scale
        if overlap < self.min_iou and distance > self.max_center_distance:
            return None
        motion_score = max(0.0, 1.0 - distance / max(0.01, self.max_center_distance))
        similarity = self._similarity(track.get("appearance"), appearance)
        if track.get("appearance") and appearance:
            return overlap * 0.45 + motion_score * 0.30 + similarity * 0.25
        return overlap * 0.60 + motion_score * 0.40

    @staticmethod
    def _maximum_assignment(scores, minimum_score):
        """Globally maximize valid track/detection matches, allowing unmatched tracks."""
        row_count = len(scores)
        if not row_count:
            return {}
        detection_count = max((len(row) for row in scores), default=0)
        if not detection_count:
            return {}
        column_count = detection_count + row_count
        invalid_cost = 1e6
        unmatched_cost = 1.0 - minimum_score
        costs = []
        for row in scores:
            converted = [
                (1.0 - value) if value is not None and value >= minimum_score else invalid_cost
                for value in row
            ]
            converted.extend([unmatched_cost] * row_count)
            costs.append(converted)

        # Rectangular Hungarian algorithm; rows <= columns because every track
        # receives dummy unmatched columns.
        u = [0.0] * (row_count + 1)
        v = [0.0] * (column_count + 1)
        matched_row = [0] * (column_count + 1)
        path = [0] * (column_count + 1)
        for row_index in range(1, row_count + 1):
            matched_row[0] = row_index
            column0 = 0
            min_cost = [float("inf")] * (column_count + 1)
            used = [False] * (column_count + 1)
            while True:
                used[column0] = True
                current_row = matched_row[column0]
                delta = float("inf")
                column1 = 0
                for column in range(1, column_count + 1):
                    if used[column]:
                        continue
                    current = costs[current_row - 1][column - 1] - u[current_row] - v[column]
                    if current < min_cost[column]:
                        min_cost[column] = current
                        path[column] = column0
                    if min_cost[column] < delta:
                        delta = min_cost[column]
                        column1 = column
                for column in range(column_count + 1):
                    if used[column]:
                        u[matched_row[column]] += delta
                        v[column] -= delta
                    else:
                        min_cost[column] -= delta
                column0 = column1
                if matched_row[column0] == 0:
                    break
            while True:
                column1 = path[column0]
                matched_row[column0] = matched_row[column1]
                column0 = column1
                if column0 == 0:
                    break

        assignment = {}
        for column in range(1, detection_count + 1):
            row = matched_row[column]
            if row and scores[row - 1][column - 1] is not None and scores[row - 1][column - 1] >= minimum_score:
                assignment[row - 1] = column - 1
        return assignment

    def _archive(self, track, timestamp):
        track_id = track["id"]
        if track.get("state") != "MISSING":
            track["state"] = "MISSING"
            track["identity_event"] = "LOST"
            self.recent_lost[track_id] = {
                "appearance": track.get("appearance"),
                "last_seen": track["last_seen"],
                "archived_at": timestamp,
                "class_id": track.get("class_id"),
                "class_name": track.get("class_name"),
            }

    def update(self, detections, frame=None, timestamp=None):
        timestamp = float(timestamp if timestamp is not None else time.monotonic())
        detections = [
            d for d in detections
            if float(d.get("confidence", 0.0)) >= self.min_detection_confidence
            and len(d.get("bbox", ())) == 4
            and float(d["bbox"][2]) > 0 and float(d["bbox"][3]) > 0
        ]
        appearance = [self._descriptor_for_detection(d, frame) for d in detections]
        for track_id, saved in list(self.recent_lost.items()):
            if timestamp - saved["last_seen"] > self.reid_window_seconds:
                self.recent_lost.pop(track_id, None)
                self.tracks.pop(track_id, None)

        used_tracks, used_detections = set(), set()
        active_track_ids = []
        score_matrix = []
        for track_id, track in self.tracks.items():
            elapsed = timestamp - track["last_seen"]
            if track["state"] == "MISSING" or elapsed > self.max_lost_seconds:
                continue
            active_track_ids.append(track_id)
            row = []
            for index, detection in enumerate(detections):
                if (track.get("class_id"), track.get("class_name")) != (detection.get("class_id"), detection.get("class_name")):
                    row.append(None)
                    continue
                score = self._association_score(track, detection, timestamp, appearance[index])
                row.append(score)
            score_matrix.append(row)

        assignment = self._maximum_assignment(score_matrix, self.min_association_score)
        for track_index, index in assignment.items():
            track_id = active_track_ids[track_index]
            track = self.tracks[track_id]
            detection = detections[index]
            previous_state = track["state"]
            old_x, old_y = self._center(track["bbox"])
            new_x, new_y = self._center(detection["bbox"])
            elapsed = max(1e-3, timestamp - track["last_seen"])
            measured_velocity = [(new_x - old_x) / elapsed, (new_y - old_y) / elapsed]
            previous_velocity = track.get("velocity_px_s", (0.0, 0.0))
            track.update(detection)
            track["velocity_px_s"] = [
                round(0.35 * measured_velocity[i] + 0.65 * previous_velocity[i], 1)
                for i in range(2)
            ]
            if math.hypot(*track["velocity_px_s"]) > 8.0:
                track["stationary_since"] = timestamp
            track["appearance"] = appearance[index] or track.get("appearance")
            track["missed"] = 0
            track["age"] += 1
            track["last_seen"] = timestamp
            track["last_seen_wall"] = time.time()
            track["state"] = "TRACKING"
            track["identity_event"] = "REACQUIRED" if previous_state == "LOST" else None
            self.recent_lost.pop(track_id, None)
            used_tracks.add(track_id)
            used_detections.add(index)

        for index, detection in enumerate(detections):
            if index in used_detections:
                continue
            candidates = []
            for lost_id, saved in self.recent_lost.items():
                if lost_id in used_tracks or timestamp - saved["last_seen"] > self.reid_window_seconds:
                    continue
                if (saved.get("class_id"), saved.get("class_name")) != (detection.get("class_id"), detection.get("class_name")):
                    continue
                similarity = self._similarity(appearance[index], saved.get("appearance"))
                if similarity >= self.reid_threshold:
                    candidates.append((similarity, lost_id))
            candidates.sort(reverse=True)
            best_id = None
            if candidates and (len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= 0.08):
                best_id = candidates[0][1]
            if best_id is not None:
                track = self.tracks[best_id]
                track.update(detection)
                track["appearance"] = appearance[index]
                track["missed"] = 0
                track["age"] += 1
                track["last_seen"] = timestamp
                track["last_seen_wall"] = time.time()
                track["velocity_px_s"] = [0.0, 0.0]
                track["stationary_since"] = timestamp
                track["state"] = "TRACKING"
                track["identity_event"] = "RE-IDENTIFIED"
                self.recent_lost.pop(best_id, None)
                used_tracks.add(best_id)
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    **detection,
                    "id": track_id,
                    "age": 1,
                    "missed": 0,
                    "velocity_px_s": [0.0, 0.0],
                    "last_seen": timestamp,
                    "last_seen_wall": time.time(),
                    "stationary_since": timestamp,
                    "state": "NEW",
                    "identity_event": "NEW",
                    "appearance": appearance[index],
                }
                used_tracks.add(track_id)
            used_detections.add(index)

        for track_id, track in list(self.tracks.items()):
            if track_id in used_tracks:
                continue
            track["missed"] += 1
            elapsed = max(0.0, timestamp - track["last_seen"])
            if elapsed >= self.max_lost_seconds or track["missed"] >= self.max_lost:
                self._archive(track, timestamp)
            else:
                track["state"] = "LOST"
                track["identity_event"] = None

        return [dict(track, id=track_id) for track_id, track in self.tracks.items()]


class VisionApp:
    def __init__(self, device=0, mode="human"):
        self.device, self.mode = device, mode
        self.server_port = int(config("network.yaml").get("port", 8765))
        self.lan_access_urls = []
        self.cap = None
        self.lock = threading.RLock()
        self.frame = None
        self.tracks = []
        self.events = deque(maxlen=200)
        self.tracking_events = deque(maxlen=200)
        self.industrial = None
        self.gpu_available = False
        self.gpu_name = "CPU"
        self.model_device = "cpu"
        self.inference_ms = 0.0
        self.frame_number = 0
        self._last_track_states = {}
        self._last_motion_states = {}
        self._status_counts = {"active": 0, "lost": 0, "warning": 0, "critical": 0}
        self.log_dir = ROOT / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.tracking_csv_path = self.log_dir / "person_tracking.csv"
        self.event_csv_path = self.log_dir / "person_events.csv"
        self.assistance_csv_path = self.log_dir / "assistance_requests.csv"
        self.quality_csv_path = self.log_dir / "quality_inspection.csv"
        self.assistance_requests = deque(maxlen=200)
        self._init_csv_logs()
        try:
            from monitor import IndustrialConfig, IndustrialMonitor
            industrial_path = next(
                path for path in (ROOT / "industrial.yaml", ROOT / "config" / "industrial.yaml")
                if path.exists()
            )
            self.industrial = IndustrialMonitor(IndustrialConfig.load(industrial_path))
        except Exception as exc:
            self.events.append({"time": now(), "type": "industrial_config_error", "message": str(exc)})
        self.quality_inspector = None
        self.quality_sample_every_n_frames = 3
        try:
            from quality_inspection import QualityInspector
            quality_settings = config("quality.yaml")
            self.quality_inspector = QualityInspector(quality_settings, self.quality_csv_path)
            self.quality_sample_every_n_frames = max(1, int(quality_settings.get("sample_every_n_frames", 3)))
        except Exception as exc:
            LOGGER.exception("Could not initialize visual quality inspection")
            self.events.append({"timestamp": now(), "type": "quality_config_error", "message": str(exc)})
        self.fps = 0.0; self.latency_ms = 0.0; self.processing_ms = 0.0
        self.camera_error = "OpenCV is not installed" if cv2 is None else "Camera not started"
        self.running = False
        self.tracker = Tracker(); self.tracker.fps = 15
        self.hog = None
        self.min_detection_confidence = 0.25
        self.custom_detector = None
        self.yolo_model = None
        self.detector_device = "cpu"
        self.detector_error = ""
        self._inference_lock = threading.Semaphore(1)
        self._init_device_and_detector()
        self.homography = None
        self._load_homography()

    def _init_device_and_detector(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_name = torch.cuda.get_device_name(0)
                self.model_device = "cuda"
            else:
                self.gpu_name = "CPU"
                self.model_device = "cpu"
        except Exception:
            self.gpu_name = "CPU"
            self.model_device = "cpu"

        if cv2 is not None:
            try:
                self.hog = cv2.HOGDescriptor(); self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except AttributeError as exc:
                self.detector_error = f"OpenCV build lacks the HOG people baseline: {exc}"

        try:
            from ultralytics import YOLO
            local_candidates = [ROOT / "models" / "yolov8n.pt", ROOT / "weights" / "yolov8n.pt", ROOT / "yolov8n.pt"]
            for candidate in local_candidates:
                if candidate.exists():
                    try:
                        self.yolo_model = YOLO(str(candidate))
                        break
                    except Exception:
                        self.yolo_model = None
                        self.detector_error = "Optional Ultralytics package unavailable; using OpenCV HOG fallback."
                    if self.yolo_model is None and not self.detector_error:
                        self.detector_error = "No local YOLO weights found; using OpenCV HOG fallback."
        except Exception:
            self.yolo_model = None

        det_cfg = config("detector.yaml")
        if det_cfg.get("backend") == "visionx_custom_grid":
            try:
                from visionx.detection import CustomGridDetector
                classes_path = ROOT / det_cfg.get("classes", "python/datasets/classes.txt")
                classes = [x.strip() for x in classes_path.read_text(encoding="utf-8").splitlines() if x.strip()]
                self.custom_detector = CustomGridDetector(str(ROOT / det_cfg.get("checkpoint", "models/detector.pt")), classes, float(det_cfg.get("confidence_threshold", .35)), float(det_cfg.get("nms_iou_threshold", .45)), str(det_cfg.get("device", self.model_device)))
                self.hog = None
            except Exception as exc:
                self.detector_error = str(exc)

    def _init_csv_logs(self):
        tracking_fields = ["Person_ID","Timestamp","Frame_Number","Detection_Confidence","BoundingBox_X","BoundingBox_Y","BoundingBox_Width","BoundingBox_Height","Center_X","Center_Y","Heading","Movement_Speed","Track_Status","Visibility_Level","Safety_Status","Reason"]
        event_fields = ["Person_ID","Timestamp","Event_Type","Safety_Status","Confidence","Reason","Recommended_Action"]
        for path, fields in ((self.tracking_csv_path, tracking_fields), (self.event_csv_path, event_fields)):
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
        if not self.assistance_csv_path.exists():
            with self.assistance_csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(
                    handle,
                    fieldnames=["Request_ID","Person_ID","Timestamp","Reason","Safety_Status","Center_X_Px","Center_Y_Px","Handoff_Status"],
                ).writeheader()

    def request_assistance(self, person_id, reason, operator_confirmed):
        if operator_confirmed is not True:
            raise ValueError("Operator confirmation is required before creating an assistance request.")
        allowed_reasons = {"possible_distress", "hazard_zone", "missing_person", "operator_observed_concern"}
        if reason not in allowed_reasons:
            raise ValueError("Unsupported assistance reason.")
        try:
            numeric_id = int(str(person_id).removeprefix("Person-").removeprefix("Human-"))
        except (TypeError, ValueError) as exc:
            raise ValueError("person_id must be a current tracked person ID.") from exc
        with self.lock:
            track = next((item for item in self.tracks if int(item.get("id", -1)) == numeric_id), None)
            if track is None:
                raise ValueError("The requested person is not in the current track list.")
            x, y, width, height = track["bbox"]
            record = {
                "request_id": f"assist-{int(time.time() * 1000)}-{numeric_id}",
                "person_id": f"Person-{numeric_id:03d}",
                "timestamp": now(),
                "reason": reason,
                "safety_status": track.get("safety_status", "UNASSESSED"),
                "last_seen": datetime.fromtimestamp(track.get("last_seen_wall", time.time()), timezone.utc).isoformat(timespec="milliseconds"),
                "image_location_px": {
                    "center_x": round(float(x + width / 2.0), 1),
                    "center_y": round(float(y + height / 2.0), 1),
                },
                "handoff_status": "LOCAL_OPERATOR_REVIEW_REQUIRED",
                "privacy": "No face image, name, biometric template, or external notification is included.",
                "recommended_action": "Operator: verify the live scene and use the approved human volunteer process if appropriate.",
            }
            try:
                with self.assistance_csv_path.open("a", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(
                        handle,
                        fieldnames=["Request_ID","Person_ID","Timestamp","Reason","Safety_Status","Center_X_Px","Center_Y_Px","Handoff_Status"],
                    ).writerow({
                        "Request_ID": record["request_id"],
                        "Person_ID": record["person_id"],
                        "Timestamp": record["timestamp"],
                        "Reason": record["reason"],
                        "Safety_Status": record["safety_status"],
                        "Center_X_Px": record["image_location_px"]["center_x"],
                        "Center_Y_Px": record["image_location_px"]["center_y"],
                        "Handoff_Status": record["handoff_status"],
                    })
            except (OSError, csv.Error, ValueError):
                LOGGER.exception("Could not append assistance request")
                raise
            self.assistance_requests.append(record)
            self._write_event_log(
                track,
                "operator assistance requested",
                record["safety_status"],
                f"Operator requested review ({reason}); no automated person-risk inference was made.",
                record["recommended_action"],
            )
            return dict(record)

    def _write_tracking_row(self, track):
        try:
            fields = ["Person_ID","Timestamp","Frame_Number","Detection_Confidence","BoundingBox_X","BoundingBox_Y","BoundingBox_Width","BoundingBox_Height","Center_X","Center_Y","Heading","Movement_Speed","Track_Status","Visibility_Level","Safety_Status","Reason"]
            payload = {
                "Person_ID": track.get("id", "N/A"),
                "Timestamp": now(),
                "Frame_Number": self.frame_number,
                "Detection_Confidence": round(float(track.get("confidence", 0.0)), 3),
                "BoundingBox_X": int(track["bbox"][0]),
                "BoundingBox_Y": int(track["bbox"][1]),
                "BoundingBox_Width": int(track["bbox"][2]),
                "BoundingBox_Height": int(track["bbox"][3]),
                "Center_X": int(track["bbox"][0] + track["bbox"][2] / 2),
                "Center_Y": int(track["bbox"][1] + track["bbox"][3] / 2),
                "Heading": self.direction(*track.get("velocity_px_s", [0.0, 0.0])),
                "Movement_Speed": round(float(math.hypot(*track.get("velocity_px_s", [0.0, 0.0]))), 2),
                "Track_Status": track.get("state", "TRACKING"),
                "Visibility_Level": round(max(0.0, min(1.0, float(track.get("confidence", 0.0)))), 3),
                "Safety_Status": track.get("safety_status", "SAFE"),
                "Reason": track.get("safety_reason", "Normal movement observed"),
            }
            with self.tracking_csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writerow(payload)
        except (OSError, csv.Error, ValueError):
            LOGGER.exception("Could not append person tracking row to %s", self.tracking_csv_path)

    def _write_event_log(self, track, event_type, safety_status, reason, recommended_action):
        timestamp = now()
        try:
            payload = {
                "Person_ID": track.get("id", "N/A"),
                "Timestamp": timestamp,
                "Event_Type": event_type,
                "Safety_Status": safety_status,
                "Confidence": round(float(track.get("confidence", 0.0)), 3),
                "Reason": reason,
                "Recommended_Action": recommended_action,
            }
            with self.event_csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Person_ID","Timestamp","Event_Type","Safety_Status","Confidence","Reason","Recommended_Action"])
                writer.writerow(payload)
        except (OSError, csv.Error, ValueError):
            LOGGER.exception("Could not append person event row to %s", self.event_csv_path)
        event = {
            "timestamp": timestamp,
            "type": event_type,
            "severity": "critical" if safety_status == "CRITICAL" else ("warning" if safety_status == "WARNING" else "info"),
            "message": f"ID {int(track.get('id', 0)):02d} {event_type}: {reason}",
            "track_id": f"Person-{int(track.get('id', 0)):03d}",
            "safety_status": safety_status,
            "confidence": round(float(track.get("confidence", 0.0)), 3),
            "recommended_action": recommended_action,
        }
        self.tracking_events.append(event)
        if self.mode != "industrial":
            self.events.append(event)

    def _get_safety_state(self, track):
        status = str(track.get("state", "TRACKING")).upper()
        speed = math.hypot(*track.get("velocity_px_s", [0.0, 0.0]))
        confidence = float(track.get("confidence", 0.0))
        if status in {"LOST", "MISSING"}:
            return "MISSING", 0.9, "Person disappeared from the webcam view and could not be reacquired within the configured timeout.", "Verify reacquisition and continue monitoring."
        safety_cfg = config("safety.yaml")
        stationary_warning_seconds = float(safety_cfg.get("stationary_warning_seconds", 30.0))
        stationary_critical_seconds = float(safety_cfg.get("stationary_critical_seconds", 120.0))
        if speed > float(safety_cfg.get("movement_threshold_px_s", 8.0)):
            track["stationary_since"] = time.monotonic()
        stationary_for = max(0.0, time.monotonic() - float(track.get("stationary_since", time.monotonic())))
        if stationary_for >= stationary_critical_seconds:
            return "CRITICAL", max(0.5, confidence), "Person has remained immobile for a prolonged period; possible danger requires verification.", "Require human verification and prioritize inspection."
        if stationary_for >= stationary_warning_seconds or confidence < float(safety_cfg.get("low_confidence_threshold", 0.45)):
            reason = "Person has remained stationary longer than the configured warning interval." if stationary_for >= stationary_warning_seconds else "Detection confidence is low; visibility may be limited."
            return "WARNING", max(0.55, confidence), reason, "Continue monitoring and re-check quickly."
        return "SAFE", max(0.7, confidence), "Normal movement and visibility detected in the webcam frame.", "Continue normal monitoring."

    def _load_homography(self):
        if cv2 is None: return
        c = config("localization.yaml")
        pts = c.get("image_points", [])
        world = c.get("world_points_m", [])
        if len(pts) == len(world) and len(pts) >= 4:
            import numpy as np
            self.homography = cv2.getPerspectiveTransform(np.array(pts[:4], dtype="float32"), np.array(world[:4], dtype="float32"))

    def _run_detector(self, frame):
        self._inference_lock.acquire()
        try:
            detections = []
            start = time.perf_counter()
            if self.yolo_model is not None:
                try:
                    detector_cfg = config("detector.yaml")
                    conf_threshold = max(0.25, float(detector_cfg.get("confidence_threshold", 0.35)))
                    results = self.yolo_model(
                        frame,
                        conf=conf_threshold,
                        iou=float(detector_cfg.get("nms_iou_threshold", 0.45)),
                        imgsz=int(detector_cfg.get("yolo_image_size", 512)),
                        max_det=int(detector_cfg.get("yolo_max_detections", 20)),
                        half=self.model_device == "cuda",
                        verbose=False,
                        device=self.model_device,
                    )
                    model = getattr(self.yolo_model, "model", None)
                    if model is not None:
                        parameter = next(model.parameters(), None)
                        if parameter is not None:
                            self.detector_device = str(parameter.device)
                    for result in results:
                        for box in getattr(result, "boxes", []):
                            if len(box.cls) == 0:
                                continue
                            cls_index = int(box.cls[0])
                            if cls_index != 0:
                                continue
                            conf = float(box.conf[0])
                            if conf < 0.25:
                                continue
                            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                            detections.append({"class_id": 0, "class_name": "person", "confidence": round(conf, 3), "bbox": [int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))]})
                except Exception as exc:
                    self.detector_error = f"YOLO inference failed; using HOG fallback: {exc}"
                    self.yolo_model = None
                    detections = self._run_fallback_detector(frame)
            elif self.custom_detector is not None:
                detections = [dict(class_id=d.class_id, class_name=d.class_name, confidence=round(float(d.confidence),3), bbox=[int(v) for v in d.bounding_box]) for d in self.custom_detector.detect(frame, time.time())]
                self.detector_device = str(getattr(self.custom_detector, "device", "cpu"))
            else:
                detections = self._run_fallback_detector(frame)
            self.inference_ms = (time.perf_counter() - start) * 1000.0
            return detections
        finally:
            self._inference_lock.release()

    def _run_fallback_detector(self, frame):
        if self.mode not in ("human", "industrial") or self.hog is None:
            self.detector_device = "cpu"
            return []
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        self.detector_device = "cpu"
        detections = [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": round(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, float(weights[i]))))), 3),
                "bbox": list(map(int, rect)),
            }
            for i, rect in enumerate(rects)
        ]
        if len(detections) > 1 and hasattr(cv2, "dnn") and hasattr(cv2.dnn, "NMSBoxes"):
            indices = cv2.dnn.NMSBoxes(
                [detection["bbox"] for detection in detections],
                [detection["confidence"] for detection in detections],
                self.tracker.min_detection_confidence,
                float(config("detector.yaml").get("nms_iou_threshold", 0.45)),
            )
            kept = set(int(index) for index in indices)
            detections = [detection for index, detection in enumerate(detections) if index in kept]
        return detections

    def start(self):
        self.running = True
        if cv2 is None: return
        self.cap = cv2.VideoCapture(self.device)
        cam = config("camera.yaml")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam.get("width", 640)))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("height", 480)))
        if not self.cap.isOpened():
            self.camera_error = f"Could not open camera device {self.device}"; self.cap.release(); self.cap = None; return
        self.camera_error = ""
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        prev = time.perf_counter()
        while self.running and self.cap:
            tick = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                self.camera_error = "Camera read failed"; time.sleep(.1); continue
            inspection_frame = frame.copy() if self.mode == "industrial" and self.quality_inspector is not None else None
            start = time.perf_counter()
            detections = self._run_detector(frame)
            self.tracker.fps = 1 / max(.001, time.perf_counter() - prev); prev = time.perf_counter()
            updated_tracks = self.tracker.update(detections, frame)
            self.frame_number += 1
            for t in updated_tracks:
                x, y, w, h = t["bbox"]
                safety, confidence, reason, action = self._get_safety_state(t)
                t["safety_status"] = safety
                t["safety_confidence"] = confidence
                t["safety_reason"] = reason
                t["recommended_action"] = action
                prior_state = self._last_track_states.get(t["id"])
                identity_event = t.get("identity_event")
                moving = math.hypot(*t.get("velocity_px_s", (0.0, 0.0))) > float(config("safety.yaml").get("movement_threshold_px_s", 8.0))
                if identity_event in {"NEW", "RE-IDENTIFIED"}:
                    event_reason = "Person detected for the first time." if identity_event == "NEW" else "Appearance match reacquired a recently lost identity."
                    self._write_event_log(t, identity_event.lower(), safety, event_reason, action)
                elif t.get("state") == "MISSING" and prior_state != "MISSING":
                    self._write_event_log(t, "missing", safety, reason, action)
                elif t.get("state") == "LOST" and prior_state not in {"LOST", "MISSING"}:
                    self._write_event_log(t, "lost", safety, reason, action)
                elif t.get("identity_event") == "REACQUIRED":
                    self._write_event_log(t, "reacquired", safety, "Previously active identity reacquired after temporary occlusion.", action)
                if moving and not self._last_motion_states.get(t["id"], False):
                    self._write_event_log(t, "moving", safety, "Image-space movement detected.", action)
                if prior_state is not None and t.get("safety_status") != t.get("previous_safety_status"):
                    self._write_event_log(t, "safety status changed", safety, reason, action)
                t["previous_safety_status"] = safety
                self._last_track_states[t["id"]] = t.get("state")
                self._last_motion_states[t["id"]] = moving
                self._write_tracking_row(t)
                if cv2 is not None and t.get("state") not in {"MISSING"}:
                    color = (70, 220, 150)
                    if safety == "WARNING":
                        color = (0, 165, 255)
                    elif safety == "CRITICAL":
                        color = (0, 0, 255)
                    elif safety == "MISSING":
                        color = (128, 128, 128)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"ID {t['id']:02d} {t.get('identity_event') or t['state']} {safety} {t['confidence']:.2f}"
                    cv2.putText(frame, label, (max(0, x), max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            self._update_industrial(updated_tracks, time.time(), frame.shape)
            if self.mode == "industrial" and self.quality_inspector is not None:
                try:
                    if self.frame_number % self.quality_sample_every_n_frames == 0:
                        self.quality_inspector.update(inspection_frame)
                    self.quality_inspector.overlay(frame)
                    combined = self.industrial.events() if self.industrial is not None else []
                    combined += list(self.tracking_events)
                    combined += self.quality_inspector.status().get("events", [])
                    combined.sort(key=lambda event: str(event.get("timestamp", "")))
                    self.events = deque(combined[-200:], maxlen=200)
                except Exception:
                    LOGGER.exception("Visual quality screening failed for frame %s", self.frame_number)
                    self.events.append({"timestamp": now(), "type": "quality_inspection_error", "message": "Visual quality screening failed; check application logs."})
            self.processing_ms = (time.perf_counter() - start) * 1000
            self.fps = 0.9 * self.fps + 0.1 / max(0.001, time.perf_counter() - tick) if self.fps else 1 / max(0.001, time.perf_counter() - tick)
            with self.lock:
                self.frame = frame
                self.tracks = updated_tracks
                self.camera_error = ""

    def _update_industrial(self, tracks, timestamp, frame_shape=None):
        if self.industrial is None or self.mode != "industrial":
            return
        configured_width = float(config("camera.yaml").get("width", 640))
        configured_height = float(config("camera.yaml").get("height", 480))
        actual_height = float(frame_shape[0]) if frame_shape is not None else configured_height
        actual_width = float(frame_shape[1]) if frame_shape is not None else configured_width
        observations = []
        for track in tracks:
            if track.get("state") not in {"NEW", "TRACKING"}:
                continue
            x, y, width, height = track["bbox"]
            center_x, center_y = x + width / 2.0, y + height
            world_x, world_y = self.localize(center_x, center_y)
            image_coordinates = world_x is None or world_y is None
            if image_coordinates:
                world_x = center_x * configured_width / max(1.0, actual_width)
                world_y = center_y * configured_height / max(1.0, actual_height)
            observations.append({
                "id": f"Person-{track['id']:03d}",
                "class_name": track.get("class_name", "person"),
                "world_x": world_x,
                "world_y": world_y,
                "state": track["state"],
                "coordinate_frame": "image_pixels" if image_coordinates else "calibrated_ground_plane",
            })
        try:
            self.industrial.update(observations, timestamp)
            combined = self.industrial.events() + list(self.tracking_events)
            combined.sort(key=lambda event: str(event.get("timestamp", "")))
            self.events = deque(combined[-200:], maxlen=200)
        except Exception as exc:
            self.events.append({"timestamp": now(), "type": "industrial_update_error", "message": str(exc)})

    def track_payload(self):
        with self.lock:
            result = []
            for t in self.tracks:
                x, y, w, h = t["bbox"]
                px = x + w / 2
                py = y + h / 2
                coord = self.localize(px, y + h)
                vx, vy = t["velocity_px_s"]
                safety, confidence, reason, action = self._get_safety_state(t)
                t["safety_status"] = safety
                t["safety_confidence"] = confidence
                t["safety_reason"] = reason
                t["recommended_action"] = action
                result.append({
                    "id": f"Person-{t['id']:03d}",
                    "class": t["class_name"],
                    "class_name": t["class_name"],
                    "confidence": t["confidence"],
                    "bbox": t["bbox"],
                    "center_x_px": round(px, 1),
                    "center_y_px": round(py, 1),
                    "x_m": coord[0] if coord[0] is not None else None,
                    "y_m": coord[1] if coord[1] is not None else None,
                    "world_x": coord[0] if coord[0] is not None else None,
                    "world_y": coord[1] if coord[1] is not None else None,
                    "speed_px_s": round(math.hypot(vx, vy), 1),
                    "direction": self.direction(vx, vy),
                    "state": t["state"],
                    "age_frames": t["age"],
                    "last_seen": datetime.fromtimestamp(t.get("last_seen_wall", time.time()), timezone.utc).isoformat(timespec="milliseconds"),
                    "timestamp": now(),
                    "safety_status": safety,
                    "safety_confidence": confidence,
                    "safety_reason": reason,
                    "recommended_action": action,
                    "visibility_level": max(0.0, min(1.0, float(t.get("confidence", 0.0)))),
                    "track_status": t["state"],
                    "identity_event": t.get("identity_event"),
                    "coordinate_frame": "calibrated_ground_plane" if coord[0] is not None else "image_pixels",
                })
            self._status_counts = {"active": sum(1 for t in result if t["state"] not in {"LOST", "MISSING"}), "lost": sum(1 for t in result if t["state"] in {"LOST", "MISSING"}), "warning": sum(1 for t in result if t["safety_status"] == "WARNING"), "critical": sum(1 for t in result if t["safety_status"] == "CRITICAL")}
            return result

    def localize(self, x, y):
        if cv2 is None or self.homography is None: return None, None
        import numpy as np
        p = cv2.perspectiveTransform(np.array([[[x, y]]], dtype="float32"), self.homography)[0][0]
        return round(float(p[0]), 2), round(float(p[1]), 2)

    def public_safety_summary(self):
        with self.lock:
            summaries = []
            counts = {"SAFE": 0, "WARNING": 0, "CRITICAL": 0, "MISSING": 0}
            for track in self.tracks:
                safety, confidence, reason, action = self._get_safety_state(track)
                track_id = int(track["id"])
                summaries.append({
                    "person_id": f"Person-{track_id:03d}",
                    "status": safety,
                    "confidence": round(float(confidence), 3),
                    "reason": reason,
                    "recommended_action": action,
                    "timestamp": datetime.fromtimestamp(track.get("last_seen_wall", time.time()), timezone.utc).isoformat(timespec="milliseconds"),
                })
                if safety in counts:
                    counts[safety] += 1
            priority = {"CRITICAL": 0, "MISSING": 1, "WARNING": 2, "SAFE": 3}
            summary = min(summaries, key=lambda item: priority[item["status"]]) if summaries else {
                "person_id": None,
                "status": "NO ACTIVE TRACKS",
                "confidence": None,
                "reason": "No person track is currently being assessed; this does not imply that the area is safe.",
                "recommended_action": "Check the camera feed and verify the scene.",
                "timestamp": now(),
            }
            industrial_alerts = self.industrial.active_alerts() if self.industrial is not None else []
            severity_priority = {"critical": 0, "warning": 2}
            if industrial_alerts:
                alert = min(industrial_alerts, key=lambda item: severity_priority.get(item["severity"], 3))
                alert_status = alert["severity"].upper()
                if priority.get(alert_status, 9) < priority.get(summary["status"], 9):
                    summary = {
                        "person_id": alert.get("track_id"),
                        "status": alert_status,
                        "confidence": None,
                        "reason": alert["message"],
                        "recommended_action": alert.get("details", {}).get(
                            "recommended_action",
                            "Notify the responsible controller and verify the hazard before taking action.",
                        ),
                        "timestamp": alert["timestamp"],
                        "source": "industrial_rule",
                    }
            counts["INDUSTRIAL"] = len(industrial_alerts)
            return {
                **summary,
                "counts": counts,
                "camera_online": bool(self.cap and self.cap.isOpened() and not self.camera_error),
                "assessment_scope": "webcam person-track heuristics; not a medical or threat-detection diagnosis",
                "human_verification_required": summary["status"] in {"WARNING", "CRITICAL", "MISSING"},
                "industrial_alert_count": len(industrial_alerts),
                "assistance_handoff": "local operator review only; no external delivery is configured",
            }

    def industrial_status(self):
        if self.industrial is None:
            return {"enabled": False, "active_alerts": [], "events": [], "quality_inspection": self.quality_inspector.status() if self.quality_inspector else {"enabled": False}}
        alerts = self.industrial.active_alerts()
        return {
            "enabled": True,
            "coordinate_frame": "calibrated_ground_plane" if self.homography is not None else "image_pixels",
            "active_alerts": alerts,
            "warning_count": sum(1 for alert in alerts if alert["severity"] == "warning"),
            "critical_count": sum(1 for alert in alerts if alert["severity"] == "critical"),
            "events": self.industrial.events(40),
            "quality_inspection": self.quality_inspector.status() if self.quality_inspector else {"enabled": False},
            "controller_action": "Advisory only: notify a human controller; this demo cannot control machinery.",
        }

    @staticmethod
    def direction(vx, vy):
        if math.hypot(vx, vy) < 8:
            return "STATIONARY"
        return ("S" if vy > 0 else "N") + ("E" if vx > 0 else "W")

    def status(self):
        online = bool(self.cap and self.cap.isOpened() and not self.camera_error)
        det_name = "YOLO person detector" if self.yolo_model is not None else ("custom trained detector" if self.custom_detector else ("OpenCV HOG people baseline" if self.hog else (self.detector_error or "Unavailable (install opencv-python)")))
        detector_active = self.yolo_model is not None or self.custom_detector is not None or (self.hog is not None and self.mode in ("human", "industrial"))
        active_people = sum(1 for t in self.tracks if str(t.get("state", "TRACKING")).upper() not in {"LOST", "MISSING"})
        lost_people = sum(1 for t in self.tracks if str(t.get("state", "TRACKING")).upper() in {"LOST", "MISSING"})
        critical = sum(1 for t in self.tracks if str(t.get("safety_status", "SAFE")).upper() == "CRITICAL")
        warning = sum(1 for t in self.tracks if str(t.get("safety_status", "SAFE")).upper() == "WARNING")
        return {
            "product": "OFFLINE VISION-X",
            "mode": self.mode,
            "lan_access_urls": list(self.lan_access_urls),
            "mobile_access_hint": "Connect this device and the laptop to the same trusted Wi-Fi, then open one listed URL on the phone.",
            "camera_online": online,
            "camera_status": "online" if online else "unavailable",
            "camera_error": self.camera_error,
            "camera_message": self.camera_error or "Camera connected",
            "detector": det_name if detector_active else ("No industrial detector configured" if self.mode == "industrial" else "Unavailable (install opencv-python)"),
            "model_status": "YOLO" if self.yolo_model is not None else ("CUSTOM TRAINED MODEL" if self.custom_detector else ("BASELINE" if self.hog and self.mode in ("human", "industrial") else "NO DETECTOR")),
            "fps": round(self.fps, 1),
            "processor": self.gpu_name,
            "device": self.gpu_name if self.detector_device.startswith("cuda") else "CPU",
            "gpu_available": self.gpu_available,
            "gpu_active": self.detector_device.startswith("cuda"),
            "device_type": self.detector_device,
            "detector_note": self.detector_error,
            "inference_time_ms": round(self.inference_ms, 2),
            "processing_ms": round(self.processing_ms, 1),
            "latency_ms": round(self.processing_ms, 1),
            "active_tracks": active_people,
            "lost_tracks": lost_people,
            "detection_count": len(self.tracks),
            "active_persons": active_people,
            "lost_persons": lost_people,
            "critical": critical,
            "warnings": warning,
            "localization": "calibrated homography" if self.homography is not None else "not calibrated",
            "localization_status": "Calibrated camera plane" if self.homography is not None else "Image pixel coordinates · camera calibration not configured",
            "coordinate_unit": "METERS" if self.homography is not None else "IMAGE PIXELS",
            "industrial_monitor": "active" if self.industrial is not None and self.mode == "industrial" else "inactive",
            "room_width_m": config("localization.yaml").get("room_width_m", 10),
            "room_height_m": config("localization.yaml").get("room_height_m", 8),
            "frame_width": int(self.frame.shape[1]) if self.frame is not None else int(config("camera.yaml").get("width", 640)),
            "frame_height": int(self.frame.shape[0]) if self.frame is not None else int(config("camera.yaml").get("height", 480)),
            "offline": True,
            "lost_track_timeout_seconds": self.tracker.max_lost_seconds,
            "reid_threshold": self.tracker.reid_threshold,
            "fps_readout": f"{round(self.fps, 1)} FPS",
        }


class Handler(BaseHTTPRequestHandler):
    app: VisionApp
    def log_message(self,*args): pass
    def send_json(self,obj):
        data=json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(data)
    def send_csv(self, path, download_name):
        if not path.is_file():
            self.send_error(404, "CSV log is not available yet")
            return
        try:
            data = path.read_bytes()
        except OSError:
            LOGGER.exception("Could not read CSV log %s", path)
            self.send_error(500, "Could not read CSV log")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/api/status": return self.send_json(self.app.status())
        if path=="/api/tracks": return self.send_json(self.app.track_payload())
        if path=="/api/events": return self.send_json(list(self.app.events))
        if path=="/api/public-safety": return self.send_json(self.app.public_safety_summary())
        if path=="/api/industrial/status": return self.send_json(self.app.industrial_status())
        if path=="/api/quality/status":
            quality = self.app.quality_inspector.status() if self.app.quality_inspector is not None else {"enabled": False}
            return self.send_json(quality)
        if path=="/api/assistance": return self.send_json(list(self.app.assistance_requests))
        if path=="/api/mode": return self.send_json({"mode":self.app.mode})
        if path=="/api/logs":
            logs = {
                "tracking": (self.app.tracking_csv_path, "person_tracking.csv"),
                "events": (self.app.event_csv_path, "person_events.csv"),
                "assistance": (self.app.assistance_csv_path, "assistance_requests.csv"),
                "quality": (self.app.quality_csv_path, "quality_inspection.csv"),
            }
            return self.send_json({
                name: {
                    "url": f"/api/logs/{name}.csv",
                    "available": file_path.is_file(),
                    "bytes": file_path.stat().st_size if file_path.is_file() else 0,
                    "updated_at": datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds") if file_path.is_file() else None,
                }
                for name, (file_path, _) in logs.items()
            })
        if path.startswith("/api/logs/") and path.endswith(".csv"):
            logs = {
                "tracking": (self.app.tracking_csv_path, "person_tracking.csv"),
                "events": (self.app.event_csv_path, "person_events.csv"),
                "assistance": (self.app.assistance_csv_path, "assistance_requests.csv"),
                "quality": (self.app.quality_csv_path, "quality_inspection.csv"),
            }
            log_name = path[len("/api/logs/"):-len(".csv")]
            if log_name not in logs:
                self.send_error(404)
                return
            file_path, download_name = logs[log_name]
            return self.send_csv(file_path, download_name)
        if path=="/video.mjpg":
            if cv2 is None or self.app.cap is None or self.app.camera_error:
                self.send_error(503, "CAMERA UNAVAILABLE"); return
            self.send_response(200); self.send_header("Cache-Control","no-store"); self.send_header("Content-Type","multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            while True:
                if self.app.camera_error or self.app.cap is None: break
                with self.app.lock: frame=self.app.frame.copy() if self.app.frame is not None else None
                if frame is None:
                    time.sleep(.25); continue
                ok,buf=cv2.imencode(".jpg",frame)
                if not ok: continue
                try: self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"); time.sleep(.05)
                except (BrokenPipeError,ConnectionResetError): break
            return
        if path.startswith("/dashboard/"):
            name = Path(path.removeprefix("/dashboard/")).name
            if name in {"app.js", "styles.css", "public-safety.css"}:
                file = ROOT / name
                if file.exists():
                    data = file.read_bytes()
                    content_type = "text/javascript; charset=utf-8" if name.endswith(".js") else "text/css; charset=utf-8"
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
        if path in ("/",""):
            file = ROOT / "index.html"
            if file.exists():
                data = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length=int(self.headers.get("Content-Length","0")); body=json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/assistance":
                request = self.app.request_assistance(
                    body.get("person_id"),
                    body.get("reason"),
                    body.get("operator_confirmed"),
                )
                return self.send_json({
                    "request": request,
                    "message": "Saved locally for operator review. No face data or external notification was sent.",
                })
            if path != "/api/mode":
                self.send_error(404); return
            mode=body.get("mode")
            if mode not in ("human","industrial"):
                self.send_error(400,"mode must be human or industrial"); return
            self.app.mode=mode
            status = self.app.status()
            self.send_json({"mode":mode,"model_status":status["model_status"],"detector":status["detector"],"device":status["device"]})
        except (ValueError,TypeError) as exc:
            self.send_error(400, str(exc) or "invalid JSON")
        except OSError:
            LOGGER.exception("Could not persist assistance request")
            self.send_error(500, "Could not save assistance request locally")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--host",default=None); ap.add_argument("--port",type=int,default=8765); ap.add_argument("--camera",type=int,default=None); ap.add_argument("--mode",choices=["human","industrial"],default=None)
    args=ap.parse_args(); nc=config("network.yaml"); cc=config("camera.yaml")
    app=VisionApp(args.camera if args.camera is not None else int(cc.get("device_id",0)), args.mode or config("system.yaml").get("mode","human"))
    Handler.app=app; app.start()
    host=args.host or nc.get("host","0.0.0.0")
    server=ThreadingHTTPServer((host,args.port),Handler)
    app.server_port = server.server_port
    app.lan_access_urls = [f"http://{address}:{server.server_port}" for address in discover_lan_ipv4_addresses()] if host in ("0.0.0.0", "::") else []
    print(f"Offline Vision-X: http://127.0.0.1:{server.server_port}  (LAN bind {host}; camera: {app.camera_error or 'online'})")
    if app.lan_access_urls:
        print("Same-Wi-Fi phone access:")
        for url in app.lan_access_urls:
            print(f"  {url}")
    elif host not in ("127.0.0.1", "localhost", "::1"):
        print("No private IPv4 address was detected. Check that laptop and phone are on the same Wi-Fi.")
    print("Allow Python through Windows Firewall on Private networks if phone access is blocked.")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        app.running=False; server.server_close()
        if app.cap: app.cap.release()

if __name__=="__main__": main()
