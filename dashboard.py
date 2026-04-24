"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Rich Analytics Dashboard
  File: dashboard.py
=============================================================================
  Drops a second panel alongside the camera feed showing live statistics:

    ┌──────────────────────────┬────────────────────────┐
    │                          │  BLIND SPOT DASHBOARD  │
    │   ANNOTATED CAMERA       │  ─────────────────────  │
    │   FEED (left panel)      │  FPS Graph             │
    │                          │  Alert Timeline        │
    │                          │  Object Count          │
    │                          │  Session Stats         │
    └──────────────────────────┴────────────────────────┘

  Run:
      python dashboard.py
      python dashboard.py --sim           (with synthetic vehicles)
      python dashboard.py --sim --rain    (simulator + rain)
=============================================================================
"""

import argparse
import collections
import datetime
import sys
import time
from typing import Deque, List, Tuple

import cv2
import numpy as np

import config
from alert_system import AlertSystem
from blind_spot_detection import (
    build_bg_subtractor,
    compute_roi,
    crop_roi,
    draw_overlay,
    extract_vehicle_contours,
    FrameDifferencer,
    open_camera,
    preprocess,
    apply_motion_detection,
)
from logger_system import get_logger
from tracker import CentroidTracker

log = get_logger("Dashboard")

# ── Optional YOLO ─────────────────────────────────────────────────────────────
yolo_detector = None
if config.USE_YOLO:
    try:
        from yolo_detector import YOLODetector
        yolo_detector = YOLODetector()
    except Exception as e:
        log.warning("YOLO not loaded: %s", e)


# =============================================================================
#  Dashboard Constants
# =============================================================================

PANEL_W       = 280                # Width of right stats panel
GRAPH_H       = 90                 # Height of FPS sparkline graph
TIMELINE_H    = 80                 # Height of alert timeline
HISTORY_LEN   = 90                 # Frames of history kept (~3s at 30fps)

BG_COLOR      = (18,  18,  30)     # Dark navy background
ACCENT_BLUE   = (255, 160,  60)    # Orange-blue accent (BGR)
ACCENT_GREEN  = ( 50, 220, 100)
ACCENT_RED    = ( 60,  60, 230)
ACCENT_YELLOW = ( 40, 210, 230)
TEXT_WHITE    = (240, 240, 240)
TEXT_DIM      = (120, 120, 140)
GRAPH_LINE    = ( 80, 200, 255)
ALERT_COLOR   = ( 60,  60, 230)
SAFE_COLOR    = ( 50, 200,  80)


# =============================================================================
#  Drawing Helpers
# =============================================================================

def draw_panel_background(panel: np.ndarray) -> np.ndarray:
    panel[:] = BG_COLOR
    return panel


def draw_header(panel: np.ndarray, y: int, title: str, color=ACCENT_BLUE) -> int:
    """Draw a section header, return next y position."""
    cv2.putText(panel, title, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.line(panel, (10, y + 4), (PANEL_W - 10, y + 4), color, 1)
    return y + 16


def draw_kv(panel: np.ndarray, y: int, key: str, value: str,
            val_color=TEXT_WHITE) -> int:
    """Draw a key: value row, return next y."""
    cv2.putText(panel, key, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(panel, value, (PANEL_W // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, val_color, 1, cv2.LINE_AA)
    return y + 16


def draw_sparkline(panel: np.ndarray, y_top: int, data: Deque,
                   label: str, max_val: float, color=GRAPH_LINE) -> int:
    """Draw a mini line graph for a deque of float values."""
    h = GRAPH_H
    w = PANEL_W - 20
    x0, y0 = 10, y_top

    # Background rect
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (30, 30, 46), -1)
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (60, 60, 80), 1)

    values = list(data)
    if len(values) >= 2 and max_val > 0:
        pts = []
        for i, v in enumerate(values):
            px = x0 + int(i * w / max(len(values) - 1, 1))
            py = y0 + h - int(min(v, max_val) / max_val * (h - 4)) - 2
            pts.append((px, py))
        for i in range(len(pts) - 1):
            cv2.line(panel, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)
        # Latest value dot
        cv2.circle(panel, pts[-1], 3, color, -1)

    # Label
    if values:
        last = values[-1]
        cv2.putText(panel, f"{label}: {last:.1f}", (x0 + 3, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return y_top + h + 8


def draw_alert_timeline(panel: np.ndarray, y_top: int,
                        history: Deque) -> int:
    """Draw a binary alert/safe timeline bar."""
    h  = TIMELINE_H
    w  = PANEL_W - 20
    x0 = 10

    cv2.rectangle(panel, (x0, y_top), (x0 + w, y_top + h), (30, 30, 46), -1)
    cv2.rectangle(panel, (x0, y_top), (x0 + w, y_top + h), (60, 60, 80), 1)

    values = list(history)
    bar_h  = h - 20  # Reserve bottom for label
    n      = len(values)

    for i, alert in enumerate(values):
        bx1 = x0 + int(i * w / max(n, 1))
        bx2 = x0 + int((i + 1) * w / max(n, 1))
        color = ALERT_COLOR if alert else (30, 30, 46)
        cv2.rectangle(panel, (bx1, y_top + 2),
                      (bx2, y_top + bar_h), color, -1)

    # Labels
    cv2.putText(panel, "ALERT HISTORY (last 90 frames)",
                (x0 + 2, y_top + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    return y_top + h + 8


def draw_object_count_bar(panel: np.ndarray, y: int,
                          count: int, max_display: int = 5) -> int:
    """Draw a simple count indicator with filled squares."""
    cv2.putText(panel, "VEHICLES IN ZONE:", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)
    y += 14
    sq_size = 16
    gap = 5
    for i in range(max_display):
        x1 = 12 + i * (sq_size + gap)
        color = ALERT_COLOR if i < count else (40, 40, 55)
        cv2.rectangle(panel, (x1, y), (x1 + sq_size, y + sq_size), color, -1)
        cv2.rectangle(panel, (x1, y), (x1 + sq_size, y + sq_size), (80, 80, 100), 1)
    if count > max_display:
        cv2.putText(panel, f"+{count - max_display}",
                    (12 + max_display * (sq_size + gap), y + sq_size - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT_YELLOW, 1, cv2.LINE_AA)
    return y + sq_size + 10


def draw_status_badge(panel: np.ndarray, y: int, alert: bool) -> int:
    text  = "  !! DANGER !!" if alert else "    SAFE"
    color = ALERT_COLOR if alert else SAFE_COLOR
    w_box = PANEL_W - 20
    cv2.rectangle(panel, (10, y), (10 + w_box, y + 28), color, -1)
    cv2.putText(panel, text, (18, y + 19),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return y + 36


# =============================================================================
#  Session Stats Tracker
# =============================================================================

class SessionStats:
    def __init__(self):
        self.start_time     = time.time()
        self.total_alerts   = 0
        self.total_frames   = 0
        self.max_vehicles   = 0
        self._was_alert     = False

    def update(self, alert: bool, vehicle_count: int):
        self.total_frames += 1
        self.max_vehicles  = max(self.max_vehicles, vehicle_count)
        if alert and not self._was_alert:
            self.total_alerts += 1
        self._was_alert = alert

    @property
    def uptime(self) -> str:
        elapsed = int(time.time() - self.start_time)
        return f"{elapsed // 60:02d}:{elapsed % 60:02d}"

    @property
    def avg_fps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_frames / max(elapsed, 1)


# =============================================================================
#  Build the stats panel image
# =============================================================================

def build_stats_panel(
    fps_history:    Deque,
    count_history:  Deque,
    alert_history:  Deque,
    vehicle_count:  int,
    alert_active:   bool,
    stats:          SessionStats,
    frame_h:        int,
) -> np.ndarray:
    panel = np.zeros((frame_h, PANEL_W, 3), dtype=np.uint8)
    draw_panel_background(panel)

    y = 14

    # ── Title ────────────────────────────────────────────────────────────
    cv2.putText(panel, "BLIND SPOT SYSTEM", (10, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, ACCENT_BLUE, 1, cv2.LINE_AA)
    y += 14
    now = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(panel, now, (PANEL_W - 62, y - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.line(panel, (10, y), (PANEL_W - 10, y), ACCENT_BLUE, 1)
    y += 10

    # ── Status badge ──────────────────────────────────────────────────────
    y = draw_status_badge(panel, y, alert_active)

    # ── Vehicle count ─────────────────────────────────────────────────────
    y = draw_object_count_bar(panel, y, vehicle_count)

    # ── FPS Sparkline ─────────────────────────────────────────────────────
    y = draw_header(panel, y, "LIVE FPS", ACCENT_GREEN)
    y = draw_sparkline(panel, y, fps_history, "FPS", 60.0, GRAPH_LINE)

    # ── Alert Timeline ────────────────────────────────────────────────────
    y = draw_header(panel, y, "ALERT TIMELINE", ACCENT_RED)
    y = draw_alert_timeline(panel, y, alert_history)

    # ── Session Stats ─────────────────────────────────────────────────────
    y = draw_header(panel, y, "SESSION STATS", ACCENT_YELLOW)
    y = draw_kv(panel, y, "Uptime",        stats.uptime,         TEXT_WHITE)
    y = draw_kv(panel, y, "Avg FPS",       f"{stats.avg_fps:.1f}", TEXT_WHITE)
    y = draw_kv(panel, y, "Total Alerts",  str(stats.total_alerts),
                ALERT_COLOR if stats.total_alerts > 0 else SAFE_COLOR)
    y = draw_kv(panel, y, "Max Vehicles",  str(stats.max_vehicles), ACCENT_YELLOW)
    y = draw_kv(panel, y, "Total Frames",  str(stats.total_frames), TEXT_DIM)
    y = draw_kv(panel, y, "Engine",
                "YOLO" if yolo_detector else config.MOTION_ENGINE,
                ACCENT_BLUE)

    # ── Keyboard shortcuts ────────────────────────────────────────────────
    y = max(y, frame_h - 80)
    cv2.line(panel, (10, y), (PANEL_W - 10, y), (50, 50, 70), 1)
    y += 10
    shortcuts = ["Q:quit  P:pause  R:reset", "M:mask  S:screenshot"]
    for s in shortcuts:
        cv2.putText(panel, s, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
        y += 13

    return panel


# =============================================================================
#  Dashboard Main Loop
# =============================================================================

def run(use_simulator: bool = False, sim_args=None):
    # ── Simulator injection ───────────────────────────────────────────────────
    if use_simulator:
        from simulate import frame_generator, VirtualCamera
        rain  = getattr(sim_args, "rain",  False)
        night = getattr(sim_args, "night", False)
        fast  = getattr(sim_args, "fast",  False)
        gen = frame_generator(config.FRAME_WIDTH, config.FRAME_HEIGHT,
                              rain=rain, night=night, fast=fast)
        cap = VirtualCamera(gen)
        log.info("Dashboard running in SIMULATOR mode.")
    else:
        cap = open_camera()

    # ── Subsystems ────────────────────────────────────────────────────────────
    alerter     = AlertSystem()
    tracker     = CentroidTracker()
    subtractor  = build_bg_subtractor()
    differencer = FrameDifferencer()
    stats       = SessionStats()

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi     = compute_roi(frame_w, frame_h)

    # ── History deques ────────────────────────────────────────────────────────
    fps_history:   Deque[float] = collections.deque(maxlen=HISTORY_LEN)
    count_history: Deque[int]   = collections.deque(maxlen=HISTORY_LEN)
    alert_history: Deque[bool]  = collections.deque(maxlen=HISTORY_LEN)

    fps_counter  = 0
    fps_display  = 0.0
    fps_timer    = time.time()
    paused       = False
    show_mask    = False

    win_name = "Blind Spot Detection — Dashboard"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    total_w = frame_w + PANEL_W
    cv2.resizeWindow(win_name, total_w, frame_h)

    log.info("Dashboard started. Press Q to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            tracker.reset()
            log.info("Tracker reset.")
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('s'):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            import os; os.makedirs("screenshots", exist_ok=True)
            path = f"screenshots/dashboard_{ts}.jpg"
            if 'composite' in dir():
                cv2.imwrite(path, composite)
                log.info("Screenshot: %s", path)

        if paused:
            continue

        # ── Capture ───────────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── Pipeline ──────────────────────────────────────────────────────────
        roi_bgr      = crop_roi(frame, roi)
        preprocessed = preprocess(roi_bgr)
        motion_mask  = apply_motion_detection(preprocessed, subtractor, differencer)

        if yolo_detector:
            bboxes = yolo_detector.detect(roi_bgr)
        else:
            bboxes = extract_vehicle_contours(motion_mask)

        active_objects  = tracker.update(bboxes)
        confirmed       = [o for o in active_objects if o.is_confirmed]
        vehicle_count   = len(confirmed)

        vehicle_in_zone = vehicle_count > 0
        alerter.update(vehicle_in_zone)

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps_display  = fps_counter / elapsed
            fps_counter  = 0
            fps_timer    = time.time()

        # ── History ───────────────────────────────────────────────────────────
        fps_history.append(fps_display)
        count_history.append(vehicle_count)
        alert_history.append(alerter.alert_active)
        stats.update(alerter.alert_active, vehicle_count)

        # ── Draw camera panel ─────────────────────────────────────────────────
        annotated = draw_overlay(frame, roi, confirmed,
                                 alerter.alert_active, fps_display)

        # ── Draw stats panel ──────────────────────────────────────────────────
        panel = build_stats_panel(
            fps_history, count_history, alert_history,
            vehicle_count, alerter.alert_active, stats, frame_h
        )

        # ── Composite ─────────────────────────────────────────────────────────
        composite = np.hstack([annotated, panel])
        cv2.imshow(win_name, composite)

        if show_mask:
            cv2.imshow("Motion Mask", motion_mask)

    cap.release()
    alerter.cleanup()
    cv2.destroyAllWindows()
    log.info("Dashboard closed. Session stats: alerts=%d frames=%d",
             stats.total_alerts, stats.total_frames)


# =============================================================================
#  Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blind Spot Dashboard")
    parser.add_argument("--sim",   action="store_true", help="Use synthetic simulator")
    parser.add_argument("--rain",  action="store_true", help="Simulator: add rain noise")
    parser.add_argument("--night", action="store_true", help="Simulator: night mode")
    parser.add_argument("--fast",  action="store_true", help="Simulator: fast vehicles")
    args = parser.parse_args()

    run(use_simulator=args.sim, sim_args=args)
