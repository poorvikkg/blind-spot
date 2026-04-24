"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Main Pipeline
  File: blind_spot_detection.py
=============================================================================
  Entry point for the real-time blind spot monitor.

  Processing pipeline (per frame):
  ─────────────────────────────────────────────────────────────────────────
   [Camera] → [ROI Mask] → [Preprocess] → [Motion Detect] →
   [Contour Filter] → [Tracker] → [Alert] → [Visualize] → [Display]
  ─────────────────────────────────────────────────────────────────────────

  Run:
      python blind_spot_detection.py

  Press  q  to quit,  p  to pause,  r  to reset tracker,
         m  to toggle mask window,  s  to save a screenshot.
=============================================================================
"""

import os
import sys
import time
import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from alert_system import AlertSystem
from logger_system import get_logger
from tracker import CentroidTracker, TrackedObject

log = get_logger("Pipeline")

# ── Optional YOLO import ──────────────────────────────────────────────────────
yolo_detector = None
if config.USE_YOLO:
    try:
        from yolo_detector import YOLODetector
        yolo_detector = YOLODetector()
        log.info("YOLO upgrade active.")
    except Exception as exc:
        log.warning("YOLO unavailable (%s). Falling back to contour mode.", exc)
        yolo_detector = None


# =============================================================================
#  STAGE 1 — Camera initialisation
# =============================================================================

def open_camera() -> cv2.VideoCapture:
    """
    Open the video source and configure resolution/FPS.

    Supports:
      • Integer index (USB webcam)
      • RTSP URL string  (IP camera)
      • File path         (recorded video for testing)

    Returns:
        Opened VideoCapture object.

    Raises:
        RuntimeError: If camera cannot be opened.
    """
    log.info("Opening camera source: %s", config.CAMERA_INDEX)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera '{config.CAMERA_INDEX}'. "
            "Check CAMERA_INDEX in config.py."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    log.info("Camera opened: %dx%d @ %.1f FPS", actual_w, actual_h, actual_fps)
    return cap


# =============================================================================
#  STAGE 2 — Region of Interest (ROI) Masking
# =============================================================================

def compute_roi(frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """
    Convert config ratio-based ROI definition to absolute pixel coordinates.

    The ROI is chosen to cover only the adjacent blind-spot lane and excludes:
      • Sky              (top portion)
      • Road far ahead   (centre of frame)
      • Truck body       (left portion for right-side camera)

    Args:
        frame_w: Full frame width in pixels.
        frame_h: Full frame height in pixels.

    Returns:
        Tuple (x, y, width, height) in pixels.

    Tuning tip:
        Set SHOW_ROI_OVERLAY = True and SHOW_MASK_WINDOW = True in config.py
        to visualise the zone live while adjusting ratios.
    """
    x = int(config.ROI_X_RATIO      * frame_w)
    y = int(config.ROI_Y_RATIO      * frame_h)
    w = int(config.ROI_WIDTH_RATIO  * frame_w)
    h = int(config.ROI_HEIGHT_RATIO * frame_h)

    # Clamp to frame boundaries
    x = min(x, frame_w - 1)
    y = min(y, frame_h - 1)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    log.debug("ROI: x=%d y=%d w=%d h=%d", x, y, w, h)
    return x, y, w, h


def crop_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract the ROI sub-image from the full frame (no copy — view)."""
    x, y, w, h = roi
    return frame[y: y + h, x: x + w]


# =============================================================================
#  STAGE 3 — Preprocessing
# =============================================================================

def preprocess(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Convert ROI to grayscale and apply Gaussian blur.

    Why Gaussian blur?
      Cameras on moving trucks pick up vibration noise and JPEG compression
      artefacts. The blur low-passes the signal so subsequent thresholding
      doesn't fire on pixels shaking by 1–2 DN values.

    Args:
        roi_bgr: Raw BGR ROI crop.

    Returns:
        Blurred grayscale image (uint8).
    """
    gray    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(
        gray,
        config.GAUSSIAN_BLUR_KERNEL,
        config.GAUSSIAN_BLUR_SIGMA,
    )
    return blurred


# =============================================================================
#  STAGE 4 — Motion Detection
# =============================================================================

def build_bg_subtractor() -> cv2.BackgroundSubtractor:
    """
    Instantiate the configured background subtraction engine.

    MOG2  — Mixture of Gaussians, handles gradual illumination change well.
             Best choice for highway conditions (shadows, passing trees).
    KNN   — K-Nearest Neighbours, slightly slower but more accurate for
             scenes with many background pixels.
    FRAME_DIFF — Raw 3-frame differencing; fastest but struggles with stopped
             vehicles or slow-moving objects.

    Returns:
        OpenCV BackgroundSubtractor instance (or None for FRAME_DIFF).
    """
    engine = config.MOTION_ENGINE.upper()
    if engine == "MOG2":
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.BG_HISTORY,
            varThreshold=config.BG_VAR_THRESHOLD,
            detectShadows=config.BG_DETECT_SHADOWS,
        )
        log.info("Motion engine: MOG2 (history=%d, thresh=%d)",
                 config.BG_HISTORY, config.BG_VAR_THRESHOLD)
    elif engine == "KNN":
        subtractor = cv2.createBackgroundSubtractorKNN(
            history=config.BG_HISTORY,
            dist2Threshold=400,
            detectShadows=config.BG_DETECT_SHADOWS,
        )
        log.info("Motion engine: KNN")
    else:
        subtractor = None
        log.info("Motion engine: Frame Differencing (threshold=%d)",
                 config.FRAME_DIFF_THRESHOLD)
    return subtractor


class FrameDifferencer:
    """
    Simple 3-frame differencing for motion detection.

    Computes:  mask = |f(t-1) - f(t)| AND |f(t) - f(t+1)| > threshold
    Using a rolling buffer of the last two preprocessed frames.
    """

    def __init__(self) -> None:
        self._prev: Optional[np.ndarray] = None

    def apply(self, gray: np.ndarray) -> np.ndarray:
        """
        Args:
            gray: Current preprocessed (grayscale + blurred) ROI frame.

        Returns:
            Binary motion mask (uint8, 0 or 255).
        """
        if self._prev is None:
            self._prev = gray
            return np.zeros_like(gray)

        diff = cv2.absdiff(self._prev, gray)
        self._prev = gray.copy()
        _, mask = cv2.threshold(
            diff, config.FRAME_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
        )
        return mask


def apply_motion_detection(
    preprocessed: np.ndarray,
    subtractor: Optional[cv2.BackgroundSubtractor],
    frame_differencer: FrameDifferencer,
) -> np.ndarray:
    """
    Generate a binary foreground mask using the configured engine.

    Args:
        preprocessed:      Blurred grayscale ROI.
        subtractor:        MOG2/KNN subtractor (None if using FRAME_DIFF).
        frame_differencer: FrameDifferencer instance.

    Returns:
        Binary mask (uint8) — 255 = foreground / motion, 0 = background.
    """
    if subtractor is not None:
        mask = subtractor.apply(preprocessed)
        # MOG2 returns 128 for shadows; binarise to 0/255
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    else:
        mask = frame_differencer.apply(preprocessed)

    # ── Morphological cleanup ─────────────────────────────────────────────
    # Erosion removes tiny noise specks; Dilation fills holes inside vehicles
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                            iterations=config.MORPH_ITERATIONS)

    return mask


# =============================================================================
#  STAGE 5 — Contour Detection & Filtering
# =============================================================================

def extract_vehicle_contours(
    mask: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """
    Find contours in the motion mask and filter to vehicle-probable shapes.

    Filtering criteria:
      1. Area   : MIN_CONTOUR_AREA ≤ area ≤ MAX_CONTOUR_AREA
      2. Aspect : MIN_ASPECT_RATIO ≤ w/h ≤ MAX_ASPECT_RATIO

    Why aspect ratio?
      A bird or leaf blowing past may have a large enough area but will have
      an extreme aspect ratio (very thin/tall or tiny square).
      Vehicles tend to be wider than they are tall → ratio 0.8–4.0.

    Args:
        mask: Binary foreground mask from motion detection stage.

    Returns:
        List of (x, y, w, h) bounding rectangles for candidate vehicles
        in ROI-relative coordinates.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (config.MIN_CONTOUR_AREA <= area <= config.MAX_CONTOUR_AREA):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if not (config.MIN_ASPECT_RATIO <= aspect <= config.MAX_ASPECT_RATIO):
            continue

        candidates.append((x, y, w, h))

    log.debug("Contours total=%d, vehicle candidates=%d",
              len(contours), len(candidates))
    return candidates


# =============================================================================
#  STAGE 6 — Visualization Helper
# =============================================================================

def draw_overlay(
    frame:      np.ndarray,
    roi:        Tuple[int, int, int, int],
    objects:    List[TrackedObject],
    alert:      bool,
    fps:        float,
) -> np.ndarray:
    """
    Annotate the full frame with:
      • ROI rectangle (green = safe, red = alert)
      • Bounding boxes for each confirmed tracked object
      • Approach indicator arrow
      • FPS counter
      • Status banner

    Args:
        frame:   Full BGR frame (will be drawn on in place).
        roi:     (x, y, w, h) of the blind spot ROI.
        objects: Active tracked objects.
        alert:   True if alert is active.
        fps:     Current FPS to display.

    Returns:
        Annotated frame (same array).
    """
    rx, ry, rw, rh = roi
    roi_color = config.COLOR_ROI_ALERT if alert else config.COLOR_ROI_NORMAL
    thickness = 3 if alert else 2

    # ── ROI rectangle ─────────────────────────────────────────────────────
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), roi_color, thickness)
    cv2.putText(
        frame, "BLIND SPOT ZONE",
        (rx + 4, ry + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA,
    )

    # ── Semi-transparent ROI fill ─────────────────────────────────────────
    overlay = frame.copy()
    alpha = 0.15 if not alert else 0.30
    fill_color = (0, 80, 0) if not alert else (0, 0, 120)
    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), fill_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Bounding boxes (ROI-relative → frame-absolute) ────────────────────
    for obj in objects:
        if not obj.is_confirmed:
            continue
        bx, by, bw, bh = obj.bbox
        abs_x1 = rx + bx
        abs_y1 = ry + by
        abs_x2 = abs_x1 + bw
        abs_y2 = abs_y1 + bh

        cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2),
                      config.COLOR_BBOX, 2)

        # Object ID tag
        label = f"ID:{obj.object_id}"
        if obj.is_approaching:
            label += " APPR."
        cv2.putText(
            frame, label,
            (abs_x1, abs_y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            config.COLOR_BBOX, 1, cv2.LINE_AA,
        )

        # Centroid dot
        cx = rx + obj.centroid[0]
        cy = ry + obj.centroid[1]
        cv2.circle(frame, (cx, cy), 4, config.COLOR_BBOX, -1)

    # ── Alert status banner ───────────────────────────────────────────────
    if alert:
        banner_text = "!! VEHICLE IN BLIND SPOT !!"
        (tw, th), _ = cv2.getTextSize(
            banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2
        )
        bx_start = (frame.shape[1] - tw) // 2
        # Red background strip
        cv2.rectangle(
            frame,
            (bx_start - 10, 8),
            (bx_start + tw + 10, 38),
            (0, 0, 200), -1,
        )
        cv2.putText(
            frame, banner_text,
            (bx_start, 30),
            cv2.FONT_HERSHEY_DUPLEX, 0.75,
            config.COLOR_TEXT, 2, cv2.LINE_AA,
        )

    # ── FPS counter ───────────────────────────────────────────────────────
    if config.SHOW_FPS:
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (8, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            config.COLOR_FPS, 1, cv2.LINE_AA,
        )

    # ── Safe / Danger badge ───────────────────────────────────────────────
    status_text  = "DANGER" if alert else "SAFE"
    status_color = config.COLOR_ROI_ALERT if alert else config.COLOR_ROI_NORMAL
    cv2.putText(
        frame, status_text,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        status_color, 2, cv2.LINE_AA,
    )

    return frame


# =============================================================================
#  Main Loop
# =============================================================================

def run() -> None:
    """
    Main entry point — initialises all subsystems and runs the per-frame loop.

    Keyboard shortcuts:
        q — quit
        p — pause / resume
        r — reset tracker
        m — toggle mask debug window
        s — save screenshot to screenshots/
    """
    # ── Init subsystems ───────────────────────────────────────────────────
    cap          = open_camera()
    alerter      = AlertSystem()
    tracker      = CentroidTracker()
    subtractor   = build_bg_subtractor()
    differencer  = FrameDifferencer()

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi     = compute_roi(frame_w, frame_h)

    os.makedirs("screenshots", exist_ok=True)

    # FPS measurement
    fps_counter   = 0
    fps_display   = 0.0
    fps_timer     = time.time()

    paused        = False
    show_mask_win = config.SHOW_MASK_WINDOW

    log.info("System running — press 'q' to quit, 'p' pause, 'r' reset, 's' screenshot")

    while True:
        # ── Keyboard input ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log.info("User quit.")
            break
        elif key == ord('p'):
            paused = not paused
            log.info("Paused = %s", paused)
        elif key == ord('r'):
            tracker.reset()
            log.info("Tracker reset.")
        elif key == ord('m'):
            show_mask_win = not show_mask_win
            if not show_mask_win:
                cv2.destroyWindow("Motion Mask")
        elif key == ord('s'):
            ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshots/bsd_{ts}.jpg"
            cv2.imwrite(path, frame if 'frame' in dir() else np.zeros((10, 10, 3), np.uint8))
            log.info("Screenshot saved: %s", path)

        if paused:
            continue

        # ── Capture frame ──────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame capture failed — trying to reconnect...")
            time.sleep(0.1)
            continue

        # ── STAGE 2: ROI crop ──────────────────────────────────────────────
        roi_bgr = crop_roi(frame, roi)

        # ── STAGE 3: Preprocessing ─────────────────────────────────────────
        preprocessed = preprocess(roi_bgr)

        # ── STAGE 4: Motion detection ──────────────────────────────────────
        motion_mask = apply_motion_detection(preprocessed, subtractor, differencer)

        # ── STAGE 5: Contour / YOLO detection ─────────────────────────────
        if yolo_detector is not None:
            # AI path — skip contour, use YOLO bboxes directly
            bboxes = yolo_detector.detect(roi_bgr)
        else:
            # Classical path
            bboxes = extract_vehicle_contours(motion_mask)

        # ── STAGE 6: Tracking ──────────────────────────────────────────────
        active_objects = tracker.update(bboxes)
        confirmed      = [o for o in active_objects if o.is_confirmed]

        # ── STAGE 7: Alert ─────────────────────────────────────────────────
        vehicle_in_blind_spot = len(confirmed) > 0
        alerter.update(vehicle_in_blind_spot)

        # ── FPS calculation ────────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display  = fps_counter / elapsed
            fps_counter  = 0
            fps_timer    = time.time()

        # ── Visualization ──────────────────────────────────────────────────
        if config.SHOW_LIVE_FEED:
            frame = draw_overlay(
                frame, roi, confirmed, alerter.alert_active, fps_display
            )
            cv2.imshow("Blind Spot Detection — Truck Safety System", frame)

        if show_mask_win:
            cv2.imshow("Motion Mask", motion_mask)

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    alerter.cleanup()
    cv2.destroyAllWindows()
    log.info("System shut down cleanly.")


# =============================================================================
#  Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        run()
    except RuntimeError as e:
        log.critical("Fatal error: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C).")
        sys.exit(0)
