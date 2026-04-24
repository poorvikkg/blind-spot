"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Interactive ROI Calibrator
  File: roi_calibrator.py
=============================================================================
  Run this ONCE when mounting the camera to calibrate the blind spot zone.

  How to use:
    1. python roi_calibrator.py
    2. A live camera frame is shown.
    3. Click and drag a rectangle to define the blind spot ROI.
    4. Press ENTER to confirm — the script prints the exact config.py
       values to paste in.
    5. Press 'r' to re-draw, 'q' to quit without saving.
    6. A reference image (roi_reference.jpg) is saved for documentation.

  The tool also overlays a grid to help align the ROI with lane boundaries.
=============================================================================
"""

import sys
import cv2
import numpy as np

import config
from logger_system import get_logger

log = get_logger("ROICalibrator")

# ── State shared with mouse callback ─────────────────────────────────────────
_drag_start  = None
_drag_end    = None
_dragging    = False
_confirmed   = False


def _mouse_callback(event, x, y, flags, param):
    global _drag_start, _drag_end, _dragging, _confirmed

    if event == cv2.EVENT_LBUTTONDOWN:
        _drag_start = (x, y)
        _drag_end   = (x, y)
        _dragging   = True
        _confirmed  = False

    elif event == cv2.EVENT_MOUSEMOVE and _dragging:
        _drag_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        _drag_end  = (x, y)
        _dragging  = False


def _draw_grid(frame: np.ndarray, cols: int = 4, rows: int = 3) -> np.ndarray:
    """Draw a faint alignment grid over the frame."""
    h, w = frame.shape[:2]
    for i in range(1, cols):
        x = w * i // cols
        cv2.line(frame, (x, 0), (x, h), (60, 60, 60), 1)
    for j in range(1, rows):
        y = h * j // rows
        cv2.line(frame, (0, y), (w, y), (60, 60, 60), 1)
    return frame


def _draw_roi_rect(frame: np.ndarray, p1, p2, confirmed: bool) -> np.ndarray:
    """Draw the in-progress or confirmed ROI rectangle."""
    if p1 is None or p2 is None:
        return frame
    color     = (0, 255, 0) if confirmed else (0, 200, 255)
    thickness = 2
    cv2.rectangle(frame, p1, p2, color, thickness)

    # Semi-transparent fill
    overlay = frame.copy()
    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 80, 0), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Dimension label
    w_px = abs(p2[0] - p1[0])
    h_px = abs(p2[1] - p1[1])
    label = f"{w_px}x{h_px}px"
    cv2.putText(frame, label, (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _compute_ratios(frame_w, frame_h, p1, p2):
    """Convert pixel coordinates to config.py ratio values."""
    x1 = min(p1[0], p2[0])
    y1 = min(p1[1], p2[1])
    x2 = max(p1[0], p2[0])
    y2 = max(p1[1], p2[1])

    roi_x_ratio      = round(x1 / frame_w, 3)
    roi_y_ratio      = round(y1 / frame_h, 3)
    roi_width_ratio  = round((x2 - x1) / frame_w, 3)
    roi_height_ratio = round((y2 - y1) / frame_h, 3)

    return roi_x_ratio, roi_y_ratio, roi_width_ratio, roi_height_ratio


def run_calibrator():
    global _drag_start, _drag_end, _dragging, _confirmed

    log.info("Opening camera for ROI calibration...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        log.error("Cannot open camera '%s'", config.CAMERA_INDEX)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    win_name = "ROI Calibrator — Drag to select blind spot zone"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _mouse_callback)

    log.info("Instructions: drag ROI, ENTER=confirm, r=reset, q=quit")

    saved_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame capture failed")
            continue

        display = frame.copy()
        _draw_grid(display, cols=4, rows=3)

        # Draw current rectangle
        display = _draw_roi_rect(display, _drag_start, _drag_end, _confirmed)

        # Help text overlay
        instructions = [
            "Drag: draw ROI  |  ENTER: confirm  |  R: reset  |  Q: quit",
        ]
        for i, txt in enumerate(instructions):
            cv2.putText(display, txt, (6, frame_h - 12 - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 0), 1, cv2.LINE_AA)

        if _confirmed:
            saved_frame = display.copy()
            cv2.putText(display, "CONFIRMED — Press ENTER again to save or R to redo",
                        (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            log.info("Calibration cancelled.")
            break

        elif key == ord('r'):
            _drag_start = None
            _drag_end   = None
            _confirmed  = False
            log.info("ROI reset.")

        elif key == 13:  # ENTER
            if _drag_start is None or _drag_end is None:
                log.warning("No ROI drawn yet. Drag a rectangle first.")
                continue

            if not _confirmed:
                _confirmed = True
                ratios = _compute_ratios(frame_w, frame_h, _drag_start, _drag_end)
                print("\n" + "=" * 60)
                print("  ✅  PASTE THESE VALUES INTO config.py")
                print("=" * 60)
                print(f"  ROI_X_RATIO      = {ratios[0]}")
                print(f"  ROI_Y_RATIO      = {ratios[1]}")
                print(f"  ROI_WIDTH_RATIO  = {ratios[2]}")
                print(f"  ROI_HEIGHT_RATIO = {ratios[3]}")
                print("=" * 60 + "\n")
                log.info("ROI confirmed. Press ENTER again to save reference image.")
            else:
                # Save reference image
                ref_path = "roi_reference.jpg"
                if saved_frame is not None:
                    cv2.imwrite(ref_path, saved_frame)
                    log.info("Reference image saved to '%s'", ref_path)
                print(f"Reference image saved: {ref_path}")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibrator()
