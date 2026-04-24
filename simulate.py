"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Synthetic Traffic Simulator
  File: simulate.py
=============================================================================
  Tests the ENTIRE detection pipeline WITHOUT a real camera.

  Generates a fake road scene with:
    • A static background (asphalt texture + lane markings)
    • Randomly spawning "vehicles" (coloured rectangles) that enter the ROI
      from the right side and move left (simulating overtaking)
    • Occasional noise (random pixels) to test robustness
    • "Rain" mode that adds Gaussian noise on top

  Run:
      python simulate.py              # Normal mode
      python simulate.py --rain       # Rain noise mode
      python simulate.py --night      # Low-light mode
      python simulate.py --fast       # Speed up vehicles

  The simulator writes frames into a virtual camera feed by monkey-patching
  cv2.VideoCapture so the main pipeline needs ZERO changes.
=============================================================================
"""

import argparse
import random
import time
import sys
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

# ── Parse CLI args early so help works before imports ────────────────────────
parser = argparse.ArgumentParser(description="Blind Spot Simulator")
parser.add_argument("--rain",  action="store_true", help="Add Gaussian noise (rain)")
parser.add_argument("--night", action="store_true", help="Low-light scene")
parser.add_argument("--fast",  action="store_true", help="Faster vehicles")
parser.add_argument("--width",  type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

import config
from logger_system import get_logger

log = get_logger("Simulator")


# =============================================================================
#  Synthetic Vehicle
# =============================================================================

VEHICLE_COLORS = [
    (60,  120, 200),   # Blue car
    (30,  200,  80),   # Green truck
    (200,  60,  60),   # Red bus
    (200, 180,  60),   # Yellow van
    (180, 180, 180),   # Silver car
]


@dataclass
class SyntheticVehicle:
    x: float                    # Current x-position (float for sub-pixel movement)
    y: int                      # Fixed y (entry row)
    width: int                  # Bounding box width
    height: int                 # Bounding box height
    speed: float                # Pixels per frame (leftward movement)
    color: tuple = field(default_factory=lambda: random.choice(VEHICLE_COLORS))
    active: bool = True

    def step(self):
        self.x -= self.speed
        if self.x + self.width < 0:
            self.active = False

    def draw(self, canvas: np.ndarray):
        x1 = int(self.x)
        y1 = self.y
        x2 = x1 + self.width
        y2 = y1 + self.height
        # Body
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.color, -1)
        # Windshield highlight
        cv2.rectangle(canvas,
                      (x1 + 4, y1 + 4),
                      (x2 - 4, y1 + self.height // 3),
                      (220, 220, 240), -1)
        # Wheels
        for wx in [x1 + 8, x2 - 8]:
            cv2.circle(canvas, (wx, y2 + 4), 6, (20, 20, 20), -1)


# =============================================================================
#  Background Generator
# =============================================================================

def build_background(w: int, h: int, night: bool) -> np.ndarray:
    """Create a static road background."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky
    sky_h = h // 3
    sky_color = (30, 20, 10) if night else (180, 160, 100)
    bg[:sky_h, :] = sky_color

    # Road (asphalt grey)
    road_color = (40, 40, 40) if night else (70, 70, 70)
    bg[sky_h:, :] = road_color

    # Lane markings (dashed white lines)
    for lane_y in [sky_h + h // 6, sky_h + h // 3, sky_h + h // 2]:
        for x in range(0, w, 60):
            cv2.line(bg, (x, lane_y), (x + 30, lane_y), (200, 200, 200), 2)

    # Horizon line
    cv2.line(bg, (0, sky_h), (w, sky_h), (100, 100, 100), 1)
    return bg


# =============================================================================
#  Frame Generator (yields numpy frames)
# =============================================================================

def frame_generator(w: int, h: int, rain: bool, night: bool, fast: bool):
    background = build_background(w, h, night)
    vehicles: List[SyntheticVehicle] = []

    spawn_cooldown = 0
    frame_idx = 0

    # ROI bounds (absolute pixels, mirroring config ratios)
    roi_x = int(config.ROI_X_RATIO * w)
    roi_y = int(config.ROI_Y_RATIO * h)
    roi_w = int(config.ROI_WIDTH_RATIO * w)
    roi_h = int(config.ROI_HEIGHT_RATIO * h)

    base_speed = 6.0 if fast else 3.5

    while True:
        canvas = background.copy()

        # ── Spawn new vehicle ─────────────────────────────────────────────
        spawn_cooldown -= 1
        if spawn_cooldown <= 0:
            vw = random.randint(80, 140)
            vh = random.randint(45, 80)
            vy = roi_y + random.randint(0, max(1, roi_h - vh - 10))
            speed = base_speed + random.uniform(-1.0, 2.0)
            veh = SyntheticVehicle(x=float(w + 10), y=vy,
                                   width=vw, height=vh, speed=speed)
            vehicles.append(veh)
            spawn_cooldown = random.randint(40, 100)
            log.debug("Spawned vehicle at x=%d y=%d speed=%.1f", w + 10, vy, speed)

        # ── Update & draw vehicles ────────────────────────────────────────
        for v in vehicles[:]:
            v.step()
            if not v.active:
                vehicles.remove(v)
            else:
                v.draw(canvas)

        # ── ROI overlay (faint) ───────────────────────────────────────────
        overlay = canvas.copy()
        cv2.rectangle(overlay, (roi_x, roi_y),
                      (roi_x + roi_w, roi_y + roi_h), (0, 60, 0), -1)
        cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
        cv2.rectangle(canvas, (roi_x, roi_y),
                      (roi_x + roi_w, roi_y + roi_h), (0, 180, 0), 1)

        # ── HUD labels ───────────────────────────────────────────────────
        label = "SIMULATED FEED"
        color = (0, 140, 255) if not night else (0, 80, 200)
        cv2.putText(canvas, label, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        mode_parts = []
        if rain:  mode_parts.append("RAIN")
        if night: mode_parts.append("NIGHT")
        if fast:  mode_parts.append("FAST")
        if mode_parts:
            cv2.putText(canvas, " + ".join(mode_parts), (6, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)

        # ── Environmental noise ───────────────────────────────────────────
        if rain:
            # Gaussian noise simulates rain/dust scatter
            noise = np.random.normal(0, 22, canvas.shape).astype(np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Rain streaks
            for _ in range(120):
                rx = random.randint(0, w - 1)
                ry = random.randint(0, h - 8)
                cv2.line(canvas, (rx, ry), (rx - 1, ry + 7), (180, 180, 220), 1)

        if night:
            # Dark vignette
            canvas = (canvas * 0.45).astype(np.uint8)
            # Vehicle headlights (bright circles)
            for v in vehicles:
                cx = int(v.x) + v.width
                cy = v.y + v.height // 2
                cv2.circle(canvas, (cx, cy), 10, (220, 220, 180), -1)
                cv2.circle(canvas, (cx, cy), 22, (180, 180, 140), 1)

        frame_idx += 1
        yield canvas


# =============================================================================
#  Virtual Camera — monkey-patches cv2.VideoCapture
# =============================================================================

class VirtualCamera:
    """Behaves exactly like cv2.VideoCapture but yields synthetic frames."""

    def __init__(self, gen):
        self._gen = gen
        self._w   = args.width
        self._h   = args.height
        self._open = True

    def isOpened(self):       return self._open
    def release(self):        self._open = False

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._h)
        if prop_id == cv2.CAP_PROP_FPS:          return 30.0
        return 0.0

    def set(self, prop_id, value):
        return True  # Accept but ignore

    def read(self):
        try:
            frame = next(self._gen)
            time.sleep(1 / 30)   # Throttle to 30 FPS
            return True, frame
        except StopIteration:
            return False, None


# =============================================================================
#  Entry Point — Patch camera and run main pipeline
# =============================================================================

def main():
    log.info("Starting simulation — rain=%s night=%s fast=%s",
             args.rain, args.night, args.fast)

    gen = frame_generator(args.width, args.height,
                          rain=args.rain, night=args.night, fast=args.fast)
    virtual_cam = VirtualCamera(gen)

    # Patch open_camera in the main pipeline module
    import blind_spot_detection as bsd
    original_open = bsd.open_camera

    def patched_open_camera():
        log.info("Virtual camera injected (simulator mode).")
        return virtual_cam

    bsd.open_camera = patched_open_camera

    try:
        bsd.run()
    finally:
        bsd.open_camera = original_open


if __name__ == "__main__":
    main()
