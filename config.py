"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Configuration
  File: config.py
=============================================================================
  All tunable parameters are centralized here. Adjust these values to match
  your camera mounting position, lane width, and environmental conditions.
=============================================================================
"""

# ---------------------------------------------------------------------------
# CAMERA SETTINGS
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0            # 0 = default USB webcam, 1 = second cam, or RTSP string
FRAME_WIDTH  = 640          # Lower resolution → faster FPS on Pi
FRAME_HEIGHT = 480
TARGET_FPS   = 30           # Camera capture FPS

# ---------------------------------------------------------------------------
# REGION OF INTEREST (ROI) — BLIND SPOT ZONE
# ---------------------------------------------------------------------------
# Defined as (x, y, width, height) in pixels relative to frame dimensions.
# Tune these to cover only the side lane next to the truck.
#
#  ┌─────────────────────────────────┐
#  │        Sky / Road Ahead         │  ← ignored
#  ├────────────┬────────────────────┤
#  │            │   BLIND SPOT ROI   │  ← ROI_* covers this
#  │  Truck     │   (right lane)     │
#  │  Body      │                    │
#  └────────────┴────────────────────┘
#
# As a fraction of frame size (0.0 – 1.0):
ROI_X_RATIO      = 0.50   # Start ROI at 50% of frame width (right half)
ROI_Y_RATIO      = 0.30   # Start ROI at 30% of frame height (skip sky)
ROI_WIDTH_RATIO  = 0.50   # ROI spans 50% of frame width
ROI_HEIGHT_RATIO = 0.55   # ROI spans 55% of frame height

# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
GAUSSIAN_BLUR_KERNEL = (5, 5)   # Must be odd. Increase for heavy noise/rain.
GAUSSIAN_BLUR_SIGMA  = 0        # 0 = auto-calculated from kernel size

# ---------------------------------------------------------------------------
# BACKGROUND SUBTRACTION / MOTION DETECTION
# ---------------------------------------------------------------------------
# Choose engine: "MOG2" | "KNN" | "FRAME_DIFF"
MOTION_ENGINE = "MOG2"

# MOG2 / KNN parameters
BG_HISTORY          = 200       # Frames used to build background model
BG_VAR_THRESHOLD    = 50        # Sensitivity (lower = more sensitive)
BG_DETECT_SHADOWS   = False     # Shadow detection (slows Pi; keep False)

# Frame differencing threshold (used when MOTION_ENGINE = "FRAME_DIFF")
FRAME_DIFF_THRESHOLD = 30       # Pixel intensity diff to count as motion

# Morphological cleanup
MORPH_KERNEL_SIZE = (5, 5)      # Kernel for dilation/erosion
MORPH_ITERATIONS  = 2           # Number of dilation passes

# ---------------------------------------------------------------------------
# CONTOUR FILTERING — Vehicle Size Heuristics
# ---------------------------------------------------------------------------
# Tune MIN_CONTOUR_AREA based on how far vehicles appear in the camera FOV.
# A car 5m away may have area=4000px²; at 15m it may be 800px².
MIN_CONTOUR_AREA   = 1500       # px² — ignore smaller blobs (birds, debris)
MAX_CONTOUR_AREA   = 200_000    # px² — ignore impossibly large blobs

# Aspect-ratio filter (width/height bounding rect)
MIN_ASPECT_RATIO   = 0.4        # Vehicles are wider than tall typically
MAX_ASPECT_RATIO   = 4.5

# ---------------------------------------------------------------------------
# TRACKING & ALERT LOGIC
# ---------------------------------------------------------------------------
ALERT_FRAME_THRESHOLD   = 5     # Frames object must persist before alert fires
ALERT_COOLDOWN_FRAMES   = 30    # Min frames between repeated alerts (avoid buzzer spam)
MAX_TRACKING_LOST_FRAMES = 10   # Remove tracker if object unseen for this many frames

# Minimum centroid movement (px) to consider object "approaching"
APPROACH_MOVEMENT_THRESHOLD = 8

# ---------------------------------------------------------------------------
# ALERT OUTPUT
# ---------------------------------------------------------------------------
ENABLE_CONSOLE_ALERT = True
ENABLE_GPIO_ALERT    = False     # Set True on Raspberry Pi with GPIO wired

# Raspberry Pi GPIO BCM pin numbers
GPIO_BUZZER_PIN = 17            # Physical pin 11
GPIO_LED_PIN    = 27            # Physical pin 13

BUZZER_DURATION_SEC = 0.5       # How long buzzer beeps per alert

# ---------------------------------------------------------------------------
# DISPLAY / VISUALIZATION
# ---------------------------------------------------------------------------
SHOW_LIVE_FEED         = True   # Show annotated camera window
SHOW_MASK_WINDOW       = False  # Debug: show motion mask (useful for tuning)
SHOW_FPS               = True
SHOW_ROI_OVERLAY       = True

# Colors (BGR)
COLOR_ROI_NORMAL  = (0, 200, 0)      # Green — safe
COLOR_ROI_ALERT   = (0, 0, 255)      # Red — danger
COLOR_BBOX        = (0, 165, 255)    # Orange bounding boxes
COLOR_TEXT        = (255, 255, 255)  # White
COLOR_FPS         = (180, 180, 0)    # Yellow-ish

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
ENABLE_FILE_LOGGING = True
LOG_FILE_PATH       = "logs/blind_spot.log"
LOG_MAX_BYTES       = 5 * 1024 * 1024  # 5 MB before rotation
LOG_BACKUP_COUNT    = 3

# ---------------------------------------------------------------------------
# OPTIONAL YOLO UPGRADE
# ---------------------------------------------------------------------------
USE_YOLO            = False           # Set True to enable AI classification
YOLO_MODEL_PATH     = "yolov8n.pt"   # Nano model — fast on Pi
YOLO_CONFIDENCE     = 0.40
YOLO_CLASSES        = [2, 3, 5, 7]   # COCO: car=2, motorcycle=3, bus=5, truck=7
YOLO_INPUT_SIZE     = 320            # Smaller = faster, less accurate
