# Blind Spot Detection System — Truck Safety AI

A real-time, computer vision-based blind spot detection system designed for large commercial vehicles (trucks, buses). 

The system uses a camera feed to monitor the "No-Zone" (blind spot) alongside the vehicle, analyses the feed for vehicle-sized objects using motion detection heuristics or AI classification (YOLOv8), and triggers visual, auditory, and hardware alerts (Raspberry Pi GPIO) when a vehicle enters the danger zone.

---

## 🚀 Features

- **Real-Time Processing**: Designed to run efficiently on limited hardware, including Raspberry Pi.
- **Dynamic Region of Interest (ROI)**: Fully configurable blind spot zone to ignore sky, the truck body, and the road ahead.
- **Motion Engine Variability**: Supports multiple classical computer vision background subtractors (`MOG2`, `KNN`, Frame Differencing) to adapt to varying lighting and environmental conditions.
- **Object Tracking**: Custom lightweight centroid tracker matching detections frame-by-frame, removing noise, and calculating "approach" vectors.
- **Hardware Integration**: Built-in GPIO support (`alert_system.py`) to trigger a buzzer and LED on Raspberry Pi.
- **AI Override (YOLOv8)**: An optional drop-in module replacing classical motion detection with an Ultralytics YOLO inference model for high-accuracy vehicle classification.
- **Simulation Mode**: Built-in synthetic traffic simulator with modifiers like rain and night modes for testing without access to a physical camera.
- **Analytics Dashboard**: Rich analytics and stats dashboard using a terminal/GUI overlay. 

---

## 📂 Project Structure

- `run.py` — The unified Command Line Interface (CLI) application entry point. Routes to different modes (e.g. camera, sim, dashboard, check, calibration).
- `blind_spot_detection.py` — The core image processing pipeline encompassing ROI cropping, motion detection, and bounding-box drawing.
- `config.py` — Centralized configuration file holding all hyperparameters (resolution, thresholds, engine selection, GPIO pins).
- `tracker.py` — A mathematically lightweight Centroid Tracker to map frame-to-frame vehicle detections.
- `alert_system.py` — Manages the alert states triggering console logs, visual cues, and optional hardware GPIO pins.
- `yolo_detector.py` — The Ultralytics YOLOv8 inference wrapper. Only activated if enabled in config or via CLI.
- `roi_calibrator.py` — Interactive tool to calibrate ROI visually.
- `dashboard.py` / `logger_system.py` / `simulate.py` / `setup_check.py` — Utility scripts for monitoring logging, simulating traffic, and system setup checks.

---

## ⚙️ How It Works: The Pipeline

The system processes video feeds through a rigorous, high-speed per-frame loop:

1. **Camera Feed & Crop**: Fetches the video frame and immediately crops it using the defined `ROI_RATIO`s to isolate solely the adjacent lane (the blind spot).
2. **Preprocessing**: The cropped Region of Interest (ROI) is converted to grayscale and blurred (Gaussian Blur) to remove environmental noise and camera vibration artifacts.
3. **Motion Detection**: The image is fed into a configured engine (`MOG2`, `KNN`, or `FRAME_DIFF`) creating a binary mask isolating moving elements from the static background. Morphological operations (erode/dilate) close holes in detection blobs.
4. **Contour Filtering / AI Override**:
    - **Classical Path**: Filters contours mathematically by Min/Max area and Aspect Ratio (vehicles are typically wider than they are tall).
    - **AI Path (YOLO)**: Bypasses classical heuristics and strictly searches for COCO classes: cars, buses, motorcycles, and trucks.
5. **Tracking**: The Centroid Tracker compares newly found objects against objects detected in the previous frame, applying an ID and tracking approaching movements based on centroid deltas.
6. **Alert System**: If a tracked vehicle passes verification criteria (detected consecutively over a frame threshold), the system triggers bounded box coloring (Red) and fires the Buzzer/LED GPIO pins. 

---

## 🖥️ Usage

The system is controlled entirely through the `run.py` interface.

### Standard Modes
```bash
# Standard live camera feed with basic window visualization
python run.py

# Live camera with rich analytic statistics dashboard
python run.py --dashboard

# Pre-flight hardware and dependency check
python run.py --check
```

### Simulation & Testing Modes
```bash
# Run the synthetic traffic simulator (does not require camera)
python run.py --sim

# Run the simulator with environmental noise variations
python run.py --sim --rain --dashboard
python run.py --sim --night --fast --dashboard
```

### Advanced Setups & Overrides
```bash
# Launch interactive ROI calibrator tool
python run.py --calibrate

# Turn on YOLO AI Vehicle Classification globally
python run.py --yolo --dashboard
```

*(Note: Advanced camera and engine overrides can also be passed like `--camera 1` or `--engine KNN`)*

---

## 🛠️ Configuration (`config.py`)

All core tuning happens inside `config.py`. Important configurations include:

### 1. Region of Interest (ROI) Settings
Tweak these values (0.0 - 1.0) to frame your specific camera angle. 
- `ROI_X_RATIO`: X coordinate starting percent (e.g. `0.50` ignores the left-half truck body).
- `ROI_Y_RATIO`: Y coordinate starting percent (e.g. `0.30` ignores the top-quarter sky).
- `ROI_WIDTH_RATIO` / `ROI_HEIGHT_RATIO`: Proportions of the total ROI block.

### 2. Motion Engine
- `MOTION_ENGINE = "MOG2"`: Recommended for general highway conditions because it actively learns and subtracts gradual shadows. Alternatives include `"KNN"` and `"FRAME_DIFF"`.

### 3. GPIO Settings (Raspberry Pi specific)
- `ENABLE_GPIO_ALERT`: Set to `True` to enable actual hardware output.
- `GPIO_BUZZER_PIN = 17`
- `GPIO_LED_PIN = 27`

---

## 🧠 Optional YOLO Upgrade

The classical motion detection handles basic situations admirably, but it may struggle with stopped vehicles (they become part of the background). The system natively incorporates an AI YOLOv8 fallback.

**Installation for YOLO**:
```bash
pip install ultralytics
```
**Activation**:
Change `USE_YOLO = True` inside `config.py` or run your script with `python run.py --yolo`.

By default, the `yolov8n.pt` (Nano) model is used because it presents the best balance of speed vs. accuracy for low-power edge computing devices.
