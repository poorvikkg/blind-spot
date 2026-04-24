"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Pre-Flight Setup Checker
  File: setup_check.py
=============================================================================
  Run this BEFORE starting blind_spot_detection.py to verify:
    ✅ Python version (≥3.8)
    ✅ OpenCV installed and correct version
    ✅ NumPy installed
    ✅ Camera accessible and returning frames
    ✅ Disk space for logs (≥50 MB free)
    ✅ GPIO library (Raspberry Pi only)
    ✅ YOLO model file (if USE_YOLO is enabled)
    ✅ Log directory writable

  Exit codes:
    0 — All checks passed
    1 — One or more critical checks failed
=============================================================================
"""

import sys
import os
import platform
import shutil


# ── ANSI colour codes (work on Linux/macOS/Windows 10+) ─────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ok(msg):    print(f"  {GREEN}✅  {msg}{RESET}")
def fail(msg):  print(f"  {RED}❌  {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠️   {msg}{RESET}")
def info(msg):  print(f"  {CYAN}ℹ️   {msg}{RESET}")


def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")


# =============================================================================
#  Individual checks
# =============================================================================

def check_python() -> bool:
    section("Python Version")
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"Python {major}.{minor}.{sys.version_info.micro}"
    if (major, minor) >= (3, 8):
        ok(f"{version_str} — OK")
        return True
    else:
        fail(f"{version_str} is too old. Requires Python ≥ 3.8")
        return False


def check_opencv() -> bool:
    section("OpenCV")
    try:
        import cv2
        ver = cv2.__version__
        major = int(ver.split(".")[0])
        if major >= 4:
            ok(f"opencv-python {ver}")
            # Check optimisations
            build = cv2.getBuildInformation()
            if "NEON" in build or "SSE" in build or "AVX" in build:
                info("Hardware SIMD optimisations detected (faster processing).")
            return True
        else:
            warn(f"OpenCV {ver} detected but ≥4.0 recommended.")
            return True  # Non-critical
    except ImportError:
        fail("OpenCV not installed. Run: pip install opencv-python")
        return False


def check_numpy() -> bool:
    section("NumPy")
    try:
        import numpy as np
        ok(f"NumPy {np.__version__}")
        return True
    except ImportError:
        fail("NumPy not installed. Run: pip install numpy")
        return False


def check_camera() -> bool:
    section("Camera / Video Source")
    try:
        import cv2
        import config

        source = config.CAMERA_INDEX
        info(f"Testing source: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            fail(f"Cannot open '{source}'. Check CAMERA_INDEX in config.py.")
            cap.release()
            return False

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            fail("Camera opened but returned no frames.")
            return False

        h, w = frame.shape[:2]
        ok(f"Camera OK — returned {w}×{h} frame.")
        return True

    except Exception as e:
        fail(f"Camera check error: {e}")
        return False


def check_disk_space() -> bool:
    section("Disk Space (for logs)")
    try:
        import config
        log_dir = os.path.dirname(os.path.abspath(config.LOG_FILE_PATH))
        total, used, free = shutil.disk_usage(log_dir if os.path.exists(log_dir) else ".")
        free_mb = free // (1024 * 1024)
        if free_mb >= 50:
            ok(f"{free_mb} MB free — sufficient.")
            return True
        else:
            warn(f"Only {free_mb} MB free. Logs may fill disk. Consider reducing LOG_MAX_BYTES.")
            return True  # Non-critical warning
    except Exception as e:
        warn(f"Disk space check failed: {e}")
        return True


def check_log_dir() -> bool:
    section("Log Directory (write access)")
    try:
        import config
        log_dir = os.path.dirname(config.LOG_FILE_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        test_file = os.path.join(log_dir or ".", ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        ok(f"Log directory '{log_dir or '.'}' is writable.")
        return True
    except Exception as e:
        fail(f"Log directory not writable: {e}")
        return False


def check_gpio() -> bool:
    section("GPIO / Raspberry Pi (optional)")
    try:
        import config
        if not config.ENABLE_GPIO_ALERT:
            info("GPIO disabled in config.py (ENABLE_GPIO_ALERT = False). Skipping.")
            return True

        import RPi.GPIO as GPIO  # type: ignore
        ok(f"RPi.GPIO available — Buzzer: BCM{config.GPIO_BUZZER_PIN}, LED: BCM{config.GPIO_LED_PIN}")
        return True

    except ImportError:
        if config.ENABLE_GPIO_ALERT:
            fail("RPi.GPIO not found but ENABLE_GPIO_ALERT=True. "
                 "Install on Pi: sudo apt install python3-rpi.gpio")
            return False
        else:
            info("RPi.GPIO not installed (expected on non-Pi systems).")
            return True
    except Exception as e:
        warn(f"GPIO check error: {e}")
        return True


def check_yolo() -> bool:
    section("YOLO (optional AI upgrade)")
    try:
        import config
        if not config.USE_YOLO:
            info("YOLO disabled (USE_YOLO = False). Skipping.")
            return True

        try:
            from ultralytics import YOLO  # type: ignore
            ok("ultralytics package found.")
        except ImportError:
            fail("USE_YOLO=True but ultralytics not installed. "
                 "Run: pip install ultralytics")
            return False

        model_path = config.YOLO_MODEL_PATH
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            ok(f"Model '{model_path}' found ({size_mb:.1f} MB).")
        else:
            warn(f"Model '{model_path}' not found — will auto-download on first run (~6 MB).")
        return True

    except Exception as e:
        warn(f"YOLO check error: {e}")
        return True


def check_platform() -> bool:
    section("Platform Info")
    plat = platform.system()
    proc = platform.processor() or platform.machine()
    info(f"OS: {plat} {platform.release()}")
    info(f"CPU: {proc}")
    info(f"Python: {sys.executable}")

    # Raspberry Pi-specific hint
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        if "Raspberry Pi" in cpuinfo or "BCM" in cpuinfo:
            warn("Raspberry Pi detected. For best performance, set "
                 "FRAME_WIDTH=320, FRAME_HEIGHT=240 in config.py.")
    except Exception:
        pass  # Not on Pi or not Linux
    return True


# =============================================================================
#  Run all checks
# =============================================================================

def main():
    # Fix unicode printing on Windows (cp1252)
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    # Enable ANSI on Windows
    if platform.system() == "Windows":
        os.system("")

    print(f"\n{BOLD}{'='*55}")
    print("  BLIND SPOT DETECTION — Pre-Flight Setup Check")
    print(f"{'='*55}{RESET}\n")

    results = {}

    results["Platform"]   = check_platform()
    results["Python"]     = check_python()
    results["OpenCV"]     = check_opencv()
    results["NumPy"]      = check_numpy()
    results["Camera"]     = check_camera()
    results["Disk Space"] = check_disk_space()
    results["Log Dir"]    = check_log_dir()
    results["GPIO"]       = check_gpio()
    results["YOLO"]       = check_yolo()

    # ── Summary ──────────────────────────────────────────────────────────
    section("Summary")
    all_pass = True
    for name, passed in results.items():
        if passed:
            ok(name)
        else:
            fail(name)
            all_pass = False

    print()
    if all_pass:
        print(f"{BOLD}{GREEN}  ✅  All checks passed! System is ready.{RESET}")
        print(f"{CYAN}  Run: python blind_spot_detection.py{RESET}\n")
        sys.exit(0)
    else:
        print(f"{BOLD}{RED}  ❌  Some checks failed. Fix the issues above before running.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
