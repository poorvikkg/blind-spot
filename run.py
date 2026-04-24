"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Unified CLI Entry Point
  File: run.py
=============================================================================
  Single command that launches any mode of the system:

    python run.py                        # Camera + basic window
    python run.py --dashboard            # Camera + rich stats dashboard
    python run.py --sim                  # Simulator + basic window
    python run.py --sim --dashboard      # Simulator + dashboard (best for demo)
    python run.py --sim --rain --dashboard
    python run.py --sim --night --fast --dashboard
    python run.py --calibrate            # Interactive ROI calibration tool
    python run.py --check                # Pre-flight system check
    python run.py --yolo                 # Enable YOLO (overrides config)
=============================================================================
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Blind Spot Detection System — Truck Safety AI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python run.py --check                   Pre-flight system check
  python run.py --calibrate               Interactive ROI calibrator
  python run.py                           Live camera, basic window
  python run.py --dashboard               Live camera, stats dashboard
  python run.py --sim --dashboard         Simulator + dashboard (no camera needed)
  python run.py --sim --rain --dashboard  Simulator in rain, with dashboard
  python run.py --yolo --dashboard        AI mode with YOLO (requires ultralytics)
        """,
    )

    # ── Mode selection ────────────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--check",     action="store_true",
                            help="Run pre-flight dependency checks and exit")
    mode_group.add_argument("--calibrate", action="store_true",
                            help="Launch interactive ROI calibrator")

    # ── Pipeline options ──────────────────────────────────────────────────────
    parser.add_argument("--dashboard", action="store_true",
                        help="Show rich analytics dashboard panel")
    parser.add_argument("--sim",       action="store_true",
                        help="Use synthetic traffic simulator (no real camera)")
    parser.add_argument("--rain",      action="store_true",
                        help="[sim] Add rain / dust noise to synthetic feed")
    parser.add_argument("--night",     action="store_true",
                        help="[sim] Low-light night scene in simulator")
    parser.add_argument("--fast",      action="store_true",
                        help="[sim] Faster synthetic vehicles")
    parser.add_argument("--yolo",      action="store_true",
                        help="Enable YOLO AI vehicle classification (overrides config)")
    parser.add_argument("--camera",    type=str, default=None,
                        help="Camera source override (int index, RTSP URL, or file path)")
    parser.add_argument("--engine",    type=str, default=None,
                        choices=["MOG2", "KNN", "FRAME_DIFF"],
                        help="Motion detection engine override")

    args = parser.parse_args()

    # ── Apply CLI overrides to config ─────────────────────────────────────────
    import config

    if args.camera is not None:
        # Try to cast to int (webcam index), keep as string if it fails
        try:
            config.CAMERA_INDEX = int(args.camera)
        except ValueError:
            config.CAMERA_INDEX = args.camera
        print(f"[run.py] Camera source overridden → {config.CAMERA_INDEX}")

    if args.engine is not None:
        config.MOTION_ENGINE = args.engine
        print(f"[run.py] Motion engine overridden → {config.MOTION_ENGINE}")

    if args.yolo:
        config.USE_YOLO = True
        print("[run.py] YOLO mode enabled (overrides config.USE_YOLO).")

    # ── Route to correct mode ─────────────────────────────────────────────────

    if args.check:
        # ── Pre-flight check ──────────────────────────────────────────────────
        import setup_check
        setup_check.main()
        return

    if args.calibrate:
        # ── ROI calibrator ────────────────────────────────────────────────────
        import roi_calibrator
        roi_calibrator.run_calibrator()
        return

    if args.dashboard:
        # ── Dashboard (+ optional simulator) ─────────────────────────────────
        import dashboard
        dashboard.run(use_simulator=args.sim, sim_args=args)
        return

    if args.sim:
        # ── Simulator → basic window ──────────────────────────────────────────
        import simulate
        # Override CLI args into simulate.args namespace
        simulate.args.rain  = args.rain
        simulate.args.night = args.night
        simulate.args.fast  = args.fast
        simulate.main()
        return

    # ── Default: live camera, basic window ────────────────────────────────────
    import blind_spot_detection
    blind_spot_detection.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[run.py] Interrupted by user.")
        sys.exit(0)
