"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Alert System
  File: alert_system.py
=============================================================================
  Handles all driver-alert outputs:
    • Console / log warning
    • GPIO buzzer  (Raspberry Pi only)
    • GPIO LED     (Raspberry Pi only)

  On a laptop (ENABLE_GPIO_ALERT = False) GPIO calls are silently skipped.
=============================================================================
"""

import time
import threading

import config
from logger_system import get_logger

log = get_logger("AlertSystem")

# ── Optional GPIO import (Raspberry Pi only) ──────────────────────────────────
GPIO = None
if config.ENABLE_GPIO_ALERT:
    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(config.GPIO_BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(config.GPIO_LED_PIN,    GPIO.OUT, initial=GPIO.LOW)
        log.info(
            "GPIO initialised — Buzzer: BCM%d | LED: BCM%d",
            config.GPIO_BUZZER_PIN,
            config.GPIO_LED_PIN,
        )
    except ImportError:
        log.warning("RPi.GPIO not found. GPIO alerts disabled (running on non-Pi?).")
        GPIO = None


# =============================================================================
#  AlertSystem Class
# =============================================================================
class AlertSystem:
    """
    Manages alert state and output.

    Design decisions
    ----------------
    * Alerts have a *cooldown* to prevent buzzer/LED from firing every frame.
    * GPIO pulses are run in a daemon thread so they never block the main loop.
    * `alert_active` flag drives the red ROI border on the video overlay.
    """

    def __init__(self) -> None:
        self.alert_active: bool = False
        self._cooldown_counter: int = 0           # Counts down frames between alerts
        self._led_on: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, object_detected: bool) -> None:
        """
        Call once per frame with the current detection state.

        Args:
            object_detected: True if a vehicle-sized object is in the ROI.
        """
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        if object_detected:
            self.alert_active = True
            if self._cooldown_counter == 0:
                self._fire_alert()
                self._cooldown_counter = config.ALERT_COOLDOWN_FRAMES
        else:
            self.alert_active = False
            self._deactivate_led()

    def cleanup(self) -> None:
        """Release GPIO resources on shutdown."""
        self._deactivate_led()
        if GPIO is not None:
            GPIO.cleanup()
            log.info("GPIO cleaned up.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fire_alert(self) -> None:
        """Trigger all enabled alert channels."""
        if config.ENABLE_CONSOLE_ALERT:
            log.warning("⚠️  BLIND SPOT ALERT — Vehicle detected in No-Zone!")

        if GPIO is not None:
            # Run GPIO in a thread to keep video loop non-blocking
            threading.Thread(
                target=self._gpio_pulse, daemon=True
            ).start()
        
        self._activate_led()

    def _gpio_pulse(self) -> None:
        """Beep buzzer for configured duration (runs in background thread)."""
        try:
            GPIO.output(config.GPIO_BUZZER_PIN, GPIO.HIGH)
            time.sleep(config.BUZZER_DURATION_SEC)
            GPIO.output(config.GPIO_BUZZER_PIN, GPIO.LOW)
        except Exception as exc:
            log.error("GPIO buzzer error: %s", exc)

    def _activate_led(self) -> None:
        if GPIO is not None and not self._led_on:
            try:
                GPIO.output(config.GPIO_LED_PIN, GPIO.HIGH)
                self._led_on = True
            except Exception as exc:
                log.error("GPIO LED ON error: %s", exc)

    def _deactivate_led(self) -> None:
        if GPIO is not None and self._led_on:
            try:
                GPIO.output(config.GPIO_LED_PIN, GPIO.LOW)
                self._led_on = False
            except Exception as exc:
                log.error("GPIO LED OFF error: %s", exc)
