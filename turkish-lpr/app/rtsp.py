"""
app/rtsp.py — Threaded RTSP frame grabber with auto-reconnect.

Provides the RTSPStream class which continuously reads frames from an
i-PRO IP camera in a background thread.  Only the most recent frame is
kept, eliminating buffering lag.

Design decisions (from camera-test.py, refined for production use):
  - TCP transport is forced via OPENCV_FFMPEG_CAPTURE_OPTIONS to avoid
    UDP routing issues on dual-NIC Windows setups.
  - A background daemon thread reads frames as fast as the camera sends
    them.  The main thread can call read() at any time to get the latest
    frame without blocking.
  - On consecutive read failures (configurable), the stream tears down
    and reconnects automatically.
  - All parameters come from app.config.Settings — no hardcoded values.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger("turkish_lpr.rtsp")


# ─────────────────────────────────────────────────────────────────────────────
# Frame container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameData:
    """A captured frame with metadata."""
    frame: np.ndarray
    timestamp: float          # time.time() when the frame was grabbed
    frame_id: int             # monotonically increasing counter
    width: int = 0
    height: int = 0

    def __post_init__(self):
        if self.frame is not None:
            self.height, self.width = self.frame.shape[:2]


# ─────────────────────────────────────────────────────────────────────────────
# RTSPStream
# ─────────────────────────────────────────────────────────────────────────────

class RTSPStream:
    """
    Continuously grabs frames from an RTSP stream in a background thread.

    Usage:
        stream = RTSPStream(settings)
        stream.start()
        ...
        frame_data = stream.read()   # latest frame or None
        ...
        stream.stop()
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[FrameData] = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._frame_counter = 0

        # Force TCP transport BEFORE any VideoCapture is created.
        # This is the single most important line for dual-NIC stability.
        transport = self._settings.rtsp_transport.lower()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
        logger.info("RTSP transport forced to %s", transport.upper())

    # ── Connection management ───────────────────────────────────────────────

    def _open(self) -> bool:
        """Attempt to open the RTSP stream.  Returns True on success."""
        url = self._settings.rtsp_url
        url_masked = self._settings.rtsp_url_masked
        logger.info("Connecting to %s", url_masked)

        # Release any previous capture
        self._release_cap()

        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            logger.error("Failed to open RTSP stream")
            return False

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("Stream opened — %dx%d @ %.1f FPS", w, h, fps)
        return True

    def _release_cap(self):
        """Safely release the current VideoCapture object."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # ── Background reader ───────────────────────────────────────────────────

    def _reader_loop(self):
        """
        Continuously read frames; store only the latest one.
        Reconnects automatically on sustained failures.
        """
        consecutive_failures = 0
        reconnect_delay = self._settings.rtsp_reconnect_delay
        max_failures = self._settings.rtsp_max_failures

        while self._running:
            # ── (Re)connect if needed ───────────────────────────────────────
            if not self._connected:
                if self._open():
                    self._connected = True
                    consecutive_failures = 0
                else:
                    logger.warning(
                        "Reconnecting in %ds …", reconnect_delay
                    )
                    # Sleep in small increments so stop() is responsive
                    for _ in range(reconnect_delay * 10):
                        if not self._running:
                            return
                        time.sleep(0.1)
                    continue

            # ── Read one frame ──────────────────────────────────────────────
            ret, frame = self._cap.read()

            if ret and frame is not None:
                consecutive_failures = 0
                self._frame_counter += 1
                fd = FrameData(
                    frame=frame,
                    timestamp=time.time(),
                    frame_id=self._frame_counter,
                )
                with self._lock:
                    self._latest = fd
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error(
                        "%d consecutive read failures — reconnecting",
                        consecutive_failures,
                    )
                    self._connected = False
                    self._release_cap()
                    consecutive_failures = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self):
        """Start the background reader thread."""
        if self._running:
            logger.warning("Stream is already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="rtsp-reader"
        )
        self._thread.start()
        logger.info("Background frame grabber started")

    def read(self) -> Optional[FrameData]:
        """
        Return the most recent FrameData, or None if no frame is ready.
        Thread-safe.  Returns a copy of the frame array.
        """
        with self._lock:
            if self._latest is None:
                return None
            return FrameData(
                frame=self._latest.frame.copy(),
                timestamp=self._latest.timestamp,
                frame_id=self._latest.frame_id,
            )

    def wait_for_frame(self, timeout: float = 15.0) -> Optional[FrameData]:
        """
        Block until a frame is available or timeout expires.
        Returns FrameData or None on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            fd = self.read()
            if fd is not None:
                return fd
            time.sleep(0.1)
        return None

    def stop(self):
        """Signal the reader thread to stop and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._release_cap()
        logger.info("Background frame grabber stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self._connected
