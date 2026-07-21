"""
camera-test.py — Stable RTSP Live Stream Viewer for i-PRO (Panasonic) IP Cameras

Designed for a Windows dual-network setup where:
  - Ethernet (192.168.1.59) connects to the camera (192.168.1.68)
  - Wi-Fi   (192.168.30.59)  connects to the internet

Key features:
  1. Forces TCP transport to avoid UDP routing/firewall issues on dual NICs.
  2. Threaded frame grabbing to eliminate buffering lag.
  3. Automatic reconnection on connection loss (no crashes).
  4. Resized display window (960×540) with clean 'q'-key exit.

Usage:
  pip install opencv-python
  python camera-test.py
"""

import os
import sys
import time
import threading
import cv2


# =============================================================================
# Configuration
# =============================================================================

# ── RTSP credentials & URL ──────────────────────────────────────────────────
CAMERA_USER = "admin"
CAMERA_PASS = "Panasonic1"          # ← Replace with your actual password
CAMERA_IP   = "192.168.1.68"
STREAM_PATH = "MediaInput/stream_1"

RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}/{STREAM_PATH}"

# ── Display settings ────────────────────────────────────────────────────────
DISPLAY_WIDTH  = 960
DISPLAY_HEIGHT = 540
WINDOW_NAME    = "i-PRO Camera — Live Feed"

# ── Reconnect settings ─────────────────────────────────────────────────────
RECONNECT_DELAY_SEC = 3           # Seconds to wait between reconnection attempts
MAX_CONSECUTIVE_FAILURES = 30     # Frames of consecutive read failures before reconnect


# =============================================================================
# Force FFmpeg to use TCP for RTSP (critical for dual-NIC stability)
# =============================================================================
# This MUST be set before any cv2.VideoCapture call.  Without it, FFmpeg
# defaults to UDP, which frequently hangs or times out when multiple network
# interfaces are active and the OS picks the wrong route for UDP packets.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
print("[INIT] RTSP transport forced to TCP.", flush=True)

# ── Verify that OpenCV was built with GUI support ───────────────────────────
# opencv-python-headless does not include imshow/destroyAllWindows.
# Detect this early so the user gets a clear message instead of a crash.
try:
    cv2.namedWindow("__gui_check__")
    cv2.destroyWindow("__gui_check__")
except cv2.error:
    print(
        "\n[ERROR] OpenCV GUI support is missing.\n"
        "  You probably have opencv-python-headless installed.\n"
        "  Fix:\n"
        "    pip uninstall opencv-python opencv-python-headless -y\n"
        "    pip install opencv-python\n",
        flush=True,
    )
    sys.exit(1)


# =============================================================================
# Threaded Frame Grabber
# =============================================================================
class RTSPStream:
    """
    Continuously grabs frames from an RTSP stream in a background thread.

    Why threading?
    cv2.VideoCapture.read() is blocking — it waits for the next frame from the
    network buffer.  If the main loop can't consume frames fast enough, the
    buffer grows and introduces visible lag (sometimes several seconds).

    By reading in a dedicated thread, we always have the *latest* frame ready
    for the main loop, and older buffered frames are silently discarded.
    """

    def __init__(self, url: str):
        self.url = url
        self.frame = None
        self.is_running = False
        self.is_connected = False
        self.lock = threading.Lock()
        self.cap = None

    # ── Connection management ───────────────────────────────────────────────
    def _open_stream(self) -> bool:
        """Attempt to open the RTSP stream.  Returns True on success."""
        print(f"[CONNECT] Opening stream: rtsp://****:****@{CAMERA_IP}/{STREAM_PATH}", flush=True)

        # Release any previous capture object
        if self.cap is not None:
            self.cap.release()

        # Use the FFMPEG backend explicitly for RTSP
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            print("[CONNECT] ✗ Failed to open stream.", flush=True)
            return False

        # Log basic stream properties
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CONNECT] ✓ Stream opened — {w}×{h} @ {fps:.1f} FPS", flush=True)
        return True

    # ── Background reader thread ────────────────────────────────────────────
    def _reader_loop(self):
        """
        Continuously reads frames and stores only the most recent one.
        Older frames are discarded to guarantee real-time display.
        """
        consecutive_failures = 0

        while self.is_running:
            # ── If not connected, attempt to (re)connect ────────────────────
            if not self.is_connected:
                if self._open_stream():
                    self.is_connected = True
                    consecutive_failures = 0
                else:
                    print(
                        f"[RECONNECT] Retrying in {RECONNECT_DELAY_SEC}s …",
                        flush=True,
                    )
                    time.sleep(RECONNECT_DELAY_SEC)
                    continue

            # ── Grab the next frame ─────────────────────────────────────────
            ret, frame = self.cap.read()

            if ret and frame is not None:
                consecutive_failures = 0
                with self.lock:
                    self.frame = frame
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(
                        f"[STREAM] ✗ {consecutive_failures} consecutive read "
                        f"failures — triggering reconnect.",
                        flush=True,
                    )
                    self.is_connected = False
                    consecutive_failures = 0

    # ── Public API ──────────────────────────────────────────────────────────
    def start(self):
        """Start the background reader thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        print("[THREAD] Background frame grabber started.", flush=True)

    def read(self):
        """
        Return the most recent frame (or None if no frame is available yet).
        Thread-safe via a lock.
        """
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Signal the reader thread to stop and release resources."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.cap is not None:
            self.cap.release()
        print("[THREAD] Background frame grabber stopped.", flush=True)


# =============================================================================
# Main Display Loop
# =============================================================================
def main():
    print("=" * 60, flush=True)
    print("  i-PRO Camera — RTSP Live Stream Test", flush=True)
    print("=" * 60, flush=True)
    print(f"  Camera IP     : {CAMERA_IP}", flush=True)
    print(f"  Stream        : {STREAM_PATH}", flush=True)
    print(f"  Display size  : {DISPLAY_WIDTH}×{DISPLAY_HEIGHT}", flush=True)
    print(f"  Transport     : TCP (forced)", flush=True)
    print(f"  Press 'q'     : Quit", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    stream = RTSPStream(RTSP_URL)
    stream.start()

    # Give the reader thread a moment to establish the first connection
    print("[MAIN] Waiting for first frame …", flush=True)
    first_frame_timeout = 15  # seconds
    start_time = time.time()

    while time.time() - start_time < first_frame_timeout:
        if stream.read() is not None:
            print("[MAIN] ✓ First frame received!", flush=True)
            break
        time.sleep(0.1)
    else:
        print(
            f"[MAIN] ⚠ No frame received within {first_frame_timeout}s. "
            "The display loop will keep retrying.",
            flush=True,
        )

    # ── Display loop ────────────────────────────────────────────────────────
    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            frame = stream.read()

            if frame is None:
                # No frame yet — brief sleep to avoid busy-waiting
                time.sleep(0.05)
                continue

            # Resize for comfortable on-screen viewing
            display_frame = cv2.resize(
                frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR
            )

            # Overlay a small FPS counter (top-left corner)
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
            else:
                fps = frame_count / max(elapsed, 0.001)

            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(WINDOW_NAME, display_frame)

            # 'q' to quit — waitKey(1) keeps the window responsive
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[MAIN] 'q' pressed — exiting.", flush=True)
                break

    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C received — exiting.", flush=True)

    finally:
        # ── Clean shutdown ──────────────────────────────────────────────────
        stream.stop()
        cv2.destroyAllWindows()
        print("[MAIN] Cleanup complete. Goodbye!", flush=True)


if __name__ == "__main__":
    main()
