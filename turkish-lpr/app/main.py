"""
app/main.py — CLI entry-point for the Turkish LPR service.

Commands:
    health          Check configuration and imports.
    stream-status   Connect to the camera and report stream properties.
    grab-frame      Capture one frame and save it to data/test_frame.jpg.
    detect-frame    Run plate detection + OCR on a frame and save annotated result.

Usage:
    python -m app.main health
    python -m app.main stream-status
    python -m app.main grab-frame
    python -m app.main detect-frame
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer

app = typer.Typer(
    name="turkish-lpr",
    help="Turkish License Plate Recognition — CLI tools.",
    add_completion=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# health — Chapter 1
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def health():
    """Check configuration, validate settings, and confirm all imports."""

    from app.config import get_settings, setup_logging

    settings = get_settings()
    log = setup_logging(settings)

    typer.echo(settings.summary())
    typer.echo()

    # ── Verify critical imports ─────────────────────────────────────────────
    checks: list[tuple[str, str]] = [
        ("cv2 (OpenCV)", "cv2"),
        ("numpy", "numpy"),
        ("pydantic_settings", "pydantic_settings"),
        ("fast_alpr", "fast_alpr"),
    ]

    # Optional imports — may be needed for fallback or future chapters
    optional_checks: list[tuple[str, str]] = [
        ("easyocr (fallback OCR)", "easyocr"),
    ]

    all_ok = True

    for label, module_name in checks:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            typer.echo(f"  ✓ {label:30s} {version}")
        except ImportError:
            typer.echo(f"  ✗ {label:30s} NOT INSTALLED (required)")
            all_ok = False

    for label, module_name in optional_checks:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            typer.echo(f"  ✓ {label:30s} {version}")
        except ImportError:
            typer.echo(f"  ⊘ {label:30s} not installed (needed later)")

    typer.echo()

    if all_ok:
        typer.echo("Health check: PASSED ✓")
        log.info("Health check passed")
        raise typer.Exit(code=0)
    else:
        typer.echo("Health check: FAILED ✗ — install missing packages")
        log.error("Health check failed — missing required packages")
        raise typer.Exit(code=1)


# ─────────────────────────────────────────────────────────────────────────────
# stream-status — Chapter 2
# ─────────────────────────────────────────────────────────────────────────────

@app.command("stream-status")
def stream_status(
    timeout: float = typer.Option(15.0, help="Seconds to wait for the first frame."),
):
    """Connect to the RTSP camera, grab one frame, and report stream info."""

    from app.config import get_settings, setup_logging
    from app.rtsp import RTSPStream

    settings = get_settings()
    log = setup_logging(settings)

    typer.echo(f"Connecting to {settings.rtsp_url_masked} …")
    stream = RTSPStream(settings)
    stream.start()

    fd = stream.wait_for_frame(timeout=timeout)
    stream.stop()

    if fd is None:
        typer.echo(f"✗ No frame received within {timeout}s.")
        log.error("stream-status failed — no frame received")
        raise typer.Exit(code=1)

    typer.echo()
    typer.echo(f"  ✓ Frame received")
    typer.echo(f"    Resolution : {fd.width}×{fd.height}")
    typer.echo(f"    Frame ID   : {fd.frame_id}")
    typer.echo(f"    Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fd.timestamp))}")
    typer.echo()
    typer.echo("Stream status: OK ✓")
    log.info("stream-status OK — %dx%d", fd.width, fd.height)
    raise typer.Exit(code=0)


# ─────────────────────────────────────────────────────────────────────────────
# grab-frame — Chapter 2
# ─────────────────────────────────────────────────────────────────────────────

@app.command("grab-frame")
def grab_frame(
    output: str = typer.Option("data/test_frame.jpg", help="Output file path."),
    timeout: float = typer.Option(15.0, help="Seconds to wait for the first frame."),
):
    """Capture one frame from the RTSP camera and save it to disk."""

    import cv2

    from app.config import PROJECT_ROOT, get_settings, setup_logging
    from app.rtsp import RTSPStream

    settings = get_settings()
    log = setup_logging(settings)

    # Resolve output path relative to project root
    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Connecting to {settings.rtsp_url_masked} …")
    stream = RTSPStream(settings)
    stream.start()

    fd = stream.wait_for_frame(timeout=timeout)
    stream.stop()

    if fd is None:
        typer.echo(f"✗ No frame received within {timeout}s.")
        log.error("grab-frame failed — no frame received")
        raise typer.Exit(code=1)

    # Save the frame
    success = cv2.imwrite(str(out_path), fd.frame)

    if not success:
        typer.echo(f"✗ Failed to write image to {out_path}")
        log.error("grab-frame failed — cv2.imwrite returned False")
        raise typer.Exit(code=1)

    file_size_kb = out_path.stat().st_size / 1024
    typer.echo()
    typer.echo(f"  ✓ Frame saved")
    typer.echo(f"    Path       : {out_path}")
    typer.echo(f"    Resolution : {fd.width}×{fd.height}")
    typer.echo(f"    File size  : {file_size_kb:.1f} KB")
    typer.echo(f"    Frame ID   : {fd.frame_id}")
    typer.echo()
    typer.echo("grab-frame: OK ✓")
    log.info("Saved frame to %s (%dx%d, %.1f KB)", out_path, fd.width, fd.height, file_size_kb)
    raise typer.Exit(code=0)



# ─────────────────────────────────────────────────────────────────────────────
# detect-frame — Chapter 3
# ─────────────────────────────────────────────────────────────────────────────

@app.command("detect-frame")
def detect_frame(
    source: str = typer.Option(
        "",
        help="Path to an image file.  If empty, grabs a live frame from the camera.",
    ),
    output: str = typer.Option(
        "data/detected_frame.jpg",
        help="Path to save the annotated image.",
    ),
    timeout: float = typer.Option(15.0, help="Camera timeout in seconds (live mode)."),
):
    """Run plate detection on a frame and save the annotated result with bounding boxes."""

    import cv2

    from app.config import PROJECT_ROOT, get_settings, setup_logging
    from app.detector import PlateDetector

    settings = get_settings()
    log = setup_logging(settings)

    # ── Get the source frame ────────────────────────────────────────────────
    if source:
        # Load from file
        src_path = Path(source)
        if not src_path.is_absolute():
            src_path = PROJECT_ROOT / src_path
        if not src_path.exists():
            typer.echo(f"✗ Source file not found: {src_path}")
            raise typer.Exit(code=1)

        frame = cv2.imread(str(src_path))
        if frame is None:
            typer.echo(f"✗ Failed to read image: {src_path}")
            raise typer.Exit(code=1)

        typer.echo(f"Loaded frame from {src_path}")
        h, w = frame.shape[:2]
        typer.echo(f"  Resolution: {w}×{h}")

    else:
        # Grab live from camera
        from app.rtsp import RTSPStream

        typer.echo(f"Connecting to {settings.rtsp_url_masked} …")
        stream = RTSPStream(settings)
        stream.start()

        fd = stream.wait_for_frame(timeout=timeout)
        stream.stop()

        if fd is None:
            typer.echo(f"✗ No frame received within {timeout}s.")
            raise typer.Exit(code=1)

        frame = fd.frame
        typer.echo(f"  Live frame captured: {fd.width}×{fd.height}")

    # ── Run detection ───────────────────────────────────────────────────────
    typer.echo("Loading detector …")
    try:
        detector = PlateDetector()
    except Exception as e:
        typer.echo(f"✗ Detector init failed: {e}")
        raise typer.Exit(code=1)

    typer.echo("Running detection …")
    detections = detector.detect(frame)

    # ── Report results ──────────────────────────────────────────────────────
    typer.echo()
    if not detections:
        typer.echo("  ⊘ No plates detected in this frame.")
    else:
        typer.echo(f"  ✓ {len(detections)} plate(s) detected:")
        for i, det in enumerate(detections, 1):
            typer.echo(
                f"    [{i}] bbox=({det.x1}, {det.y1}, {det.x2}, {det.y2})  "
                f"conf={det.confidence:.1%}  "
                f"size={det.width}×{det.height}px"
            )
            if det.ocr_text:
                typer.echo(
                    f"        OCR: \"{det.ocr_text}\"  "
                    f"ocr_conf={det.ocr_confidence:.1%}"
                )

    # ── Save annotated frame ────────────────────────────────────────────────
    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = PlateDetector.draw(frame, detections)
    cv2.imwrite(str(out_path), annotated)

    file_size_kb = out_path.stat().st_size / 1024
    typer.echo()
    typer.echo(f"  ✓ Annotated frame saved")
    typer.echo(f"    Path : {out_path}")
    typer.echo(f"    Size : {file_size_kb:.1f} KB")

    # ── Save detection JSON alongside the image ─────────────────────────────
    json_path = out_path.with_suffix(".json")
    result = {
        "source": source or "live_camera",
        "detections": [d.to_dict() for d in detections],
        "count": len(detections),
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    typer.echo(f"    JSON : {json_path}")

    typer.echo()
    typer.echo("detect-frame: OK ✓")
    log.info(
        "detect-frame OK — %d plate(s), saved to %s",
        len(detections),
        out_path,
    )
    raise typer.Exit(code=0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
