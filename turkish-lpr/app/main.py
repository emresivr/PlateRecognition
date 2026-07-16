"""
app/main.py — CLI entry-point for the Turkish LPR service.

Commands:
    health          Check configuration and imports.
    stream-status   Connect to the camera and report stream properties.
    grab-frame      Capture one frame and save it to data/test_frame.jpg.

Usage:
    python -m app.main health
    python -m app.main stream-status
    python -m app.main grab-frame
"""

from __future__ import annotations

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
    ]

    # Optional imports — may not be installed yet in early chapters
    optional_checks: list[tuple[str, str]] = [
        ("ultralytics (YOLO)", "ultralytics"),
        ("easyocr", "easyocr"),
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
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
