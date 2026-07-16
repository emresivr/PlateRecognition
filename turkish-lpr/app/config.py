"""
app/config.py — Centralized configuration for the Turkish LPR service.

All settings are loaded from environment variables (or a .env file).
Pydantic-settings validates types and provides sensible defaults.

Usage:
    from app.config import get_settings
    settings = get_settings()
    print(settings.rtsp_url)
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is one level above the `app/` package
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    All configurable parameters for the LPR pipeline.

    Values are read from environment variables first, then from a .env file
    located in the project root.  See .env.example for documentation.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # don't crash on unknown env vars
    )

    # ── RTSP Camera ─────────────────────────────────────────────────────────
    rtsp_user: str = "admin"
    rtsp_pass: str = "Panasonic1"
    rtsp_host: str = "192.168.1.68"
    rtsp_port: int = 554
    rtsp_path: str = "MediaInput/stream_1"
    rtsp_transport: str = "tcp"
    rtsp_reconnect_delay: int = 3
    rtsp_max_failures: int = 30

    # ── Plate Detector ──────────────────────────────────────────────────────
    detector_model_path: str = "models/plate_detect.pt"
    detector_confidence: float = 0.25

    # ── OCR ─────────────────────────────────────────────────────────────────
    ocr_languages: str = "en"
    ocr_pad_ratio: float = 0.15

    # ── Multi-frame Voting ──────────────────────────────────────────────────
    voting_frame_count: int = 5
    voting_min_agreement: float = 0.6

    # ── Storage ─────────────────────────────────────────────────────────────
    db_path: str = "data/plates.db"

    # ── Privacy ─────────────────────────────────────────────────────────────
    save_plate_images: bool = False
    plate_image_dir: str = "data/plates"

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived properties ──────────────────────────────────────────────────

    @property
    def rtsp_url(self) -> str:
        """Build the full RTSP URL from components."""
        return (
            f"rtsp://{self.rtsp_user}:{self.rtsp_pass}"
            f"@{self.rtsp_host}:{self.rtsp_port}/{self.rtsp_path}"
        )

    @property
    def rtsp_url_masked(self) -> str:
        """RTSP URL with password masked — safe for logging."""
        return (
            f"rtsp://{self.rtsp_user}:****"
            f"@{self.rtsp_host}:{self.rtsp_port}/{self.rtsp_path}"
        )

    @property
    def detector_model_abs_path(self) -> Path:
        """Resolve the detector model path relative to project root."""
        p = Path(self.detector_model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def db_abs_path(self) -> Path:
        """Resolve the database path relative to project root."""
        p = Path(self.db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def plate_image_abs_dir(self) -> Path:
        """Resolve the plate image directory relative to project root."""
        p = Path(self.plate_image_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def summary(self) -> str:
        """Human-readable summary of current settings (password masked)."""
        lines = [
            "┌─────────────────────────────────────────────────┐",
            "│         Turkish LPR — Configuration             │",
            "├─────────────────────────────────────────────────┤",
            f"│  RTSP URL       : {self.rtsp_url_masked}",
            f"│  Transport      : {self.rtsp_transport.upper()}",
            f"│  Reconnect      : {self.rtsp_reconnect_delay}s / {self.rtsp_max_failures} failures",
            f"│  Detector model : {self.detector_model_path}",
            f"│  Confidence     : {self.detector_confidence}",
            f"│  OCR languages  : {self.ocr_languages}",
            f"│  OCR pad ratio  : {self.ocr_pad_ratio}",
            f"│  Voting frames  : {self.voting_frame_count}",
            f"│  Min agreement  : {self.voting_min_agreement}",
            f"│  Database       : {self.db_path}",
            f"│  Save images    : {self.save_plate_images}",
            f"│  Log level      : {self.log_level}",
            "└─────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    """
    Configure the root logger with a structured format.

    Returns the 'turkish_lpr' logger for the application.
    """
    if settings is None:
        settings = get_settings()

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)-18s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # override any previous basicConfig
    )

    logger = logging.getLogger("turkish_lpr")
    logger.setLevel(level)
    return logger
