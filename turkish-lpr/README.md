# Turkish LPR Service

Local license plate recognition for Turkish plates, using an i-PRO (Panasonic) WV-U11550-V3 IP camera.

## Features

- **RTSP stream** with threaded frame capture and auto-reconnect
- **YOLOv8** plate detection (swappable model)
- **EasyOCR** text recognition
- **Turkish plate validation** (province codes 01–81, format regex)
- **Multi-frame voting** for reliable readings
- **SQLite/JSON** local storage — no cloud dependency
- **Privacy-first**: image saving disabled by default

## Quick Start

```bash
# 1. Clone and enter the project
cd turkish-lpr

# 2. Create virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env from the example
cp .env.example .env
# Edit .env with your camera credentials

# 5. Run health check
python -m app.main health

# 6. Test camera connection
python -m app.main stream-status

# 7. Grab a test frame
python -m app.main grab-frame
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `python -m app.main health` | Verify config and imports |
| `python -m app.main stream-status` | Connect to camera, report resolution/FPS |
| `python -m app.main grab-frame` | Save one frame to `data/test_frame.jpg` |

## Configuration

All settings are loaded from `.env`. See [.env.example](.env.example) for the full list with defaults.

## Project Structure

```
turkish-lpr/
  app/
    config.py          # Pydantic-settings configuration
    rtsp.py            # Threaded RTSP frame grabber
    detector.py        # YOLOv8 plate detection
    ocr.py             # EasyOCR text reading
    plate_rules.py     # Turkish plate normalization & validation
    voting.py          # Multi-frame consensus voting
    storage.py         # SQLite/JSON result persistence
    main.py            # Typer CLI entry-point
  tests/               # Unit and integration tests
  models/              # YOLOv8 .pt weight files
  data/                # Output frames, database, plate images
  requirements.txt
  .env.example
  README.md
```

## License

Private — internal use only.
