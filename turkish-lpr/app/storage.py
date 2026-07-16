"""
app/storage.py — Local result storage (SQLite + JSON).

Persists plate recognition results locally.  No cloud services.
Privacy-aware: image saving is disabled by default.

Will be implemented in Chapter 7.
"""

# TODO: Chapter 7 — Implement ResultStore class
#   - SQLite schema: id, timestamp, plate_text, raw_text, confidence, bbox
#   - save(result) → row ID
#   - query(filters) → list of results
#   - Optional: save cropped plate image when SAVE_PLATE_IMAGES=true
