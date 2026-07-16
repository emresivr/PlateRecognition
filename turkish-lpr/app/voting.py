"""
app/voting.py — Multi-frame voting for reliable plate readings.

Captures multiple frames per vehicle event and selects the most
consistent (highest-agreement) plate text across readings.

Will be implemented in Chapter 6.
"""

# TODO: Chapter 6 — Implement FrameBurstProcessor and vote() logic
#   - Collect N frames (configurable)
#   - Run detection + OCR on each
#   - Pick the plate text with highest frequency / confidence
