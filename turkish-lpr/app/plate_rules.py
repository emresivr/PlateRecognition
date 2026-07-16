"""
app/plate_rules.py — Turkish license plate normalization and validation.

Applies Turkish plate format rules:
  - Normalize to uppercase, strip spaces/dashes.
  - Province code 01–81, then 1–3 letters, then 2–4 digits.
  - Context-aware OCR ambiguity fixes (O→0 in digit segments, etc.).

Will be implemented in Chapter 5.
"""

# TODO: Chapter 5 — Implement normalize() and validate() functions
#   - PLATE_REGEX = r"^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}\d{2,4}$"
#   - normalize(raw_text) → cleaned text
#   - validate(normalized_text) → bool
#   - fix_ambiguities(text) → text with segment-aware substitutions
