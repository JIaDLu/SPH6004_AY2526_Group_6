"""
Global constants: clinical feature ranges, missing-rate thresholds,
categorical mappings, and path config.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/origin"
CACHE_DIR = "data/cache"
STATIC_FILE = "MIMIC-IV-static(Group Assignment).csv"
TEXT_FILE   = "MIMIC-IV-text(Group Assignment).csv"
TS_FILE     = "MIMIC-IV-time_series(Group Assignment).csv"

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Time-series windowing ────────────────────────────────────────────────────
WINDOW_SIZE = 24   # hours

# ─── Missing-rate thresholds for TS treatment strategy ───────────────────────
LOW_MISSING_THRESH    = 0.15   # ≤ 15 %  → LOCF + mask
HIGH_MISSING_THRESH   = 0.90   # ≥ 90 %  → ever_measured binary
# Between the two thresholds → LOCF + mask (same code path as low, different semantics)

# ─── Clinical feature clip ranges (domain knowledge) ─────────────────────────
# Keys must match column names in the TS table.
CLINICAL_CLIP_RANGES: dict[str, tuple[float, float]] = {
    "hr":               (0.0,   300.0),   # heart rate, bpm
    "rr":               (0.0,    60.0),   # respiratory rate, breaths/min
    "map":              (0.0,   300.0),   # mean arterial pressure, mmHg
    "temp":             (25.0,   45.0),   # temperature, °C (MIMIC stores in Celsius)
    "gcs":              (3.0,    15.0),   # Glasgow Coma Scale
    "sofa_cardio":      (0.0,     4.0),
    "sofa_resp":        (0.0,     4.0),
    "urine_output":     (0.0,  5000.0),   # mL / hr
    "lactate":          (0.0,    30.0),   # mmol/L
    "creatinine":       (0.0,    30.0),   # mg/dL
    "bilirubin":        (0.0,    50.0),   # mg/dL
    "wbc":              (0.0,   200.0),   # 10³/µL
    "inr":              (0.0,    30.0),
    "fluid_input":      (0.0, 10000.0),   # mL
    "ventilation_flag": (0.0,     1.0),
}

# ─── Static feature categorical definitions ───────────────────────────────────
# Substring rules for race mapping; checked in order (first match wins).
RACE_MAPPING_RULES: list[tuple[str, str]] = [
    ("WHITE",             "White"),
    ("BLACK",             "Black"),
    ("HISPANIC",          "Hispanic"),
    ("LATINO",            "Hispanic"),
    ("SOUTH AMERICAN",    "Hispanic"),
    ("ASIAN",             "Asian"),
    ("AMERICAN INDIAN",   "Native_American"),
    ("ALASKA",            "Native_American"),
    ("HAWAIIAN",          "Pacific_Islander"),
    ("PACIFIC",           "Pacific_Islander"),
    ("UNKNOWN",           "Unknown"),
    ("UNABLE TO OBTAIN",  "Unknown"),
    ("PATIENT DECLINED",  "Unknown"),
]

# All possible category values (order defines one-hot column order).
RACE_CATEGORIES       = ["White", "Black", "Hispanic", "Asian",
                          "Native_American", "Pacific_Islander", "Unknown", "Other"]
INSURANCE_CATEGORIES  = ["Medicaid", "Medicare", "Unknown", "No charge", "Other", "Private"]
MARITAL_CATEGORIES    = ["MARRIED", "SINGLE", "WIDOWED", "DIVORCED", "Unknown"]

# ─── Text / BERT ──────────────────────────────────────────────────────────────
BERT_MODEL_NAME  = "emilyalsentzer/Bio_ClinicalBERT"
BERT_EMBED_DIM   = 768
NOTE_SEPARATOR   = "-----"
BERT_MAX_LENGTH  = 512
BERT_BATCH_SIZE  = 32