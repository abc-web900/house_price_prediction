"""Validate raw CSVs, preserve IDs, and keep targets separate from inputs."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ID, RAW_FEATURES, TARGET


def require_columns(frame, columns=RAW_FEATURES):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")


def read_csv(path):
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Input must be a .csv file")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    names = [c.strip() for c in header]
    if not names or any(not c for c in names) or len(names) != len(set(names)):
        raise ValueError("CSV requires nonempty, unique column names")
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={c: "string" for c, stripped in zip(header, names) if stripped == ID},
    )
    frame.columns = names
    if frame.empty:
        raise ValueError("CSV contains no data rows")
    return frame


def identifiers(frame):
    values = (
        frame[ID].astype("string")
        if ID in frame
        else pd.Series(frame.index.astype(str), index=frame.index, dtype="string")
    )
    if values.isna().any() or values.str.strip().eq("").any() or values.duplicated().any():
        raise ValueError("Id values must be nonempty and unique")
    return values


def parse_target(frame):
    if TARGET not in frame:
        raise ValueError(f"Labeled data must contain {TARGET}")
    y = pd.to_numeric(
        frame[TARGET].astype("string").str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    ).astype(float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("SalePrice must contain only finite, positive prices")
    return y


@dataclass
class TrainingData:
    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series
    summary: dict


def load_training_data(path):
    frame = read_csv(path)
    loaded = len(frame)
    frame = frame.drop_duplicates().copy()
    require_columns(frame)
    ids = identifiers(frame)
    y = parse_target(frame)
    if len(frame) < 80:
        raise ValueError("At least 80 valid labeled rows are required")
    return TrainingData(
        frame[RAW_FEATURES].copy(),
        y,
        ids,
        {
            "rows_loaded": loaded,
            "duplicates_removed": loaded - len(frame),
            "usable_rows": len(frame),
        },
    )
