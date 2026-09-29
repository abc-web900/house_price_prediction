"""Synthetic Ames-shaped rows for execution checks, not model-quality claims."""

import numpy as np
import pandas as pd

from .schema import CAT_COLS, NUM_COLS, RAW_FEATURES, TARGET


def make_data(rows=240, seed=42):
    if rows < 1:
        raise ValueError("rows must be positive")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.uniform(0, 10, rows) for c in NUM_COLS})
    for col in CAT_COLS:
        df[col] = rng.choice(["TA", "Gd", "Ex"], rows)
    df["Neighborhood"] = rng.choice(["CollgCr", "OldTown", "NAmes", "NridgHt"], rows)
    df["MSZoning"] = rng.choice(["RL", "RM"], rows)
    df["OverallQual"] = rng.integers(3, 10, rows)
    df["OverallCond"] = rng.integers(3, 9, rows)
    df["GrLivArea"] = rng.uniform(700, 3200, rows)
    df["TotalBsmtSF"] = rng.uniform(0, 1500, rows)
    df["1stFlrSF"] = 0.7 * df["GrLivArea"]
    df["2ndFlrSF"] = 0.3 * df["GrLivArea"]
    df["YearBuilt"] = rng.integers(1940, 2006, rows)
    df["YrSold"] = rng.integers(2006, 2011, rows)
    df["YearRemodAdd"] = np.maximum(df["YearBuilt"], 1980)
    df["GarageYrBlt"] = df["YearBuilt"]
    df["FullBath"] = rng.integers(1, 4, rows)
    df["HalfBath"] = rng.integers(0, 2, rows)
    df["BsmtFullBath"] = rng.integers(0, 2, rows)
    df["BsmtHalfBath"] = rng.integers(0, 2, rows)
    df["GarageCars"] = rng.integers(0, 4, rows)
    df["LotArea"] = rng.uniform(3000, 18000, rows)
    price = (
        35000
        + 22000 * df["OverallQual"]
        + 50 * df["GrLivArea"]
        + 20 * df["TotalBsmtSF"]
        - 600 * (df["YrSold"] - df["YearBuilt"])
        + df["Neighborhood"].map(
            {"CollgCr": 10000, "OldTown": -15000, "NAmes": 0, "NridgHt": 25000}
        )
        + rng.normal(0, 12000, rows)
    )
    df = df[RAW_FEATURES].copy()
    df.insert(0, "Id", [f"{i + 1:06d}" for i in range(rows)])
    df[TARGET] = np.maximum(price, 15000)
    for col in ["LotFrontage", "GarageYrBlt", "MasVnrArea"]:
        df.loc[rng.random(rows) < 0.12, col] = np.nan
    df.loc[rng.random(rows) < 0.15, "Alley"] = None
    return df


def write_csv(path, rows=240, seed=42):
    from pathlib import Path

    path = Path(path)
    if path.exists():
        raise FileExistsError("Demo output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    make_data(rows, seed).to_csv(path, index=False)
    return path
