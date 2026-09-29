"""Raw-data summaries and descriptive associations; no target-derived selection."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .artifacts import write_json
from .config import OUTPUT_DIR, TRAIN_PATH
from .data import load_training_data, read_csv
from .schema import CAT_COLS, NUM_COLS, TARGET


def run(data_path=None, output_dir=None):
    data_path = TRAIN_PATH if data_path is None else data_path
    out = Path(OUTPUT_DIR / "eda" if output_dir is None else output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new or empty EDA directory")
    raw = read_csv(data_path)
    data = load_training_data(data_path)
    numeric = data.X[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    out.mkdir(parents=True, exist_ok=True)
    missing = raw.isna().sum().rename("missing_rows").to_frame()
    missing["missing_percent"] = 100 * missing["missing_rows"] / len(raw)
    missing.to_csv(out / "missing_values.csv", index_label="column")
    numeric.describe().T.to_csv(out / "numeric_summary.csv", index_label="column")
    numeric.corr().to_csv(out / "numeric_correlations.csv", index_label="column")
    category_rows = []
    for col in CAT_COLS:
        category_rows.extend(
            {"feature": col, "category": value, "rows": int(count)}
            for value, count in data.X[col].fillna("Missing").value_counts().items()
        )
    pd.DataFrame(category_rows).to_csv(out / "category_counts.csv", index=False)
    correlations = []
    for col in NUM_COLS:
        valid = numeric[col].notna()
        if numeric.loc[valid, col].nunique() > 1 and data.y.loc[valid].nunique() > 1:
            result = spearmanr(numeric.loc[valid, col], data.y.loc[valid])
            correlations.append(
                {
                    "feature": col,
                    "spearman": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )
    pd.DataFrame(correlations).to_csv(out / "target_associations.csv", index=False)
    summary = {**data.summary, "target": TARGET, "target_summary": data.y.describe().to_dict()}
    write_json(out / "summary.json", summary)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data.y, bins=30)
    axes[0].set_title("SalePrice")
    axes[1].hist(np.log1p(data.y), bins=30)
    axes[1].set_title("log1p(SalePrice)")
    fig.tight_layout()
    fig.savefig(out / "target_distribution.png", dpi=150)
    plt.close(fig)
    return {"summary": summary, "output_dir": out}
