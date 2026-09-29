"""Run the public scripts from outside the project, including a fresh inference process."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_all_scripts_and_fresh_process_reload(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    train = tmp_path / "train.csv"
    out = tmp_path / "run"
    test = tmp_path / "test.csv"
    commands = [
        ["make_demo_data.py", "--output", str(train), "--rows", "100"],
        ["run_eda.py", "--data", str(train), "--output", str(tmp_path / "eda")],
        [
            "run_train.py",
            "--data",
            str(train),
            "--output",
            str(out),
            "--models",
            "ridge",
            "--feature-sets",
            "engineered",
            "--cv-folds",
            "2",
            "--n-jobs",
            "1",
        ],
        ["build_report.py", "--run", str(out)],
    ]
    for args in commands:
        result = subprocess.run(
            [sys.executable, str(scripts / args[0]), *args[1:]],
            cwd=tmp_path,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    frame = pd.read_csv(train, dtype={"Id": "string"}).iloc[:5].drop(columns="SalePrice")
    frame.to_csv(test, index=False)
    result = subprocess.run(
        [
            sys.executable,
            str(scripts / "run_predict.py"),
            "--model",
            str(out / "model.joblib"),
            "--data",
            str(test),
            "--output",
            str(tmp_path / "submission.csv"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    prediction = pd.read_csv(tmp_path / "submission.csv", dtype={"Id": "string"})
    assert prediction.columns.tolist() == ["Id", "SalePrice"]
    assert prediction.Id.tolist() == frame.Id.tolist()
    assert not (tmp_path / "submission.metrics.json").exists()
    assert (out / "REPORT.html").is_file()
    assert json.loads((tmp_path / "eda/summary.json").read_text())["usable_rows"] == 100
