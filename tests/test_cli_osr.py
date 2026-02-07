"""CLI tests for OSR command."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dsgekit.cli.main import main


def test_cli_osr_runs_and_writes_csv(models_dir, tmp_path: Path):
    output = tmp_path / "osr.csv"
    rc = main(
        [
            "osr",
            str(models_dir / "ar1.yaml"),
            "--grid", "rho=0.0,0.5,0.9",
            "--loss", "y=1.0",
            "--top", "3",
            "--output", str(output),
        ]
    )

    assert rc == 0
    assert output.exists()
    df = pd.read_csv(output)
    assert "rho" in df.columns
    assert "loss" in df.columns
    assert "status" in df.columns
    assert float(df.iloc[0]["rho"]) == 0.0
    assert str(df.iloc[0]["status"]) == "ok"


def test_cli_osr_returns_nonzero_when_no_admissible_candidate(models_dir, tmp_path: Path):
    output = tmp_path / "osr_fail.csv"
    rc = main(
        [
            "osr",
            str(models_dir / "ar1.yaml"),
            "--grid", "rho=1.05,1.10",
            "--loss", "y=1.0",
            "--include-failed",
            "--output", str(output),
        ]
    )

    assert rc == 1
    assert output.exists()
    df = pd.read_csv(output)
    assert set(df["status"]) == {"non_determinate"}


def test_cli_osr_rejects_invalid_grid_spec(models_dir):
    rc = main(
        [
            "osr",
            str(models_dir / "ar1.yaml"),
            "--grid", "rho",
            "--loss", "y=1.0",
        ]
    )
    assert rc == 1
