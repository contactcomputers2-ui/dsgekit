"""CLI tests for forecast and decomposition commands."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dsgekit.cli.main import create_parser, main


def _subcommand_choices(parser) -> dict:
    for action in parser._actions:  # noqa: SLF001
        if hasattr(action, "choices") and action.choices:
            return action.choices
    return {}


def test_parser_registers_forecast_and_decompose():
    parser = create_parser()
    choices = _subcommand_choices(parser)
    assert "forecast" in choices
    assert "decompose" in choices
    assert "estimation" in choices
    assert "osr" in choices
    assert "baseline_regression" in choices


def test_cli_forecast_runs_and_writes_csv(models_dir, tmp_path: Path):
    output = tmp_path / "forecast.csv"
    rc = main(
        [
            "forecast",
            str(models_dir / "ar1.yaml"),
            "-p", "6",
            "--train-periods", "40",
            "--seed", "2025",
            "--observables", "y",
            "--output", str(output),
        ]
    )
    assert rc == 0
    assert output.exists()
    df = pd.read_csv(output)
    assert "y" in df.columns
    assert len(df) == 6


def test_cli_decompose_runs_and_writes_csv(models_dir, tmp_path: Path):
    output = tmp_path / "decomposition.csv"
    rc = main(
        [
            "decompose",
            str(models_dir / "ar1.yaml"),
            "--train-periods", "40",
            "--seed", "2026",
            "--observables", "y",
            "--output", str(output),
        ]
    )
    assert rc == 0
    assert output.exists()
    assert output.stat().st_size > 0


def test_cli_simulate_runs(models_dir, tmp_path: Path):
    output = tmp_path / "sim.png"
    rc = main(
        [
            "simulate",
            str(models_dir / "ar1.yaml"),
            "-n", "25",
            "--seed", "2027",
            "--output", str(output),
        ]
    )
    assert rc == 0


def test_cli_estimation_mle_runs_and_writes_csv(models_dir, tmp_path: Path):
    output = tmp_path / "estimation.csv"
    rc = main(
        [
            "estimation",
            str(models_dir / "ar1.yaml"),
            "--method", "mle",
            "--params", "rho",
            "--train-periods", "80",
            "--seed", "2031",
            "--max-iter", "40",
            "--output", str(output),
        ]
    )
    assert rc == 0
    assert output.exists()
    df = pd.read_csv(output)
    assert set(df.columns) == {"parameter", "estimate"}
    assert "rho" in set(df["parameter"])
