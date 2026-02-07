"""Tests for OSR grid-search optimizer."""

from __future__ import annotations

import pytest

from dsgekit import load_model
from dsgekit.solvers import osr_grid_search


def test_osr_grid_search_picks_low_variance_candidate(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    result = osr_grid_search(
        model,
        ss,
        cal,
        parameter_grid={"rho": [0.0, 0.5, 0.9]},
        loss_weights={"y": 1.0},
        require_determinate=True,
    )

    assert result.n_candidates == 3
    assert result.n_admissible >= 1
    assert result.best is not None
    assert result.best.parameters["rho"] == pytest.approx(0.0)
    assert result.best.loss >= 0.0
    assert "OSR Grid Search" in result.summary()


def test_osr_rejects_non_determinate_when_required(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    result = osr_grid_search(
        model,
        ss,
        cal,
        parameter_grid={"rho": [1.05, 1.10]},
        loss_weights={"y": 1.0},
        require_determinate=True,
    )

    assert result.best is None
    assert result.n_admissible == 0
    frame = result.to_frame(include_failed=True)
    assert set(frame["status"]) == {"non_determinate"}


def test_osr_validates_loss_variables(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    with pytest.raises(ValueError, match="loss variable 'z' not found"):
        osr_grid_search(
            model,
            ss,
            cal,
            parameter_grid={"rho": [0.0, 0.5]},
            loss_weights={"z": 1.0},
        )
