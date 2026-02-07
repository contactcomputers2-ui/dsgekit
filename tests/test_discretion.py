"""Tests for discretionary linear-quadratic policy solver."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.solvers import (
    simulate_discretion_path,
    solve_discretion_from_linear_solution,
    solve_discretion_lq,
    solve_linear,
    solve_ramsey_lq,
)
from dsgekit.transforms import linearize


def test_discretion_horizon_converges_toward_ramsey():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[1.0]])
    beta = 0.95

    discretion_h1 = solve_discretion_lq(A, B, Q, R, beta=beta, horizon=1)
    discretion_h40 = solve_discretion_lq(A, B, Q, R, beta=beta, horizon=40)
    ramsey = solve_ramsey_lq(A, B, Q, R, beta=beta)

    err_h1 = abs(float(discretion_h1.K[0, 0] - ramsey.K[0, 0]))
    err_h40 = abs(float(discretion_h40.K[0, 0] - ramsey.K[0, 0]))

    assert err_h40 < err_h1
    assert discretion_h1.horizon == 1
    assert discretion_h40.horizon == 40
    assert discretion_h1.is_stable
    assert "Discretion LQ Solution" in discretion_h1.summary()


def test_discretion_path_reduces_state_magnitude():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[0.2]])

    discretion = solve_discretion_lq(A, B, Q, R, beta=0.99, horizon=10)
    path = simulate_discretion_path(discretion, [2.0], n_periods=20)

    assert path.states.shape[0] == 21
    assert path.controls.shape[0] == 20
    assert abs(path.states.iloc[-1, 0]) < abs(path.states.iloc[0, 0])
    assert path.discounted_loss > 0.0
    assert "Discretion Policy Path" in path.summary()


def test_discretion_from_linear_solution_ar1_fixture(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))

    discretion = solve_discretion_from_linear_solution(
        solution,
        control_shocks=["e"],
        state_weights={"y": 1.0},
        control_weights={"e": 0.5},
        beta=0.99,
        horizon=8,
    )

    assert discretion.n_states == 1
    assert discretion.n_controls == 1
    assert discretion.control_names == ["e"]
    assert discretion.state_names == ["y"]
    assert np.isfinite(discretion.K[0, 0])
    assert discretion.is_stable


def test_discretion_validates_controls_and_horizon(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))

    with pytest.raises(ValueError, match="Unknown control shocks"):
        solve_discretion_from_linear_solution(solution, control_shocks=["unknown"])

    with pytest.raises(ValueError, match="control_shocks cannot be empty"):
        solve_discretion_from_linear_solution(solution, control_shocks=[])

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        solve_discretion_from_linear_solution(solution, control_shocks=["e"], horizon=0)
