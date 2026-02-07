"""Tests for Ramsey linear-quadratic policy solver."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.solvers import (
    simulate_ramsey_path,
    solve_linear,
    solve_ramsey_from_linear_solution,
    solve_ramsey_lq,
)
from dsgekit.transforms import linearize


def _riccati_residual(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    beta: float,
) -> np.ndarray:
    term = R + beta * B.T @ P @ B
    rhs = Q + beta * A.T @ P @ A - (beta * A.T @ P @ B) @ np.linalg.solve(term, beta * B.T @ P @ A)
    return P - rhs


def test_scalar_ramsey_lq_matches_riccati_equation():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[1.0]])
    beta = 0.95

    result = solve_ramsey_lq(A, B, Q, R, beta=beta)
    resid = _riccati_residual(A, B, Q, R, result.P, beta)

    assert result.K.shape == (1, 1)
    assert result.P.shape == (1, 1)
    np.testing.assert_allclose(resid, np.zeros((1, 1)), atol=1e-8)
    assert result.is_stable
    assert "Ramsey LQ Solution" in result.summary()


def test_ramsey_path_reduces_state_magnitude():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[0.2]])
    result = solve_ramsey_lq(A, B, Q, R, beta=0.99)

    path = simulate_ramsey_path(result, [2.0], n_periods=20)

    assert path.states.shape[0] == 21
    assert path.controls.shape[0] == 20
    assert abs(path.states.iloc[-1, 0]) < abs(path.states.iloc[0, 0])
    assert path.discounted_loss > 0.0
    assert "Ramsey Policy Path" in path.summary()


def test_ramsey_from_linear_solution_ar1_fixture(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))

    ramsey = solve_ramsey_from_linear_solution(
        solution,
        control_shocks=["e"],
        state_weights={"y": 1.0},
        control_weights={"e": 0.5},
        beta=0.99,
    )

    assert ramsey.n_states == 1
    assert ramsey.n_controls == 1
    assert ramsey.control_names == ["e"]
    assert ramsey.state_names == ["y"]
    assert np.isfinite(ramsey.K[0, 0])
    assert ramsey.is_stable


def test_ramsey_from_linear_solution_validates_controls(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))

    with pytest.raises(ValueError, match="Unknown control shocks"):
        solve_ramsey_from_linear_solution(solution, control_shocks=["unknown"])
