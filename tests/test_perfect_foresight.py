"""Tests for deterministic perfect-foresight solver."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.exceptions import SolverError
from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.solvers import (
    PerfectForesightResult,
    anticipated_shock,
    build_news_shock_path,
    solve_perfect_foresight,
    unanticipated_shock,
)


def test_ar1_impulse_matches_closed_form(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    n_periods = 10
    shock = np.zeros(n_periods, dtype=np.float64)
    shock[0] = 0.1

    result = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks={"e": shock},
        initial_state={"y": 0.0},
        terminal_state={"y": 0.0},
        tol=1e-12,
        max_iter=20,
    )
    assert isinstance(result, PerfectForesightResult)
    assert result.converged
    expected = 0.1 * (0.9 ** np.arange(n_periods))
    np.testing.assert_allclose(result.path["y"].values, expected, atol=1e-10)
    assert result.max_abs_residual < 1e-10


def test_forward_looking_single_variable_converges():
    model, cal, ss = (
        ModelBuilder("pf_lead_lag")
        .var("x")
        .varexo("e")
        .param("a", 0.4)
        .param("b", 0.5)
        .equation("x = a * x(-1) + b * x(+1) + e")
        .initval(x=0.0)
        .shock_stderr(e=0.0)
        .build()
    )

    n_periods = 16
    shock = np.zeros(n_periods, dtype=np.float64)
    shock[0] = 0.2
    result = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks={"e": shock},
        initial_state={"x": 0.0},
        terminal_state={"x": 0.0},
        tol=1e-10,
        max_iter=40,
    )
    assert result.converged
    assert np.all(np.isfinite(result.path["x"].values))
    assert result.path["x"].iloc[0] > 0.0
    assert result.max_abs_residual < 1e-8


def test_raises_for_higher_order_timing():
    model, cal, ss = (
        ModelBuilder("lag2")
        .var("y")
        .varexo("e")
        .param("rho", 0.2)
        .equation("y = rho * y(-2) + e")
        .initval(y=0.0)
        .shock_stderr(e=0.0)
        .build()
    )
    with pytest.raises(SolverError, match="supports timings up to one lag/lead"):
        solve_perfect_foresight(
            model,
            ss,
            cal,
            n_periods=8,
        )


def test_invalid_horizon_raises(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    with pytest.raises(SolverError, match="n_periods"):
        solve_perfect_foresight(model, ss, cal, n_periods=0)


def test_non_convergence_can_return_result(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    result = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=5,
        shocks={"e": np.array([1.0, 0.0, 0.0, 0.0, 0.0])},
        max_iter=1,
        tol=1e-18,
        raise_on_fail=False,
    )
    assert isinstance(result, PerfectForesightResult)
    assert result.converged is False


def test_defaults_use_histval_endval_and_deterministic_shocks():
    model, cal, ss = (
        ModelBuilder("pf_defaults")
        .var("x")
        .varexo("e")
        .param("a", 0.4)
        .param("b", 0.5)
        .equation("x = a * x(-1) + b * x(+1) + e")
        .initval(x=0.0)
        .shock_stderr(e=0.0)
        .build()
    )
    ss.histval[("x", 0)] = 0.2
    ss.endval["x"] = 1.0
    ss.deterministic_shocks[("e", 2)] = 0.3

    n_periods = 8
    result_auto = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        tol=1e-10,
        max_iter=40,
    )

    shock_path = np.zeros(n_periods, dtype=np.float64)
    shock_path[1] = 0.3
    result_explicit = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks={"e": shock_path},
        initial_state={"x": 0.2},
        terminal_state={"x": 1.0},
        tol=1e-10,
        max_iter=40,
    )

    assert result_auto.converged
    assert result_explicit.converged
    np.testing.assert_allclose(
        result_auto.path["x"].values,
        result_explicit.path["x"].values,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result_auto.shocks["e"].values,
        shock_path,
        atol=1e-12,
    )


def test_news_shock_path_builder_supports_anticipated_and_unanticipated():
    events = [
        unanticipated_shock("e", period=1, value=0.1),
        anticipated_shock("e", announcement_period=1, horizon=3, value=0.2),
    ]

    assert events[0].is_anticipated is False
    assert events[1].is_anticipated is True
    assert events[1].anticipation_horizon == 3

    path = build_news_shock_path(
        n_periods=6,
        shock_names=["e"],
        events=events,
    )
    expected = np.array([[0.1], [0.0], [0.0], [0.2], [0.0], [0.0]], dtype=np.float64)
    np.testing.assert_allclose(path, expected, atol=1e-12)


def test_news_shock_builder_rejects_out_of_horizon_impacts():
    events = [
        anticipated_shock("e", announcement_period=3, horizon=4, value=0.2),
    ]
    with pytest.raises(SolverError, match="outside the simulation horizon"):
        build_news_shock_path(n_periods=6, shock_names=["e"], events=events)


def test_news_shock_path_drives_perfect_foresight_simulation(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    n_periods = 8

    events = [
        unanticipated_shock("e", period=1, value=0.1),
        anticipated_shock("e", announcement_period=1, horizon=3, value=0.2),
    ]
    shock_path = build_news_shock_path(
        n_periods=n_periods,
        shock_names=model.shock_names,
        events=events,
    )

    result = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks=shock_path,
        initial_state={"y": 0.0},
        terminal_state={"y": 0.0},
        tol=1e-12,
        max_iter=30,
    )
    assert result.converged
    np.testing.assert_allclose(result.shocks["e"].values, shock_path[:, 0], atol=1e-12)

    expected = np.zeros(n_periods, dtype=np.float64)
    for t in range(n_periods):
        shock_t = shock_path[t, 0]
        expected[t] = 0.9 * (expected[t - 1] if t > 0 else 0.0) + shock_t
    np.testing.assert_allclose(result.path["y"].values, expected, atol=1e-10)


def test_sparse_linear_solver_matches_dense(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    n_periods = 24
    shock = np.zeros(n_periods, dtype=np.float64)
    shock[0] = 0.15

    dense = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks={"e": shock},
        initial_state={"y": 0.0},
        terminal_state={"y": 0.0},
        linear_solver="dense",
        tol=1e-12,
        max_iter=40,
    )
    sparse = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks={"e": shock},
        initial_state={"y": 0.0},
        terminal_state={"y": 0.0},
        linear_solver="sparse",
        tol=1e-12,
        max_iter=40,
    )

    assert dense.converged
    assert sparse.converged
    assert dense.solver_meta["linear_solver_used"] == "dense"
    assert sparse.solver_meta["linear_solver_used"] == "sparse"
    np.testing.assert_allclose(sparse.path.values, dense.path.values, atol=1e-10)
    np.testing.assert_allclose(
        sparse.residuals.values,
        dense.residuals.values,
        atol=1e-10,
    )


def test_invalid_linear_solver_mode_raises(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    with pytest.raises(SolverError, match="linear_solver"):
        solve_perfect_foresight(
            model,
            ss,
            cal,
            n_periods=6,
            linear_solver="invalid",
        )


def test_invalid_jit_backend_raises(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    with pytest.raises(SolverError, match="jit_backend"):
        solve_perfect_foresight(
            model,
            ss,
            cal,
            n_periods=6,
            jit_backend="invalid",
        )


def test_numba_jit_requires_optional_dependency_if_dense(models_dir):
    if importlib.util.find_spec("numba") is not None:
        pytest.skip("numba is installed in this environment")

    model, cal, ss = load_model(models_dir / "ar1.yaml")
    with pytest.raises(SolverError, match="jit_backend='numba' requires"):
        solve_perfect_foresight(
            model,
            ss,
            cal,
            n_periods=8,
            linear_solver="dense",
            jit_backend="numba",
        )


def test_numba_request_is_ignored_on_sparse_linear_path(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    n_periods = 12
    shock = np.zeros(n_periods, dtype=np.float64)
    shock[0] = 0.1

    result = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks={"e": shock},
        linear_solver="sparse",
        jit_backend="numba",
        tol=1e-12,
        max_iter=30,
    )
    assert result.converged
    assert result.solver_meta["linear_solver_used"] == "sparse"
    assert result.solver_meta["jit_backend_used"] == "none"


@pytest.mark.skipif(
    importlib.util.find_spec("numba") is None,
    reason="numba not installed",
)
def test_numba_jit_backend_runs_when_available(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    n_periods = 10
    shock = np.zeros(n_periods, dtype=np.float64)
    shock[0] = 0.1

    result = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks={"e": shock},
        linear_solver="dense",
        jit_backend="numba",
        tol=1e-12,
        max_iter=30,
    )

    assert result.converged
    assert result.solver_meta["jit_backend_used"] == "numba"
