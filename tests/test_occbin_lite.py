"""Tests for one-constraint OccBin-lite solver."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.exceptions import SolverError
from dsgekit.solvers import solve_occbin_lite, solve_perfect_foresight


def _zlb_shock_path(model, n_periods: int, e_i_values: list[float]) -> np.ndarray:
    path = np.zeros((n_periods, model.n_shocks), dtype=np.float64)
    idx = model.shock_names.index("e_i")
    for t, value in enumerate(e_i_values):
        path[t, idx] = value
    return path


def test_occbin_zlb_toy_binds_and_enforces_floor(models_dir):
    model, cal, ss = load_model(models_dir / "zlb_toy.yaml")
    n_periods = 16
    shocks = _zlb_shock_path(
        model,
        n_periods=n_periods,
        e_i_values=[-0.22, -0.14, -0.09, -0.05, -0.02],
    )

    result = solve_occbin_lite(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks=shocks,
        switch_equation="effective_rate_floor",
        relaxed_equation="i = i_shadow",
        binding_equation="i = 0.0",
        constraint_var="i_shadow",
        constraint_operator="<=",
        constraint_value=0.0,
        constraint_timing=0,
        tol=1e-10,
        max_iter=60,
        max_regime_iter=40,
    )

    assert result.converged
    assert result.binding_regime.any()
    assert np.all(np.isfinite(result.path.values))
    assert np.all(np.isfinite(result.residuals.values))
    np.testing.assert_allclose(
        result.path.loc[result.binding_regime, "i"].values,
        0.0,
        atol=1e-8,
    )

    relaxed = ~result.binding_regime.values
    if np.any(relaxed):
        np.testing.assert_allclose(
            result.path["i"].values[relaxed],
            result.path["i_shadow"].values[relaxed],
            atol=1e-7,
        )


def test_occbin_matches_perfect_foresight_if_never_binding(models_dir):
    model, cal, ss = load_model(models_dir / "zlb_toy.yaml")
    n_periods = 10
    shocks = _zlb_shock_path(
        model,
        n_periods=n_periods,
        e_i_values=[0.05, 0.03, 0.02],
    )

    baseline = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks=shocks,
        tol=1e-10,
        max_iter=60,
    )
    occbin = solve_occbin_lite(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks=shocks,
        switch_equation="effective_rate_floor",
        relaxed_equation="i = i_shadow",
        binding_equation="i = 0.0",
        constraint_var="i_shadow",
        constraint_operator="<=",
        constraint_value=0.0,
        constraint_timing=0,
        tol=1e-10,
        max_iter=60,
        max_regime_iter=20,
    )

    assert baseline.converged
    assert occbin.converged
    assert not occbin.binding_regime.any()
    np.testing.assert_allclose(occbin.path.values, baseline.path.values, atol=1e-8)


def test_occbin_rejects_unknown_constraint_variable(models_dir):
    model, cal, ss = load_model(models_dir / "zlb_toy.yaml")
    with pytest.raises(SolverError, match="constraint_var"):
        solve_occbin_lite(
            model,
            ss,
            cal,
            n_periods=8,
            switch_equation="effective_rate_floor",
            relaxed_equation="i = i_shadow",
            binding_equation="i = 0.0",
            constraint_var="does_not_exist",
        )


def test_occbin_sparse_linear_solver_matches_dense(models_dir):
    model, cal, ss = load_model(models_dir / "zlb_toy.yaml")
    n_periods = 14
    shocks = _zlb_shock_path(
        model,
        n_periods=n_periods,
        e_i_values=[-0.22, -0.14, -0.09, -0.05, -0.02],
    )

    kwargs = {
        "n_periods": n_periods,
        "shocks": shocks,
        "switch_equation": "effective_rate_floor",
        "relaxed_equation": "i = i_shadow",
        "binding_equation": "i = 0.0",
        "constraint_var": "i_shadow",
        "constraint_operator": "<=",
        "constraint_value": 0.0,
        "constraint_timing": 0,
        "tol": 1e-10,
        "max_iter": 60,
        "max_regime_iter": 40,
    }

    dense = solve_occbin_lite(model, ss, cal, linear_solver="dense", **kwargs)
    sparse = solve_occbin_lite(model, ss, cal, linear_solver="sparse", **kwargs)

    assert dense.converged
    assert sparse.converged
    assert dense.solver_meta["linear_solver_used"] == "dense"
    assert sparse.solver_meta["linear_solver_used"] == "sparse"
    np.testing.assert_array_equal(
        sparse.binding_regime.values,
        dense.binding_regime.values,
    )
    np.testing.assert_allclose(sparse.path.values, dense.path.values, atol=1e-8)
    np.testing.assert_allclose(
        sparse.residuals.values,
        dense.residuals.values,
        atol=1e-8,
    )


def test_occbin_invalid_linear_solver_raises(models_dir):
    model, cal, ss = load_model(models_dir / "zlb_toy.yaml")
    with pytest.raises(SolverError, match="linear_solver"):
        solve_occbin_lite(
            model,
            ss,
            cal,
            n_periods=8,
            switch_equation="effective_rate_floor",
            relaxed_equation="i = i_shadow",
            binding_equation="i = 0.0",
            constraint_var="i_shadow",
            linear_solver="bad_mode",
        )


def test_occbin_invalid_jit_backend_raises(models_dir):
    model, cal, ss = load_model(models_dir / "zlb_toy.yaml")
    with pytest.raises(SolverError, match="jit_backend"):
        solve_occbin_lite(
            model,
            ss,
            cal,
            n_periods=8,
            switch_equation="effective_rate_floor",
            relaxed_equation="i = i_shadow",
            binding_equation="i = 0.0",
            constraint_var="i_shadow",
            jit_backend="bad_jit",
        )


def test_occbin_numba_jit_requires_optional_dependency_if_dense(models_dir):
    if importlib.util.find_spec("numba") is not None:
        pytest.skip("numba is installed in this environment")

    model, cal, ss = load_model(models_dir / "zlb_toy.yaml")
    with pytest.raises(SolverError, match="jit_backend='numba' requires"):
        solve_occbin_lite(
            model,
            ss,
            cal,
            n_periods=8,
            switch_equation="effective_rate_floor",
            relaxed_equation="i = i_shadow",
            binding_equation="i = 0.0",
            constraint_var="i_shadow",
            linear_solver="dense",
            jit_backend="numba",
        )
