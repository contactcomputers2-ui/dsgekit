"""Tests for third-order perturbation solver and pruning simulation."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.exceptions import SolverError
from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.simulate import (
    simulate,
    simulate_pruned_third_order,
    simulate_pruned_third_order_path,
)
from dsgekit.solvers import solve_third_order


def _build_cubic_ar1_model(
    rho: float = 0.8,
    a: float = 0.2,
    b: float = 0.05,
):
    return (
        ModelBuilder("cubic_ar1")
        .var("y")
        .varexo("e")
        .param("rho", rho)
        .param("a", a)
        .param("b", b)
        .equation("y = rho * y(-1) + a * y(-1)^2 + b * y(-1)^3 + e")
        .initval(y=0.0)
        .shock_stderr(e=0.02)
        .build()
    )


def test_third_order_coefficients_match_cubic_ar1():
    model, cal, ss = _build_cubic_ar1_model(rho=0.8, a=0.2, b=0.05)
    solution = solve_third_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    # y = rho*x + a*x^2 + b*x^3 + e => d2y/dx2 = 2a, d3y/dx3 = 6b
    np.testing.assert_allclose(solution.T, np.array([[0.8]]), atol=1e-8)
    np.testing.assert_allclose(solution.R, np.array([[1.0]]), atol=1e-8)
    np.testing.assert_allclose(solution.quadratic_tensor[0, 0, 0], 0.4, atol=1e-5)
    np.testing.assert_allclose(solution.cubic_tensor[0, 0, 0, 0], 0.3, atol=3e-4)
    np.testing.assert_allclose(solution.quadratic_tensor[0, 1, 1], 0.0, atol=1e-5)
    np.testing.assert_allclose(solution.cubic_tensor[0, 1, 1, 1], 0.0, atol=1e-5)
    assert "Third-Order Perturbation Solution:" in solution.summary()


def test_pruned_third_order_deterministic_components_match_recursion():
    rho = 0.8
    a = 0.2
    b = 0.05
    x0 = 0.2
    periods = 12

    model, cal, ss = _build_cubic_ar1_model(rho=rho, a=a, b=b)
    cal.set_shock_stderr("e", 0.0)
    solution = solve_third_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    sim = simulate_pruned_third_order(
        solution,
        cal,
        n_periods=periods,
        initial_state=np.array([x0]),
        seed=42,
    )

    x1_prev = x0
    x2_prev = 0.0
    x3_prev = 0.0
    x1_path = np.zeros(periods, dtype=np.float64)
    x2_path = np.zeros(periods, dtype=np.float64)
    x3_path = np.zeros(periods, dtype=np.float64)
    for t in range(periods):
        x1_now = rho * x1_prev
        x2_now = rho * x2_prev + a * (x1_prev**2)
        x3_now = rho * x3_prev + 2.0 * a * x1_prev * x2_prev + b * (x1_prev**3)
        x1_path[t] = x1_now
        x2_path[t] = x2_now
        x3_path[t] = x3_now
        x1_prev = x1_now
        x2_prev = x2_now
        x3_prev = x3_now

    np.testing.assert_allclose(sim.first_order_component["y"].values, x1_path, atol=1e-8)
    np.testing.assert_allclose(sim.second_order_component["y"].values, x2_path, atol=1e-8)
    np.testing.assert_allclose(sim.third_order_component["y"].values, x3_path, atol=5e-7)
    np.testing.assert_allclose(sim["y"], x1_path + x2_path + x3_path, atol=5e-7)


def test_pruned_third_order_simulation_stays_finite_for_strong_nonlinearity():
    model, cal, ss = _build_cubic_ar1_model(rho=0.92, a=0.25, b=0.08)
    solution = solve_third_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    sim = simulate_pruned_third_order(
        solution,
        cal,
        n_periods=1200,
        burn_in=200,
        seed=2026,
    )

    assert np.all(np.isfinite(sim.data.values))
    assert np.all(np.isfinite(sim.first_order_component.values))
    assert np.all(np.isfinite(sim.second_order_component.values))
    assert np.all(np.isfinite(sim.third_order_component.values))
    assert np.max(np.abs(sim.data.values)) < 1e5
    assert np.mean(np.abs(sim.third_order_component.values)) > 0.0


def test_third_order_reduces_to_linear_when_nonlinear_terms_zero(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_third_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    np.testing.assert_allclose(solution.quadratic_tensor, 0.0, atol=1e-8)
    np.testing.assert_allclose(solution.cubic_tensor, 0.0, atol=2e-4)

    sim_linear = simulate(solution.linear_solution, cal, n_periods=80, seed=77)
    sim_third = simulate_pruned_third_order(solution, cal, n_periods=80, seed=77)
    np.testing.assert_allclose(sim_third.data.values, sim_linear.data.values, atol=1e-8)


def test_third_order_raises_for_forward_looking_model(models_dir):
    model, cal, ss = load_model(models_dir / "nk.yaml")
    with pytest.raises(SolverError, match="backward-looking models with max_lead=0"):
        solve_third_order(model, ss, cal)


def test_third_order_path_api_matches_random_simulation_shocks():
    model, cal, ss = _build_cubic_ar1_model(rho=0.85, a=0.15, b=0.04)
    solution = solve_third_order(model, ss, cal, derivative_backend="numeric", eps=1e-6)

    sim_random = simulate_pruned_third_order(solution, cal, n_periods=40, seed=123, burn_in=0)
    sim_path = simulate_pruned_third_order_path(solution, sim_random.shocks.values)

    np.testing.assert_allclose(sim_path.data.values, sim_random.data.values, atol=1e-12)
    np.testing.assert_allclose(
        sim_path.first_order_component.values,
        sim_random.first_order_component.values,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sim_path.second_order_component.values,
        sim_random.second_order_component.values,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sim_path.third_order_component.values,
        sim_random.third_order_component.values,
        atol=1e-12,
    )
