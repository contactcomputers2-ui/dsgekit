"""End-to-end integration tests for core dsgekit pipelines."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.estimation import (
    build_objective,
    estimate_map,
    estimate_mcmc,
    estimate_mdd_harmonic_mean,
    estimate_mdd_laplace,
    estimate_mle,
)
from dsgekit.filters import (
    forecast,
    historical_decomposition,
    kalman_filter,
    kalman_smoother,
)
from dsgekit.model.calibration import EstimatedParam
from dsgekit.simulate import irf, moments, simulate
from dsgekit.solvers import diagnose_bk, solve_linear, solve_perfect_foresight
from dsgekit.transforms import linearize, to_state_space


def _ar1_dict_source() -> dict:
    return {
        "name": "AR1_dict",
        "variables": ["y"],
        "shocks": ["e"],
        "parameters": {"rho": 0.9},
        "equations": [{"name": "ar1", "expr": "y = rho * y(-1) + e"}],
        "steady_state": {"y": 0.0},
        "shocks_config": {"e": {"stderr": 0.01}},
    }


def _load_source(models_dir, source_key: str):
    if source_key == "yaml":
        return load_model(models_dir / "ar1.yaml")
    if source_key == "mod":
        return load_model(models_dir / "ar1.mod")
    if source_key == "dict":
        return load_model(_ar1_dict_source())
    raise ValueError(f"Unknown source key: {source_key}")


@pytest.mark.parametrize("source_key", ["yaml", "mod", "dict"])
def test_ar1_end_to_end_pipeline(models_dir, source_key):
    """Load -> linearize -> solve -> simulate -> moments -> filter -> smoother."""
    model, cal, ss = _load_source(models_dir, source_key)
    lin = linearize(model, ss, cal)
    solution = solve_linear(lin)

    # Core solve outputs
    np.testing.assert_allclose(solution.T, np.array([[0.9]]), atol=1e-12)
    np.testing.assert_allclose(solution.R, np.array([[1.0]]), atol=1e-12)
    assert solution.n_predetermined == 1
    assert solution.n_stable == 1

    # Dynamics outputs
    irf_result = irf(solution, "e", periods=8)
    sim = simulate(solution, cal, n_periods=120, seed=123)
    m = moments(solution, cal, max_lag=4)

    np.testing.assert_allclose(irf_result["y"], 0.9 ** np.arange(8), rtol=1e-12)
    assert sim.data.shape == (120, 1)
    assert m.variance["y"] == pytest.approx((0.01**2) / (1 - 0.9**2), rel=1e-10)

    # State-space + filter + smoother
    ss_model = to_state_space(solution, cal, observables=["y"])
    kf = kalman_filter(ss_model, sim.data[["y"]])
    sm = kalman_smoother(ss_model, kf)

    assert np.isfinite(kf.log_likelihood)
    assert sm.smoothed_states.shape == (120, 1)
    np.testing.assert_allclose(
        sm.smoothed_states.iloc[-1].values,
        kf.filtered_states.iloc[-1].values,
        atol=1e-10,
    )


def test_ar1_integration_with_mle(models_dir):
    """End-to-end estimation: simulate data -> objective -> MLE."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    data = simulate(solution, cal, n_periods=200, seed=2026).data[["y"]]

    obj = build_objective(
        model,
        ss,
        cal,
        data,
        observables=["y"],
        param_names=["rho"],
    )

    at_true = obj(np.array([0.9]))
    at_wrong = obj(np.array([0.5]))
    assert np.isfinite(at_true)
    assert at_true < at_wrong

    result = estimate_mle(
        model,
        ss,
        cal,
        data,
        observables=["y"],
        param_names=["rho"],
        bounds={"rho": (0.01, 0.99)},
        compute_se=False,
    )
    assert result.success
    assert abs(result.parameters["rho"] - 0.9) < 0.1


def test_ar1_integration_with_map(models_dir):
    """End-to-end MAP: simulate data -> posterior mode with prior."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    data = simulate(solution, cal, n_periods=200, seed=2027).data[["y"]]

    cal_map = cal.copy()
    cal_map.estimated_params = [
        EstimatedParam.from_dict(
            {
                "type": "param",
                "name": "rho",
                "init": 0.5,
                "lower": 0.01,
                "upper": 0.99,
                "prior": {"distribution": "normal", "mean": 0.8, "std": 0.1},
            }
        )
    ]

    result = estimate_map(
        model,
        ss,
        cal_map,
        data,
        observables=["y"],
        param_names=None,
        bounds=None,
        compute_se=False,
    )
    assert result.success
    assert np.isfinite(result.log_posterior)
    assert abs(result.parameters["rho"] - 0.9) < 0.12


def test_ar1_integration_with_mcmc(models_dir):
    """End-to-end MCMC: posterior recovery plus diagnostics report."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    data = simulate(solution, cal, n_periods=100, seed=2029).data[["y"]]

    cal_post = cal.copy()
    cal_post.estimated_params = [
        EstimatedParam.from_dict(
            {
                "type": "param",
                "name": "rho",
                "init": 0.7,
                "lower": 0.01,
                "upper": 0.99,
                "prior": {"distribution": "normal", "mean": 0.8, "std": 0.12},
            }
        )
    ]

    result = estimate_mcmc(
        model,
        ss,
        cal_post,
        data,
        observables=["y"],
        param_names=None,
        n_draws=300,
        burn_in=80,
        thin=4,
        proposal_scale=0.035,
        seed=13,
    )
    assert np.isfinite(result.log_posterior_samples).all()
    assert abs(result.posterior_mean()["rho"] - 0.9) < 0.2
    diag = result.diagnostics()
    assert np.isfinite(diag.ess["rho"])
    assert 1.0 <= diag.ess["rho"] <= float(result.samples.shape[0])


def test_ar1_integration_with_marginal_data_density(models_dir):
    """End-to-end MDD: Laplace and harmonic-mean estimators."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    data = simulate(solution, cal, n_periods=110, seed=2032).data[["y"]]

    cal_post = cal.copy()
    cal_post.estimated_params = [
        EstimatedParam.from_dict(
            {
                "type": "param",
                "name": "rho",
                "init": 0.7,
                "lower": 0.01,
                "upper": 0.99,
                "prior": {"distribution": "normal", "mean": 0.8, "std": 0.12},
            }
        )
    ]

    laplace = estimate_mdd_laplace(
        model,
        ss,
        cal_post,
        data,
        observables=["y"],
        param_names=None,
        hessian_eps=1e-4,
    )
    mcmc = estimate_mcmc(
        model,
        ss,
        cal_post,
        data,
        observables=["y"],
        param_names=None,
        n_draws=220,
        burn_in=60,
        thin=2,
        proposal_scale=0.035,
        seed=23,
    )
    harmonic = estimate_mdd_harmonic_mean(
        model,
        ss,
        cal_post,
        data,
        observables=["y"],
        param_names=None,
        mcmc_result=mcmc,
        max_samples=80,
        seed=3,
    )

    assert np.isfinite(laplace.log_mdd)
    assert np.isfinite(harmonic.log_mdd)


def test_ar1_integration_with_forecast_and_decomposition(models_dir):
    """End-to-end filter/smoother -> forecast and historical decomposition."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    data = simulate(solution, cal, n_periods=150, seed=2034).data[["y"]]

    ss_model = to_state_space(solution, cal, observables=["y"])
    kf = kalman_filter(ss_model, data)
    sm = kalman_smoother(ss_model, kf)

    fc = forecast(ss_model, kf, steps=10, smoother_result=sm)
    assert fc.out_of_sample_observables.shape == (10, 1)
    assert np.isfinite(fc.out_of_sample_observables.values).all()

    hd = historical_decomposition(ss_model, sm)
    assert hd.max_abs_state_error < 1e-8
    assert hd.max_abs_obs_error < 1e-8


def test_ar1_integration_with_perfect_foresight(models_dir):
    """End-to-end deterministic perfect foresight path on AR(1)."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    shocks = np.zeros(12, dtype=np.float64)
    shocks[0] = 0.05

    result = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=12,
        shocks={"e": shocks},
        initial_state={"y": 0.0},
        terminal_state={"y": 0.0},
        tol=1e-12,
        max_iter=20,
    )
    assert result.converged
    expected = 0.05 * (0.9 ** np.arange(12))
    np.testing.assert_allclose(result.path["y"].values, expected, atol=1e-10)


def test_ar1_transition_with_histval_endval_and_deterministic_shocks(models_dir):
    """Deterministic transition uses histval/endval and parsed deterministic shocks."""
    model, cal, ss = load_model(models_dir / "ar1_deterministic_transition.mod")
    result = solve_perfect_foresight(
        model,
        ss,
        cal,
        n_periods=6,
        tol=1e-12,
        max_iter=30,
    )
    assert result.converged

    expected_shock = np.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(result.shocks["e"].values, expected_shock, atol=1e-12)

    expected = np.zeros(6, dtype=np.float64)
    y_prev = ss.histval[("y", 0)]
    for t in range(6):
        expected[t] = 0.9 * y_prev + expected_shock[t]
        y_prev = expected[t]

    np.testing.assert_allclose(result.path["y"].values, expected, atol=1e-10)


@pytest.mark.parametrize("fmt", ["nk.yaml", "nk.mod"])
def test_nk_forward_looking_integration(models_dir, fmt):
    """Forward-looking model should satisfy BK and be analyzable."""
    model, cal, ss = load_model(models_dir / fmt)
    lin = linearize(model, ss, cal)

    solved = solve_linear(lin)
    assert solved.n_stable == 1

    solution = solve_linear(lin, check_bk=False)
    diag = diagnose_bk(solution)

    assert diag.status == "determinate"
    assert solution.n_predetermined == 1
    assert solution.n_stable == 1

    for shock_name in solution.shock_names:
        responses = irf(solution, shock_name, periods=8).data
        assert responses.shape == (8, 3)
        assert np.all(np.isfinite(responses.values))
