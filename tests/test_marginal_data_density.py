"""Tests for marginal data density estimators."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.estimation import (
    MarginalDataDensityResult,
    estimate_map,
    estimate_mcmc,
    estimate_mdd_harmonic_mean,
    estimate_mdd_laplace,
)
from dsgekit.estimation.marginal_data_density import _harmonic_mean_log_mdd
from dsgekit.model.calibration import EstimatedParam
from dsgekit.simulate import simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize


@pytest.fixture
def ar1_bayes_setup(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    data = simulate(solution, cal, n_periods=180, seed=2031).data[["y"]]

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
    return model, cal_post, ss, data


class TestHarmonicMeanMath:
    def test_closed_form_two_point_case(self):
        # Likelihood values: [1, 2] -> harmonic mean = 4/3
        ll = np.log(np.array([1.0, 2.0], dtype=np.float64))
        got = _harmonic_mean_log_mdd(ll)
        expected = np.log(4.0 / 3.0)
        assert got == pytest.approx(expected, abs=1e-12)


class TestMarginalDataDensity:
    def test_laplace_returns_finite_result(self, ar1_bayes_setup):
        model, cal, ss, data = ar1_bayes_setup
        result = estimate_mdd_laplace(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            hessian_eps=1e-4,
        )
        assert isinstance(result, MarginalDataDensityResult)
        assert result.method == "laplace"
        assert np.isfinite(result.log_mdd)
        assert np.isfinite(result.log_posterior_mode)
        assert np.isfinite(result.hessian_logdet)
        assert result.n_params == 1
        assert result.n_obs == data.shape[0]
        assert "laplace" in result.summary().lower()

    def test_laplace_accepts_external_map_result(self, ar1_bayes_setup):
        model, cal, ss, data = ar1_bayes_setup
        map_result = estimate_map(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            bounds=None,
            compute_se=False,
        )
        result = estimate_mdd_laplace(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            map_result=map_result,
            hessian_eps=1e-4,
        )
        assert np.isfinite(result.log_mdd)
        assert any("externally supplied MAP result" in n for n in result.notes)

    def test_harmonic_mean_returns_finite_result(self, ar1_bayes_setup):
        model, cal, ss, data = ar1_bayes_setup
        mcmc = estimate_mcmc(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            n_draws=260,
            burn_in=60,
            thin=2,
            proposal_scale=0.035,
            seed=21,
        )
        result = estimate_mdd_harmonic_mean(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            mcmc_result=mcmc,
            max_samples=80,
            seed=99,
        )
        assert isinstance(result, MarginalDataDensityResult)
        assert result.method == "harmonic_mean"
        assert np.isfinite(result.log_mdd)
        assert result.n_samples == 80
        assert any("Harmonic mean can be unstable" in note for note in result.notes)

    def test_laplace_and_harmonic_not_grossly_inconsistent(self, ar1_bayes_setup):
        model, cal, ss, data = ar1_bayes_setup
        laplace = estimate_mdd_laplace(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            hessian_eps=1e-4,
        )
        mcmc = estimate_mcmc(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            n_draws=280,
            burn_in=80,
            thin=2,
            proposal_scale=0.035,
            seed=22,
        )
        harmonic = estimate_mdd_harmonic_mean(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            mcmc_result=mcmc,
            max_samples=100,
            seed=101,
        )
        assert abs(laplace.log_mdd - harmonic.log_mdd) < 35.0
