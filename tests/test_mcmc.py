"""Tests for Metropolis-Hastings posterior sampling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.estimation import (
    MCMCDiagnostics,
    MCMCResult,
    estimate_mcmc,
    sample_posterior_mh,
)
from dsgekit.model.calibration import EstimatedParam
from dsgekit.simulate import simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize


@pytest.fixture
def ar1_posterior_setup(models_dir):
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    data = simulate(solution, cal, n_periods=120, seed=2028).data[["y"]]

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


class TestMCMC:
    def test_seed_reproducibility(self, ar1_posterior_setup):
        model, cal, ss, data = ar1_posterior_setup

        r1 = sample_posterior_mh(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            n_draws=120,
            burn_in=20,
            thin=5,
            proposal_scale=0.04,
            seed=123,
        )
        r2 = sample_posterior_mh(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            n_draws=120,
            burn_in=20,
            thin=5,
            proposal_scale=0.04,
            seed=123,
        )
        np.testing.assert_allclose(r1.chain, r2.chain)
        np.testing.assert_allclose(r1.log_posterior_chain, r2.log_posterior_chain)

    def test_burnin_thinning_controls_shapes(self, ar1_posterior_setup):
        model, cal, ss, data = ar1_posterior_setup
        draws = 131
        burn_in = 31
        thin = 6

        result = sample_posterior_mh(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            n_draws=draws,
            burn_in=burn_in,
            thin=thin,
            proposal_scale=0.05,
            seed=7,
        )
        expected = len(range(burn_in, draws, thin))
        assert result.chain.shape == (draws, 1)
        assert result.samples.shape == (expected, 1)
        assert result.log_posterior_samples.shape == (expected,)
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_recover_rho_on_simulated_data(self, ar1_posterior_setup):
        model, cal, ss, data = ar1_posterior_setup

        result = estimate_mcmc(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            n_draws=500,
            burn_in=120,
            thin=4,
            proposal_scale=0.03,
            seed=11,
        )
        assert isinstance(result, MCMCResult)
        assert result.samples.shape[0] > 50
        rho_mean = result.posterior_mean()["rho"]
        assert abs(rho_mean - 0.9) < 0.15

    def test_invalid_controls_raise(self, ar1_posterior_setup):
        model, cal, ss, data = ar1_posterior_setup
        with pytest.raises(Exception, match="burn_in"):
            sample_posterior_mh(
                model,
                ss,
                cal,
                data,
                observables=["y"],
                param_names=None,
                n_draws=50,
                burn_in=50,
                thin=1,
                proposal_scale=0.05,
            )
        with pytest.raises(Exception, match="proposal_scale"):
            sample_posterior_mh(
                model,
                ss,
                cal,
                data,
                observables=["y"],
                param_names=None,
                n_draws=50,
                burn_in=10,
                thin=1,
                proposal_scale=-0.1,
            )

    def test_trace_dict_shapes(self, ar1_posterior_setup):
        model, cal, ss, data = ar1_posterior_setup
        result = sample_posterior_mh(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            n_draws=120,
            burn_in=20,
            thin=5,
            proposal_scale=0.04,
            seed=31,
        )
        full_trace = result.trace_dict(post_burn=False)
        saved_trace = result.trace_dict(post_burn=True)
        assert set(full_trace) == {"rho"}
        assert set(saved_trace) == {"rho"}
        assert full_trace["rho"].shape == (120,)
        assert saved_trace["rho"].shape == result.samples[:, 0].shape

    def test_diagnostics_from_estimated_chain(self, ar1_posterior_setup):
        model, cal, ss, data = ar1_posterior_setup
        result = sample_posterior_mh(
            model,
            ss,
            cal,
            data,
            observables=["y"],
            param_names=None,
            n_draws=320,
            burn_in=80,
            thin=2,
            proposal_scale=0.035,
            seed=19,
        )
        diag = result.diagnostics()
        assert isinstance(diag, MCMCDiagnostics)
        assert diag.acceptance_rate == pytest.approx(result.acceptance_rate)
        assert diag.n_samples == result.samples.shape[0]
        assert 1.0 <= diag.ess["rho"] <= float(result.samples.shape[0])
        assert np.isfinite(diag.r_hat["rho"])
        assert 0.9 <= diag.r_hat["rho"] <= 1.2

    def test_diagnostics_rhat_nan_when_too_few_samples(self):
        samples = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
        result = MCMCResult(
            param_names=["theta"],
            chain=samples.copy(),
            log_posterior_chain=np.zeros(3, dtype=np.float64),
            samples=samples,
            log_posterior_samples=np.zeros(3, dtype=np.float64),
            accepted=1,
            acceptance_rate=1.0 / 3.0,
            n_draws=3,
            burn_in=0,
            thin=1,
            seed=7,
            proposal_scale=np.array([0.1], dtype=np.float64),
        )
        diag = result.diagnostics()
        assert math.isnan(diag.r_hat["theta"])
        assert any("R-hat unavailable" in note for note in diag.notes)

    def test_diagnostics_for_iid_chain_has_high_ess(self):
        rng = np.random.default_rng(123)
        n = 1200
        samples = rng.normal(size=(n, 2))
        result = MCMCResult(
            param_names=["a", "b"],
            chain=samples.copy(),
            log_posterior_chain=np.zeros(n, dtype=np.float64),
            samples=samples,
            log_posterior_samples=np.zeros(n, dtype=np.float64),
            accepted=n // 3,
            acceptance_rate=1.0 / 3.0,
            n_draws=n,
            burn_in=0,
            thin=1,
            seed=123,
            proposal_scale=np.array([0.1, 0.1], dtype=np.float64),
        )
        diag = result.diagnostics()
        assert diag.ess["a"] > n / 6.0
        assert diag.ess["b"] > n / 6.0
        assert abs(diag.r_hat["a"] - 1.0) < 0.15
        assert abs(diag.r_hat["b"] - 1.0) < 0.15
