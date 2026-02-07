"""Tests for likelihood objective and MLE estimation."""

from __future__ import annotations

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.estimation import (
    MAPResult,
    MLEResult,
    build_objective,
    estimate_map,
    estimate_mle,
)
from dsgekit.estimation import (
    likelihood as likelihood_module,
)
from dsgekit.estimation.mle import _compute_std_errors
from dsgekit.exceptions import EstimationError
from dsgekit.model.calibration import EstimatedParam
from dsgekit.simulate import simulate
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ar1_model(models_dir):
    """Load AR(1) model, solve, and generate data."""
    model, cal, ss = load_model(models_dir / "ar1.yaml")
    solution = solve_linear(linearize(model, ss, cal))
    data = simulate(solution, cal, n_periods=500, seed=42).data[["y"]]
    return model, cal, ss, data


# ---------------------------------------------------------------------------
# build_objective
# ---------------------------------------------------------------------------


class TestBuildObjective:
    def test_returns_callable(self, ar1_model):
        model, cal, ss, data = ar1_model
        obj = build_objective(
            model, ss, cal, data, observables=["y"], param_names=["rho"]
        )
        assert callable(obj)

    def test_returns_finite_at_true_params(self, ar1_model):
        model, cal, ss, data = ar1_model
        obj = build_objective(
            model, ss, cal, data, observables=["y"], param_names=["rho"]
        )
        val = obj(np.array([0.9]))
        assert np.isfinite(val)

    def test_true_params_better_than_wrong(self, ar1_model):
        """Negative log-likelihood should be lower at true params."""
        model, cal, ss, data = ar1_model
        obj = build_objective(
            model, ss, cal, data, observables=["y"], param_names=["rho"]
        )
        at_true = obj(np.array([0.9]))
        at_wrong = obj(np.array([0.5]))
        assert at_true < at_wrong

    def test_unstable_param_returns_penalty(self, ar1_model):
        """rho > 1 violates BK → should return penalty."""
        model, cal, ss, data = ar1_model
        obj = build_objective(
            model, ss, cal, data, observables=["y"], param_names=["rho"]
        )
        val = obj(np.array([1.5]))
        assert val >= 1e9  # penalty value

    def test_shock_stderr_as_param(self, ar1_model):
        """Can use shock_stderr name as estimable parameter."""
        model, cal, ss, data = ar1_model
        obj = build_objective(
            model, ss, cal, data, observables=["y"], param_names=["e"]
        )
        val = obj(np.array([0.01]))
        assert np.isfinite(val)

    def test_invalid_param_name_raises(self, ar1_model):
        model, cal, ss, data = ar1_model
        with pytest.raises(Exception, match="not found"):
            build_objective(
                model, ss, cal, data,
                observables=["y"], param_names=["nonexistent"]
            )

    def test_memoizes_repeated_theta(self, ar1_model, monkeypatch):
        model, cal, ss, data = ar1_model

        calls = {"n": 0}
        orig_linearize = likelihood_module.linearize

        def _counted_linearize(*args, **kwargs):
            calls["n"] += 1
            return orig_linearize(*args, **kwargs)

        monkeypatch.setattr(likelihood_module, "linearize", _counted_linearize)

        obj = build_objective(
            model, ss, cal, data, observables=["y"], param_names=["rho"], cache=True
        )
        t1 = np.array([0.9])
        t2 = np.array([0.9])

        v1 = obj(t1)
        v2 = obj(t2)

        assert np.isfinite(v1)
        assert v1 == v2
        assert calls["n"] == 1
        info = obj.cache_info()  # type: ignore[attr-defined]
        assert info["enabled"] == 1
        assert info["hits"] >= 1
        assert info["misses"] == 1

    def test_cache_can_be_disabled(self, ar1_model, monkeypatch):
        model, cal, ss, data = ar1_model

        calls = {"n": 0}
        orig_linearize = likelihood_module.linearize

        def _counted_linearize(*args, **kwargs):
            calls["n"] += 1
            return orig_linearize(*args, **kwargs)

        monkeypatch.setattr(likelihood_module, "linearize", _counted_linearize)

        obj = build_objective(
            model, ss, cal, data, observables=["y"], param_names=["rho"], cache=False
        )

        _ = obj(np.array([0.9]))
        _ = obj(np.array([0.9]))

        assert calls["n"] == 2
        info = obj.cache_info()  # type: ignore[attr-defined]
        assert info["enabled"] == 0
        assert info["hits"] == 0
        assert info["size"] == 0

    def test_invalid_cache_size_raises(self, ar1_model):
        model, cal, ss, data = ar1_model
        with pytest.raises(EstimationError, match="cache_max_size"):
            build_objective(
                model,
                ss,
                cal,
                data,
                observables=["y"],
                param_names=["rho"],
                cache=True,
                cache_max_size=0,
            )


# ---------------------------------------------------------------------------
# estimate_mle
# ---------------------------------------------------------------------------


class TestEstimateMLE:
    def test_recovers_rho(self, ar1_model):
        """MLE should recover rho close to 0.9."""
        model, cal, ss, data = ar1_model
        result = estimate_mle(
            model, ss, cal, data,
            observables=["y"],
            param_names=["rho"],
            bounds={"rho": (0.01, 0.99)},
            compute_se=False,
        )
        assert isinstance(result, MLEResult)
        assert result.success
        assert abs(result.parameters["rho"] - 0.9) < 0.1

    def test_recovers_sigma(self, ar1_model):
        """MLE should recover sigma_e close to 0.01."""
        model, cal, ss, data = ar1_model
        result = estimate_mle(
            model, ss, cal, data,
            observables=["y"],
            param_names=["e"],
            bounds={"e": (0.001, 0.1)},
            compute_se=False,
        )
        assert result.success
        assert abs(result.parameters["e"] - 0.01) < 0.005

    def test_joint_estimation(self, ar1_model):
        """Estimate both rho and sigma_e jointly."""
        model, cal, ss, data = ar1_model
        result = estimate_mle(
            model, ss, cal, data,
            observables=["y"],
            param_names=["rho", "e"],
            bounds={"rho": (0.01, 0.99), "e": (0.001, 0.1)},
            compute_se=False,
        )
        assert result.success
        assert abs(result.parameters["rho"] - 0.9) < 0.1
        assert abs(result.parameters["e"] - 0.01) < 0.005

    def test_log_likelihood_finite(self, ar1_model):
        model, cal, ss, data = ar1_model
        result = estimate_mle(
            model, ss, cal, data,
            observables=["y"],
            param_names=["rho"],
            bounds={"rho": (0.01, 0.99)},
            compute_se=False,
        )
        assert np.isfinite(result.log_likelihood)

    def test_aic_bic(self, ar1_model):
        model, cal, ss, data = ar1_model
        result = estimate_mle(
            model, ss, cal, data,
            observables=["y"],
            param_names=["rho"],
            bounds={"rho": (0.01, 0.99)},
            compute_se=False,
        )
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        # AIC = -2*ll + 2*k, BIC = -2*ll + k*log(n)
        assert result.aic == pytest.approx(-2 * result.log_likelihood + 2)
        assert result.bic == pytest.approx(
            -2 * result.log_likelihood + np.log(500)
        )

    def test_std_errors(self, ar1_model):
        """Standard errors should be computed when requested."""
        model, cal, ss, data = ar1_model
        result = estimate_mle(
            model, ss, cal, data,
            observables=["y"],
            param_names=["rho"],
            bounds={"rho": (0.01, 0.99)},
            compute_se=True,
        )
        assert result.std_errors is not None
        assert "rho" in result.std_errors
        assert result.std_errors["rho"] > 0

    def test_summary(self, ar1_model):
        model, cal, ss, data = ar1_model
        result = estimate_mle(
            model, ss, cal, data,
            observables=["y"],
            param_names=["rho"],
            bounds={"rho": (0.01, 0.99)},
            compute_se=False,
        )
        s = result.summary()
        assert "Maximum Likelihood" in s
        assert "rho" in s
        assert "Log-likelihood" in s

    def test_std_error_hessian_reuses_eval_points(self):
        calls = {"n": 0}

        def quad_obj(theta):
            calls["n"] += 1
            x, y = theta
            return float(x * x + 3.0 * y * y + x * y)

        se = _compute_std_errors(
            quad_obj,
            np.array([0.2, -0.3]),
            ["x", "y"],
        )

        assert se is not None
        assert set(se) == {"x", "y"}
        # For k=2 with cached points: f0 + 2 diagonal pairs + 1 off-diagonal block.
        assert calls["n"] == 9

    def test_mle_with_x0_outside_bounds_is_robust(self, ar1_model):
        model, cal, ss, data = ar1_model
        cal2 = cal.copy()
        cal2.parameters["rho"] = 3.0  # intentionally outside bounds

        result = estimate_mle(
            model,
            ss,
            cal2,
            data,
            observables=["y"],
            param_names=["rho"],
            bounds={"rho": (0.01, 0.99)},
            compute_se=False,
        )
        assert result.success
        assert 0.01 <= result.parameters["rho"] <= 0.99


class TestEstimateMAP:
    def _calibration_with_rho_prior(self, cal):
        out = cal.copy()
        out.estimated_params = [
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
        return out

    def test_map_recovers_rho_with_prior(self, ar1_model):
        model, cal, ss, data = ar1_model
        cal_map = self._calibration_with_rho_prior(cal)

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
        assert isinstance(result, MAPResult)
        assert result.success
        assert np.isfinite(result.log_posterior)
        assert abs(result.parameters["rho"] - 0.9) < 0.1

    def test_map_prior_regularizes_vs_mle(self, ar1_model):
        model, cal, ss, data = ar1_model
        short_data = data.iloc[:60].copy()

        mle = estimate_mle(
            model,
            ss,
            cal,
            short_data,
            observables=["y"],
            param_names=["rho"],
            bounds={"rho": (0.01, 0.99)},
            compute_se=False,
        )

        cal_map = cal.copy()
        cal_map.estimated_params = [
            EstimatedParam.from_dict(
                {
                    "type": "param",
                    "name": "rho",
                    "init": 0.6,
                    "lower": 0.01,
                    "upper": 0.99,
                    "prior": {"distribution": "normal", "mean": 0.6, "std": 0.03},
                }
            )
        ]

        map_result = estimate_map(
            model,
            ss,
            cal_map,
            short_data,
            observables=["y"],
            param_names=None,
            bounds=None,
            compute_se=False,
        )
        assert map_result.success
        assert abs(map_result.parameters["rho"] - 0.6) < abs(mle.parameters["rho"] - 0.6)

    def test_map_requires_priors_when_requested(self, ar1_model):
        model, cal, ss, data = ar1_model
        cal_map = cal.copy()
        cal_map.estimated_params = [
            EstimatedParam.from_dict(
                {
                    "type": "param",
                    "name": "rho",
                    "init": 0.8,
                    "lower": 0.01,
                    "upper": 0.99,
                }
            )
        ]

        with pytest.raises(EstimationError, match="Missing prior"):
            estimate_map(
                model,
                ss,
                cal_map,
                data.iloc[:50],
                observables=["y"],
                param_names=None,
                bounds=None,
                compute_se=False,
                require_priors=True,
            )

    def test_map_infers_bounds_and_handles_out_of_bounds_init(self, ar1_model):
        model, cal, ss, data = ar1_model
        cal_map = cal.copy()
        cal_map.parameters["rho"] = 2.0  # outside inferred bounds
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
        assert 0.01 <= result.parameters["rho"] <= 0.99

    def test_map_rejects_corr_entries_in_auto_inference(self, ar1_model):
        model, cal, ss, data = ar1_model
        cal_map = cal.copy()
        cal_map.estimated_params = [
            EstimatedParam.from_dict(
                {
                    "type": "corr",
                    "name": "e",
                    "name2": "u",
                    "init": 0.1,
                    "lower": -0.5,
                    "upper": 0.5,
                    "prior": {"distribution": "normal", "mean": 0.0, "std": 0.2},
                }
            )
        ]
        with pytest.raises(EstimationError, match="Correlation estimated parameters"):
            estimate_map(
                model,
                ss,
                cal_map,
                data.iloc[:30],
                observables=["y"],
                param_names=None,
                bounds=None,
                compute_se=False,
            )
