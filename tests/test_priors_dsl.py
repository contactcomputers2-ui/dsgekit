"""Tests for prior DSL and estimated parameter validation/serialization."""

from __future__ import annotations

import pytest

from dsgekit import load_model
from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.io.formats.yaml_format import model_to_yaml
from dsgekit.model.calibration import Calibration, EstimatedParam
from dsgekit.model.priors import PriorSpec


def _base_builder() -> ModelBuilder:
    return (
        ModelBuilder("ar1")
        .var("y")
        .varexo("e")
        .param("rho", 0.9)
        .equation("y = rho * y(-1) + e")
        .initval(y=0.0)
        .shock_stderr(e=0.01)
    )


def test_prior_spec_normalizes_aliases():
    prior = PriorSpec("Normal", 0.9, 0.05)
    assert prior.distribution == "normal_pdf"


def test_prior_spec_rejects_invalid_beta_parameters():
    with pytest.raises(ValueError, match="Beta prior mean must be in"):
        PriorSpec("beta", 1.2, 0.1)
    with pytest.raises(ValueError, match="too large"):
        PriorSpec("beta_pdf", 0.4, 0.5)


def test_estimated_param_from_dict_parses_nested_prior():
    ep = EstimatedParam.from_dict(
        {
            "type": "param",
            "name": "rho",
            "init": 0.9,
            "lower": 0.0,
            "upper": 0.99,
            "prior": {"distribution": "normal", "mean": 0.9, "std": 0.05},
        }
    )
    assert ep.prior_shape == "normal_pdf"
    assert ep.prior is not None
    assert ep.prior.distribution == "normal_pdf"


def test_calibration_roundtrip_preserves_prior_dsl():
    cal = Calibration(
        parameters={"rho": 0.9},
        estimated_params=[
            EstimatedParam.from_dict(
                {
                    "type": "param",
                    "name": "rho",
                    "init": 0.9,
                    "lower": 0.0,
                    "upper": 0.99,
                    "prior": {"distribution": "normal_pdf", "mean": 0.9, "std": 0.05},
                }
            )
        ],
    )
    dumped = cal.to_dict()
    assert dumped["estimated_params"][0]["prior"]["distribution"] == "normal_pdf"

    loaded = Calibration.from_dict(dumped)
    assert len(loaded.estimated_params) == 1
    ep = loaded.estimated_params[0]
    assert ep.prior_shape == "normal_pdf"
    assert ep.prior is not None
    assert ep.prior.mean == pytest.approx(0.9)


def test_model_builder_estimation_api_accepts_prior_objects():
    model, cal, _ss = (
        _base_builder()
        .estimate_param(
            "rho",
            init=0.9,
            lower=0.0,
            upper=0.99,
            prior=PriorSpec.normal(0.9, 0.05),
        )
        .estimate_stderr(
            "e",
            init=0.01,
            lower=0.001,
            upper=0.1,
            prior=PriorSpec.inv_gamma(0.02, 0.01),
        )
        .build()
    )
    assert model.name == "ar1"
    assert len(cal.estimated_params) == 2
    assert cal.estimated_params[0].prior_shape == "normal_pdf"
    assert cal.estimated_params[1].prior_shape == "inv_gamma_pdf"


def test_model_builder_validates_estimated_param_references():
    with pytest.raises(ValueError, match="not declared"):
        (
            _base_builder()
            .estimate_param(
                "missing_param",
                init=0.9,
                lower=0.0,
                upper=0.99,
                prior=PriorSpec.normal(0.9, 0.05),
            )
            .build()
        )


def test_load_model_dict_parses_prior_dsl():
    source = {
        "name": "ar1",
        "variables": ["y"],
        "shocks": ["e"],
        "parameters": {"rho": 0.9},
        "equations": ["y = rho * y(-1) + e"],
        "steady_state": {"y": 0.0},
        "shocks_config": {"e": {"stderr": 0.01}},
        "estimated_params": [
            {
                "type": "param",
                "name": "rho",
                "init": 0.9,
                "lower": 0.0,
                "upper": 0.99,
                "prior": {"distribution": "beta", "mean": 0.8, "std": 0.05},
            },
            {
                "type": "stderr",
                "name": "e",
                "init": 0.01,
                "lower": 0.001,
                "upper": 0.1,
                "prior": "inv_gamma",
                "prior_mean": 0.02,
                "prior_std": 0.01,
            },
        ],
    }
    _model, cal, _ss = load_model(source)
    assert [ep.prior_shape for ep in cal.estimated_params] == ["beta_pdf", "inv_gamma_pdf"]


def test_model_to_yaml_serializes_estimated_params_with_nested_prior():
    model, cal, ss = (
        _base_builder()
        .estimate_param(
            "rho",
            init=0.9,
            lower=0.0,
            upper=0.99,
            prior=PriorSpec.normal(0.9, 0.05),
        )
        .build()
    )
    content = model_to_yaml(model, cal, ss)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError("PyYAML is required for this test") from exc

    data = yaml.safe_load(content)
    assert "estimated_params" in data
    assert data["estimated_params"][0]["prior"]["distribution"] == "normal_pdf"
