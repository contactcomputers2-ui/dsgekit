"""Calibration for DSGE models.

Calibration holds:
- Parameter values
- Shock covariance matrix (or standard deviations + correlations)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dsgekit.model.priors import PriorSpec, parse_prior_spec

if TYPE_CHECKING:
    from typing import Any

    from dsgekit.model.ir import ModelIR


@dataclass
class EstimatedParam:
    """A single entry from an estimated_params block.

    Attributes:
        entry_type: One of "param", "stderr", "corr".
        name: Parameter name (for "param") or shock name (for "stderr").
        name2: Second shock name (only for "corr").
        init_value: Starting value for the optimizer (NaN if unspecified).
        lower_bound: Lower bound constraint.
        upper_bound: Upper bound constraint.
        prior_shape: Prior distribution name (e.g. "normal_pdf"), empty for MLE.
        prior_mean: Prior mean (NaN if unspecified).
        prior_std: Prior std deviation (NaN if unspecified).
    """

    entry_type: str
    name: str
    name2: str = ""
    init_value: float = field(default_factory=lambda: float("nan"))
    lower_bound: float = float("-inf")
    upper_bound: float = float("inf")
    prior_shape: str = ""
    prior_mean: float = field(default_factory=lambda: float("nan"))
    prior_std: float = field(default_factory=lambda: float("nan"))

    def __post_init__(self) -> None:
        self.entry_type = self.entry_type.strip().lower()
        self.name = self.name.strip()
        self.name2 = self.name2.strip()
        self.init_value = float(self.init_value)
        self.lower_bound = float(self.lower_bound)
        self.upper_bound = float(self.upper_bound)
        self.prior_shape = self.prior_shape.strip()
        self.prior_mean = float(self.prior_mean)
        self.prior_std = float(self.prior_std)

        if self.entry_type not in {"param", "stderr", "corr"}:
            raise ValueError(
                f"Invalid estimated parameter type '{self.entry_type}'. "
                "Supported: param, stderr, corr"
            )
        if not self.name:
            raise ValueError("Estimated parameter name cannot be empty")
        if self.entry_type == "corr":
            if not self.name2:
                raise ValueError("Correlation estimated parameter requires a second name")
            if self.name == self.name2:
                raise ValueError("Correlation estimated parameter must reference two shocks")
        elif self.name2:
            raise ValueError(f"Estimated parameter type '{self.entry_type}' cannot have name2")

        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"Invalid bounds for '{self.name}': "
                f"lower={self.lower_bound} > upper={self.upper_bound}"
            )
        if math.isfinite(self.init_value) and (
            self.init_value < self.lower_bound or self.init_value > self.upper_bound
        ):
            raise ValueError(
                f"Initial value for '{self.name}' is outside bounds: "
                f"init={self.init_value}, [{self.lower_bound}, {self.upper_bound}]"
            )

        has_shape = bool(self.prior_shape)
        has_mean = not math.isnan(self.prior_mean)
        has_std = not math.isnan(self.prior_std)
        if has_shape or has_mean or has_std:
            prior = parse_prior_spec(
                self.prior_shape or None,
                mean=self.prior_mean if has_mean else None,
                std=self.prior_std if has_std else None,
            )
            if prior is None:
                raise ValueError("Invalid prior configuration")
            self.prior_shape = prior.distribution
            self.prior_mean = prior.mean
            self.prior_std = prior.std
        else:
            self.prior_shape = ""
            self.prior_mean = float("nan")
            self.prior_std = float("nan")

    @property
    def prior(self) -> PriorSpec | None:
        """Return prior specification if available."""
        if not self.prior_shape:
            return None
        return PriorSpec(self.prior_shape, self.prior_mean, self.prior_std)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EstimatedParam:
        """Build an estimated parameter from dict (legacy + DSL formats)."""
        prior_shape = data.get("prior_shape", "")
        prior_mean = data.get("prior_mean", data.get("mean", float("nan")))
        prior_std = data.get("prior_std", data.get("std", float("nan")))
        if "prior" in data:
            prior = parse_prior_spec(
                data["prior"],
                mean=None if prior_mean is None else float(prior_mean),
                std=None if prior_std is None else float(prior_std),
            )
            if prior is not None:
                prior_shape = prior.distribution
                prior_mean = prior.mean
                prior_std = prior.std

        return cls(
            entry_type=str(data.get("type", data.get("entry_type", "param"))),
            name=str(data["name"]),
            name2=str(data.get("name2", "")),
            init_value=float(data.get("init", data.get("init_value", "nan"))),
            lower_bound=float(data.get("lower", data.get("lower_bound", "-inf"))),
            upper_bound=float(data.get("upper", data.get("upper_bound", "inf"))),
            prior_shape=str(prior_shape) if prior_shape else "",
            prior_mean=float(prior_mean) if prior_mean is not None else float("nan"),
            prior_std=float(prior_std) if prior_std is not None else float("nan"),
        )

    def to_dict(self) -> dict[str, float | str | dict[str, float | str]]:
        """Serialize to dict with legacy and nested prior representations."""
        d: dict[str, float | str | dict[str, float | str]] = {
            "type": self.entry_type,
            "name": self.name,
            "name2": self.name2,
            "init": self.init_value,
            "lower": self.lower_bound,
            "upper": self.upper_bound,
        }
        prior = self.prior
        if prior is not None:
            d["prior_shape"] = prior.distribution
            d["prior_mean"] = prior.mean
            d["prior_std"] = prior.std
            d["prior"] = prior.to_dict()
        return d


@dataclass
class Calibration:
    """Parameter values and shock covariances for a DSGE model.

    Attributes:
        parameters: Mapping param_name -> value
        shock_stderr: Mapping shock_name -> standard deviation
        shock_corr: Mapping (shock1, shock2) -> correlation (optional)
        estimated_params: Entries from estimated_params block (optional)
    """

    parameters: dict[str, float] = field(default_factory=dict)
    shock_stderr: dict[str, float] = field(default_factory=dict)
    shock_corr: dict[tuple[str, str], float] = field(default_factory=dict)
    estimated_params: list[EstimatedParam] = field(default_factory=list)

    def set_parameter(self, name: str, value: float) -> None:
        """Set a parameter value."""
        self.parameters[name] = value

    def set_parameters(self, params: dict[str, float]) -> None:
        """Set multiple parameter values."""
        self.parameters.update(params)

    def get_parameter(self, name: str) -> float:
        """Get a parameter value."""
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not calibrated")
        return self.parameters[name]

    def set_shock_stderr(self, name: str, stderr: float) -> None:
        """Set standard deviation for a shock."""
        if stderr < 0:
            raise ValueError(f"Standard deviation must be non-negative, got {stderr}")
        self.shock_stderr[name] = stderr

    def set_shock_corr(self, shock1: str, shock2: str, corr: float) -> None:
        """Set correlation between two shocks."""
        if not -1 <= corr <= 1:
            raise ValueError(f"Correlation must be in [-1, 1], got {corr}")
        # Store in canonical order
        key = tuple(sorted([shock1, shock2]))
        self.shock_corr[key] = corr

    def get_shock_stderr(self, name: str, default: float = 0.0) -> float:
        """Get standard deviation for a shock."""
        return self.shock_stderr.get(name, default)

    def get_shock_corr(self, shock1: str, shock2: str) -> float:
        """Get correlation between two shocks (0 if not specified)."""
        if shock1 == shock2:
            return 1.0
        key = tuple(sorted([shock1, shock2]))
        return self.shock_corr.get(key, 0.0)

    def shock_cov_matrix(self, shock_names: list[str]) -> NDArray[np.float64]:
        """Build covariance matrix for shocks.

        Args:
            shock_names: Ordered list of shock names

        Returns:
            n_shocks x n_shocks covariance matrix
        """
        n = len(shock_names)
        cov = np.zeros((n, n))

        for i, s1 in enumerate(shock_names):
            for j, s2 in enumerate(shock_names):
                std1 = self.get_shock_stderr(s1)
                std2 = self.get_shock_stderr(s2)
                corr = self.get_shock_corr(s1, s2)
                cov[i, j] = std1 * std2 * corr

        return cov

    def shock_cholesky(self, shock_names: list[str]) -> NDArray[np.float64]:
        """Get Cholesky decomposition of shock covariance matrix.

        Useful for simulation: eps = chol @ standard_normal

        Args:
            shock_names: Ordered list of shock names

        Returns:
            Lower triangular Cholesky factor
        """
        cov = self.shock_cov_matrix(shock_names)
        return np.linalg.cholesky(cov)

    def validate(self, model: ModelIR) -> list[str]:
        """Validate calibration against a model.

        Args:
            model: The model to validate against

        Returns:
            List of warning/error messages (empty if valid)
        """
        errors = []

        # Check all parameters are calibrated
        for name in model.parameter_names:
            if name not in self.parameters:
                errors.append(f"Parameter '{name}' not calibrated")

        # Check for extra parameters (warning)
        for name in self.parameters:
            if name not in model.parameter_names:
                errors.append(f"Warning: Parameter '{name}' calibrated but not in model")

        # Check all shocks have stderr (use 0 as default if not specified)
        for name in model.shock_names:
            if name not in self.shock_stderr:
                # This is OK, we default to 0
                pass

        # Check shock correlations reference valid shocks
        for (s1, s2), _ in self.shock_corr.items():
            if s1 not in model.shock_names:
                errors.append(f"Shock '{s1}' in correlation not in model")
            if s2 not in model.shock_names:
                errors.append(f"Shock '{s2}' in correlation not in model")

        # Check estimated params
        errors.extend(self.validate_estimated_params(model))

        return errors

    def validate_estimated_params(self, model: ModelIR | None = None) -> list[str]:
        """Validate estimated_params entries and optional model references."""
        errors: list[str] = []
        for i, ep in enumerate(self.estimated_params):
            label = f"estimated_params[{i}] ({ep.entry_type}:{ep.name})"
            try:
                # Recreate to enforce __post_init__ checks even for copied/mutated instances.
                EstimatedParam.from_dict(ep.to_dict())
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{label}: {exc}")
                continue

            if model is None:
                continue

            if ep.entry_type == "param":
                if ep.name not in model.parameter_names:
                    errors.append(f"{label}: parameter '{ep.name}' not declared in model")
            elif ep.entry_type == "stderr":
                if ep.name not in model.shock_names:
                    errors.append(f"{label}: shock '{ep.name}' not declared in model")
            else:  # corr
                if ep.name not in model.shock_names:
                    errors.append(f"{label}: shock '{ep.name}' not declared in model")
                if ep.name2 not in model.shock_names:
                    errors.append(f"{label}: shock '{ep.name2}' not declared in model")

        return errors

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d: dict = {
            "parameters": self.parameters.copy(),
            "shock_stderr": self.shock_stderr.copy(),
            "shock_corr": {f"{k[0]},{k[1]}": v for k, v in self.shock_corr.items()},
        }
        if self.estimated_params:
            d["estimated_params"] = [ep.to_dict() for ep in self.estimated_params]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Calibration:
        """Create from dictionary."""
        cal = cls()
        cal.parameters = data.get("parameters", {}).copy()
        cal.shock_stderr = data.get("shock_stderr", {}).copy()

        # Parse correlations
        for key, val in data.get("shock_corr", {}).items():
            parts = key.split(",")
            if len(parts) == 2:
                cal.shock_corr[(parts[0], parts[1])] = val

        # Parse estimated params
        for item in data.get("estimated_params", []):
            cal.estimated_params.append(EstimatedParam.from_dict(item))

        return cal

    def copy(self) -> Calibration:
        """Create a copy of this calibration."""
        return Calibration(
            parameters=self.parameters.copy(),
            shock_stderr=self.shock_stderr.copy(),
            shock_corr=self.shock_corr.copy(),
            estimated_params=list(self.estimated_params),
        )

    def __str__(self) -> str:
        lines = ["Calibration:"]
        lines.append("  Parameters:")
        for name, val in sorted(self.parameters.items()):
            lines.append(f"    {name} = {val}")
        lines.append("  Shock std. dev.:")
        for name, val in sorted(self.shock_stderr.items()):
            lines.append(f"    {name} = {val}")
        if self.shock_corr:
            lines.append("  Shock correlations:")
            for (s1, s2), val in sorted(self.shock_corr.items()):
                lines.append(f"    corr({s1}, {s2}) = {val}")
        return "\n".join(lines)
