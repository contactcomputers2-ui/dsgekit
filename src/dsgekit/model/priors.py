"""Prior distribution DSL and validation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_PRIOR_ALIASES: dict[str, str] = {
    "normal": "normal_pdf",
    "normal_pdf": "normal_pdf",
    "gaussian": "normal_pdf",
    "gaussian_pdf": "normal_pdf",
    "beta": "beta_pdf",
    "beta_pdf": "beta_pdf",
    "gamma": "gamma_pdf",
    "gamma_pdf": "gamma_pdf",
    "inv_gamma": "inv_gamma_pdf",
    "inv_gamma_pdf": "inv_gamma_pdf",
    "invgamma": "inv_gamma_pdf",
    "invgamma_pdf": "inv_gamma_pdf",
    "inverse_gamma": "inv_gamma_pdf",
    "inverse_gamma_pdf": "inv_gamma_pdf",
}

_SUPPORTED_PRIORS = frozenset({"normal_pdf", "beta_pdf", "gamma_pdf", "inv_gamma_pdf"})


def normalize_prior_distribution(name: str) -> str:
    """Map prior aliases to canonical distribution names."""
    normalized = name.strip().lower()
    if normalized not in _PRIOR_ALIASES:
        supported = ", ".join(sorted(_SUPPORTED_PRIORS))
        raise ValueError(f"Unknown prior distribution '{name}'. Supported: {supported}")
    return _PRIOR_ALIASES[normalized]


@dataclass(frozen=True, slots=True)
class PriorSpec:
    """Prior specification for estimated parameters."""

    distribution: str
    mean: float
    std: float

    def __post_init__(self) -> None:
        distribution = normalize_prior_distribution(self.distribution)
        mean = float(self.mean)
        std = float(self.std)

        if not math.isfinite(mean):
            raise ValueError(f"Prior mean must be finite, got {mean}")
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError(f"Prior std must be finite and > 0, got {std}")

        if distribution == "beta_pdf":
            if not (0.0 < mean < 1.0):
                raise ValueError(f"Beta prior mean must be in (0, 1), got {mean}")
            if std * std >= mean * (1.0 - mean):
                raise ValueError(
                    "Beta prior std is too large for the given mean "
                    f"(mean={mean}, std={std})"
                )
        elif distribution in {"gamma_pdf", "inv_gamma_pdf"}:
            if mean <= 0.0:
                raise ValueError(f"{distribution} prior mean must be > 0, got {mean}")

        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def normal(cls, mean: float, std: float) -> PriorSpec:
        return cls("normal_pdf", mean, std)

    @classmethod
    def beta(cls, mean: float, std: float) -> PriorSpec:
        return cls("beta_pdf", mean, std)

    @classmethod
    def gamma(cls, mean: float, std: float) -> PriorSpec:
        return cls("gamma_pdf", mean, std)

    @classmethod
    def inv_gamma(cls, mean: float, std: float) -> PriorSpec:
        return cls("inv_gamma_pdf", mean, std)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "distribution": self.distribution,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorSpec:
        distribution = data.get(
            "distribution",
            data.get("dist", data.get("family", data.get("shape"))),
        )
        if distribution is None:
            raise ValueError("Prior dict must include 'distribution'")
        if "mean" not in data or "std" not in data:
            raise ValueError("Prior dict must include 'mean' and 'std'")
        return cls(str(distribution), float(data["mean"]), float(data["std"]))


def parse_prior_spec(
    prior: PriorSpec | dict[str, Any] | str | None,
    *,
    mean: float | None = None,
    std: float | None = None,
) -> PriorSpec | None:
    """Parse prior input from DSL/legacy forms."""
    if prior is None:
        if mean is None and std is None:
            return None
        raise ValueError("Prior mean/std provided without a prior distribution")

    if isinstance(prior, PriorSpec):
        return prior

    if isinstance(prior, dict):
        return PriorSpec.from_dict(prior)

    if isinstance(prior, str):
        if mean is None or std is None:
            raise ValueError(
                "Prior distribution string requires both mean and std "
                "(e.g. prior='normal', mean=0.9, std=0.05)"
            )
        return PriorSpec(prior, float(mean), float(std))

    raise TypeError(f"Unsupported prior specification type: {type(prior)!r}")
