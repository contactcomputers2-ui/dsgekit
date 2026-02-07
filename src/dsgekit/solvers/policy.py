"""Policy-rule optimization helpers (OSR-style grid search)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dsgekit.simulate import compute_variance
from dsgekit.solvers.linear import solve_linear
from dsgekit.transforms import linearize

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


@dataclass
class OSRCandidate:
    """Single candidate in an OSR grid search."""

    parameters: dict[str, float]
    loss: float
    status: str
    determinate: bool
    n_stable: int | None = None
    n_predetermined: int | None = None
    max_abs_eigenvalue: float | None = None
    message: str = ""

    @property
    def admissible(self) -> bool:
        return self.status == "ok" and np.isfinite(self.loss)

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        payload: dict[str, float | int | str | bool | None] = {
            **{name: float(value) for name, value in self.parameters.items()},
            "loss": float(self.loss),
            "status": self.status,
            "admissible": self.admissible,
            "determinate": self.determinate,
            "n_stable": self.n_stable,
            "n_predetermined": self.n_predetermined,
            "max_abs_eigenvalue": self.max_abs_eigenvalue,
            "message": self.message,
        }
        return payload


@dataclass
class OSRResult:
    """OSR grid search summary."""

    parameter_names: list[str]
    loss_weights: dict[str, float]
    require_determinate: bool
    candidates: list[OSRCandidate] = field(default_factory=list)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_admissible(self) -> int:
        return sum(1 for c in self.candidates if c.admissible)

    @property
    def best(self) -> OSRCandidate | None:
        admissible = [c for c in self.candidates if c.admissible]
        if not admissible:
            return None
        return min(admissible, key=lambda c: c.loss)

    def to_frame(self, *, include_failed: bool = True) -> pd.DataFrame:
        rows = [c.to_dict() for c in self.candidates]
        if not rows:
            cols = self.parameter_names + [
                "loss",
                "status",
                "admissible",
                "determinate",
                "n_stable",
                "n_predetermined",
                "max_abs_eigenvalue",
                "message",
            ]
            return pd.DataFrame(columns=cols)

        table = pd.DataFrame(rows)
        if not include_failed:
            table = table[table["admissible"]]
        table = table.sort_values(
            by=["admissible", "loss"],
            ascending=[False, True],
            na_position="last",
            kind="stable",
        )
        return table.reset_index(drop=True)

    def summary(self) -> str:
        lines = [
            "OSR Grid Search",
            "=" * 50,
            f"  Parameters:         {', '.join(self.parameter_names)}",
            f"  Loss weights:       {self.loss_weights}",
            f"  Candidates:         {self.n_candidates}",
            f"  Admissible:         {self.n_admissible}",
            f"  Require determ.:    {self.require_determinate}",
        ]
        best = self.best
        if best is None:
            lines.append("  Best candidate:     none")
        else:
            lines.append(f"  Best loss:          {best.loss:.8f}")
            lines.append(f"  Best parameters:    {best.parameters}")
        return "\n".join(lines)


def osr_grid_search(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    parameter_grid: dict[str, list[float]],
    loss_weights: dict[str, float],
    *,
    require_determinate: bool = True,
    n_predetermined: int | None = None,
    tol: float = 1e-10,
) -> OSRResult:
    """Run a grid-search optimizer for simple policy rules.

    The objective is a quadratic loss on unconditional variances:
        loss = sum_i w_i * Var(x_i)
    """
    if not parameter_grid:
        raise ValueError("parameter_grid must contain at least one parameter")
    if not loss_weights:
        raise ValueError("loss_weights must contain at least one variable")

    parameter_names = list(parameter_grid)
    values_per_parameter: list[list[float]] = []
    for name in parameter_names:
        if name not in calibration.parameters:
            raise ValueError(f"parameter '{name}' not found in calibration")
        raw_values = parameter_grid[name]
        if not raw_values:
            raise ValueError(f"parameter grid for '{name}' is empty")
        values = [float(v) for v in raw_values]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"parameter grid for '{name}' contains non-finite values")
        values_per_parameter.append(values)

    var_index = {name: i for i, name in enumerate(model.variable_names)}
    for var_name, weight in loss_weights.items():
        if var_name not in var_index:
            raise ValueError(f"loss variable '{var_name}' not found in model variables")
        if weight < 0.0 or not np.isfinite(weight):
            raise ValueError(f"loss weight for '{var_name}' must be finite and >= 0")

    result = OSRResult(
        parameter_names=parameter_names,
        loss_weights={name: float(w) for name, w in loss_weights.items()},
        require_determinate=require_determinate,
    )

    for values in product(*values_per_parameter):
        params = dict(zip(parameter_names, values, strict=True))
        cal_i = calibration.copy()
        cal_i.set_parameters(params)

        try:
            linear_sys = linearize(model, steady_state, cal_i)
            solution = solve_linear(
                linear_sys,
                n_predetermined=n_predetermined,
                tol=tol,
                check_bk=False,
            )
        except Exception as err:  # pragma: no cover - defensive branch
            result.candidates.append(
                OSRCandidate(
                    parameters=params,
                    loss=float("inf"),
                    status="error",
                    determinate=False,
                    message=str(err),
                )
            )
            continue

        eig = np.asarray(solution.eigenvalues)
        finite_eig = eig[np.isfinite(eig)]
        max_abs = float(np.max(np.abs(finite_eig))) if finite_eig.size > 0 else None
        determinate = bool(solution.is_determinate())

        if require_determinate and not determinate:
            result.candidates.append(
                OSRCandidate(
                    parameters=params,
                    loss=float("inf"),
                    status="non_determinate",
                    determinate=False,
                    n_stable=int(solution.n_stable),
                    n_predetermined=int(solution.n_predetermined),
                    max_abs_eigenvalue=max_abs,
                    message="Candidate rejected: non-determinate solution",
                )
            )
            continue

        try:
            variance = compute_variance(solution, cal_i)
            loss = 0.0
            for var_name, weight in loss_weights.items():
                idx = var_index[var_name]
                loss += float(weight) * float(variance[idx, idx])
            if not np.isfinite(loss):
                raise ValueError("non-finite loss")
        except Exception as err:
            result.candidates.append(
                OSRCandidate(
                    parameters=params,
                    loss=float("inf"),
                    status="error",
                    determinate=determinate,
                    n_stable=int(solution.n_stable),
                    n_predetermined=int(solution.n_predetermined),
                    max_abs_eigenvalue=max_abs,
                    message=str(err),
                )
            )
            continue

        result.candidates.append(
            OSRCandidate(
                parameters=params,
                loss=float(loss),
                status="ok",
                determinate=determinate,
                n_stable=int(solution.n_stable),
                n_predetermined=int(solution.n_predetermined),
                max_abs_eigenvalue=max_abs,
            )
        )

    return result
