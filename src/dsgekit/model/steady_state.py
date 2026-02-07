"""Steady state computation and validation for DSGE models.

The steady state is the fixed point where:
- All variables are constant: x_{t-1} = x_t = x_{t+1} = x_ss
- All shocks are zero: u_t = 0
- Model equations hold: f(x_ss, x_ss, x_ss, 0, θ) = 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR


@dataclass
class SteadyState:
    """Steady state values for a DSGE model.

    Attributes:
        values: Mapping var_name -> steady state value
        residuals: Mapping eq_name -> residual at steady state
        converged: Whether numerical solver converged (if used)
    """

    values: dict[str, float] = field(default_factory=dict)
    residuals: dict[str, float] = field(default_factory=dict)
    converged: bool = True
    histval: dict[tuple[str, int], float] = field(default_factory=dict)
    endval: dict[str, float] = field(default_factory=dict)
    deterministic_shocks: dict[tuple[str, int], float] = field(default_factory=dict)
    analytical_equations: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, name: str) -> float:
        """Get steady state value for a variable."""
        return self.values[name]

    def __setitem__(self, name: str, value: float) -> None:
        """Set steady state value for a variable."""
        self.values[name] = value

    def get(self, name: str, default: float = 0.0) -> float:
        """Get steady state value with default."""
        return self.values.get(name, default)

    def to_array(self, var_names: list[str]) -> NDArray[np.float64]:
        """Convert to array in given variable order."""
        return np.array([self.values[name] for name in var_names])

    @classmethod
    def from_array(
        cls, values: NDArray[np.float64], var_names: list[str]
    ) -> SteadyState:
        """Create from array and variable names."""
        return cls(values=dict(zip(var_names, values, strict=True)))

    def max_residual(self) -> float:
        """Maximum absolute residual."""
        if not self.residuals:
            return 0.0
        return max(abs(r) for r in self.residuals.values())

    def is_valid(self, tol: float = 1e-6) -> bool:
        """Check if steady state is valid (residuals below tolerance)."""
        return self.converged and self.max_residual() < tol

    def __str__(self) -> str:
        lines = ["Steady State:"]
        for name, val in sorted(self.values.items()):
            lines.append(f"  {name} = {val:.6g}")
        if self.residuals:
            max_res = self.max_residual()
            lines.append(f"  Max residual: {max_res:.2e}")
        return "\n".join(lines)


def validate_steady_state(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    tol: float = 1e-6,
) -> None:
    """Validate that steady state values satisfy model equations.

    Args:
        model: The model
        steady_state: Steady state values to validate
        calibration: Parameter calibration
        tol: Tolerance for residuals

    Raises:
        SteadyStateValidationError: If any residual exceeds tolerance
    """
    from dsgekit.exceptions import SteadyStateValidationError

    residuals = model.residuals_at_steady_state(
        steady_state=steady_state.values,
        parameters=calibration.parameters,
    )

    # Store residuals
    residual_dict = {}
    for i, eq in enumerate(model.equations):
        name = eq.name or f"eq_{i+1}"
        residual_dict[name] = residuals[i]

    steady_state.residuals = residual_dict

    # Check for violations
    violations = {name: res for name, res in residual_dict.items() if abs(res) > tol}
    if violations:
        raise SteadyStateValidationError(residual_dict, tol)


def solve_steady_state(
    model: ModelIR,
    calibration: Calibration,
    initial_guess: dict[str, float] | None = None,
    method: str = "hybr",
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> SteadyState:
    """Numerically solve for steady state.

    Uses scipy.optimize.root to find values where all residuals = 0.

    Args:
        model: The model (must be validated)
        calibration: Parameter calibration
        initial_guess: Initial values for solver (default: all ones)
        method: Scipy root method ('hybr', 'lm', 'broyden1', etc.)
        tol: Solver tolerance
        max_iter: Maximum iterations

    Returns:
        SteadyState with computed values

    Raises:
        SteadyStateNotFoundError: If solver fails to converge
    """
    from dsgekit.exceptions import SteadyStateNotFoundError

    var_names = model.variable_names
    n_vars = len(var_names)

    # Initial guess
    if initial_guess is None:
        x0 = np.ones(n_vars)
    else:
        x0 = np.array([initial_guess.get(name, 1.0) for name in var_names])

    # Define residual function
    def residual_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        ss_dict = dict(zip(var_names, x, strict=True))
        return model.residuals_at_steady_state(
            steady_state=ss_dict,
            parameters=calibration.parameters,
        )

    # Solve
    result = optimize.root(
        residual_func,
        x0,
        method=method,
        tol=tol,
        options={"maxfev": max_iter},
    )

    # Build steady state
    ss = SteadyState.from_array(result.x, var_names)
    ss.converged = result.success

    # Compute final residuals
    final_residuals = residual_func(result.x)
    for i, eq in enumerate(model.equations):
        name = eq.name or f"eq_{i+1}"
        ss.residuals[name] = final_residuals[i]

    if not result.success:
        raise SteadyStateNotFoundError(
            message=f"Steady state solver did not converge: {result.message}",
            residuals=ss.residuals,
            iterations=result.nfev if hasattr(result, "nfev") else None,
        )

    return ss


def compute_steady_state(
    model: ModelIR,
    calibration: Calibration,
    provided_values: dict[str, float] | None = None,
    solve_for: list[str] | None = None,
    initial_guess: dict[str, float] | None = None,
    tol: float = 1e-10,
) -> SteadyState:
    """Compute steady state, optionally solving for some variables.

    This allows a hybrid approach:
    - Some values provided directly (e.g., normalized variables)
    - Others solved numerically

    Args:
        model: The model
        calibration: Parameter calibration
        provided_values: Values given directly (not solved)
        solve_for: Variables to solve for (default: all not provided)
        initial_guess: Initial guess for solver
        tol: Solver tolerance

    Returns:
        Complete SteadyState
    """
    from dsgekit.exceptions import SteadyStateNotFoundError

    provided_values = provided_values or {}
    var_names = model.variable_names

    # Determine what to solve for
    if solve_for is None:
        solve_for = [v for v in var_names if v not in provided_values]

    if not solve_for:
        # All values provided, just validate
        ss = SteadyState(values=provided_values.copy())
        validate_steady_state(model, ss, calibration, tol)
        return ss

    # Build partial residual function
    n_solve = len(solve_for)

    def partial_residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        # Build full steady state dict
        ss_dict = provided_values.copy()
        for i, name in enumerate(solve_for):
            ss_dict[name] = x[i]

        return model.residuals_at_steady_state(
            steady_state=ss_dict,
            parameters=calibration.parameters,
        )

    # Initial guess for unknowns
    if initial_guess is None:
        x0 = np.ones(n_solve)
    else:
        x0 = np.array([initial_guess.get(name, 1.0) for name in solve_for])

    # Solve
    result = optimize.root(partial_residual, x0, method="hybr", tol=tol)

    # Build complete steady state
    values = provided_values.copy()
    for i, name in enumerate(solve_for):
        values[name] = result.x[i]

    ss = SteadyState(values=values, converged=result.success)

    # Compute final residuals
    final_residuals = partial_residual(result.x)
    for i, eq in enumerate(model.equations):
        name = eq.name or f"eq_{i+1}"
        ss.residuals[name] = final_residuals[i]

    if not result.success:
        raise SteadyStateNotFoundError(
            message=f"Steady state solver did not converge: {result.message}",
            residuals=ss.residuals,
        )

    return ss
