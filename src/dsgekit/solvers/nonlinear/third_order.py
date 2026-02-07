"""Third-order perturbation solver (with pruning-ready policy tensors).

Current scope:
- Non-linear models with timings up to one lag and no leads (`max_lead=0`).
- Computes first-order policy (`T`, `R`) plus second/third-order tensors over
  state-shock coordinates `z_t = [y_{t-1}, u_t]`.

Policy approximation:
    y_t = T y_{t-1} + R u_t
          + 0.5 * G2(z_t, z_t)
          + (1/6) * G3(z_t, z_t, z_t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from dsgekit.derivatives import (
    DerivativeCoordinate,
    DerivativeStack,
    shock_coord,
    var_coord,
)

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState
    from dsgekit.solvers.linear import LinearSolution


@dataclass(slots=True)
class ThirdOrderSolution:
    """Third-order perturbation policy around steady state."""

    linear_solution: LinearSolution
    quadratic_tensor: NDArray[np.float64]
    cubic_tensor: NDArray[np.float64]
    state_shock_names: list[str]
    backend_name: str

    @property
    def T(self) -> NDArray[np.float64]:
        return self.linear_solution.T

    @property
    def R(self) -> NDArray[np.float64]:
        return self.linear_solution.R

    @property
    def var_names(self) -> list[str]:
        return self.linear_solution.var_names

    @property
    def shock_names(self) -> list[str]:
        return self.linear_solution.shock_names

    @property
    def steady_state(self) -> dict[str, float]:
        return self.linear_solution.steady_state

    @property
    def n_variables(self) -> int:
        return len(self.var_names)

    @property
    def n_shocks(self) -> int:
        return len(self.shock_names)

    @property
    def n_state_shock(self) -> int:
        return self.n_variables + self.n_shocks

    def _state_shock_vector(
        self,
        state: NDArray[np.float64],
        shock: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        state_vec = np.asarray(state, dtype=np.float64).reshape(-1)
        shock_vec = np.asarray(shock, dtype=np.float64).reshape(-1)

        if state_vec.shape[0] != self.n_variables:
            raise ValueError(
                f"state has length {state_vec.shape[0]}, expected {self.n_variables}"
            )
        if shock_vec.shape[0] != self.n_shocks:
            raise ValueError(
                f"shock has length {shock_vec.shape[0]}, expected {self.n_shocks}"
            )
        return np.concatenate([state_vec, shock_vec])

    def quadratic_effect(
        self,
        state: NDArray[np.float64],
        shock: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate 0.5 * G2(z, z), with z=[state, shock]."""
        z = self._state_shock_vector(state, shock)
        return 0.5 * np.einsum("kij,i,j->k", self.quadratic_tensor, z, z)

    def quadratic_cross_effect(
        self,
        state_left: NDArray[np.float64],
        shock_left: NDArray[np.float64],
        state_right: NDArray[np.float64],
        shock_right: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate G2(z_left, z_right)."""
        z_left = self._state_shock_vector(state_left, shock_left)
        z_right = self._state_shock_vector(state_right, shock_right)
        return np.einsum("kij,i,j->k", self.quadratic_tensor, z_left, z_right)

    def cubic_effect(
        self,
        state: NDArray[np.float64],
        shock: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate (1/6) * G3(z, z, z), with z=[state, shock]."""
        z = self._state_shock_vector(state, shock)
        return (1.0 / 6.0) * np.einsum("kijl,i,j,l->k", self.cubic_tensor, z, z, z)

    def summary(self) -> str:
        """Human-readable summary."""
        max_abs_quad = float(np.max(np.abs(self.quadratic_tensor)))
        max_abs_cubic = float(np.max(np.abs(self.cubic_tensor)))
        lines = [
            "Third-Order Perturbation Solution:",
            f"  Variables: {self.n_variables}",
            f"  Shocks: {self.n_shocks}",
            f"  State-shock coordinates: {self.n_state_shock}",
            f"  Derivative backend: {self.backend_name}",
            f"  Max |quadratic coefficient|: {max_abs_quad:.4e}",
            f"  Max |cubic coefficient|: {max_abs_cubic:.4e}",
        ]
        lines.append(self.linear_solution.summary())
        return "\n".join(lines)


def _build_eval_context(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
):
    from dsgekit.model.equations import EvalContext

    # Force one lag timing in context so derivatives wrt y(-1) are always defined.
    timings = [-1, 0]
    shocks = dict.fromkeys(model.shock_names, 0.0)
    return EvalContext.from_steady_state(
        steady_state=steady_state.values,
        parameters=calibration.parameters,
        shocks=shocks,
        timings=timings,
    )


def _state_shock_coordinates(model: ModelIR) -> list[DerivativeCoordinate]:
    coords: list[DerivativeCoordinate] = []
    for var_name in model.variable_names:
        coords.append(var_coord(var_name, -1))
    for var_name in model.variable_names:
        coords.append(var_coord(var_name, 0))
    for shock_name in model.shock_names:
        coords.append(shock_coord(shock_name))
    return coords


def _symmetrize_last_three_axes(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    return (
        tensor
        + np.transpose(tensor, (0, 1, 3, 2))
        + np.transpose(tensor, (0, 2, 1, 3))
        + np.transpose(tensor, (0, 2, 3, 1))
        + np.transpose(tensor, (0, 3, 1, 2))
        + np.transpose(tensor, (0, 3, 2, 1))
    ) / 6.0


def solve_third_order(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    *,
    derivative_backend: str = "numeric",
    eps: float = 1e-6,
    n_predetermined: int | None = None,
    tol: float = 1e-10,
    check_bk: bool = True,
) -> ThirdOrderSolution:
    """Compute third-order policy tensors for backward-looking models.

    Raises:
        SolverError: If model has leads or higher-order timings.
    """
    from dsgekit.exceptions import SolverError
    from dsgekit.solvers.linear import solve_linear
    from dsgekit.solvers.nonlinear.first_order import linearize_first_order

    model.validate()
    if model.lead_lag.max_lag > 1:
        raise SolverError(
            "Third-order perturbation currently supports timings up to one lag "
            f"(max_lag={model.lead_lag.max_lag})."
        )
    if model.lead_lag.max_lead > 0:
        raise SolverError(
            "Third-order perturbation currently supports backward-looking models "
            f"with max_lead=0 (got max_lead={model.lead_lag.max_lead})."
        )

    approximation = linearize_first_order(
        model,
        steady_state,
        calibration,
        derivative_backend=derivative_backend,
        eps=eps,
    )
    linear_system = approximation.linear_system

    if np.any(np.abs(linear_system.C) > tol):
        raise SolverError(
            "Third-order perturbation currently requires no lead terms "
            "(C matrix must be numerically zero)."
        )

    if n_predetermined is None:
        n_predetermined = model.n_predetermined

    linear_solution = solve_linear(
        linear_system,
        n_predetermined=n_predetermined,
        tol=tol,
        check_bk=check_bk,
    )

    coords = _state_shock_coordinates(model)
    stack = DerivativeStack(derivative_backend, eps=eps)
    context = _build_eval_context(model, steady_state, calibration)

    H = stack.hessian(model, context, coords)
    T3 = stack.third_order(model, context, coords)

    n_vars = model.n_variables
    n_shocks = model.n_shocks
    n_state_shock = n_vars + n_shocks
    n_q = (2 * n_vars) + n_shocks

    # q = [y(-1), y(0), u], z = [y(-1), u], and q_z = dq/dz at steady state.
    G1 = np.hstack([linear_solution.T, linear_solution.R])
    q_z = np.zeros((n_q, n_state_shock), dtype=np.float64)
    q_z[:n_vars, :n_vars] = np.eye(n_vars, dtype=np.float64)
    q_z[n_vars:2 * n_vars, :] = G1
    q_z[2 * n_vars:, n_vars:] = np.eye(n_shocks, dtype=np.float64)

    # Second-order policy tensor (same implicit system used by solve_second_order).
    rhs2 = np.einsum("eab,ai,bj->eij", H, q_z, q_z)
    rhs2_2d = rhs2.reshape(n_vars, n_state_shock * n_state_shock)
    try:
        solved2 = linalg.solve(linear_system.B, rhs2_2d, assume_a="gen")
    except linalg.LinAlgError as err:
        raise SolverError("Failed to solve second-order implicit system (singular B).") from err
    g2 = -solved2.reshape(n_vars, n_state_shock, n_state_shock)
    g2 = 0.5 * (g2 + np.swapaxes(g2, 1, 2))

    # q_zz is non-zero only on the y(0) block.
    q_zz = np.zeros((n_q, n_state_shock, n_state_shock), dtype=np.float64)
    q_zz[n_vars:2 * n_vars, :, :] = g2

    pure_third = np.einsum("eabc,ai,bj,ck->eijk", T3, q_z, q_z, q_z)
    mixed = (
        np.einsum("eab,aij,bk->eijk", H, q_zz, q_z)
        + np.einsum("eab,aik,bj->eijk", H, q_zz, q_z)
        + np.einsum("eab,ajk,bi->eijk", H, q_zz, q_z)
    )
    rhs3 = pure_third + mixed
    rhs3_2d = rhs3.reshape(n_vars, n_state_shock * n_state_shock * n_state_shock)

    try:
        solved3 = linalg.solve(linear_system.B, rhs3_2d, assume_a="gen")
    except linalg.LinAlgError as err:
        raise SolverError("Failed to solve third-order implicit system (singular B).") from err
    g3 = -solved3.reshape(n_vars, n_state_shock, n_state_shock, n_state_shock)
    g3 = _symmetrize_last_three_axes(g3)

    linear_solution.bk_meta = {
        **linear_solution.bk_meta,
        "solver": "third_order_perturbation",
        "derivative_backend": stack.backend_name,
        "n_derivative_coordinates": len(coords),
        "pruning_recommended": True,
    }

    state_shock_names = [
        *(f"{name}(-1)" for name in model.variable_names),
        *model.shock_names,
    ]

    return ThirdOrderSolution(
        linear_solution=linear_solution,
        quadratic_tensor=g2,
        cubic_tensor=g3,
        state_shock_names=state_shock_names,
        backend_name=stack.backend_name,
    )
