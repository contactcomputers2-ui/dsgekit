"""Discretionary linear-quadratic policy solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    from dsgekit.solvers.linear import LinearSolution


def _validate_square_matrix(name: str, mat: NDArray[np.float64], size: int | None = None) -> None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix")
    if size is not None and mat.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}, got {mat.shape}")


def _validate_real_matrix(name: str, mat: NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"{name} contains non-finite values")


def _ensure_symmetric(name: str, mat: NDArray[np.float64], tol: float = 1e-10) -> NDArray[np.float64]:
    sym = 0.5 * (mat + mat.T)
    if np.max(np.abs(sym - mat)) > tol:
        raise ValueError(f"{name} must be symmetric within tolerance {tol}")
    return sym


@dataclass
class DiscretionLQResult:
    """Result of receding-horizon discretionary LQ policy.

    The problem solved each period is finite-horizon LQ with fixed horizon `H`:
        min E sum_{j=0}^{H-1} beta^j [x'Qx + u'Ru] + beta^H x_H' W x_H

    Only the first action is implemented, yielding a time-consistent feedback:
        u_t = -K x_t
    """

    A: NDArray[np.float64]
    B: NDArray[np.float64]
    Q: NDArray[np.float64]
    R: NDArray[np.float64]
    terminal_weight: NDArray[np.float64]
    beta: float
    horizon: int
    K: NDArray[np.float64]
    closed_loop: NDArray[np.float64]
    eigenvalues: NDArray[np.complex128]
    state_names: list[str]
    control_names: list[str]

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def n_controls(self) -> int:
        return self.B.shape[1]

    @property
    def is_stable(self) -> bool:
        finite = self.eigenvalues[np.isfinite(self.eigenvalues)]
        if finite.size == 0:
            return False
        return bool(np.max(np.abs(finite)) < 1.0 - 1e-12)

    def control(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_states:
            raise ValueError(f"state must have length {self.n_states}, got {x.shape[0]}")
        return -(self.K @ x)

    def summary(self) -> str:
        finite = self.eigenvalues[np.isfinite(self.eigenvalues)]
        max_abs = float(np.max(np.abs(finite))) if finite.size > 0 else float("nan")
        lines = [
            "Discretion LQ Solution",
            "=" * 50,
            f"  States:              {self.n_states}",
            f"  Controls:            {self.n_controls}",
            f"  Discount factor:     {self.beta:.4f}",
            f"  Horizon:             {self.horizon}",
            f"  Closed-loop stable:  {self.is_stable}",
            f"  Max |eig(A-BK)|:     {max_abs:.6f}",
        ]
        return "\n".join(lines)


@dataclass
class DiscretionPath:
    """Simulated path under discretionary LQ feedback policy."""

    states: pd.DataFrame
    controls: pd.DataFrame
    loss: pd.Series
    discounted_loss: float
    mean_loss: float
    beta: float
    horizon: int

    def summary(self) -> str:
        lines = [
            "Discretion Policy Path",
            "=" * 50,
            f"  Periods:            {len(self.loss)}",
            f"  Discount factor:    {self.beta:.4f}",
            f"  Horizon:            {self.horizon}",
            f"  Mean period loss:   {self.mean_loss:.6f}",
            f"  Discounted loss:    {self.discounted_loss:.6f}",
        ]
        return "\n".join(lines)


def solve_discretion_lq(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
    *,
    beta: float = 0.99,
    horizon: int = 1,
    terminal_weight: NDArray[np.float64] | None = None,
    state_names: list[str] | None = None,
    control_names: list[str] | None = None,
) -> DiscretionLQResult:
    """Solve receding-horizon discretionary LQ policy.

    Notes:
    - horizon=1 with terminal_weight=Q gives a one-step-ahead discretionary rule.
    - as horizon increases, the feedback converges toward the infinite-horizon Ramsey LQ rule.
    """
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    Q_arr = np.asarray(Q, dtype=np.float64)
    R_arr = np.asarray(R, dtype=np.float64)

    _validate_square_matrix("A", A_arr)
    _validate_real_matrix("A", A_arr)

    if B_arr.ndim != 2 or B_arr.shape[0] != A_arr.shape[0]:
        raise ValueError(
            f"B must have shape (n_states, n_controls) with n_states={A_arr.shape[0]}"
        )
    _validate_real_matrix("B", B_arr)

    _validate_square_matrix("Q", Q_arr, size=A_arr.shape[0])
    _validate_real_matrix("Q", Q_arr)
    Q_sym = _ensure_symmetric("Q", Q_arr)

    _validate_square_matrix("R", R_arr, size=B_arr.shape[1])
    _validate_real_matrix("R", R_arr)
    R_sym = _ensure_symmetric("R", R_arr)

    if terminal_weight is None:
        W_sym = Q_sym.copy()
    else:
        W_arr = np.asarray(terminal_weight, dtype=np.float64)
        _validate_square_matrix("terminal_weight", W_arr, size=A_arr.shape[0])
        _validate_real_matrix("terminal_weight", W_arr)
        W_sym = _ensure_symmetric("terminal_weight", W_arr)

    if not np.isfinite(beta) or beta <= 0.0 or beta > 1.0:
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    n_states = A_arr.shape[0]
    n_controls = B_arr.shape[1]
    resolved_state_names = (
        [f"x{i}" for i in range(n_states)] if state_names is None else list(state_names)
    )
    resolved_control_names = (
        [f"u{i}" for i in range(n_controls)] if control_names is None else list(control_names)
    )
    if len(resolved_state_names) != n_states:
        raise ValueError("state_names length mismatch")
    if len(resolved_control_names) != n_controls:
        raise ValueError("control_names length mismatch")

    # Backward Riccati for finite horizon with terminal penalty W.
    P_next = W_sym
    K0 = np.zeros((n_controls, n_states), dtype=np.float64)
    for _ in range(horizon):
        denom = R_sym + beta * (B_arr.T @ P_next @ B_arr)
        numer = beta * (B_arr.T @ P_next @ A_arr)
        try:
            K_curr = np.asarray(linalg.solve(denom, numer, assume_a="sym"), dtype=np.float64)
        except linalg.LinAlgError:
            K_curr = np.asarray(np.linalg.solve(denom, numer), dtype=np.float64)

        P_curr = Q_sym + beta * (A_arr.T @ P_next @ A_arr) - beta * (A_arr.T @ P_next @ B_arr @ K_curr)
        K0 = K_curr
        P_next = P_curr

    closed_loop = A_arr - B_arr @ K0
    eig = np.asarray(np.linalg.eigvals(closed_loop), dtype=np.complex128)

    return DiscretionLQResult(
        A=A_arr,
        B=B_arr,
        Q=Q_sym,
        R=R_sym,
        terminal_weight=W_sym,
        beta=float(beta),
        horizon=int(horizon),
        K=K0,
        closed_loop=closed_loop,
        eigenvalues=eig,
        state_names=resolved_state_names,
        control_names=resolved_control_names,
    )


def solve_discretion_from_linear_solution(
    solution: LinearSolution,
    *,
    control_shocks: list[str] | None = None,
    state_weights: dict[str, float] | None = None,
    control_weights: dict[str, float] | None = None,
    beta: float = 0.99,
    horizon: int = 1,
    terminal_state_weights: dict[str, float] | None = None,
) -> DiscretionLQResult:
    """Build and solve discretionary LQ policy from solved linear model."""
    state_names = list(solution.var_names)
    shock_names = list(solution.shock_names)
    if control_shocks is None:
        controls = shock_names
    else:
        controls = list(control_shocks)
    if not controls:
        raise ValueError("control_shocks cannot be empty")

    shock_index = {name: i for i, name in enumerate(shock_names)}
    missing_controls = [name for name in controls if name not in shock_index]
    if missing_controls:
        raise ValueError(f"Unknown control shocks: {missing_controls}")

    control_idx = [shock_index[name] for name in controls]
    A = np.asarray(solution.T, dtype=np.float64)
    B = np.asarray(solution.R[:, control_idx], dtype=np.float64)

    Q = np.zeros((len(state_names), len(state_names)), dtype=np.float64)
    if state_weights is None:
        Q[0, 0] = 1.0
    else:
        for i, var_name in enumerate(state_names):
            weight = float(state_weights.get(var_name, 0.0))
            if weight < 0.0:
                raise ValueError(f"state weight for '{var_name}' must be >= 0")
            Q[i, i] = weight

    R = np.zeros((len(controls), len(controls)), dtype=np.float64)
    if control_weights is None:
        np.fill_diagonal(R, 1.0)
    else:
        for i, shock_name in enumerate(controls):
            weight = float(control_weights.get(shock_name, 1.0))
            if weight <= 0.0:
                raise ValueError(f"control weight for '{shock_name}' must be > 0")
            R[i, i] = weight

    W = np.zeros_like(Q)
    if terminal_state_weights is None:
        np.fill_diagonal(W, np.diag(Q))
    else:
        for i, var_name in enumerate(state_names):
            weight = float(terminal_state_weights.get(var_name, 0.0))
            if weight < 0.0:
                raise ValueError(f"terminal state weight for '{var_name}' must be >= 0")
            W[i, i] = weight

    return solve_discretion_lq(
        A,
        B,
        Q,
        R,
        beta=beta,
        horizon=horizon,
        terminal_weight=W,
        state_names=state_names,
        control_names=controls,
    )


def simulate_discretion_path(
    result: DiscretionLQResult,
    initial_state: NDArray[np.float64] | list[float],
    *,
    n_periods: int = 40,
) -> DiscretionPath:
    """Simulate closed-loop dynamics under discretionary policy feedback."""
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")

    x = np.asarray(initial_state, dtype=np.float64).reshape(-1)
    if x.shape[0] != result.n_states:
        raise ValueError(
            f"initial_state length mismatch: expected {result.n_states}, got {x.shape[0]}"
        )

    states = np.zeros((n_periods + 1, result.n_states), dtype=np.float64)
    controls = np.zeros((n_periods, result.n_controls), dtype=np.float64)
    loss = np.zeros(n_periods, dtype=np.float64)

    states[0, :] = x
    for t in range(n_periods):
        u_t = result.control(states[t, :])
        controls[t, :] = u_t
        loss[t] = float(states[t, :] @ result.Q @ states[t, :] + u_t @ result.R @ u_t)
        states[t + 1, :] = result.A @ states[t, :] + result.B @ u_t

    discount = np.power(result.beta, np.arange(n_periods, dtype=np.float64))
    discounted_loss = float(np.dot(discount, loss))

    states_df = pd.DataFrame(states, columns=result.state_names)
    states_df.index.name = "period"
    controls_df = pd.DataFrame(controls, columns=result.control_names)
    controls_df.index.name = "period"
    loss_series = pd.Series(loss, name="loss")
    loss_series.index.name = "period"

    return DiscretionPath(
        states=states_df,
        controls=controls_df,
        loss=loss_series,
        discounted_loss=discounted_loss,
        mean_loss=float(np.mean(loss)),
        beta=result.beta,
        horizon=result.horizon,
    )
