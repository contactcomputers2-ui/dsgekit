"""OccBin-lite solver for one occasionally binding constraint.

This implementation solves deterministic perfect-foresight paths with two
regimes (relaxed vs binding), iterating on the regime sequence until
self-consistency:

1) solve the piecewise system for a fixed regime path
2) update regime from a simple constraint rule on one variable
3) repeat until no regime changes
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from dsgekit.derivatives import DerivativeStack, var_coord
from dsgekit.exceptions import SolverError
from dsgekit.model.equations import (
    EQUATION_FUNCTIONS,
    KNOWN_FUNCTIONS,
    Equation,
    EvalContext,
    param,
    shock,
    var,
)
from dsgekit.solvers.nonlinear._newton_linear import (
    BlockTridiagonalJacobianBuilder,
    JitBackendMode,
    LinearSolverMode,
    resolve_jit_backend_mode,
    resolve_linear_solver_mode,
    solve_newton_step,
)
from dsgekit.solvers.nonlinear.perfect_foresight import (
    PerfectForesightResult,
    _default_initial_state_from_steady_state,
    _default_shock_path_from_steady_state,
    _default_terminal_state_from_steady_state,
    _resolve_initial_path,
    _resolve_shock_path,
    _resolve_state_vector,
    _set_period_context,
)

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


ConstraintOperator = Literal["<", "<=", ">", ">="]


@dataclass(slots=True)
class OccBinResult:
    """Output for OccBin-lite trajectories."""

    path: pd.DataFrame
    shocks: pd.DataFrame
    residuals: pd.DataFrame
    binding_regime: pd.Series
    converged: bool
    n_regime_iterations: int
    n_newton_iterations: int
    max_abs_residual: float
    residual_history: list[float] = field(default_factory=list)
    regime_binding_counts: list[int] = field(default_factory=list)
    solver_meta: dict[str, float | int | str] = field(default_factory=dict)

    def __getitem__(self, var_name: str) -> NDArray[np.float64]:
        return self.path[var_name].values

    def summary(self) -> str:
        n_binding = int(np.sum(self.binding_regime.values))
        lines = [
            "OccBin-lite Solver",
            "=" * 50,
            f"  Converged:            {self.converged}",
            f"  Regime iterations:    {self.n_regime_iterations}",
            f"  Newton iterations:    {self.n_newton_iterations}",
            f"  Max |residual|:       {self.max_abs_residual:.3e}",
            f"  Binding periods:      {n_binding}",
            f"  Periods:              {self.path.shape[0]}",
            f"  Variables:            {self.path.shape[1]}",
            f"  Shocks:               {self.shocks.shape[1]}",
        ]
        return "\n".join(lines)


def _resolve_equation_index(model: ModelIR, switch_equation: str | int) -> int:
    if isinstance(switch_equation, int):
        eq_idx = int(switch_equation)
        if eq_idx < 0 or eq_idx >= model.n_equations:
            raise SolverError(
                f"switch_equation index out of range: {eq_idx} "
                f"(n_equations={model.n_equations})"
            )
        return eq_idx

    matches = [i for i, eq in enumerate(model.equations) if eq.name == switch_equation]
    if not matches:
        raise SolverError(
            f"switch_equation '{switch_equation}' was not found in equation names"
        )
    return matches[0]


def _parse_equation_for_model(model: ModelIR, eq_str: str):
    """Parse equation text into an Expression aligned with model symbols."""
    from dsgekit.exceptions import ParseError

    if "=" in eq_str:
        lhs, rhs = eq_str.split("=", 1)
        expr_str = f"({lhs.strip()}) - ({rhs.strip()})"
    else:
        expr_str = eq_str.strip()

    var_objs = {v.name: v for v in model.symbols.variables}
    shock_objs = {s.name: s for s in model.symbols.shocks}
    param_objs = {p.name: p for p in model.symbols.parameters}

    local_ns: dict[str, object] = {"var": var, "shock": shock, "param": param}
    local_ns.update(EQUATION_FUNCTIONS)

    converted = expr_str
    for func in KNOWN_FUNCTIONS:
        converted = re.sub(
            rf"\b{func}\b",
            f"__FUNC_{func}__",
            converted,
            flags=re.IGNORECASE,
        )

    for vname in var_objs:
        converted = re.sub(
            rf"\b{vname}\s*\(\s*(-\d+)\s*\)",
            rf"var(_v_{vname}, \1)",
            converted,
        )
        converted = re.sub(
            rf"\b{vname}\s*\(\s*\+(\d+)\s*\)",
            rf"var(_v_{vname}, \1)",
            converted,
        )
        converted = re.sub(
            rf"\b{vname}\s*\(\s*(\d+)\s*\)",
            rf"var(_v_{vname}, \1)",
            converted,
        )
        converted = re.sub(
            rf"(?<![_\w]){vname}(?![_\w\(])",
            rf"var(_v_{vname})",
            converted,
        )

    for sname in shock_objs:
        converted = re.sub(
            rf"(?<![_\w]){sname}(?![_\w])",
            rf"shock(_s_{sname})",
            converted,
        )
    for pname in param_objs:
        converted = re.sub(
            rf"(?<![_\w]){pname}(?![_\w])",
            rf"param(_p_{pname})",
            converted,
        )

    for func in KNOWN_FUNCTIONS:
        converted = converted.replace(f"__FUNC_{func}__", func)
    converted = converted.replace("^", "**")

    for vname, vobj in var_objs.items():
        local_ns[f"_v_{vname}"] = vobj
    for sname, sobj in shock_objs.items():
        local_ns[f"_s_{sname}"] = sobj
    for pname, pobj in param_objs.items():
        local_ns[f"_p_{pname}"] = pobj

    try:
        return eval(converted, {"__builtins__": {}}, local_ns)
    except Exception as exc:
        raise ParseError(
            f"Failed to parse equation: {eq_str}\n"
            f"Converted to: {converted}\n"
            f"Error: {exc}"
        ) from exc


def _build_regime_model(
    base_model: ModelIR,
    eq_idx: int,
    replacement_equation: str | None,
) -> ModelIR:
    model_out = copy.deepcopy(base_model)
    if replacement_equation is None:
        return model_out

    expr = _parse_equation_for_model(model_out, replacement_equation)
    original_name = model_out.equations[eq_idx].name
    model_out.equations[eq_idx] = Equation(expr, name=original_name)
    model_out.validate()
    return model_out


def _timed_value_from_path(
    *,
    path: NDArray[np.float64],
    var_idx: int,
    period: int,
    timing: int,
    x_initial: NDArray[np.float64],
    x_terminal: NDArray[np.float64],
) -> float:
    if timing == 0:
        return float(path[period, var_idx])
    if timing == -1:
        if period == 0:
            return float(x_initial[var_idx])
        return float(path[period - 1, var_idx])
    if timing == 1:
        if period == (path.shape[0] - 1):
            return float(x_terminal[var_idx])
        return float(path[period + 1, var_idx])
    raise SolverError(f"constraint_timing must be one of -1, 0, 1; got {timing}")


def _is_binding(
    value: float,
    *,
    operator: ConstraintOperator,
    threshold: float,
    tol: float,
) -> bool:
    if operator == "<":
        return value < (threshold - tol)
    if operator == "<=":
        return value <= (threshold + tol)
    if operator == ">":
        return value > (threshold + tol)
    return value >= (threshold - tol)


def _regime_from_path(
    *,
    path: NDArray[np.float64],
    var_idx: int,
    x_initial: NDArray[np.float64],
    x_terminal: NDArray[np.float64],
    timing: int,
    operator: ConstraintOperator,
    threshold: float,
    tol: float,
) -> NDArray[np.bool_]:
    n_periods = path.shape[0]
    out = np.zeros(n_periods, dtype=np.bool_)
    for t in range(n_periods):
        value_t = _timed_value_from_path(
            path=path,
            var_idx=var_idx,
            period=t,
            timing=timing,
            x_initial=x_initial,
            x_terminal=x_terminal,
        )
        out[t] = _is_binding(
            value_t,
            operator=operator,
            threshold=threshold,
            tol=tol,
        )
    return out


def _solve_fixed_regime(
    *,
    relaxed_model: ModelIR,
    binding_model: ModelIR,
    calibration: Calibration,
    n_periods: int,
    shock_path: NDArray[np.float64],
    x_initial: NDArray[np.float64],
    x_terminal: NDArray[np.float64],
    x_initial_guess: NDArray[np.float64],
    regime: NDArray[np.bool_],
    derivative_backend: str,
    eps: float,
    tol: float,
    max_iter: int,
    line_search_max_steps: int,
    line_search_shrink: float,
    resolved_linear_solver: Literal["dense", "sparse"],
    linear_solver_requested: LinearSolverMode | str,
    resolved_jit_backend: Literal["none", "numba"],
    jit_backend_requested: JitBackendMode | str,
    sparse_threshold: int,
    raise_on_fail: bool,
) -> PerfectForesightResult:
    n_vars = relaxed_model.n_variables
    n_eq = relaxed_model.n_equations
    var_names = relaxed_model.variable_names
    shock_names = relaxed_model.shock_names
    eq_names = [eq.name or f"eq_{i+1}" for i, eq in enumerate(relaxed_model.equations)]

    coords = (
        [var_coord(name, -1) for name in var_names]
        + [var_coord(name, 0) for name in var_names]
        + [var_coord(name, 1) for name in var_names]
    )
    deriv_stack = DerivativeStack(derivative_backend, eps=eps)
    context = EvalContext(
        variables={},
        shocks=dict.fromkeys(shock_names, 0.0),
        parameters=calibration.parameters.copy(),
    )

    def evaluate_system(
        x_flat: NDArray[np.float64],
        *,
        compute_jacobian: bool,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64] | sparse.csr_matrix | None,
        int | None,
    ]:
        x_path = np.asarray(x_flat, dtype=np.float64).reshape(n_periods, n_vars)
        residual_vec = np.zeros(n_periods * n_eq, dtype=np.float64)
        jac_builder = None
        if compute_jacobian:
            jac_builder = BlockTridiagonalJacobianBuilder(
                n_periods=n_periods,
                n_eq=n_eq,
                n_vars=n_vars,
                use_sparse=(resolved_linear_solver == "sparse"),
                jit_backend=resolved_jit_backend,
            )

        for t in range(n_periods):
            y_prev = x_initial if t == 0 else x_path[t - 1, :]
            y_curr = x_path[t, :]
            y_next = x_terminal if t == (n_periods - 1) else x_path[t + 1, :]
            _set_period_context(
                context=context,
                var_names=var_names,
                shock_names=shock_names,
                y_prev=y_prev,
                y_curr=y_curr,
                y_next=y_next,
                u_t=shock_path[t, :],
            )

            model_t = binding_model if regime[t] else relaxed_model
            row_slice = slice(t * n_eq, (t + 1) * n_eq)

            f_t = model_t.residuals(context)
            if not np.all(np.isfinite(f_t)):
                raise SolverError(
                    f"Non-finite residuals encountered at period {t + 1}"
                )
            residual_vec[row_slice] = f_t

            if jac_builder is not None:
                j_t = deriv_stack.jacobian(model_t, context, coords)
                if not np.all(np.isfinite(j_t)):
                    raise SolverError(
                        f"Non-finite Jacobian entries at period {t + 1}"
                    )
                a_t = j_t[:, :n_vars]
                b_t = j_t[:, n_vars : 2 * n_vars]
                c_t = j_t[:, 2 * n_vars :]

                jac_builder.add_period_blocks(
                    period=t,
                    a_t=a_t,
                    b_t=b_t,
                    c_t=c_t,
                )

        if jac_builder is None:
            return residual_vec, None, None
        jac_assembly = jac_builder.build()
        return residual_vec, jac_assembly.matrix, jac_assembly.nnz

    x = np.asarray(x_initial_guess, dtype=np.float64).reshape(-1).copy()
    residual_history: list[float] = []
    converged = False
    n_iterations = 0
    stalled = False
    last_jacobian_nnz = 0

    for it in range(1, max_iter + 1):
        n_iterations = it
        try:
            f_val, jac, jac_nnz = evaluate_system(x, compute_jacobian=True)
        except Exception as exc:
            if raise_on_fail:
                raise SolverError(
                    f"OccBin fixed-regime evaluation failed at iteration {it}: {exc}"
                ) from exc
            break

        assert jac is not None  # for type checkers
        if jac_nnz is not None:
            last_jacobian_nnz = jac_nnz
        current_norm = float(np.max(np.abs(f_val)))
        residual_history.append(current_norm)
        if current_norm < tol:
            converged = True
            break

        rhs = -f_val
        delta = solve_newton_step(
            jacobian=jac,
            rhs=rhs,
            solver_mode=resolved_linear_solver,
        )

        if not np.all(np.isfinite(delta)):
            if raise_on_fail:
                raise SolverError(
                    f"Newton step became non-finite at iteration {it}"
                )
            break

        accepted = False
        alpha = 1.0
        for _ in range(line_search_max_steps):
            x_trial = x + alpha * delta
            try:
                f_trial, _, _ = evaluate_system(x_trial, compute_jacobian=False)
            except Exception:
                alpha *= line_search_shrink
                continue

            trial_norm = float(np.max(np.abs(f_trial)))
            if np.isfinite(trial_norm) and trial_norm < current_norm:
                x = x_trial
                accepted = True
                break
            alpha *= line_search_shrink

        if not accepted:
            stalled = True
            break

    f_final, _, _ = evaluate_system(x, compute_jacobian=False)
    max_abs_residual = float(np.max(np.abs(f_final)))
    converged = converged or (max_abs_residual < tol)

    x_final = x.reshape(n_periods, n_vars)
    period_index = pd.RangeIndex(start=1, stop=n_periods + 1, name="period")
    path_df = pd.DataFrame(x_final, index=period_index, columns=var_names)
    shocks_df = pd.DataFrame(shock_path, index=period_index, columns=shock_names)
    residuals_df = pd.DataFrame(
        f_final.reshape(n_periods, n_eq),
        index=period_index,
        columns=eq_names,
    )

    jac_rows = n_periods * n_eq
    jac_cols = n_periods * n_vars
    jac_density = (
        float(last_jacobian_nnz) / float(jac_rows * jac_cols)
        if jac_rows > 0 and jac_cols > 0
        else 0.0
    )

    return PerfectForesightResult(
        path=path_df,
        shocks=shocks_df,
        residuals=residuals_df,
        converged=converged,
        n_iterations=n_iterations,
        max_abs_residual=max_abs_residual,
        residual_history=residual_history,
        solver_meta={
            "solver": "occbin_lite_fixed_regime_newton",
            "derivative_backend": derivative_backend,
            "eps": float(eps),
            "tol": float(tol),
            "max_iter": int(max_iter),
            "line_search_max_steps": int(line_search_max_steps),
            "line_search_shrink": float(line_search_shrink),
            "linear_solver_requested": str(linear_solver_requested),
            "linear_solver_used": resolved_linear_solver,
            "jit_backend_requested": str(jit_backend_requested),
            "jit_backend_used": resolved_jit_backend,
            "sparse_threshold": int(sparse_threshold),
            "jacobian_nnz_last": int(last_jacobian_nnz),
            "jacobian_density_last": float(jac_density),
            "stalled": int(stalled),
            "n_binding_periods": int(np.sum(regime)),
        },
    )


def solve_occbin_lite(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    *,
    n_periods: int,
    switch_equation: str | int,
    binding_equation: str,
    relaxed_equation: str | None = None,
    constraint_var: str,
    constraint_operator: ConstraintOperator = "<=",
    constraint_value: float = 0.0,
    constraint_timing: int = 0,
    shocks: dict[str, NDArray[np.float64]] | NDArray[np.float64] | None = None,
    initial_state: dict[str, float] | NDArray[np.float64] | None = None,
    terminal_state: dict[str, float] | NDArray[np.float64] | None = None,
    initial_path: pd.DataFrame | NDArray[np.float64] | None = None,
    initial_regime: NDArray[np.bool_] | list[bool] | None = None,
    derivative_backend: str = "numeric",
    eps: float = 1e-6,
    tol: float = 1e-8,
    max_iter: int = 50,
    max_regime_iter: int = 30,
    line_search_max_steps: int = 12,
    line_search_shrink: float = 0.5,
    linear_solver: LinearSolverMode = "auto",
    jit_backend: JitBackendMode = "none",
    sparse_threshold: int = 200,
    constraint_tol: float = 1e-10,
    raise_on_fail: bool = True,
) -> OccBinResult:
    """Solve one-constraint OccBin-lite deterministic path.

    The equation identified by `switch_equation` is replaced by:
    - `relaxed_equation` when the constraint is not binding
    - `binding_equation` when it is binding

    If `relaxed_equation` is omitted, the original model equation is used.
    The binding regime is updated from `constraint_var` and the chosen operator.
    """
    if n_periods < 1:
        raise SolverError(f"n_periods must be >= 1, got {n_periods}")
    if max_iter < 1:
        raise SolverError(f"max_iter must be >= 1, got {max_iter}")
    if max_regime_iter < 1:
        raise SolverError(f"max_regime_iter must be >= 1, got {max_regime_iter}")
    if tol <= 0.0 or not np.isfinite(tol):
        raise SolverError(f"tol must be finite and > 0, got {tol}")
    if constraint_operator not in {"<", "<=", ">", ">="}:
        raise SolverError(
            "constraint_operator must be one of '<', '<=', '>', '>='; "
            f"got '{constraint_operator}'"
        )
    if constraint_timing not in {-1, 0, 1}:
        raise SolverError(
            f"constraint_timing must be one of -1, 0, 1; got {constraint_timing}"
        )
    if line_search_max_steps < 1:
        raise SolverError(
            f"line_search_max_steps must be >= 1, got {line_search_max_steps}"
        )
    if not (0.0 < line_search_shrink < 1.0):
        raise SolverError(
            "line_search_shrink must be in (0, 1), "
            f"got {line_search_shrink}"
        )

    model.validate()
    if model.lead_lag.max_lag > 1 or model.lead_lag.max_lead > 1:
        raise SolverError(
            "OccBin-lite solver supports timings up to one lag/lead "
            f"(max_lag={model.lead_lag.max_lag}, max_lead={model.lead_lag.max_lead})."
        )

    n_vars = model.n_variables
    n_eq = model.n_equations
    if n_eq != n_vars:
        raise SolverError(
            "OccBin-lite solver currently requires a square system: "
            f"{n_eq} equations vs {n_vars} variables"
        )
    resolved_linear_solver = resolve_linear_solver_mode(
        linear_solver=linear_solver,
        n_unknowns=n_periods * n_vars,
        sparse_threshold=sparse_threshold,
    )
    resolved_jit_backend = resolve_jit_backend_mode(
        jit_backend=jit_backend,
        linear_solver=resolved_linear_solver,
    )

    if constraint_var not in model.variable_names:
        raise SolverError(f"constraint_var '{constraint_var}' is not a model variable")

    eq_idx = _resolve_equation_index(model, switch_equation)
    relaxed_model = _build_regime_model(model, eq_idx, relaxed_equation)
    binding_model = _build_regime_model(model, eq_idx, binding_equation)

    var_names = model.variable_names
    shock_names = model.shock_names
    steady_vec = np.array(
        [float(steady_state.values.get(name, 0.0)) for name in var_names],
        dtype=np.float64,
    )
    initial_default = _default_initial_state_from_steady_state(
        steady_state=steady_state,
        var_names=var_names,
        baseline=steady_vec,
    )
    terminal_default = _default_terminal_state_from_steady_state(
        steady_state=steady_state,
        var_names=var_names,
        baseline=steady_vec,
    )
    x_initial = _resolve_state_vector(
        value=initial_state,
        var_names=var_names,
        baseline=initial_default,
        label="initial_state",
    )
    x_terminal = _resolve_state_vector(
        value=terminal_state,
        var_names=var_names,
        baseline=terminal_default,
        label="terminal_state",
    )
    if shocks is None:
        shock_path = _default_shock_path_from_steady_state(
            steady_state=steady_state,
            n_periods=n_periods,
            shock_names=shock_names,
        )
    else:
        shock_path = _resolve_shock_path(
            shocks=shocks,
            n_periods=n_periods,
            shock_names=shock_names,
        )

    x_guess = _resolve_initial_path(
        initial_path=initial_path,
        n_periods=n_periods,
        var_names=var_names,
        x_initial=x_initial,
        x_terminal=x_terminal,
    )

    if initial_regime is None:
        regime = np.zeros(n_periods, dtype=np.bool_)
    else:
        regime = np.asarray(initial_regime, dtype=np.bool_).reshape(-1)
        if regime.shape[0] != n_periods:
            raise SolverError(
                f"initial_regime must have length {n_periods}, got {regime.shape[0]}"
            )

    var_idx = var_names.index(constraint_var)
    seen_regimes = {regime.tobytes()}
    regime_binding_counts: list[int] = []
    regime_converged = False
    repeated_regime = False
    last_fixed_result: PerfectForesightResult | None = None

    for regime_it in range(1, max_regime_iter + 1):
        fixed_result = _solve_fixed_regime(
            relaxed_model=relaxed_model,
            binding_model=binding_model,
            calibration=calibration,
            n_periods=n_periods,
            shock_path=shock_path,
            x_initial=x_initial,
            x_terminal=x_terminal,
            x_initial_guess=x_guess,
            regime=regime,
            derivative_backend=derivative_backend,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            line_search_max_steps=line_search_max_steps,
            line_search_shrink=line_search_shrink,
            resolved_linear_solver=resolved_linear_solver,
            linear_solver_requested=linear_solver,
            resolved_jit_backend=resolved_jit_backend,
            jit_backend_requested=jit_backend,
            sparse_threshold=sparse_threshold,
            raise_on_fail=raise_on_fail,
        )
        last_fixed_result = fixed_result
        x_guess = fixed_result.path.values

        if not fixed_result.converged:
            if raise_on_fail:
                raise SolverError(
                    "OccBin-lite fixed-regime solve did not converge "
                    f"(regime_iteration={regime_it}, "
                    f"newton_iterations={fixed_result.n_iterations}, "
                    f"max_abs_residual={fixed_result.max_abs_residual:.3e})"
                )
            break

        next_regime = _regime_from_path(
            path=fixed_result.path.values,
            var_idx=var_idx,
            x_initial=x_initial,
            x_terminal=x_terminal,
            timing=constraint_timing,
            operator=constraint_operator,
            threshold=float(constraint_value),
            tol=float(constraint_tol),
        )
        regime_binding_counts.append(int(np.sum(next_regime)))

        if np.array_equal(next_regime, regime):
            regime = next_regime
            regime_converged = True
            break

        key = next_regime.tobytes()
        if key in seen_regimes:
            repeated_regime = True
            break

        seen_regimes.add(key)
        regime = next_regime

    if last_fixed_result is None:
        raise SolverError("OccBin-lite did not execute any fixed-regime iteration")

    period_index = last_fixed_result.path.index
    binding_series = pd.Series(
        regime.astype(bool),
        index=period_index,
        name="binding",
    )
    converged = bool(last_fixed_result.converged and regime_converged)

    result = OccBinResult(
        path=last_fixed_result.path.copy(),
        shocks=last_fixed_result.shocks.copy(),
        residuals=last_fixed_result.residuals.copy(),
        binding_regime=binding_series,
        converged=converged,
        n_regime_iterations=len(regime_binding_counts),
        n_newton_iterations=last_fixed_result.n_iterations,
        max_abs_residual=last_fixed_result.max_abs_residual,
        residual_history=last_fixed_result.residual_history.copy(),
        regime_binding_counts=regime_binding_counts,
        solver_meta={
            "solver": "occbin_lite",
            "derivative_backend": derivative_backend,
            "eps": float(eps),
            "tol": float(tol),
            "max_iter": int(max_iter),
            "max_regime_iter": int(max_regime_iter),
            "line_search_max_steps": int(line_search_max_steps),
            "line_search_shrink": float(line_search_shrink),
            "linear_solver_requested": str(linear_solver),
            "linear_solver_used": resolved_linear_solver,
            "jit_backend_requested": str(jit_backend),
            "jit_backend_used": resolved_jit_backend,
            "sparse_threshold": int(sparse_threshold),
            "constraint_var": constraint_var,
            "constraint_operator": constraint_operator,
            "constraint_value": float(constraint_value),
            "constraint_timing": int(constraint_timing),
            "constraint_tol": float(constraint_tol),
            "repeated_regime": int(repeated_regime),
            "fixed_regime_solver": str(
                last_fixed_result.solver_meta.get("solver", "")
            ),
        },
    )

    if not result.converged and raise_on_fail:
        reason = "repeated regime sequence" if repeated_regime else "no fixed point"
        raise SolverError(
            "OccBin-lite solver did not converge "
            f"(reason={reason}, "
            f"regime_iterations={result.n_regime_iterations}, "
            f"max_abs_residual={result.max_abs_residual:.3e})"
        )
    return result
