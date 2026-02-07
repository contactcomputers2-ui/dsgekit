"""Solvers for DSGE models: linear (Gensys, Klein) and nonlinear."""

from dsgekit.solvers.diagnostics import (
    BKDiagnostics,
    EigenInfo,
    diagnose_bk,
)
from dsgekit.solvers.discretion import (
    DiscretionLQResult,
    DiscretionPath,
    simulate_discretion_path,
    solve_discretion_from_linear_solution,
    solve_discretion_lq,
)
from dsgekit.solvers.linear import (
    LinearSolution,
    eigenvalue_analysis,
    solve_linear,
)
from dsgekit.solvers.nonlinear import (
    FirstOrderApproximation,
    NewsShock,
    OccBinResult,
    PerfectForesightResult,
    SecondOrderSolution,
    ThirdOrderSolution,
    anticipated_shock,
    build_news_shock_path,
    linearize_first_order,
    solve_first_order,
    solve_occbin_lite,
    solve_perfect_foresight,
    solve_second_order,
    solve_third_order,
    unanticipated_shock,
)
from dsgekit.solvers.policy import (
    OSRCandidate,
    OSRResult,
    osr_grid_search,
)
from dsgekit.solvers.ramsey import (
    RamseyLQResult,
    RamseyPath,
    simulate_ramsey_path,
    solve_ramsey_from_linear_solution,
    solve_ramsey_lq,
)

__all__ = [
    "LinearSolution",
    "solve_linear",
    "eigenvalue_analysis",
    "DiscretionLQResult",
    "DiscretionPath",
    "solve_discretion_lq",
    "solve_discretion_from_linear_solution",
    "simulate_discretion_path",
    "OSRCandidate",
    "OSRResult",
    "osr_grid_search",
    "RamseyLQResult",
    "RamseyPath",
    "solve_ramsey_lq",
    "solve_ramsey_from_linear_solution",
    "simulate_ramsey_path",
    "FirstOrderApproximation",
    "linearize_first_order",
    "solve_first_order",
    "OccBinResult",
    "solve_occbin_lite",
    "SecondOrderSolution",
    "solve_second_order",
    "ThirdOrderSolution",
    "solve_third_order",
    "NewsShock",
    "PerfectForesightResult",
    "anticipated_shock",
    "unanticipated_shock",
    "build_news_shock_path",
    "solve_perfect_foresight",
    "BKDiagnostics",
    "EigenInfo",
    "diagnose_bk",
]
