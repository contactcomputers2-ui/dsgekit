"""Command-line interface for dsgekit.

Usage:
    dsgekit run model.mod         Run full pipeline and show results
    dsgekit solve model.yaml      Solve model and show solution
    dsgekit irf model.mod         Compute and display IRFs
    dsgekit simulate model.yaml   Run stochastic simulation
    dsgekit estimation model.mod  Run MLE/MAP/MCMC estimation
    dsgekit forecast model.yaml   In/out-of-sample forecasts
    dsgekit decompose model.yaml  Historical shock decomposition
    dsgekit osr model.mod         Grid-search optimization for policy-rule params
    dsgekit baseline_regression   Batch baseline regression checks
    dsgekit info model.mod        Show model information
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="dsgekit",
        description="A Python toolkit for DSGE models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dsgekit run model.mod              Run full analysis pipeline
  dsgekit irf model.yaml -s e        IRFs for shock 'e'
  dsgekit simulate model.mod -n 200  Simulate 200 periods
  dsgekit estimation model.mod --method mle --params rho
  dsgekit forecast model.yaml -p 12  12-step forecast
  dsgekit osr model.mod --grid rho=0.1,0.5,0.9 --loss y=1.0
  dsgekit baseline_regression --baselines path/to/baselines
  dsgekit info model.yaml            Show model structure
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========================================================================
    # run command
    # =========================================================================
    run_parser = subparsers.add_parser(
        "run",
        help="Run full pipeline: load, solve, IRFs, moments",
        description="Load a model, solve it, and display key results.",
    )
    run_parser.add_argument("model", help="Model file (.mod, .yaml)")
    run_parser.add_argument(
        "-o", "--output",
        help="Output directory for plots",
    )
    run_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    run_parser.add_argument(
        "--format",
        choices=["plain", "markdown"],
        default="plain",
        help="Output format for tables",
    )

    # =========================================================================
    # solve command
    # =========================================================================
    solve_parser = subparsers.add_parser(
        "solve",
        help="Solve model and show solution matrices",
        description="Load and solve a model, display solution matrices.",
    )
    solve_parser.add_argument("model", help="Model file (.mod, .yaml)")
    solve_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed eigenvalue analysis",
    )

    # =========================================================================
    # irf command
    # =========================================================================
    irf_parser = subparsers.add_parser(
        "irf",
        help="Compute impulse response functions",
        description="Compute and display IRFs for model shocks.",
    )
    irf_parser.add_argument("model", help="Model file (.mod, .yaml)")
    irf_parser.add_argument(
        "-s", "--shock",
        help="Shock name (default: all shocks)",
    )
    irf_parser.add_argument(
        "-p", "--periods",
        type=int,
        default=40,
        help="Number of periods (default: 40)",
    )
    irf_parser.add_argument(
        "-o", "--output",
        help="Output file for plot (e.g., irf.png)",
    )
    irf_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting, only show table",
    )

    # =========================================================================
    # simulate command
    # =========================================================================
    sim_parser = subparsers.add_parser(
        "simulate",
        help="Run stochastic simulation",
        description="Simulate the model with random shocks.",
    )
    sim_parser.add_argument("model", help="Model file (.mod, .yaml)")
    sim_parser.add_argument(
        "-n", "--periods",
        type=int,
        default=100,
        help="Number of periods (default: 100)",
    )
    sim_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    sim_parser.add_argument(
        "-o", "--output",
        help="Output file for plot (e.g., sim.png)",
    )

    # =========================================================================
    # estimation command
    # =========================================================================
    estimation_parser = subparsers.add_parser(
        "estimation",
        help="Run parameter estimation (MLE/MAP/MCMC)",
        description="Estimate model parameters from data or synthetic observations.",
    )
    estimation_parser.add_argument("model", help="Model file (.mod, .yaml)")
    estimation_parser.add_argument(
        "--method",
        choices=["mle", "map", "mcmc"],
        default="mle",
        help="Estimation method (default: mle)",
    )
    estimation_parser.add_argument(
        "--data",
        help="Optional CSV with observed data. If omitted, synthetic data is simulated.",
    )
    estimation_parser.add_argument(
        "--train-periods",
        type=int,
        default=120,
        help="Periods for synthetic data when --data is not provided (default: 120)",
    )
    estimation_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for synthetic data and MCMC proposals "
            "(default: 42)"
        ),
    )
    estimation_parser.add_argument(
        "--observables",
        help="Comma-separated observable names (default: model varobs or first variable)",
    )
    estimation_parser.add_argument(
        "--params",
        help=(
            "Comma-separated parameter/shock-stderr names to estimate "
            "(default: infer from estimated_params)"
        ),
    )
    estimation_parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum optimizer iterations for MLE/MAP (default: 200)",
    )
    estimation_parser.add_argument(
        "--compute-se",
        action="store_true",
        help="Compute Hessian-based standard errors for MLE/MAP",
    )
    estimation_parser.add_argument(
        "--no-demean",
        action="store_true",
        help="Do not demean observations before likelihood evaluation",
    )
    estimation_parser.add_argument(
        "--prior-weight",
        type=float,
        default=1.0,
        help="Prior weight for MAP/MCMC (default: 1.0)",
    )
    estimation_parser.add_argument(
        "--allow-missing-priors",
        action="store_true",
        help="Allow MAP/MCMC even if some estimated parameters have no prior",
    )
    estimation_parser.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="MCMC draws when --method mcmc (default: 1000)",
    )
    estimation_parser.add_argument(
        "--burn-in",
        type=int,
        default=250,
        help="MCMC burn-in when --method mcmc (default: 250)",
    )
    estimation_parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="MCMC thinning when --method mcmc (default: 1)",
    )
    estimation_parser.add_argument(
        "--proposal-scale",
        type=float,
        default=0.05,
        help="MCMC random-walk proposal scale (default: 0.05)",
    )
    estimation_parser.add_argument(
        "-o", "--output",
        help=(
            "Optional CSV output path. "
            "MLE/MAP: parameter estimates. MCMC: saved posterior samples."
        ),
    )

    # =========================================================================
    # forecast command
    # =========================================================================
    forecast_parser = subparsers.add_parser(
        "forecast",
        help="Compute in/out-of-sample forecasts via Kalman filter",
        description="Filter data and produce one-step in-sample plus h-step forecasts.",
    )
    forecast_parser.add_argument("model", help="Model file (.mod, .yaml)")
    forecast_parser.add_argument(
        "-p", "--periods",
        type=int,
        default=12,
        help="Out-of-sample forecast horizon (default: 12)",
    )
    forecast_parser.add_argument(
        "--data",
        help="Optional CSV with observed data. If omitted, synthetic data is simulated.",
    )
    forecast_parser.add_argument(
        "--train-periods",
        type=int,
        default=120,
        help="Periods for synthetic data when --data is not provided (default: 120)",
    )
    forecast_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data (default: 42)",
    )
    forecast_parser.add_argument(
        "--observables",
        help="Comma-separated observable names (default: model varobs or first variable)",
    )
    forecast_parser.add_argument(
        "--use-filtered",
        action="store_true",
        help="Use final filtered state (default uses final smoothed state)",
    )
    forecast_parser.add_argument(
        "-o", "--output",
        help="Optional CSV output path for out-of-sample forecast means",
    )

    # =========================================================================
    # decompose command
    # =========================================================================
    decompose_parser = subparsers.add_parser(
        "decompose",
        help="Historical shock decomposition from smoothed states",
        description="Run Kalman smoother and decompose trajectories by structural shock.",
    )
    decompose_parser.add_argument("model", help="Model file (.mod, .yaml)")
    decompose_parser.add_argument(
        "--data",
        help="Optional CSV with observed data. If omitted, synthetic data is simulated.",
    )
    decompose_parser.add_argument(
        "--train-periods",
        type=int,
        default=120,
        help="Periods for synthetic data when --data is not provided (default: 120)",
    )
    decompose_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data (default: 42)",
    )
    decompose_parser.add_argument(
        "--observables",
        help="Comma-separated observable names (default: model varobs or first variable)",
    )
    decompose_parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Disable residual balancing component in decomposition",
    )
    decompose_parser.add_argument(
        "--no-steady-state",
        action="store_true",
        help="Do not include steady-state level in observable decomposition",
    )
    decompose_parser.add_argument(
        "-o", "--output",
        help="Optional CSV output path for observable contributions",
    )

    # =========================================================================
    # osr command
    # =========================================================================
    osr_parser = subparsers.add_parser(
        "osr",
        help="Grid-search optimization for policy-rule parameters",
        description=(
            "Run OSR-style parameter sweeps over user-defined rule grids and "
            "minimize weighted unconditional variances."
        ),
    )
    osr_parser.add_argument("model", help="Model file (.mod, .yaml)")
    osr_parser.add_argument(
        "--grid",
        action="append",
        required=True,
        help=(
            "Parameter grid spec (repeatable): "
            "'name=v1,v2,...' or 'name=start:stop:num'."
        ),
    )
    osr_parser.add_argument(
        "--loss",
        action="append",
        help=(
            "Loss weight spec (repeatable): 'var=weight'. "
            "If omitted, defaults to first model variable with weight 1.0."
        ),
    )
    osr_parser.add_argument(
        "--allow-indeterminate",
        action="store_true",
        help="Allow non-determinate candidates (default rejects them)",
    )
    osr_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Rows to display from ranking table (default: 10)",
    )
    osr_parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed/non-admissible candidates in displayed/output table",
    )
    osr_parser.add_argument(
        "-o", "--output",
        help="Optional CSV output path for ranked OSR candidates",
    )

    # =========================================================================
    # baseline_regression command
    # =========================================================================
    regression_parser = subparsers.add_parser(
        "baseline_regression",
        help="Run batch regression checks against reference baselines",
        description=(
            "Run compatibility checks against one or more baseline JSON files "
            "and generate a compact deviation dashboard."
        ),
    )
    regression_parser.add_argument(
        "--baselines",
        nargs="+",
        required=True,
        help=(
            "Baseline JSON paths, directories, or glob patterns. "
            "Directories expand to '*.json'."
        ),
    )
    regression_parser.add_argument(
        "--model",
        help=(
            "Optional model source override used for all baselines "
            "(default: baseline model_path)."
        ),
    )
    regression_parser.add_argument(
        "-p", "--periods",
        type=int,
        help="Optional IRF horizon override for all baselines",
    )
    regression_parser.add_argument(
        "--dashboard",
        help="Optional output path for Markdown dashboard",
    )
    regression_parser.add_argument(
        "--json-output",
        help="Optional output path for JSON suite report",
    )

    # =========================================================================
    # info command
    # =========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Show model information",
        description="Display model structure: variables, shocks, parameters.",
    )
    info_parser.add_argument("model", help="Model file (.mod, .yaml)")

    return parser


def _get_version() -> str:
    """Get package version."""
    try:
        from dsgekit._version import __version__

        return __version__
    except ImportError:
        return "unknown"


def _load_model(model_path: str):
    """Load model from file."""
    from dsgekit import load_model

    path = Path(model_path)
    if not path.exists():
        print(f"Error: Model file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        return load_model(path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def _solve_model(model, calibration, steady_state):
    """Linearize and solve model."""
    from dsgekit.solvers import solve_linear
    from dsgekit.transforms import linearize

    try:
        linear_sys = linearize(model, steady_state, calibration)
        solution = solve_linear(linear_sys)
        return solution
    except Exception as e:
        print(f"Error solving model: {e}", file=sys.stderr)
        sys.exit(1)


def _resolve_observables(model, observables_arg: str | None) -> list[str]:
    """Resolve observable names from CLI arg or model defaults."""
    if observables_arg:
        observables = [token.strip() for token in observables_arg.split(",") if token.strip()]
    elif model.observables:
        observables = list(model.observables)
    elif model.variable_names:
        observables = [model.variable_names[0]]
    else:
        observables = []

    if not observables:
        print("Error: Could not resolve observables", file=sys.stderr)
        sys.exit(1)

    unknown = [name for name in observables if name not in model.variable_names]
    if unknown:
        print(
            f"Error: Unknown observables {unknown}. Available variables: {model.variable_names}",
            file=sys.stderr,
        )
        sys.exit(1)

    return observables


def _resolve_param_names(params_arg: str | None) -> list[str] | None:
    """Resolve parameter names from comma-separated CLI input."""
    if params_arg is None:
        return None

    names = [token.strip() for token in params_arg.split(",") if token.strip()]
    if not names:
        print("Error: --params must include at least one name", file=sys.stderr)
        sys.exit(1)

    if len(set(names)) != len(names):
        print("Error: --params contains duplicates", file=sys.stderr)
        sys.exit(1)

    return names


def _parse_grid_specs(specs: list[str]) -> dict[str, list[float]]:
    """Parse repeatable --grid specs into parameter -> candidate values."""
    import numpy as np

    grid: dict[str, list[float]] = {}
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"Invalid --grid spec '{raw}'. Use name=v1,v2 or name=a:b:n")
        name, rhs = raw.split("=", 1)
        name = name.strip()
        rhs = rhs.strip()
        if not name or not rhs:
            raise ValueError(f"Invalid --grid spec '{raw}'. Empty name or value part")
        if name in grid:
            raise ValueError(f"Duplicate --grid parameter '{name}'")

        values: list[float]
        if ":" in rhs and "," not in rhs:
            parts = [token.strip() for token in rhs.split(":")]
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid range spec '{raw}'. Use name=start:stop:num"
                )
            start = float(parts[0])
            stop = float(parts[1])
            num = int(parts[2])
            if num < 2:
                raise ValueError(
                    f"Range spec '{raw}' requires num >= 2"
                )
            values = [float(v) for v in np.linspace(start, stop, num=num)]
        else:
            tokens = [token.strip() for token in rhs.split(",") if token.strip()]
            if not tokens:
                raise ValueError(f"Invalid --grid spec '{raw}'. No values found")
            values = [float(token) for token in tokens]

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values in --grid spec '{raw}'")
        grid[name] = values

    return grid


def _parse_weight_specs(specs: list[str] | None) -> dict[str, float]:
    """Parse repeatable --loss specs into variable -> weight."""
    weights: dict[str, float] = {}
    if not specs:
        return weights

    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"Invalid --loss spec '{raw}'. Use var=weight")
        name, rhs = raw.split("=", 1)
        name = name.strip()
        rhs = rhs.strip()
        if not name or not rhs:
            raise ValueError(f"Invalid --loss spec '{raw}'. Empty name or weight")
        if name in weights:
            raise ValueError(f"Duplicate --loss variable '{name}'")
        weight = float(rhs)
        if weight < 0.0:
            raise ValueError(f"Loss weight must be >= 0 for '{name}'")
        weights[name] = weight

    return weights


def _load_or_simulate_observations(args: Namespace, solution, cal, observables: list[str]):
    """Load CSV data if provided, otherwise simulate synthetic observations."""
    import pandas as pd

    data_path = getattr(args, "data", None)
    if data_path:
        path = Path(data_path)
        if not path.exists():
            print(f"Error: Data file not found: {path}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(path)
        if "period" in df.columns:
            df = df.set_index("period")
        missing = [name for name in observables if name not in df.columns]
        if missing:
            print(
                f"Error: Data file missing observables columns {missing}",
                file=sys.stderr,
            )
            sys.exit(1)
        return df[observables], f"csv:{path}"

    from dsgekit.simulate import simulate

    train_periods = int(getattr(args, "train_periods", 120))
    seed = getattr(args, "seed", 42)
    sim_result = simulate(solution, cal, n_periods=train_periods, seed=seed)
    return sim_result.data[observables], f"simulated(seed={seed}, periods={train_periods})"


def cmd_run(args: Namespace) -> int:
    """Run full pipeline command."""
    from dsgekit.report import (
        TableFormat,
        format_model_summary,
        format_moments,
    )
    from dsgekit.simulate import irf_all_shocks, moments

    print(f"Loading model: {args.model}")
    model, cal, ss = _load_model(args.model)

    print("Solving model...")
    solution = _solve_model(model, cal, ss)

    print()

    # Model summary
    fmt = TableFormat.MARKDOWN if args.format == "markdown" else TableFormat.PLAIN
    print(format_model_summary(model, cal, ss, solution, format=fmt))
    print()

    # Moments
    m = moments(solution, cal)
    print(format_moments(m, format=fmt))
    print()

    # IRFs
    if not args.no_plots:
        try:
            from dsgekit.report import plot_irf_comparison, save_figure

            irfs = irf_all_shocks(solution, periods=40)

            fig = plot_irf_comparison(irfs, title=f"IRFs - {model.name}")

            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "irf.png"
                save_figure(fig, str(output_path))
                print(f"IRF plot saved to: {output_path}")
            else:
                from dsgekit.report import show_figures

                show_figures()

        except ImportError:
            print("Note: matplotlib not installed, skipping plots")

    return 0


def cmd_solve(args: Namespace) -> int:
    """Solve command."""
    from dsgekit.report import blanchard_kahn_summary
    from dsgekit.solvers.linear.gensys import eigenvalue_analysis

    print(f"Loading model: {args.model}")
    model, cal, ss = _load_model(args.model)

    print("Solving model...")
    solution = _solve_model(model, cal, ss)

    print()
    print("=" * 50)
    print("SOLUTION")
    print("=" * 50)
    print()

    # Blanchard-Kahn summary
    print(blanchard_kahn_summary(solution).to_string(index=False))
    print()

    # Transition matrix T
    print("Transition matrix T (y_t = T * y_{t-1} + R * u_t):")
    print("-" * 40)
    import pandas as pd

    T_df = pd.DataFrame(
        solution.T,
        index=solution.var_names,
        columns=solution.var_names,
    )
    print(T_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    # Impact matrix R
    print("Impact matrix R:")
    print("-" * 40)
    R_df = pd.DataFrame(
        solution.R,
        index=solution.var_names,
        columns=solution.shock_names,
    )
    print(R_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    if args.verbose:
        print("Eigenvalue analysis:")
        print("-" * 40)
        print(eigenvalue_analysis(solution.eigenvalues))

    return 0


def cmd_irf(args: Namespace) -> int:
    """IRF command."""
    from dsgekit.report import irf_table
    from dsgekit.simulate import irf, irf_all_shocks

    print(f"Loading model: {args.model}")
    model, cal, ss = _load_model(args.model)

    print("Solving model...")
    solution = _solve_model(model, cal, ss)

    print()

    if args.shock:
        # Single shock
        if args.shock not in solution.shock_names:
            print(
                f"Error: Shock '{args.shock}' not found. "
                f"Available: {solution.shock_names}",
                file=sys.stderr,
            )
            return 1

        irf_result = irf(solution, args.shock, periods=args.periods)
        print(f"IRF to shock '{args.shock}':")
        print("-" * 40)
        print(irf_table(irf_result).to_string(float_format=lambda x: f"{x:.6f}"))

        if not args.no_plot:
            try:
                from dsgekit.report import plot_irf, save_figure, show_figures

                fig = plot_irf(irf_result)
                if args.output:
                    save_figure(fig, args.output)
                    print(f"\nPlot saved to: {args.output}")
                else:
                    show_figures()
            except ImportError:
                pass

    else:
        # All shocks
        irfs = irf_all_shocks(solution, periods=args.periods)
        for shock_name, irf_result in irfs.items():
            print(f"IRF to shock '{shock_name}':")
            print("-" * 40)
            print(irf_table(irf_result).to_string(float_format=lambda x: f"{x:.6f}"))
            print()

        if not args.no_plot:
            try:
                from dsgekit.report import (
                    plot_irf_comparison,
                    save_figure,
                    show_figures,
                )

                fig = plot_irf_comparison(irfs)
                if args.output:
                    save_figure(fig, args.output)
                    print(f"Plot saved to: {args.output}")
                else:
                    show_figures()
            except ImportError:
                pass

    return 0


def cmd_simulate(args: Namespace) -> int:
    """Simulate command."""
    from dsgekit.simulate import simulate

    print(f"Loading model: {args.model}")
    model, cal, ss = _load_model(args.model)

    print("Solving model...")
    solution = _solve_model(model, cal, ss)

    print(f"Simulating {args.periods} periods...")
    sim_result = simulate(solution, cal, n_periods=args.periods, seed=args.seed)

    print()
    print("Simulation statistics:")
    print("-" * 40)
    print(sim_result.data.describe().to_string(float_format=lambda x: f"{x:.6f}"))

    try:
        from dsgekit.report import plot_simulation, save_figure, show_figures

        fig = plot_simulation(sim_result)
        if args.output:
            save_figure(fig, args.output)
            print(f"\nPlot saved to: {args.output}")
        else:
            show_figures()
    except ImportError:
        pass

    return 0


def cmd_forecast(args: Namespace) -> int:
    """Forecast command."""
    from dsgekit.filters import forecast, kalman_filter, kalman_smoother
    from dsgekit.transforms import to_state_space

    if args.periods < 1:
        print("Error: --periods must be >= 1", file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}")
    model, cal, ss = _load_model(args.model)
    observables = _resolve_observables(model, args.observables)

    print("Solving model...")
    solution = _solve_model(model, cal, ss)
    data, data_label = _load_or_simulate_observations(args, solution, cal, observables)

    print(f"Filtering data ({data_label})...")
    ss_model = to_state_space(solution, cal, observables=observables)
    kf = kalman_filter(ss_model, data)
    sm = kalman_smoother(ss_model, kf)

    result = forecast(
        ss_model,
        kf,
        steps=args.periods,
        smoother_result=None if args.use_filtered else sm,
    )

    print()
    print(result.summary())
    print()
    print("Out-of-sample forecast means:")
    print("-" * 40)
    print(result.out_of_sample_observables.to_string(float_format=lambda x: f"{x:.6f}"))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.out_of_sample_observables.to_csv(out_path)
        print(f"\nForecast means saved to: {out_path}")

    return 0


def cmd_decompose(args: Namespace) -> int:
    """Historical decomposition command."""
    from dsgekit.filters import historical_decomposition, kalman_filter, kalman_smoother
    from dsgekit.transforms import to_state_space

    print(f"Loading model: {args.model}")
    model, cal, ss = _load_model(args.model)
    observables = _resolve_observables(model, args.observables)

    print("Solving model...")
    solution = _solve_model(model, cal, ss)
    data, data_label = _load_or_simulate_observations(args, solution, cal, observables)

    print(f"Smoothing data ({data_label})...")
    ss_model = to_state_space(solution, cal, observables=observables)
    kf = kalman_filter(ss_model, data)
    sm = kalman_smoother(ss_model, kf)

    result = historical_decomposition(
        ss_model,
        sm,
        include_residual=not args.no_residual,
        include_steady_state=not args.no_steady_state,
    )

    print()
    print(result.summary())
    print()
    obs_name = observables[0]
    print(f"Observable decomposition preview for '{obs_name}':")
    print("-" * 40)
    preview = result.observable_contributions.xs(obs_name, axis=1, level="observable")
    print(preview.head(10).to_string(float_format=lambda x: f"{x:.6f}"))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.observable_contributions.to_csv(out_path)
        print(f"\nObservable contributions saved to: {out_path}")

    return 0


def cmd_estimation(args: Namespace) -> int:
    """Estimation command (MLE/MAP/MCMC)."""
    import pandas as pd

    from dsgekit.estimation import estimate_map, estimate_mcmc, estimate_mle

    if args.max_iter < 1:
        print("Error: --max-iter must be >= 1", file=sys.stderr)
        return 1
    if args.draws < 1:
        print("Error: --draws must be >= 1", file=sys.stderr)
        return 1
    if args.burn_in < 0 or args.burn_in >= args.draws:
        print("Error: --burn-in must satisfy 0 <= burn_in < draws", file=sys.stderr)
        return 1
    if args.thin < 1:
        print("Error: --thin must be >= 1", file=sys.stderr)
        return 1
    if args.proposal_scale <= 0.0:
        print("Error: --proposal-scale must be > 0", file=sys.stderr)
        return 1
    if args.prior_weight <= 0.0:
        print("Error: --prior-weight must be > 0", file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}")
    model, cal, ss = _load_model(args.model)
    observables = _resolve_observables(model, args.observables)
    param_names = _resolve_param_names(args.params)

    print("Solving model...")
    solution = _solve_model(model, cal, ss)
    data, data_label = _load_or_simulate_observations(args, solution, cal, observables)
    demean = not args.no_demean

    method = args.method
    print(f"Running {method.upper()} estimation ({data_label})...")

    try:
        if method == "mle":
            result = estimate_mle(
                model,
                ss,
                cal,
                data,
                observables=observables,
                param_names=param_names,
                bounds=None,
                demean=demean,
                options={"maxiter": args.max_iter},
                compute_se=args.compute_se,
            )
        elif method == "map":
            result = estimate_map(
                model,
                ss,
                cal,
                data,
                observables=observables,
                param_names=param_names,
                bounds=None,
                demean=demean,
                options={"maxiter": args.max_iter},
                compute_se=args.compute_se,
                prior_weight=args.prior_weight,
                require_priors=not args.allow_missing_priors,
            )
        else:  # method == "mcmc"
            result = estimate_mcmc(
                model,
                ss,
                cal,
                data,
                observables=observables,
                param_names=param_names,
                bounds=None,
                demean=demean,
                prior_weight=args.prior_weight,
                require_priors=not args.allow_missing_priors,
                n_draws=args.draws,
                burn_in=args.burn_in,
                thin=args.thin,
                proposal_scale=args.proposal_scale,
                seed=args.seed,
            )
    except Exception as e:
        print(f"Error during estimation: {e}", file=sys.stderr)
        return 1

    print()
    print(result.summary())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if method == "mcmc":
            output_df = pd.DataFrame(result.samples, columns=result.param_names)
        else:
            output_df = pd.DataFrame(
                {
                    "parameter": list(result.parameters.keys()),
                    "estimate": list(result.parameters.values()),
                }
            )
        output_df.to_csv(out_path, index=False)
        print(f"\nEstimation output saved to: {out_path}")

    return 0


def cmd_osr(args: Namespace) -> int:
    """OSR command."""
    from dsgekit.solvers import osr_grid_search

    if args.top < 1:
        print("Error: --top must be >= 1", file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}")
    model, cal, ss = _load_model(args.model)

    try:
        parameter_grid = _parse_grid_specs(args.grid)
        loss_weights = _parse_weight_specs(args.loss)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if not loss_weights:
        if not model.variable_names:
            print("Error: model has no variables for default --loss", file=sys.stderr)
            return 1
        loss_weights = {model.variable_names[0]: 1.0}

    n_candidates = 1
    for values in parameter_grid.values():
        n_candidates *= len(values)

    print(
        "Running OSR grid search "
        f"({n_candidates} candidates; require_determinate={not args.allow_indeterminate})..."
    )
    result = osr_grid_search(
        model,
        ss,
        cal,
        parameter_grid=parameter_grid,
        loss_weights=loss_weights,
        require_determinate=not args.allow_indeterminate,
    )

    print()
    print(result.summary())
    print()

    ranked = result.to_frame(include_failed=args.include_failed)
    if ranked.empty:
        print("No candidates available after filtering.")
    else:
        print("Top OSR candidates:")
        print("-" * 40)
        preview = ranked.head(args.top)
        print(preview.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(out_path, index=False)
        print(f"\nOSR candidates saved to: {out_path}")

    return 0 if result.best is not None else 1


def cmd_baseline_regression(args: Namespace) -> int:
    """Run batch baseline regression checks."""
    import json

    from dsgekit.diagnostics import (
        expand_baseline_sources,
        run_linear_regression_suite,
    )

    if args.periods is not None and args.periods < 1:
        print("Error: --periods must be >= 1", file=sys.stderr)
        return 1

    baseline_sources = expand_baseline_sources(args.baselines)
    if not baseline_sources:
        print(
            "Error: no baseline files found. Check --baselines patterns.",
            file=sys.stderr,
        )
        return 1

    suite = run_linear_regression_suite(
        baseline_sources=baseline_sources,
        model_source=args.model,
        periods=args.periods,
    )

    print(suite.summary())

    if args.dashboard:
        out_path = Path(args.dashboard)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(suite.dashboard_markdown(), encoding="utf-8")
        print(f"\nDashboard saved to: {out_path}")

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(suite.to_dict(), indent=2),
            encoding="utf-8",
        )
        print(f"JSON report saved to: {out_path}")

    return 0 if suite.passed else 1


def cmd_info(args: Namespace) -> int:
    """Info command."""
    print(f"Loading model: {args.model}")
    model, cal, ss = _load_model(args.model)

    print()
    print("=" * 50)
    print(f"MODEL: {model.name}")
    print("=" * 50)
    print()

    # Variables
    print("Variables:")
    print("-" * 30)
    for var in model.variable_names:
        ss_val = ss.values.get(var, "?")
        print(f"  {var:<15} (ss = {ss_val})")
    print()

    # Shocks
    print("Shocks:")
    print("-" * 30)
    for shock in model.shock_names:
        stderr = cal.shock_stderr.get(shock, 0.0)
        print(f"  {shock:<15} (σ = {stderr})")
    print()

    # Parameters
    print("Parameters:")
    print("-" * 30)
    for param, value in sorted(cal.parameters.items()):
        print(f"  {param:<15} = {value}")
    print()

    # Equations
    print("Equations:")
    print("-" * 30)
    for i, eq in enumerate(model.equations, 1):
        name = eq.name or f"eq_{i}"
        print(f"  [{name}] {eq}")
    print()

    # Lead-lag structure
    print("Timing structure:")
    print("-" * 30)
    ll = model.lead_lag
    var_names = model.variable_names
    predetermined = ll.get_predetermined_vars(var_names)
    forward_looking = ll.get_forward_looking_vars(var_names)
    static = [v for v in var_names if ll.is_static(v)]

    if predetermined:
        print(f"  Predetermined: {', '.join(predetermined)}")
    if forward_looking:
        print(f"  Forward-looking: {', '.join(forward_looking)}")
    if static:
        print(f"  Static: {', '.join(static)}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "run": cmd_run,
        "solve": cmd_solve,
        "irf": cmd_irf,
        "simulate": cmd_simulate,
        "estimation": cmd_estimation,
        "forecast": cmd_forecast,
        "decompose": cmd_decompose,
        "osr": cmd_osr,
        "baseline_regression": cmd_baseline_regression,
        "info": cmd_info,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
