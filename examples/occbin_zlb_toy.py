"""OccBin-lite example on the ZLB toy model.

Run from repository root:
    python examples/occbin_zlb_toy.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dsgekit import load_model
from dsgekit.solvers import solve_occbin_lite


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "tests" / "fixtures" / "models" / "zlb_toy.yaml"
    model, cal, ss = load_model(model_path)

    n_periods = 16
    shocks = np.zeros((n_periods, model.n_shocks), dtype=np.float64)
    e_i_idx = model.shock_names.index("e_i")
    shocks[:5, e_i_idx] = np.array([-0.22, -0.14, -0.09, -0.05, -0.02])

    result = solve_occbin_lite(
        model,
        ss,
        cal,
        n_periods=n_periods,
        shocks=shocks,
        switch_equation="effective_rate_floor",
        relaxed_equation="i = i_shadow",
        binding_equation="i = 0.0",
        constraint_var="i_shadow",
        constraint_operator="<=",
        constraint_value=0.0,
        constraint_timing=0,
        tol=1e-10,
        max_iter=60,
        max_regime_iter=40,
    )

    print(result.summary())
    print()
    out = result.path.copy()
    out["binding"] = result.binding_regime.values.astype(int)
    print(out[["i_shadow", "i", "binding"]].to_string())


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    main()
