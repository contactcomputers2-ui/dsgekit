# dsgekit

A Python toolkit for Dynamic Stochastic General Equilibrium (DSGE) modeling.

## Status

- Version: `0.1.0.dev0` (alpha pre-release)
- API stability: evolving
- Minimum Python version: `3.13` (from `pyproject.toml`)

## What Is Included

- Model loading from `.mod`, YAML, and Python dict/API sources
- Linear workflow: linearization, QZ/BK solving, IRFs, simulation, moments, FEVD
- State-space pipeline: Kalman filter, RTS smoother, forecasting, historical decomposition
- Estimation: MLE, MAP, MCMC (with diagnostics), marginal data density
- Nonlinear solvers: first-, second-, and third-order perturbation (with pruning)
- Deterministic workflows: perfect foresight and one-constraint `occbin`-lite
- Policy and welfare tooling: OSR sweeps, Ramsey/discretionary LQ, welfare comparisons
- Baseline regression harness and CI workflow for compatibility checks

## Installation

### From source (recommended at this stage)

```bash
git clone https://github.com/gustavovargas/dsgekit.git
cd dsgekit
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Optional extras

```bash
pip install -e ".[plot]"   # plotting
pip install -e ".[sym]"    # symbolic derivatives
pip install -e ".[speed]"  # numba acceleration
pip install -e ".[all]"    # all optional extras
```

## Quick Start

```python
from dsgekit import load_model
from dsgekit.simulate import irf
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize

model, cal, ss = load_model("tests/fixtures/models/ar1.yaml")
solution = solve_linear(linearize(model, ss, cal))
responses = irf(solution, "e", periods=20)

print(responses.data.head())
```

## CLI Overview

```bash
dsgekit info model.mod
dsgekit solve model.mod
dsgekit irf model.mod -s e -p 20
dsgekit simulate model.mod -n 200 --seed 42
dsgekit estimation model.mod --method mle --params rho
dsgekit forecast model.mod -p 12
dsgekit decompose model.mod
dsgekit osr model.mod --grid phi_pi=1.0:3.0:9 --loss pi=1.0 --loss y=0.5
dsgekit baseline_regression --baselines tests/fixtures/baselines
```

## Documentation Map

- `docs/quickstart.md`
- `docs/formats.md`
- `docs/first_order_perturbation.md`
- `docs/second_order_perturbation.md`
- `docs/third_order_perturbation.md`
- `docs/perfect_foresight.md`
- `docs/occbin_lite.md`
- `docs/forecast_decomposition.md`
- `docs/osr.md`
- `docs/ramsey_lq.md`
- `docs/discretion_lq.md`
- `docs/welfare.md`
- `docs/performance.md`
- `docs/mod_workflow_migration.md`

Full index: `docs/index.md`

## Examples

```bash
python examples/ar1_pipeline.py
python examples/nk_forward_looking.py
python examples/nonlinear_girf.py
python examples/occbin_zlb_toy.py
python examples/reference_model_zoo.py
```

## Contributing and Security

- Contribution guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`

## Legal

- `dsgekit` is an independent project and is not affiliated with third-party tool maintainers.
- Implementation provenance and attribution policy: `docs/legal_and_attribution.md`.
- Naming policy for third-party references: `docs/brand_naming_policy.md`.
- Dependency notices: `THIRD_PARTY_NOTICES.md`.

## License

MIT
