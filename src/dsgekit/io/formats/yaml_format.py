"""YAML format for DSGE models.

A simple, readable format for specifying DSGE models:

```yaml
name: AR1

variables:
  - y: Output

shocks:
  - e: Technology shock

parameters:
  rho: 0.9

equations:
  - name: ar1
    expr: y = rho * y(-1) + e

steady_state:
  y: 0

shocks_config:
  e:
    stderr: 0.01
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.model.calibration import Calibration, EstimatedParam
from dsgekit.model.ir import ModelIR
from dsgekit.model.steady_state import SteadyState


def _parse_yaml_content(data: dict[str, Any]) -> tuple[ModelIR, Calibration, SteadyState]:
    """Parse YAML dict into model components."""
    from dsgekit.exceptions import ParseError

    name = data.get("name", "model")
    builder = ModelBuilder(name)

    # Parse variables
    variables = data.get("variables", [])
    for item in variables:
        if isinstance(item, str):
            builder.var(item)
        elif isinstance(item, dict):
            for vname, desc in item.items():
                builder.var(vname, desc or "")
        else:
            raise ParseError(f"Invalid variable specification: {item}")

    # Parse shocks
    shocks = data.get("shocks", [])
    for item in shocks:
        if isinstance(item, str):
            builder.varexo(item)
        elif isinstance(item, dict):
            for sname, desc in item.items():
                builder.varexo(sname, desc or "")
        else:
            raise ParseError(f"Invalid shock specification: {item}")

    # Parse parameters
    parameters = data.get("parameters", {})
    if isinstance(parameters, dict):
        for pname, value in parameters.items():
            if isinstance(value, (int, float)):
                builder.param(pname, float(value))
            else:
                builder.param(pname, description=str(value) if value else "")
    elif isinstance(parameters, list):
        for item in parameters:
            if isinstance(item, str):
                builder.param(item)
            elif isinstance(item, dict):
                for pname, value in item.items():
                    if isinstance(value, (int, float)):
                        builder.param(pname, float(value))
                    else:
                        builder.param(pname)

    # Parse equations
    equations = data.get("equations", [])
    for item in equations:
        if isinstance(item, str):
            builder.equation(item)
        elif isinstance(item, dict):
            expr = item.get("expr", item.get("equation", ""))
            name = item.get("name", "")
            if expr:
                builder.equation(expr, name=name)
        else:
            raise ParseError(f"Invalid equation specification: {item}")

    # Parse steady state / initval
    steady_state = data.get("steady_state", data.get("initval", {}))
    if isinstance(steady_state, dict):
        for vname, value in steady_state.items():
            builder.initval(**{vname: float(value)})

    # Parse shock configuration
    shocks_config = data.get("shocks_config", data.get("shock_config", {}))
    if isinstance(shocks_config, dict):
        for sname, config in shocks_config.items():
            if isinstance(config, dict):
                stderr = config.get("stderr", config.get("std", None))
                if stderr is not None:
                    builder.shock_stderr(**{sname: float(stderr)})
            elif isinstance(config, (int, float)):
                # Direct stderr value
                builder.shock_stderr(**{sname: float(config)})

    # Parse calibration (alternative to inline parameter values)
    calibration_data = data.get("calibration", {})
    if isinstance(calibration_data, dict):
        for pname, value in calibration_data.items():
            if isinstance(value, (int, float)):
                builder._param_values[pname] = float(value)

    # Collect new-block data before build
    obs_data = data.get("observables", [])
    endval_data = data.get("endval", {})
    histval_data = data.get("histval", {})
    ssm_data = data.get("steady_state_model", {})
    ep_data = data.get("estimated_params", [])

    model, calibration, ss = builder.build()

    # Set observables
    if obs_data:
        model.observables = list(obs_data) if isinstance(obs_data, list) else [obs_data]

    # Set endval
    if isinstance(endval_data, dict):
        for vname, value in endval_data.items():
            ss.endval[vname] = float(value)

    # Set histval
    if isinstance(histval_data, dict):
        for vname, timings in histval_data.items():
            if isinstance(timings, dict):
                for timing, value in timings.items():
                    ss.histval[(vname, int(timing))] = float(value)

    # Set steady_state_model
    if isinstance(ssm_data, dict):
        ss.analytical_equations = {k: str(v) for k, v in ssm_data.items()}

    # Set estimated_params
    if isinstance(ep_data, list):
        for item in ep_data:
            if not isinstance(item, dict):
                raise ParseError(f"Invalid estimated_params entry: {item!r}")
            try:
                calibration.estimated_params.append(EstimatedParam.from_dict(item))
            except Exception as exc:
                raise ParseError(f"Invalid estimated_params entry: {item!r}") from exc

    ep_errors = calibration.validate_estimated_params(model)
    if ep_errors:
        raise ParseError("; ".join(ep_errors))

    return model, calibration, ss


def load_yaml(path: str | Path) -> tuple[ModelIR, Calibration, SteadyState]:
    """Load a DSGE model from a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Tuple of (ModelIR, Calibration, SteadyState)
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML format. "
            "Install with: pip install pyyaml"
        ) from exc

    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    return _parse_yaml_content(data)


def yaml_to_ir(content: str, name: str = "model") -> tuple[ModelIR, Calibration, SteadyState]:
    """Parse YAML string into model components.

    Args:
        content: YAML content as string
        name: Default model name if not in YAML

    Returns:
        Tuple of (ModelIR, Calibration, SteadyState)
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML format. "
            "Install with: pip install pyyaml"
        ) from exc

    data = yaml.safe_load(content)
    if "name" not in data:
        data["name"] = name

    return _parse_yaml_content(data)


def model_to_yaml(
    model: ModelIR,
    calibration: Calibration,
    steady_state: SteadyState,
) -> str:
    """Export model to YAML format.

    Args:
        model: ModelIR instance
        calibration: Calibration instance
        steady_state: SteadyState instance

    Returns:
        YAML string representation
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML format. "
            "Install with: pip install pyyaml"
        ) from exc

    data: dict[str, Any] = {"name": model.name}

    # Variables
    if model.variable_names:
        data["variables"] = model.variable_names

    # Shocks
    if model.shock_names:
        data["shocks"] = model.shock_names

    # Parameters with values
    if calibration.parameters:
        data["parameters"] = calibration.parameters.copy()

    # Equations (simplified - just names, can't reverse-engineer expressions)
    if model.equations:
        data["equations"] = [
            {"name": eq.name} for eq in model.equations if eq.name
        ]

    # Steady state
    if steady_state.values:
        data["steady_state"] = steady_state.values.copy()

    # Shock config
    if calibration.shock_stderr:
        data["shocks_config"] = {
            name: {"stderr": stderr}
            for name, stderr in calibration.shock_stderr.items()
        }

    # Observables
    if model.observables:
        data["observables"] = list(model.observables)

    # Endval
    if steady_state.endval:
        data["endval"] = steady_state.endval.copy()

    # Histval
    if steady_state.histval:
        hv: dict[str, dict[int, float]] = {}
        for (vname, timing), value in steady_state.histval.items():
            hv.setdefault(vname, {})[timing] = value
        data["histval"] = hv

    # Steady state model
    if steady_state.analytical_equations:
        data["steady_state_model"] = dict(steady_state.analytical_equations)

    # Estimated params + priors
    if calibration.estimated_params:
        data["estimated_params"] = [ep.to_dict() for ep in calibration.estimated_params]

    return yaml.dump(data, default_flow_style=False, sort_keys=False)
