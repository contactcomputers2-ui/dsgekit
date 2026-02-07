"""Fluent Python API for building DSGE models.

Provides a builder pattern for constructing models programmatically:

    model, cal, ss = (
        ModelBuilder("MyModel")
        .var("y", "Output")
        .var("c", "Consumption")
        .varexo("e", "Technology shock")
        .param("beta", 0.99)
        .param("rho", 0.9)
        .equation("y = rho * y(-1) + e", name="output")
        .equation("c = beta * c(+1)", name="euler")
        .calibrate(beta=0.99, rho=0.9)
        .shock_stderr(e=0.01)
        .initval(y=0, c=1)
        .build()
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dsgekit.model.calibration import Calibration, EstimatedParam
from dsgekit.model.equations import (
    EQUATION_FUNCTIONS,
    KNOWN_FUNCTIONS,
    Equation,
    param,
    shock,
    var,
)
from dsgekit.model.ir import ModelIR
from dsgekit.model.priors import PriorSpec, parse_prior_spec
from dsgekit.model.steady_state import SteadyState
from dsgekit.model.symbols import Parameter, Shock, Variable


@dataclass
class ModelBuilder:
    """Fluent builder for DSGE models.

    Example:
        model, cal, ss = (
            ModelBuilder("AR1")
            .var("y")
            .varexo("e")
            .param("rho", 0.9)
            .equation("y = rho * y(-1) + e")
            .shock_stderr(e=0.01)
            .initval(y=0)
            .build()
        )
    """

    name: str
    _variables: list[tuple[str, str]] = field(default_factory=list)
    _shocks: list[tuple[str, str]] = field(default_factory=list)
    _parameters: list[tuple[str, str]] = field(default_factory=list)
    _equations: list[tuple[str, str]] = field(default_factory=list)
    _param_values: dict[str, float] = field(default_factory=dict)
    _shock_stderr: dict[str, float] = field(default_factory=dict)
    _shock_corr: dict[tuple[str, str], float] = field(default_factory=dict)
    _initval: dict[str, float] = field(default_factory=dict)
    _estimated_params: list[EstimatedParam] = field(default_factory=list)

    def var(self, name: str, description: str = "") -> ModelBuilder:
        """Add an endogenous variable."""
        self._variables.append((name, description))
        return self

    def vars(self, *names: str) -> ModelBuilder:
        """Add multiple endogenous variables."""
        for name in names:
            self._variables.append((name, ""))
        return self

    def varexo(self, name: str, description: str = "") -> ModelBuilder:
        """Add an exogenous shock."""
        self._shocks.append((name, description))
        return self

    def varexos(self, *names: str) -> ModelBuilder:
        """Add multiple exogenous shocks."""
        for name in names:
            self._shocks.append((name, ""))
        return self

    def param(self, name: str, value: float | None = None, description: str = "") -> ModelBuilder:
        """Add a parameter, optionally with its value."""
        self._parameters.append((name, description))
        if value is not None:
            self._param_values[name] = value
        return self

    def params(self, **name_values: float) -> ModelBuilder:
        """Add multiple parameters with values."""
        for name, value in name_values.items():
            self._parameters.append((name, ""))
            self._param_values[name] = value
        return self

    def equation(self, expr: str, name: str = "") -> ModelBuilder:
        """Add a model equation.

        The equation can be in the form:
        - "lhs = rhs" (converted to lhs - rhs = 0)
        - "expression" (treated as expression = 0)

        Timing notation:
        - x(-1): lagged variable
        - x or x(0): current variable
        - x(+1) or x(1): lead variable
        """
        self._equations.append((expr, name))
        return self

    def calibrate(self, **param_values: float) -> ModelBuilder:
        """Set parameter values."""
        self._param_values.update(param_values)
        return self

    def shock_stderr(self, **shock_stderrs: float) -> ModelBuilder:
        """Set shock standard deviations."""
        self._shock_stderr.update(shock_stderrs)
        return self

    def shock_corr(self, shock1: str, shock2: str, corr: float) -> ModelBuilder:
        """Set correlation between two shocks."""
        self._shock_corr[(shock1, shock2)] = corr
        return self

    def initval(self, **values: float) -> ModelBuilder:
        """Set initial/steady state values."""
        self._initval.update(values)
        return self

    def estimate_param(
        self,
        name: str,
        *,
        init: float | None = None,
        lower: float = float("-inf"),
        upper: float = float("inf"),
        prior: PriorSpec | dict[str, float | str] | str | None = None,
        prior_mean: float | None = None,
        prior_std: float | None = None,
    ) -> ModelBuilder:
        """Add an estimable model parameter with optional prior."""
        prior_spec = parse_prior_spec(prior, mean=prior_mean, std=prior_std)
        self._estimated_params.append(
            EstimatedParam(
                entry_type="param",
                name=name,
                init_value=float("nan") if init is None else float(init),
                lower_bound=lower,
                upper_bound=upper,
                prior_shape="" if prior_spec is None else prior_spec.distribution,
                prior_mean=float("nan") if prior_spec is None else prior_spec.mean,
                prior_std=float("nan") if prior_spec is None else prior_spec.std,
            )
        )
        return self

    def estimate_stderr(
        self,
        shock_name: str,
        *,
        init: float | None = None,
        lower: float = float("-inf"),
        upper: float = float("inf"),
        prior: PriorSpec | dict[str, float | str] | str | None = None,
        prior_mean: float | None = None,
        prior_std: float | None = None,
    ) -> ModelBuilder:
        """Add an estimable shock standard deviation with optional prior."""
        prior_spec = parse_prior_spec(prior, mean=prior_mean, std=prior_std)
        self._estimated_params.append(
            EstimatedParam(
                entry_type="stderr",
                name=shock_name,
                init_value=float("nan") if init is None else float(init),
                lower_bound=lower,
                upper_bound=upper,
                prior_shape="" if prior_spec is None else prior_spec.distribution,
                prior_mean=float("nan") if prior_spec is None else prior_spec.mean,
                prior_std=float("nan") if prior_spec is None else prior_spec.std,
            )
        )
        return self

    def estimate_corr(
        self,
        shock1: str,
        shock2: str,
        *,
        init: float | None = None,
        lower: float = -1.0,
        upper: float = 1.0,
        prior: PriorSpec | dict[str, float | str] | str | None = None,
        prior_mean: float | None = None,
        prior_std: float | None = None,
    ) -> ModelBuilder:
        """Add an estimable shock correlation with optional prior."""
        prior_spec = parse_prior_spec(prior, mean=prior_mean, std=prior_std)
        self._estimated_params.append(
            EstimatedParam(
                entry_type="corr",
                name=shock1,
                name2=shock2,
                init_value=float("nan") if init is None else float(init),
                lower_bound=lower,
                upper_bound=upper,
                prior_shape="" if prior_spec is None else prior_spec.distribution,
                prior_mean=float("nan") if prior_spec is None else prior_spec.mean,
                prior_std=float("nan") if prior_spec is None else prior_spec.std,
            )
        )
        return self

    def build(self) -> tuple[ModelIR, Calibration, SteadyState]:
        """Build the model, calibration, and steady state.

        Returns:
            Tuple of (ModelIR, Calibration, SteadyState)

        Raises:
            ParseError: If equations cannot be parsed
            ModelSpecError: If model is invalid
        """
        model = ModelIR(name=self.name)

        # Add variables
        var_objs: dict[str, Variable] = {}
        for name, desc in self._variables:
            var_objs[name] = model.add_variable(name, desc)

        # Add shocks
        shock_objs: dict[str, Shock] = {}
        for name, desc in self._shocks:
            shock_objs[name] = model.add_shock(name, desc)

        # Add parameters
        param_objs: dict[str, Parameter] = {}
        for name, desc in self._parameters:
            param_objs[name] = model.add_parameter(name, desc)

        # Parse and add equations
        var_names = set(var_objs.keys())
        shock_names = set(shock_objs.keys())
        param_names = set(param_objs.keys())

        for i, (eq_str, eq_name) in enumerate(self._equations):
            if not eq_name:
                eq_name = f"eq_{i + 1}"

            expr = self._parse_equation(
                eq_str, var_objs, shock_objs, param_objs,
                var_names, shock_names, param_names
            )
            model.add_equation(Equation(expr, name=eq_name))

        # Validate model
        model.validate()

        # Build calibration
        calibration = Calibration()
        for name, value in self._param_values.items():
            calibration.set_parameter(name, value)
        for name, stderr in self._shock_stderr.items():
            calibration.set_shock_stderr(name, stderr)
        for (s1, s2), corr in self._shock_corr.items():
            calibration.set_shock_corr(s1, s2, corr)
        calibration.estimated_params.extend(self._estimated_params)

        calibration_errors = calibration.validate_estimated_params(model)
        if calibration_errors:
            raise ValueError("; ".join(calibration_errors))

        # Build steady state
        steady_state = SteadyState(values=self._initval.copy())

        return model, calibration, steady_state

    def _parse_equation(
        self,
        eq_str: str,
        var_objs: dict[str, Variable],
        shock_objs: dict[str, Shock],
        param_objs: dict[str, Parameter],
        var_names: set[str],
        shock_names: set[str],
        param_names: set[str],
    ):
        """Parse equation string into Expression."""
        from dsgekit.exceptions import ParseError

        # Handle LHS = RHS format
        if "=" in eq_str:
            parts = eq_str.split("=", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            expr_str = f"({lhs}) - ({rhs})"
        else:
            expr_str = eq_str.strip()

        # Build evaluation namespace
        local_ns: dict[str, object] = {
            "var": var,
            "shock": shock,
            "param": param,
        }
        local_ns.update(EQUATION_FUNCTIONS)

        # Convert variable references with timing
        # y(-1) -> var(y_obj, -1)
        # y(+1) -> var(y_obj, 1)
        # y -> var(y_obj)

        converted = expr_str

        # First protect function names
        functions = KNOWN_FUNCTIONS
        for func in functions:
            converted = re.sub(
                rf"\b{func}\b",
                f"__FUNC_{func}__",
                converted,
                flags=re.IGNORECASE
            )

        # Convert variable timings
        for vname in var_names:
            # var(-1) or var(-2) etc
            converted = re.sub(
                rf"\b{vname}\s*\(\s*(-\d+)\s*\)",
                rf"var(_v_{vname}, \1)",
                converted,
            )
            # var(+1) or var(+2) etc
            converted = re.sub(
                rf"\b{vname}\s*\(\s*\+(\d+)\s*\)",
                rf"var(_v_{vname}, \1)",
                converted,
            )
            # var(1) or var(2) etc (without +)
            converted = re.sub(
                rf"\b{vname}\s*\(\s*(\d+)\s*\)",
                rf"var(_v_{vname}, \1)",
                converted,
            )
            # var alone -> var(obj, 0)
            converted = re.sub(
                rf"(?<![_\w]){vname}(?![_\w\(])",
                rf"var(_v_{vname})",
                converted,
            )

        # Convert shock references
        for sname in shock_names:
            converted = re.sub(
                rf"(?<![_\w]){sname}(?![_\w])",
                rf"shock(_s_{sname})",
                converted,
            )

        # Convert parameter references
        for pname in param_names:
            converted = re.sub(
                rf"(?<![_\w]){pname}(?![_\w])",
                rf"param(_p_{pname})",
                converted,
            )

        # Restore function names
        for func in functions:
            converted = converted.replace(f"__FUNC_{func}__", func)

        # Convert ^ to **
        converted = converted.replace("^", "**")

        # Add objects to namespace
        for vname, vobj in var_objs.items():
            local_ns[f"_v_{vname}"] = vobj
        for sname, sobj in shock_objs.items():
            local_ns[f"_s_{sname}"] = sobj
        for pname, pobj in param_objs.items():
            local_ns[f"_p_{pname}"] = pobj

        try:
            return eval(converted, {"__builtins__": {}}, local_ns)
        except Exception as e:
            raise ParseError(
                f"Failed to parse equation: {eq_str}\n"
                f"Converted to: {converted}\n"
                f"Error: {e}"
            ) from e


def model_builder(name: str) -> ModelBuilder:
    """Create a new model builder.

    Convenience function for creating ModelBuilder instances.

    Example:
        model, cal, ss = (
            model_builder("AR1")
            .var("y")
            .varexo("e")
            .param("rho", 0.9)
            .equation("y = rho * y(-1) + e")
            .shock_stderr(e=0.01)
            .initval(y=0)
            .build()
        )
    """
    return ModelBuilder(name)
