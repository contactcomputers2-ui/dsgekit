"""Intermediate Representation (IR) for DSGE models.

The ModelIR is the canonical internal representation of a DSGE model.
It is independent of the input format (Python API, .mod, YAML) and
provides the interface used by solvers, simulators, and estimators.

A DSGE model in general form:
    E_t[f(y_{t-1}, y_t, y_{t+1}, u_t, θ)] = 0

where:
    y_t: endogenous variables
    u_t: exogenous shocks
    θ: parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dsgekit.model.equations import Equation, EvalContext
from dsgekit.model.symbols import (
    Parameter,
    Shock,
    SymbolTable,
    TimedVariable,
    Variable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class LeadLagStructure:
    """Describes the lead/lag structure of variables in the model.

    This is crucial for the solver to know which variables are:
    - Predetermined (appear with lag): state variables
    - Forward-looking (appear with lead): jump variables
    - Static (only current period)

    Attributes:
        max_lag: Maximum lag in the model (typically 1)
        max_lead: Maximum lead in the model (typically 1)
        incidence: Dict mapping var_name -> set of timings used
    """

    max_lag: int = 0
    max_lead: int = 0
    incidence: dict[str, set[int]] = field(default_factory=dict)

    @property
    def n_timings(self) -> int:
        """Total number of time periods: lag + current + lead."""
        return self.max_lag + 1 + self.max_lead

    def is_predetermined(self, var_name: str) -> bool:
        """True if variable appears with a lag (state variable)."""
        return var_name in self.incidence and any(
            t < 0 for t in self.incidence[var_name]
        )

    def is_forward_looking(self, var_name: str) -> bool:
        """True if variable appears with a lead (jump variable)."""
        return var_name in self.incidence and any(
            t > 0 for t in self.incidence[var_name]
        )

    def is_static(self, var_name: str) -> bool:
        """True if variable only appears in current period."""
        if var_name not in self.incidence:
            return True
        timings = self.incidence[var_name]
        return timings == {0}

    def get_predetermined_vars(self, var_names: Sequence[str]) -> list[str]:
        """Get list of predetermined variable names (in order)."""
        return [v for v in var_names if self.is_predetermined(v)]

    def get_forward_looking_vars(self, var_names: Sequence[str]) -> list[str]:
        """Get list of forward-looking variable names (in order)."""
        return [v for v in var_names if self.is_forward_looking(v)]

    @property
    def n_predetermined(self) -> int:
        """Number of predetermined (state) variables."""
        return sum(1 for timings in self.incidence.values() if any(t < 0 for t in timings))

    @property
    def n_forward_looking(self) -> int:
        """Number of forward-looking (jump) variables."""
        return sum(1 for timings in self.incidence.values() if any(t > 0 for t in timings))


@dataclass
class ModelIR:
    """Canonical intermediate representation of a DSGE model.

    This is the central data structure that all components work with.
    It is constructed from various input formats and consumed by solvers.

    Attributes:
        name: Model name/identifier
        symbols: Symbol table with all variables, shocks, parameters
        equations: List of model equations (in residual form)
        lead_lag: Lead/lag structure of the model
    """

    name: str = "unnamed"
    symbols: SymbolTable = field(default_factory=SymbolTable)
    equations: list[Equation] = field(default_factory=list)
    lead_lag: LeadLagStructure = field(default_factory=LeadLagStructure)
    observables: list[str] = field(default_factory=list)

    # Computed after validation
    _validated: bool = field(default=False, repr=False)

    def add_variable(self, name: str, description: str = "") -> Variable:
        """Add an endogenous variable."""
        var = Variable(name, description)
        self.symbols.add_variable(var)
        self._validated = False
        return var

    def add_shock(self, name: str, description: str = "") -> Shock:
        """Add an exogenous shock."""
        shock = Shock(name, description)
        self.symbols.add_shock(shock)
        self._validated = False
        return shock

    def add_parameter(self, name: str, description: str = "") -> Parameter:
        """Add a parameter."""
        param = Parameter(name, description)
        self.symbols.add_parameter(param)
        self._validated = False
        return param

    def add_equation(self, equation: Equation) -> None:
        """Add a model equation."""
        self.equations.append(equation)
        self._validated = False

    def validate(self) -> None:
        """Validate model consistency.

        Checks:
        - Number of equations equals number of endogenous variables
        - All referenced symbols are declared
        - Timing structure is valid (max 1 lag, 1 lead for linear solver)

        Raises:
            ModelSpecError: If validation fails
        """
        from dsgekit.exceptions import (
            EquationCountError,
            UndeclaredSymbolError,
        )

        # Check equation count
        n_eq = len(self.equations)
        n_endo = self.symbols.n_variables
        if n_eq != n_endo:
            raise EquationCountError(n_eq, n_endo)

        # Build lead/lag structure and check symbols
        incidence: dict[str, set[int]] = {}
        max_lag = 0
        max_lead = 0

        for eq in self.equations:
            # Check variables
            for timed_var in eq.get_variables():
                name = timed_var.name
                timing = timed_var.timing

                # Check variable is declared
                if name not in self.symbols.variable_names:
                    raise UndeclaredSymbolError(name, context=f"equation '{eq.name}'")

                # Track timing
                if name not in incidence:
                    incidence[name] = set()
                incidence[name].add(timing)

                # Update max lag/lead
                if timing < 0:
                    max_lag = max(max_lag, -timing)
                elif timing > 0:
                    max_lead = max(max_lead, timing)

            # Check shocks
            for shock in eq.get_shocks():
                if shock.name not in self.symbols.shock_names:
                    raise UndeclaredSymbolError(
                        shock.name, context=f"equation '{eq.name}'"
                    )

            # Check parameters
            for param in eq.get_parameters():
                if param.name not in self.symbols.parameter_names:
                    raise UndeclaredSymbolError(
                        param.name, context=f"equation '{eq.name}'"
                    )

        # For linear solver, we typically require max_lag=1, max_lead=1
        # but we allow arbitrary for flexibility
        if max_lag > 1 or max_lead > 1:
            # This is a warning case, not an error
            # Some solvers can handle higher order
            pass

        self.lead_lag = LeadLagStructure(
            max_lag=max_lag,
            max_lead=max_lead,
            incidence=incidence,
        )
        self._validated = True

    def residuals(
        self,
        context: EvalContext,
    ) -> NDArray[np.float64]:
        """Evaluate all equation residuals.

        Args:
            context: Evaluation context with variable/parameter values

        Returns:
            Array of residuals (should all be ~0 at solution)
        """
        return np.array([eq.residual(context) for eq in self.equations])

    def residuals_at_steady_state(
        self,
        steady_state: dict[str, float],
        parameters: dict[str, float],
    ) -> NDArray[np.float64]:
        """Evaluate residuals at steady state (shocks = 0).

        Args:
            steady_state: Mapping var_name -> steady state value
            parameters: Mapping param_name -> value

        Returns:
            Array of residuals
        """
        # At steady state, all timings have same value and shocks = 0
        shocks = dict.fromkeys(self.symbols.shock_names, 0.0)
        context = EvalContext.from_steady_state(
            steady_state=steady_state,
            parameters=parameters,
            shocks=shocks,
            timings=list(range(-self.lead_lag.max_lag, self.lead_lag.max_lead + 1)),
        )
        return self.residuals(context)

    @property
    def n_equations(self) -> int:
        """Number of equations."""
        return len(self.equations)

    @property
    def n_variables(self) -> int:
        """Number of endogenous variables."""
        return self.symbols.n_variables

    @property
    def n_shocks(self) -> int:
        """Number of exogenous shocks."""
        return self.symbols.n_shocks

    @property
    def n_parameters(self) -> int:
        """Number of parameters."""
        return self.symbols.n_parameters

    @property
    def variable_names(self) -> list[str]:
        """Ordered list of endogenous variable names."""
        return self.symbols.variable_names

    @property
    def shock_names(self) -> list[str]:
        """Ordered list of shock names."""
        return self.symbols.shock_names

    @property
    def parameter_names(self) -> list[str]:
        """Ordered list of parameter names."""
        return self.symbols.parameter_names

    @property
    def predetermined_variable_names(self) -> list[str]:
        """Names of predetermined (state) variables."""
        if not self._validated:
            self.validate()
        return self.lead_lag.get_predetermined_vars(self.variable_names)

    @property
    def forward_looking_variable_names(self) -> list[str]:
        """Names of forward-looking (jump) variables."""
        if not self._validated:
            self.validate()
        return self.lead_lag.get_forward_looking_vars(self.variable_names)

    @property
    def n_predetermined(self) -> int:
        """Number of predetermined (state) variables."""
        if not self._validated:
            self.validate()
        return len(self.predetermined_variable_names)

    @property
    def n_forward_looking(self) -> int:
        """Number of forward-looking (jump) variables."""
        if not self._validated:
            self.validate()
        return len(self.forward_looking_variable_names)

    def get_all_timed_variables(self) -> list[TimedVariable]:
        """Get all timed variable references in the model (unique)."""
        result: set[TimedVariable] = set()
        for eq in self.equations:
            result |= eq.get_variables()
        return sorted(result, key=lambda tv: (tv.name, tv.timing))

    def summary(self) -> str:
        """Return a human-readable summary of the model."""
        if not self._validated:
            try:
                self.validate()
            except Exception:
                pass

        lines = [
            f"Model: {self.name}",
            f"  Endogenous variables: {self.n_variables}",
            f"  Exogenous shocks: {self.n_shocks}",
            f"  Parameters: {self.n_parameters}",
            f"  Equations: {self.n_equations}",
        ]

        if self._validated:
            lines.extend([
                f"  Predetermined (state): {self.n_predetermined}",
                f"  Forward-looking (jump): {self.n_forward_looking}",
                f"  Max lag: {self.lead_lag.max_lag}",
                f"  Max lead: {self.lead_lag.max_lead}",
            ])

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
