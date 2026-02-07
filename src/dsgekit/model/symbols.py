"""Symbol system for DSGE models.

This module defines the core symbolic elements:
- Variable: endogenous variables with timing (leads/lags)
- Shock: exogenous stochastic shocks
- Parameter: model parameters

Variables can appear with different timings:
- x(-1): lagged (predetermined)
- x or x(0): current
- x(+1) or x(1): lead (forward-looking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class SymbolType(Enum):
    """Type of symbol in the model."""

    ENDOGENOUS = auto()
    EXOGENOUS = auto()  # Shocks
    PARAMETER = auto()


@dataclass(frozen=True, slots=True)
class Symbol:
    """Base class for all symbols.

    Attributes:
        name: Unique identifier for the symbol.
        description: Optional human-readable description.
    """

    name: str
    description: str = ""

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


@dataclass(frozen=True, slots=True)
class Parameter(Symbol):
    """A model parameter (constant during simulation).

    Example:
        beta = Parameter("beta", description="Discount factor")
    """

    pass


@dataclass(frozen=True, slots=True)
class Shock(Symbol):
    """An exogenous stochastic shock.

    Example:
        e_a = Shock("e_a", description="Technology shock")
    """

    pass


@dataclass(frozen=True, slots=True)
class Variable(Symbol):
    """An endogenous variable.

    Variables can appear with different timings in equations.
    This class represents the variable itself, not a specific timing.

    Example:
        y = Variable("y", description="Output")
    """

    pass


@dataclass(frozen=True, slots=True)
class TimedVariable:
    """A variable at a specific time (with lead or lag).

    This represents x(t+k) where k is the timing offset:
    - k = -1: x(-1), lagged/predetermined
    - k = 0: x or x(0), current
    - k = +1: x(+1), lead/forward-looking

    Attributes:
        variable: The underlying Variable.
        timing: Time offset (-1, 0, +1, etc.).
    """

    variable: Variable
    timing: int = 0

    @property
    def name(self) -> str:
        """Variable name."""
        return self.variable.name

    @property
    def is_lagged(self) -> bool:
        """True if this is a lagged variable (timing < 0)."""
        return self.timing < 0

    @property
    def is_current(self) -> bool:
        """True if this is a current-period variable (timing == 0)."""
        return self.timing == 0

    @property
    def is_lead(self) -> bool:
        """True if this is a forward-looking variable (timing > 0)."""
        return self.timing > 0

    def __str__(self) -> str:
        if self.timing == 0:
            return self.name
        elif self.timing > 0:
            return f"{self.name}(+{self.timing})"
        else:
            return f"{self.name}({self.timing})"

    def __repr__(self) -> str:
        return f"TimedVariable({self.name!r}, timing={self.timing})"


@dataclass
class SymbolTable:
    """Container for all symbols in a model.

    Provides lookup and validation of symbol references.
    Ensures uniqueness of symbol names across all types.
    """

    _variables: dict[str, Variable] = field(default_factory=dict)
    _shocks: dict[str, Shock] = field(default_factory=dict)
    _parameters: dict[str, Parameter] = field(default_factory=dict)

    def add_variable(self, var: Variable) -> None:
        """Add an endogenous variable."""
        self._check_unique(var.name, "variable")
        self._variables[var.name] = var

    def add_shock(self, shock: Shock) -> None:
        """Add an exogenous shock."""
        self._check_unique(shock.name, "shock")
        self._shocks[shock.name] = shock

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter."""
        self._check_unique(param.name, "parameter")
        self._parameters[param.name] = param

    def _check_unique(self, name: str, new_type: str) -> None:
        """Check that a symbol name is not already used."""
        from dsgekit.exceptions import DuplicateSymbolError

        if name in self._variables:
            raise DuplicateSymbolError(name, "variable", new_type)
        if name in self._shocks:
            raise DuplicateSymbolError(name, "shock", new_type)
        if name in self._parameters:
            raise DuplicateSymbolError(name, "parameter", new_type)

    def get_variable(self, name: str) -> Variable:
        """Get a variable by name, raising if not found."""
        from dsgekit.exceptions import UndeclaredSymbolError

        if name not in self._variables:
            raise UndeclaredSymbolError(name, context="variables")
        return self._variables[name]

    def get_shock(self, name: str) -> Shock:
        """Get a shock by name, raising if not found."""
        from dsgekit.exceptions import UndeclaredSymbolError

        if name not in self._shocks:
            raise UndeclaredSymbolError(name, context="shocks")
        return self._shocks[name]

    def get_parameter(self, name: str) -> Parameter:
        """Get a parameter by name, raising if not found."""
        from dsgekit.exceptions import UndeclaredSymbolError

        if name not in self._parameters:
            raise UndeclaredSymbolError(name, context="parameters")
        return self._parameters[name]

    def get_symbol(self, name: str) -> Symbol:
        """Get any symbol by name."""
        from dsgekit.exceptions import UndeclaredSymbolError

        if name in self._variables:
            return self._variables[name]
        if name in self._shocks:
            return self._shocks[name]
        if name in self._parameters:
            return self._parameters[name]
        raise UndeclaredSymbolError(name)

    def get_symbol_type(self, name: str) -> SymbolType:
        """Get the type of a symbol."""
        from dsgekit.exceptions import UndeclaredSymbolError

        if name in self._variables:
            return SymbolType.ENDOGENOUS
        if name in self._shocks:
            return SymbolType.EXOGENOUS
        if name in self._parameters:
            return SymbolType.PARAMETER
        raise UndeclaredSymbolError(name)

    def has_symbol(self, name: str) -> bool:
        """Check if a symbol exists."""
        return (
            name in self._variables
            or name in self._shocks
            or name in self._parameters
        )

    @property
    def variables(self) -> list[Variable]:
        """List of all endogenous variables (ordered by insertion)."""
        return list(self._variables.values())

    @property
    def shocks(self) -> list[Shock]:
        """List of all exogenous shocks (ordered by insertion)."""
        return list(self._shocks.values())

    @property
    def parameters(self) -> list[Parameter]:
        """List of all parameters (ordered by insertion)."""
        return list(self._parameters.values())

    @property
    def variable_names(self) -> list[str]:
        """Names of all endogenous variables."""
        return list(self._variables.keys())

    @property
    def shock_names(self) -> list[str]:
        """Names of all exogenous shocks."""
        return list(self._shocks.keys())

    @property
    def parameter_names(self) -> list[str]:
        """Names of all parameters."""
        return list(self._parameters.keys())

    @property
    def n_variables(self) -> int:
        """Number of endogenous variables."""
        return len(self._variables)

    @property
    def n_shocks(self) -> int:
        """Number of exogenous shocks."""
        return len(self._shocks)

    @property
    def n_parameters(self) -> int:
        """Number of parameters."""
        return len(self._parameters)

    def __iter__(self) -> Iterator[Symbol]:
        """Iterate over all symbols."""
        yield from self._variables.values()
        yield from self._shocks.values()
        yield from self._parameters.values()

    def __len__(self) -> int:
        """Total number of symbols."""
        return len(self._variables) + len(self._shocks) + len(self._parameters)

    def __contains__(self, name: str) -> bool:
        """Check if symbol name exists."""
        return self.has_symbol(name)


def timed(var: Variable, timing: int = 0) -> TimedVariable:
    """Create a timed variable reference.

    Convenience function for creating TimedVariable instances.

    Examples:
        y = Variable("y")
        y_lag = timed(y, -1)   # y(-1)
        y_cur = timed(y, 0)    # y
        y_lead = timed(y, 1)   # y(+1)
    """
    return TimedVariable(variable=var, timing=timing)
