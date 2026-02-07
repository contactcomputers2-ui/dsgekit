"""Equation representation and evaluation for DSGE models.

This module provides:
- Expression: AST nodes for mathematical expressions
- Equation: A single model equation (LHS = RHS, stored as LHS - RHS = 0)
- Evaluation of expressions given variable/parameter values
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsgekit.model.symbols import Parameter, Shock, TimedVariable, Variable


# =============================================================================
# Expression AST Nodes
# =============================================================================


class Expression(ABC):
    """Abstract base class for expression nodes."""

    @abstractmethod
    def evaluate(self, context: EvalContext) -> float:
        """Evaluate the expression given variable/parameter values."""
        pass

    @abstractmethod
    def get_variables(self) -> set[TimedVariable]:
        """Get all timed variables referenced in this expression."""
        pass

    @abstractmethod
    def get_shocks(self) -> set[Shock]:
        """Get all shocks referenced in this expression."""
        pass

    @abstractmethod
    def get_parameters(self) -> set[Parameter]:
        """Get all parameters referenced in this expression."""
        pass

    def __add__(self, other: Expression | float | int) -> Expression:
        return BinaryOp("+", self, _to_expr(other))

    def __radd__(self, other: float | int) -> Expression:
        return BinaryOp("+", _to_expr(other), self)

    def __sub__(self, other: Expression | float | int) -> Expression:
        return BinaryOp("-", self, _to_expr(other))

    def __rsub__(self, other: float | int) -> Expression:
        return BinaryOp("-", _to_expr(other), self)

    def __mul__(self, other: Expression | float | int) -> Expression:
        return BinaryOp("*", self, _to_expr(other))

    def __rmul__(self, other: float | int) -> Expression:
        return BinaryOp("*", _to_expr(other), self)

    def __truediv__(self, other: Expression | float | int) -> Expression:
        return BinaryOp("/", self, _to_expr(other))

    def __rtruediv__(self, other: float | int) -> Expression:
        return BinaryOp("/", _to_expr(other), self)

    def __pow__(self, other: Expression | float | int) -> Expression:
        return BinaryOp("^", self, _to_expr(other))

    def __rpow__(self, other: float | int) -> Expression:
        return BinaryOp("^", _to_expr(other), self)

    def __neg__(self) -> Expression:
        return UnaryOp("-", self)

    def __pos__(self) -> Expression:
        return self


def _to_expr(value: Expression | float | int) -> Expression:
    """Convert a value to an Expression if needed."""
    if isinstance(value, Expression):
        return value
    return Constant(float(value))


@dataclass(frozen=True, slots=True)
class Constant(Expression):
    """A numeric constant."""

    value: float

    def evaluate(self, context: EvalContext) -> float:
        return self.value

    def get_variables(self) -> set[TimedVariable]:
        return set()

    def get_shocks(self) -> set[Shock]:
        return set()

    def get_parameters(self) -> set[Parameter]:
        return set()

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class VariableRef(Expression):
    """Reference to a timed variable."""

    timed_var: TimedVariable

    def evaluate(self, context: EvalContext) -> float:
        return context.get_variable(self.timed_var)

    def get_variables(self) -> set[TimedVariable]:
        return {self.timed_var}

    def get_shocks(self) -> set[Shock]:
        return set()

    def get_parameters(self) -> set[Parameter]:
        return set()

    def __str__(self) -> str:
        return str(self.timed_var)


@dataclass(frozen=True, slots=True)
class ShockRef(Expression):
    """Reference to a shock."""

    shock: Shock

    def evaluate(self, context: EvalContext) -> float:
        return context.get_shock(self.shock)

    def get_variables(self) -> set[TimedVariable]:
        return set()

    def get_shocks(self) -> set[Shock]:
        return {self.shock}

    def get_parameters(self) -> set[Parameter]:
        return set()

    def __str__(self) -> str:
        return str(self.shock)


@dataclass(frozen=True, slots=True)
class ParameterRef(Expression):
    """Reference to a parameter."""

    param: Parameter

    def evaluate(self, context: EvalContext) -> float:
        return context.get_parameter(self.param)

    def get_variables(self) -> set[TimedVariable]:
        return set()

    def get_shocks(self) -> set[Shock]:
        return set()

    def get_parameters(self) -> set[Parameter]:
        return {self.param}

    def __str__(self) -> str:
        return str(self.param)


@dataclass(frozen=True, slots=True)
class BinaryOp(Expression):
    """Binary operation (e.g., +, -, *, /, ^)."""

    op: str
    left: Expression
    right: Expression

    _OP_FUNCS: dict[str, Callable[[float, float], float]] = field(
        default_factory=lambda: {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "^": lambda a, b: a**b,
        },
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.op not in self._OP_FUNCS:
            raise ValueError(f"Unknown binary operator: {self.op}")

    def evaluate(self, context: EvalContext) -> float:
        left_val = self.left.evaluate(context)
        right_val = self.right.evaluate(context)
        return self._OP_FUNCS[self.op](left_val, right_val)

    def get_variables(self) -> set[TimedVariable]:
        return self.left.get_variables() | self.right.get_variables()

    def get_shocks(self) -> set[Shock]:
        return self.left.get_shocks() | self.right.get_shocks()

    def get_parameters(self) -> set[Parameter]:
        return self.left.get_parameters() | self.right.get_parameters()

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True, slots=True)
class UnaryOp(Expression):
    """Unary operation (e.g., negation)."""

    op: str
    operand: Expression

    _OP_FUNCS: dict[str, Callable[[float], float]] = field(
        default_factory=lambda: {
            "-": lambda x: -x,
            "+": lambda x: x,
        },
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.op not in self._OP_FUNCS:
            raise ValueError(f"Unknown unary operator: {self.op}")

    def evaluate(self, context: EvalContext) -> float:
        val = self.operand.evaluate(context)
        return self._OP_FUNCS[self.op](val)

    def get_variables(self) -> set[TimedVariable]:
        return self.operand.get_variables()

    def get_shocks(self) -> set[Shock]:
        return self.operand.get_shocks()

    def get_parameters(self) -> set[Parameter]:
        return self.operand.get_parameters()

    def __str__(self) -> str:
        return f"({self.op}{self.operand})"


@dataclass(frozen=True, slots=True)
class FunctionCall(Expression):
    """Built-in function call (log, exp, sqrt, etc.)."""

    name: str
    args: tuple[Expression, ...]

    _FUNCS: dict[str, Callable[..., float]] = field(
        default_factory=lambda: {
            "log": math.log,
            "ln": math.log,
            "exp": math.exp,
            "sqrt": math.sqrt,
            "abs": abs,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pow": pow,
            "min": min,
            "max": max,
        },
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.name not in self._FUNCS:
            raise ValueError(f"Unknown function: {self.name}")

    def evaluate(self, context: EvalContext) -> float:
        arg_vals = [arg.evaluate(context) for arg in self.args]
        return self._FUNCS[self.name](*arg_vals)

    def get_variables(self) -> set[TimedVariable]:
        result: set[TimedVariable] = set()
        for arg in self.args:
            result |= arg.get_variables()
        return result

    def get_shocks(self) -> set[Shock]:
        result: set[Shock] = set()
        for arg in self.args:
            result |= arg.get_shocks()
        return result

    def get_parameters(self) -> set[Parameter]:
        result: set[Parameter] = set()
        for arg in self.args:
            result |= arg.get_parameters()
        return result

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"


# =============================================================================
# Evaluation Context
# =============================================================================


@dataclass
class EvalContext:
    """Context for evaluating expressions.

    Holds current values of variables, shocks, and parameters.
    """

    variables: dict[str, dict[int, float]] = field(default_factory=dict)
    """Mapping: var_name -> {timing -> value}"""

    shocks: dict[str, float] = field(default_factory=dict)
    """Mapping: shock_name -> value"""

    parameters: dict[str, float] = field(default_factory=dict)
    """Mapping: param_name -> value"""

    def get_variable(self, timed_var: TimedVariable) -> float:
        """Get value of a timed variable."""
        name = timed_var.name
        timing = timed_var.timing
        if name not in self.variables:
            raise KeyError(f"Variable '{name}' not in context")
        if timing not in self.variables[name]:
            raise KeyError(f"Variable '{name}' timing {timing} not in context")
        return self.variables[name][timing]

    def get_shock(self, shock: Shock) -> float:
        """Get value of a shock."""
        if shock.name not in self.shocks:
            raise KeyError(f"Shock '{shock.name}' not in context")
        return self.shocks[shock.name]

    def get_parameter(self, param: Parameter) -> float:
        """Get value of a parameter."""
        if param.name not in self.parameters:
            raise KeyError(f"Parameter '{param.name}' not in context")
        return self.parameters[param.name]

    def set_variable(self, name: str, timing: int, value: float) -> None:
        """Set value of a timed variable."""
        if name not in self.variables:
            self.variables[name] = {}
        self.variables[name][timing] = value

    def set_shock(self, name: str, value: float) -> None:
        """Set value of a shock."""
        self.shocks[name] = value

    def set_parameter(self, name: str, value: float) -> None:
        """Set value of a parameter."""
        self.parameters[name] = value

    @classmethod
    def from_steady_state(
        cls,
        steady_state: dict[str, float],
        parameters: dict[str, float],
        shocks: dict[str, float] | None = None,
        timings: list[int] | None = None,
    ) -> EvalContext:
        """Create context from steady state values.

        In steady state, variables have the same value at all timings.

        Args:
            steady_state: Mapping var_name -> steady state value
            parameters: Mapping param_name -> value
            shocks: Mapping shock_name -> value (default 0)
            timings: List of timings to populate (default [-1, 0, 1])
        """
        if timings is None:
            timings = [-1, 0, 1]
        if shocks is None:
            shocks = {}

        ctx = cls(parameters=parameters.copy(), shocks=shocks.copy())

        for name, value in steady_state.items():
            for t in timings:
                ctx.set_variable(name, t, value)

        return ctx


# =============================================================================
# Equation
# =============================================================================


@dataclass
class Equation:
    """A single model equation.

    Equations are stored in residual form: expression = 0
    where expression = LHS - RHS from the original equation.

    Attributes:
        expression: The residual expression (should equal 0).
        name: Optional name/identifier for the equation.
    """

    expression: Expression
    name: str = ""

    def residual(self, context: EvalContext) -> float:
        """Compute the residual (should be 0 if equation holds)."""
        return self.expression.evaluate(context)

    def get_variables(self) -> set[TimedVariable]:
        """Get all timed variables in this equation."""
        return self.expression.get_variables()

    def get_shocks(self) -> set[Shock]:
        """Get all shocks in this equation."""
        return self.expression.get_shocks()

    def get_parameters(self) -> set[Parameter]:
        """Get all parameters in this equation."""
        return self.expression.get_parameters()

    def __str__(self) -> str:
        name_str = f"[{self.name}] " if self.name else ""
        return f"{name_str}{self.expression} = 0"


# =============================================================================
# Helper functions for building expressions
# =============================================================================


def log(x: Expression | float) -> Expression:
    """Natural logarithm."""
    return FunctionCall("log", (_to_expr(x),))


def exp(x: Expression | float) -> Expression:
    """Exponential."""
    return FunctionCall("exp", (_to_expr(x),))


def sqrt(x: Expression | float) -> Expression:
    """Square root."""
    return FunctionCall("sqrt", (_to_expr(x),))


def ln(x: Expression | float) -> Expression:
    """Natural logarithm (alias for log)."""
    return FunctionCall("ln", (_to_expr(x),))


def abs_(x: Expression | float) -> Expression:
    """Absolute value."""
    return FunctionCall("abs", (_to_expr(x),))


def sin_(x: Expression | float) -> Expression:
    """Sine."""
    return FunctionCall("sin", (_to_expr(x),))


def cos_(x: Expression | float) -> Expression:
    """Cosine."""
    return FunctionCall("cos", (_to_expr(x),))


def tan_(x: Expression | float) -> Expression:
    """Tangent."""
    return FunctionCall("tan", (_to_expr(x),))


def pow_(x: Expression | float, y: Expression | float) -> Expression:
    """Power function."""
    return FunctionCall("pow", (_to_expr(x), _to_expr(y)))


def min_(x: Expression | float, y: Expression | float) -> Expression:
    """Minimum of two values."""
    return FunctionCall("min", (_to_expr(x), _to_expr(y)))


def max_(x: Expression | float, y: Expression | float) -> Expression:
    """Maximum of two values."""
    return FunctionCall("max", (_to_expr(x), _to_expr(y)))


# Canonical list of function names recognized by FunctionCall.
KNOWN_FUNCTIONS: tuple[str, ...] = (
    "log", "ln", "exp", "sqrt", "abs", "sin", "cos", "tan", "pow", "min", "max",
)

# Mapping: `.mod` function name -> Expression-building callable.
# Used by parsers to populate eval() namespaces.
EQUATION_FUNCTIONS: dict[str, object] = {
    "log": log,
    "ln": ln,
    "exp": exp,
    "sqrt": sqrt,
    "abs": abs_,
    "sin": sin_,
    "cos": cos_,
    "tan": tan_,
    "pow": pow_,
    "min": min_,
    "max": max_,
}


def var(variable: Variable, timing: int = 0) -> VariableRef:
    """Create a variable reference with timing."""
    from dsgekit.model.symbols import TimedVariable

    return VariableRef(TimedVariable(variable, timing))


def shock(s: Shock) -> ShockRef:
    """Create a shock reference."""
    return ShockRef(s)


def param(p: Parameter) -> ParameterRef:
    """Create a parameter reference."""
    return ParameterRef(p)
