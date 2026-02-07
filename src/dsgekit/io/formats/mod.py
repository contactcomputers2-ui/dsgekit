"""Parser for `.mod` files.

Supports a subset of .mod syntax:
- var: endogenous variables
- varexo: exogenous shocks
- varobs: observed variables
- parameters: model parameters
- model; ... end;: model equations
- initval; ... end;: initial/steady state values
- endval; ... end;: terminal steady state values
- histval; ... end;: historical values with timing
- shocks; ... end;: shock variances and deterministic paths
- estimated_params; ... end;: estimation parameter specs
- steady_state_model; ... end;: analytical steady state
- Basic macro preprocessor:
  - @#define
  - @#if / @#elseif / @#else / @#endif
  - @#include "file.mod"
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


@dataclass
class ModFileAST:
    """Abstract syntax tree for a .mod file."""

    var: list[str] = field(default_factory=list)
    varexo: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    equation_names: list[str] = field(default_factory=list)
    initval: dict[str, str] = field(default_factory=dict)
    param_values: dict[str, str] = field(default_factory=dict)
    shock_stderr: dict[str, str] = field(default_factory=dict)
    deterministic_shocks: dict[tuple[str, int], str] = field(default_factory=dict)
    varobs: list[str] = field(default_factory=list)
    estimated_params_raw: list[dict[str, str]] = field(default_factory=list)
    histval: dict[tuple[str, int], str] = field(default_factory=dict)
    endval: dict[str, str] = field(default_factory=dict)
    steady_state_model: dict[str, str] = field(default_factory=dict)


def _remove_comments(content: str) -> str:
    """Remove line and block comments from a .mod file."""
    content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)


def _coerce_macro_literal(raw: str, defines: dict[str, object]) -> object:
    """Coerce a macro define value into bool/int/float/string when possible."""
    value = raw.strip()
    lower = value.lower()

    if lower == "true":
        return True
    if lower == "false":
        return False

    if value in defines:
        return defines[value]

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _replace_macro_tokens(expr: str, defines: dict[str, object]) -> str:
    """Replace macro identifiers with Python literals for safe eval."""

    def replace_token(match: re.Match) -> str:
        token = match.group(0)
        if token in {"and", "or", "not", "True", "False"}:
            return token
        if token in defines:
            return repr(defines[token])
        return token

    return re.sub(r"\b[A-Za-z_]\w*\b", replace_token, expr)


def _eval_macro_condition(expr: str, defines: dict[str, object]) -> bool:
    """Evaluate @#if/@#elseif condition using current macro definitions."""
    from dsgekit.exceptions import ParseError

    text = expr.strip()
    if not text:
        raise ParseError("Empty macro condition")

    # .mod logical operators.
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"(?<![=!<>])!(?!=)", " not ", text)
    text = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfalse\b", "False", text, flags=re.IGNORECASE)

    # defined(NAME) helper.
    def repl_defined(match: re.Match) -> str:
        arg = match.group(1).strip()
        if (arg.startswith('"') and arg.endswith('"')) or (
            arg.startswith("'") and arg.endswith("'")
        ):
            key = arg[1:-1]
        else:
            key = arg
        return "True" if key in defines else "False"

    text = re.sub(
        r"\bdefined\s*\(\s*([A-Za-z_]\w*|\"[^\"]+\"|'[^']+')\s*\)",
        repl_defined,
        text,
        flags=re.IGNORECASE,
    )

    text = _replace_macro_tokens(text, defines)

    try:
        value = eval(text, {"__builtins__": {}}, {})
    except Exception as exc:
        raise ParseError(f"Invalid macro condition: {expr}") from exc
    return bool(value)


def _resolve_include_path(include_target: str, source_path: Path | None) -> Path:
    """Resolve include path relative to current file."""
    from dsgekit.exceptions import ParseError

    include_path = Path(include_target)
    if include_path.is_absolute():
        return include_path

    if source_path is None:
        raise ParseError(
            f"Cannot resolve relative include '{include_target}' without source path"
        )

    return (source_path.parent / include_path).resolve()


def _preprocess_macros(
    content: str,
    source_path: Path | None = None,
    defines: dict[str, object] | None = None,
    include_stack: tuple[Path, ...] = (),
) -> str:
    """Process supported .mod macro directives before parsing blocks."""
    from dsgekit.exceptions import ParseError, UnsupportedFormatFeatureError

    if defines is None:
        defines = {}

    content = _remove_comments(content)
    output: list[str] = []
    lines = content.splitlines(keepends=True)

    # Per-level state for nested @#if blocks.
    # Keys: parent_active, branch_taken, in_else
    if_stack: list[dict[str, bool]] = []
    active = True

    for line in lines:
        stripped = line.strip()

        if not stripped.startswith("@#"):
            if active:
                output.append(line)
            continue

        directive = stripped[2:].strip()
        if not directive:
            raise ParseError("Empty macro directive '@#'")

        parts = directive.split(None, 1)
        keyword = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if keyword == "define":
            if active:
                match = re.match(
                    r"^([A-Za-z_]\w*)(?:\s*=\s*(.+)|\s+(.+))?$",
                    rest,
                )
                if not match:
                    raise ParseError(f"Invalid @#define directive: {directive}")

                name = match.group(1)
                raw_value = match.group(2) if match.group(2) is not None else match.group(3)
                value = True if raw_value is None else _coerce_macro_literal(raw_value, defines)
                defines[name] = value
            continue

        if keyword == "include":
            if active:
                include_match = re.match(r'^"([^"]+)"$', rest)
                if include_match is None:
                    include_match = re.match(r"^'([^']+)'$", rest)
                if include_match is None:
                    raise ParseError(f"Invalid @#include directive: {directive}")

                include_target = include_match.group(1)
                include_path = _resolve_include_path(include_target, source_path)

                if include_path in include_stack:
                    chain = " -> ".join(str(p) for p in (*include_stack, include_path))
                    raise ParseError(f"Cyclic @#include detected: {chain}")

                try:
                    include_content = include_path.read_text()
                except OSError as exc:
                    raise ParseError(
                        f"Could not read include file '{include_target}'"
                    ) from exc

                included = _preprocess_macros(
                    include_content,
                    source_path=include_path,
                    defines=defines,
                    include_stack=(*include_stack, include_path),
                )
                output.append(included)
                if included and not included.endswith("\n"):
                    output.append("\n")
            continue

        if keyword == "if":
            parent_active = active
            cond = _eval_macro_condition(rest, defines) if parent_active else False
            if_stack.append(
                {
                    "parent_active": parent_active,
                    "branch_taken": cond,
                    "in_else": False,
                }
            )
            active = parent_active and cond
            continue

        if keyword == "elseif":
            if not if_stack:
                raise ParseError("@#elseif without matching @#if")

            frame = if_stack[-1]
            if frame["in_else"]:
                raise ParseError("@#elseif after @#else is not allowed")

            if not frame["parent_active"] or frame["branch_taken"]:
                cond = False
            else:
                cond = _eval_macro_condition(rest, defines)

            frame["branch_taken"] = frame["branch_taken"] or cond
            active = frame["parent_active"] and cond
            continue

        if keyword == "else":
            if not if_stack:
                raise ParseError("@#else without matching @#if")

            frame = if_stack[-1]
            if frame["in_else"]:
                raise ParseError("Multiple @#else for the same @#if block")

            frame["in_else"] = True
            cond = frame["parent_active"] and not frame["branch_taken"]
            frame["branch_taken"] = frame["branch_taken"] or cond
            active = cond
            continue

        if keyword == "endif":
            if not if_stack:
                raise ParseError("@#endif without matching @#if")
            frame = if_stack.pop()
            active = frame["parent_active"]
            continue

        raise UnsupportedFormatFeatureError(f".mod macro '@#{keyword}'")

    if if_stack:
        raise ParseError("Unclosed @#if block (missing @#endif)")

    return "".join(output)


def parse_mod_file(content: str, source_path: str | Path | None = None) -> ModFileAST:
    """Parse .mod file content into AST.

    Args:
        content: Contents of .mod file
        source_path: Optional source file path (used for resolving @#include)

    Returns:
        ModFileAST with parsed elements

    Raises:
        ParseError: On syntax errors
        UnsupportedFormatFeatureError: On unsupported features
    """
    ast = ModFileAST()

    path_obj = Path(source_path).resolve() if source_path is not None else None
    content = _preprocess_macros(content, source_path=path_obj)

    # Parse var block
    var_match = re.search(r"\bvar\b\s+(.*?);", content, re.DOTALL | re.IGNORECASE)
    if var_match:
        ast.var = _parse_symbol_list(var_match.group(1))

    # Parse varexo block
    varexo_match = re.search(r"\bvarexo\b\s+(.*?);", content, re.DOTALL | re.IGNORECASE)
    if varexo_match:
        ast.varexo = _parse_symbol_list(varexo_match.group(1))

    # Parse parameters block
    params_match = re.search(
        r"\bparameters\b\s+(.*?);", content, re.DOTALL | re.IGNORECASE
    )
    if params_match:
        ast.parameters = _parse_symbol_list(params_match.group(1))

    # Parse parameter assignments (outside blocks)
    # Pattern: param_name = value;
    param_assigns = re.findall(
        r"^\s*(\w+)\s*=\s*([^;]+);",
        content,
        re.MULTILINE,
    )
    for name, value in param_assigns:
        if name in ast.parameters:
            ast.param_values[name] = value.strip()

    # Parse model block
    model_match = re.search(
        r"\bmodel\b\s*;(.*?)\bend\b\s*;", content, re.DOTALL | re.IGNORECASE
    )
    if model_match:
        model_content = model_match.group(1)
        ast.equations, ast.equation_names = _parse_equations(model_content)

    # Parse initval block
    initval_match = re.search(
        r"\binitval\b\s*;(.*?)\bend\b\s*;", content, re.DOTALL | re.IGNORECASE
    )
    if initval_match:
        ast.initval = _parse_assignments(initval_match.group(1))

    # Parse shocks block
    shocks_match = re.search(
        r"\bshocks\b\s*;(.*?)\bend\b\s*;", content, re.DOTALL | re.IGNORECASE
    )
    if shocks_match:
        ast.shock_stderr, ast.deterministic_shocks = _parse_shocks(shocks_match.group(1))

    # Parse varobs
    varobs_match = re.search(r"\bvarobs\b\s+(.*?);", content, re.DOTALL | re.IGNORECASE)
    if varobs_match:
        ast.varobs = _parse_symbol_list(varobs_match.group(1))

    # Parse endval block
    endval_match = re.search(
        r"\bendval\b\s*;(.*?)\bend\b\s*;", content, re.DOTALL | re.IGNORECASE
    )
    if endval_match:
        ast.endval = _parse_assignments(endval_match.group(1))

    # Parse histval block
    histval_match = re.search(
        r"\bhistval\b\s*;(.*?)\bend\b\s*;", content, re.DOTALL | re.IGNORECASE
    )
    if histval_match:
        ast.histval = _parse_timed_assignments(histval_match.group(1))

    # Parse estimated_params block
    ep_match = re.search(
        r"\bestimated_params\b\s*;(.*?)\bend\b\s*;", content, re.DOTALL | re.IGNORECASE
    )
    if ep_match:
        ast.estimated_params_raw = _parse_estimated_params(ep_match.group(1))

    # Parse steady_state_model block
    ssm_match = re.search(
        r"\bsteady_state_model\b\s*;(.*?)\bend\b\s*;", content, re.DOTALL | re.IGNORECASE
    )
    if ssm_match:
        ast.steady_state_model = _parse_assignments(ssm_match.group(1))

    return ast


def _parse_symbol_list(text: str) -> list[str]:
    """Parse a list of symbol names."""
    # Split by whitespace or commas
    symbols = re.split(r"[\s,]+", text.strip())
    # Filter empty strings and clean
    return [s.strip() for s in symbols if s.strip()]


def _parse_equations(text: str) -> tuple[list[str], list[str]]:
    """Parse equations from model block.

    Returns:
        Tuple of (equations, equation_names)
    """
    equations = []
    names = []

    # Split by semicolons
    lines = text.split(";")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Check for equation name: [name] equation
        name_match = re.match(r"\[(\w+)\]\s*(.*)", line, re.DOTALL)
        if name_match:
            name = name_match.group(1)
            eq = name_match.group(2).strip()
        else:
            name = f"eq_{i + 1}"
            eq = line

        if eq:
            equations.append(eq)
            names.append(name)

    return equations, names


def _parse_assignments(text: str) -> dict[str, str]:
    """Parse name = value; assignments."""
    result = {}
    matches = re.findall(r"(\w+)\s*=\s*([^;]+);?", text)
    for name, value in matches:
        result[name.strip()] = value.strip()
    return result


def _parse_period_spec(spec: str) -> list[int]:
    """Parse deterministic period spec into explicit integer periods."""
    from dsgekit.exceptions import ParseError

    tokens = [tok.strip() for tok in re.split(r"[,\s]+", spec.strip()) if tok.strip()]
    periods: list[int] = []

    for token in tokens:
        if ":" not in token:
            try:
                period = int(token)
            except ValueError as exc:
                raise ParseError(f"Invalid deterministic period token '{token}'") from exc
            if period < 1:
                raise ParseError(
                    f"Deterministic periods must be >= 1, got {period}"
                )
            periods.append(period)
            continue

        parts = [p.strip() for p in token.split(":")]
        try:
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                step = 1 if end >= start else -1
            elif len(parts) == 3:
                start, step, end = int(parts[0]), int(parts[1]), int(parts[2])
                if step == 0:
                    raise ParseError(f"Invalid deterministic period token '{token}'")
            else:
                raise ParseError(f"Invalid deterministic period token '{token}'")
        except ValueError as exc:
            raise ParseError(f"Invalid deterministic period token '{token}'") from exc

        expanded = list(range(start, end + (1 if step > 0 else -1), step))
        if any(period < 1 for period in expanded):
            raise ParseError(
                f"Deterministic periods must be >= 1, got range '{token}'"
            )
        periods.extend(expanded)

    if not periods:
        raise ParseError("Empty deterministic periods specification")

    return periods


def _parse_value_spec(spec: str) -> list[str]:
    """Parse deterministic values spec into expression tokens."""
    tokens = [tok.strip() for tok in re.split(r"[,\s]+", spec.strip()) if tok.strip()]
    return tokens


def _parse_shocks(text: str) -> tuple[dict[str, str], dict[tuple[str, int], str]]:
    """Parse shocks block for stderr values and deterministic paths.

    Supports:
        var e; stderr 0.01;
        var e = 0.01;
        var e; periods 1:4; values 0.05;
        var e; periods 1 3; values 0.1 -0.2;
    """
    from dsgekit.exceptions import ParseError

    stderrs: dict[str, str] = {}
    deterministic: dict[tuple[str, int], str] = {}

    starts = list(re.finditer(r"\bvar\b\s+(\w+)\s*(=|;)", text, re.IGNORECASE))
    for i, start_match in enumerate(starts):
        shock_name = start_match.group(1).strip()
        delimiter = start_match.group(2)
        section_start = start_match.end()
        section_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        section = text[section_start:section_end]

        if delimiter == "=":
            # .mod shorthand in shocks block: "var e = <variance>;"
            variance_expr = section.split(";", 1)[0].strip()
            if variance_expr:
                stderrs[shock_name] = f"sqrt({variance_expr})"
            continue

        stderr_match = re.search(r"\bstderr\b\s+([^;]+);", section, re.IGNORECASE)
        if stderr_match:
            stderrs[shock_name] = stderr_match.group(1).strip()

        periods_match = re.search(r"\bperiods\b\s+([^;]+);", section, re.IGNORECASE)
        values_match = re.search(r"\bvalues\b\s+([^;]+);", section, re.IGNORECASE)
        if periods_match is None and values_match is None:
            continue
        if periods_match is None or values_match is None:
            raise ParseError(
                f"Deterministic shocks for '{shock_name}' require both periods and values"
            )

        periods = _parse_period_spec(periods_match.group(1))
        values = _parse_value_spec(values_match.group(1))
        if not values:
            raise ParseError(f"Empty deterministic values for '{shock_name}'")
        if len(values) == 1:
            values = values * len(periods)
        elif len(values) != len(periods):
            raise ParseError(
                f"Deterministic shocks for '{shock_name}' require 1 value or "
                f"as many values as periods (got {len(values)} vs {len(periods)})"
            )

        for period, value_expr in zip(periods, values, strict=True):
            deterministic[(shock_name, period)] = value_expr

    return stderrs, deterministic


def _parse_timed_assignments(text: str) -> dict[tuple[str, int], str]:
    """Parse name(timing) = value; assignments for histval blocks."""
    result: dict[tuple[str, int], str] = {}
    matches = re.findall(r"(\w+)\s*\(\s*(-?\d+)\s*\)\s*=\s*([^;]+);?", text)
    for name, timing, value in matches:
        result[(name.strip(), int(timing))] = value.strip()
    return result


def _parse_estimated_params(text: str) -> list[dict[str, str]]:
    """Parse estimated_params block entries.

    Handles three entry types:
    - PARAM, INIT, LOWER, UPPER[, PRIOR_SHAPE, PRIOR_MEAN, PRIOR_STD];
    - stderr SHOCK, INIT, LOWER, UPPER[, ...];
    - corr SHOCK1, SHOCK2, INIT, LOWER, UPPER[, ...];
    """
    result: list[dict[str, str]] = []
    lines = text.split(";")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue

        entry: dict[str, str] = {}

        first = parts[0]
        if first.lower().startswith("stderr "):
            entry["type"] = "stderr"
            entry["name"] = first.split(None, 1)[1].strip()
            fields = parts[1:]
        elif first.lower().startswith("corr "):
            entry["type"] = "corr"
            entry["name"] = first.split(None, 1)[1].strip()
            if len(parts) < 2:
                continue
            entry["name2"] = parts[1].strip()
            fields = parts[2:]
        else:
            entry["type"] = "param"
            entry["name"] = first.strip()
            fields = parts[1:]

        field_names = ["init", "lower", "upper", "prior_shape", "prior_mean", "prior_std"]
        for i, field_name in enumerate(field_names):
            if i < len(fields) and fields[i].strip():
                entry[field_name] = fields[i].strip()

        result.append(entry)
    return result


def _convert_equation(eq_str: str, var_names: set[str]) -> str:
    """Convert .mod equation string to Python expression.

    Args:
        eq_str: Equation string like "y = rho*y(-1) + e"
        var_names: Set of variable names (to distinguish from functions)

    Returns:
        Python expression for the residual (LHS - RHS)
    """
    # Split by = to get LHS and RHS
    if "=" in eq_str:
        parts = eq_str.split("=", 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        expr = f"({lhs}) - ({rhs})"
    else:
        expr = eq_str

    # Convert timing notation for variables
    # We need to identify variables vs functions

    # First, protect function calls (log, exp, sqrt, etc.)
    from dsgekit.model.equations import KNOWN_FUNCTIONS

    functions = KNOWN_FUNCTIONS
    for func in functions:
        # Temporarily replace function names
        expr = re.sub(rf"\b{func}\b", f"__FUNC_{func}__", expr, flags=re.IGNORECASE)

    # Convert variable timings
    for var in var_names:
        # var(-1) -> var(var_obj, -1)
        expr = re.sub(
            rf"\b{var}\s*\(\s*(-?\d+)\s*\)",
            rf"var({var}_v, \1)",
            expr,
        )
        # var(+1) -> var(var_obj, 1)
        expr = re.sub(
            rf"\b{var}\s*\(\s*\+(\d+)\s*\)",
            rf"var({var}_v, \1)",
            expr,
        )
        # var alone (no parentheses) -> var(var_obj)
        # But not if already converted or part of another word
        expr = re.sub(
            rf"(?<![_\w]){var}(?![_\w\(])",
            rf"var({var}_v)",
            expr,
        )

    # Restore function names
    for func in functions:
        expr = expr.replace(f"__FUNC_{func}__", func)

    # Convert ^ to **
    expr = expr.replace("^", "**")

    return expr


def _normalize_numeric_expr(expr: str) -> str:
    """Normalize numeric expressions from .mod syntax to Python syntax."""
    return expr.replace("^", "**")


def _resolve_parameter_values(param_values: dict[str, str]) -> dict[str, float]:
    """Resolve parameter expressions to numeric values.

    Supports references across parameters in multiple passes, e.g.:
        alpha = 0.3
        beta = 1 - alpha
    """
    if not param_values:
        return {}

    remaining = {
        name: _normalize_numeric_expr(expr.strip())
        for name, expr in param_values.items()
    }
    resolved: dict[str, float] = {}

    math_ns = {
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
    }

    while remaining:
        progressed = False

        for name in list(remaining.keys()):
            expr = remaining[name]

            try:
                value = float(
                    eval(expr, {"__builtins__": {}}, {**math_ns, **resolved})
                )
            except Exception:
                try:
                    value = float(expr)
                except Exception:
                    continue

            resolved[name] = value
            del remaining[name]
            progressed = True

        if not progressed:
            break

    return resolved


def mod_to_ir(
    content: str, name: str = "model", source_path: str | Path | None = None
) -> tuple[ModelIR, Calibration, SteadyState]:
    """Convert .mod file content to ModelIR.

    Args:
        content: .mod file content
        name: Model name
        source_path: Optional source file path (used for resolving @#include)

    Returns:
        Tuple of (ModelIR, Calibration, SteadyState)
    """
    from dsgekit.exceptions import ParseError
    from dsgekit.model.calibration import Calibration, EstimatedParam
    from dsgekit.model.equations import EQUATION_FUNCTIONS, Equation, param, shock, var
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState

    ast = parse_mod_file(content, source_path=source_path)

    model = ModelIR(name=name)

    # Add variables
    var_objs = {}
    for v in ast.var:
        var_objs[v] = model.add_variable(v)

    # Add shocks
    shock_objs = {}
    for s in ast.varexo:
        shock_objs[s] = model.add_shock(s)

    # Add parameters
    param_objs = {}
    for p in ast.parameters:
        param_objs[p] = model.add_parameter(p)

    # Build equations
    var_names = set(ast.var)
    for eq_str, eq_name in zip(ast.equations, ast.equation_names, strict=True):
        try:
            # Create local namespace for eval
            local_ns: dict[str, object] = {
                "var": var,
                "shock": shock,
                "param": param,
            }
            local_ns.update(EQUATION_FUNCTIONS)

            # Add variable objects with _v suffix
            for v, obj in var_objs.items():
                local_ns[f"{v}_v"] = obj

            # Add shock references
            for s, obj in shock_objs.items():
                local_ns[s] = shock(obj)

            # Add parameter references
            for p, obj in param_objs.items():
                local_ns[p] = param(obj)

            # Convert equation
            expr_str = _convert_equation(eq_str, var_names)

            # Evaluate to get expression
            expr = eval(expr_str, {"__builtins__": {}}, local_ns)

            model.add_equation(Equation(expr, name=eq_name))

        except Exception as e:
            raise ParseError(f"Error parsing equation '{eq_name}': {eq_str}") from e

    # Build calibration
    calibration = Calibration()

    resolved_params = _resolve_parameter_values(ast.param_values)
    for p, value in resolved_params.items():
        calibration.set_parameter(p, value)

    value_ns = {
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
        "nan": math.nan,
        **resolved_params,
    }

    for s, value_str in ast.shock_stderr.items():
        expr = _normalize_numeric_expr(value_str)
        try:
            value = eval(expr, {"__builtins__": {}}, value_ns)
            calibration.set_shock_stderr(s, float(value))
        except Exception:
            try:
                calibration.set_shock_stderr(s, float(expr))
            except ValueError:
                pass

    # Build steady state from initval
    steady_state = SteadyState()
    for v, value_str in ast.initval.items():
        expr = _normalize_numeric_expr(value_str)
        try:
            value = eval(expr, {"__builtins__": {}}, value_ns)
            steady_state[v] = float(value)
        except Exception:
            try:
                steady_state[v] = float(expr)
            except ValueError:
                pass

    # Evaluate steady_state_model (overwrites initval values)
    if ast.steady_state_model:
        steady_state.analytical_equations = ast.steady_state_model
        ss_ns = {**value_ns}
        for var_name, expr_str in ast.steady_state_model.items():
            normalized = _normalize_numeric_expr(expr_str)
            try:
                val = eval(normalized, {"__builtins__": {}}, ss_ns)
                steady_state[var_name] = float(val)
                ss_ns[var_name] = float(val)
            except Exception:
                pass

    # Evaluate endval
    for v, value_str in ast.endval.items():
        expr = _normalize_numeric_expr(value_str)
        try:
            value = eval(expr, {"__builtins__": {}}, value_ns)
            steady_state.endval[v] = float(value)
        except Exception:
            try:
                steady_state.endval[v] = float(expr)
            except ValueError:
                pass

    # Evaluate histval
    for (var_name, timing), value_str in ast.histval.items():
        expr = _normalize_numeric_expr(value_str)
        try:
            value = eval(expr, {"__builtins__": {}}, value_ns)
            steady_state.histval[(var_name, timing)] = float(value)
        except Exception:
            try:
                steady_state.histval[(var_name, timing)] = float(expr)
            except ValueError:
                pass

    # Evaluate deterministic shock path declared in shocks block
    for (shock_name, period), value_str in ast.deterministic_shocks.items():
        expr = _normalize_numeric_expr(value_str)
        try:
            value = eval(expr, {"__builtins__": {}}, value_ns)
            steady_state.deterministic_shocks[(shock_name, period)] = float(value)
        except Exception:
            try:
                steady_state.deterministic_shocks[(shock_name, period)] = float(expr)
            except ValueError:
                pass

    # Build estimated_params
    for raw in ast.estimated_params_raw:
        try:
            parsed: dict[str, object] = {
                "type": raw["type"],
                "name": raw["name"],
                "name2": raw.get("name2", ""),
            }
            for key in ("init", "lower", "upper", "prior_mean", "prior_std"):
                if key in raw:
                    expr = _normalize_numeric_expr(raw[key])
                    parsed[key] = float(eval(expr, {"__builtins__": {}}, value_ns))
            if "prior_shape" in raw:
                parsed["prior_shape"] = raw["prior_shape"]

            calibration.estimated_params.append(EstimatedParam.from_dict(parsed))
        except Exception as exc:
            entry_name = raw.get("name", "<unknown>")
            raise ParseError(
                f"Invalid estimated_params entry for '{entry_name}': {exc}"
            ) from exc

    # Set observables
    model.observables = ast.varobs

    # Validate model
    model.validate()
    ep_errors = calibration.validate_estimated_params(model)
    if ep_errors:
        raise ParseError("; ".join(ep_errors))

    return model, calibration, steady_state


def load_mod_file(path: str) -> tuple[ModelIR, Calibration, SteadyState]:
    """Load a .mod file.

    Args:
        path: Path to .mod file

    Returns:
        Tuple of (ModelIR, Calibration, SteadyState)
    """
    from pathlib import Path

    p = Path(path)
    content = p.read_text()
    name = p.stem

    return mod_to_ir(content, name, source_path=p)
