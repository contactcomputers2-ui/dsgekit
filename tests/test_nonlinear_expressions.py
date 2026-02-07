"""Tests for non-linear expression parsing and evaluation (DYN-A03)."""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from dsgekit import load_model
from dsgekit.io.formats.mod import mod_to_ir
from dsgekit.io.formats.python_api import ModelBuilder
from dsgekit.model.equations import (
    EQUATION_FUNCTIONS,
    KNOWN_FUNCTIONS,
    EvalContext,
    FunctionCall,
    abs_,
    cos_,
    exp,
    ln,
    log,
    max_,
    min_,
    pow_,
    sin_,
    sqrt,
    tan_,
)
from dsgekit.transforms import linearize

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "models"


# =========================================================================
# Helper / registry consistency
# =========================================================================


class TestEquationHelpers:
    """All expression-building helpers produce correct AST and evaluate."""

    def test_known_functions_matches_function_call(self):
        """KNOWN_FUNCTIONS should contain all FunctionCall._FUNCS keys."""
        # Instantiate a dummy to read _FUNCS keys
        dummy = FunctionCall("log", ())
        fc_keys = set(dummy._FUNCS.keys())
        assert set(KNOWN_FUNCTIONS) == fc_keys

    def test_equation_functions_covers_all_known(self):
        """EQUATION_FUNCTIONS should have an entry for every KNOWN_FUNCTIONS name."""
        assert set(EQUATION_FUNCTIONS.keys()) == set(KNOWN_FUNCTIONS)

    @pytest.mark.parametrize(
        "helper,arg,expected",
        [
            (log, 1.0, 0.0),
            (ln, math.e, 1.0),
            (exp, 0.0, 1.0),
            (sqrt, 4.0, 2.0),
            (abs_, -3.0, 3.0),
            (sin_, 0.0, 0.0),
            (cos_, 0.0, 1.0),
            (tan_, 0.0, 0.0),
        ],
    )
    def test_unary_helper_evaluates(self, helper, arg, expected):
        expr = helper(arg)
        ctx = EvalContext()
        assert expr.evaluate(ctx) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "helper,args,expected",
        [
            (pow_, (2.0, 3.0), 8.0),
            (min_, (3.0, 5.0), 3.0),
            (max_, (3.0, 5.0), 5.0),
        ],
    )
    def test_binary_helper_evaluates(self, helper, args, expected):
        expr = helper(*args)
        ctx = EvalContext()
        assert expr.evaluate(ctx) == pytest.approx(expected)

    def test_nested_functions(self):
        """log(exp(x)) should equal x."""
        expr = log(exp(2.5))
        ctx = EvalContext()
        assert expr.evaluate(ctx) == pytest.approx(2.5)

    def test_function_of_expression(self):
        """exp(log(x) + log(y)) = x * y."""
        expr = exp(log(2.0) + log(3.0))
        ctx = EvalContext()
        assert expr.evaluate(ctx) == pytest.approx(6.0)


# =========================================================================
# .mod parser: all functions parse correctly
# =========================================================================


class TestModParserFunctions:
    """All supported functions parse correctly via .mod syntax."""

    @pytest.mark.parametrize(
        "func",
        ["log", "ln", "exp", "sqrt", "abs", "sin", "cos", "tan"],
    )
    def test_unary_function_parses(self, func):
        content = dedent(f"""\
            var y;
            varexo e;
            parameters rho;
            rho = 0.9;
            model;
                y = rho * {func}(y(-1)) + e;
            end;
            initval; y = 1; end;
            shocks; var e; stderr 0.01; end;
        """)
        model, cal, ss = mod_to_ir(content)
        assert model.n_equations == 1

    @pytest.mark.parametrize("func", ["pow", "min", "max"])
    def test_binary_function_parses(self, func):
        content = dedent(f"""\
            var y;
            varexo e;
            parameters rho;
            rho = 0.9;
            model;
                y = {func}(rho, 2) * y(-1) + e;
            end;
            initval; y = 0; end;
            shocks; var e; stderr 0.01; end;
        """)
        model, cal, ss = mod_to_ir(content)
        assert model.n_equations == 1

    def test_nested_functions_in_mod(self):
        content = dedent("""\
            var y;
            varexo e;
            parameters rho;
            rho = 0.9;
            model;
                y = log(exp(rho * y(-1))) + e;
            end;
            initval; y = 0; end;
            shocks; var e; stderr 0.01; end;
        """)
        model, cal, ss = mod_to_ir(content)
        # log(exp(rho * 0)) + 0 = 0  => residual: 0 - 0 = 0
        residuals = model.residuals_at_steady_state(ss.values, cal.parameters)
        assert abs(residuals[0]) < 1e-12

    def test_function_of_timed_variable(self):
        content = dedent("""\
            var y;
            varexo e;
            parameters rho;
            rho = 0.9;
            model;
                y = exp(rho * log(y(-1))) + e;
            end;
            initval; y = 1; end;
            shocks; var e; stderr 0.01; end;
        """)
        model, cal, ss = mod_to_ir(content)
        assert model.n_equations == 1
        # exp(rho * log(1)) + 0 = 1, y_ss = 1 => residual: 1 - 1 = 0
        residuals = model.residuals_at_steady_state(ss.values, cal.parameters)
        assert abs(residuals[0]) < 1e-12

    def test_power_expressions(self):
        """Powers via ^ operator (.mod syntax)."""
        content = dedent("""\
            var y k;
            varexo e;
            parameters alpha;
            alpha = 0.33;
            model;
                y = k(-1)^alpha + e;
                k = y;
            end;
            initval; y = 1; k = 1; end;
            shocks; var e; stderr 0.01; end;
        """)
        model, cal, ss = mod_to_ir(content)
        assert model.n_equations == 2


# =========================================================================
# Python API (ModelBuilder): all functions parse correctly
# =========================================================================


class TestPythonAPIFunctions:
    """Function support via ModelBuilder."""

    @pytest.mark.parametrize(
        "func",
        ["log", "ln", "exp", "sqrt", "abs", "sin", "cos", "tan"],
    )
    def test_unary_function_parses(self, func):
        m, c, s = (
            ModelBuilder("test")
            .var("y")
            .varexo("e")
            .param("rho", 0.9)
            .equation(f"y = rho * {func}(y(-1)) + e")
            .shock_stderr(e=0.01)
            .initval(y=1)
            .build()
        )
        assert m.n_equations == 1

    @pytest.mark.parametrize("func", ["pow", "min", "max"])
    def test_binary_function_parses(self, func):
        m, c, s = (
            ModelBuilder("test")
            .var("y")
            .varexo("e")
            .param("rho", 0.9)
            .equation(f"y = {func}(rho, 2) * y(-1) + e")
            .shock_stderr(e=0.01)
            .initval(y=0)
            .build()
        )
        assert m.n_equations == 1


# =========================================================================
# RBC non-linear model integration
# =========================================================================


class TestRBCNonlinearModel:
    """Canonical RBC model with non-linear equations (rbc.mod fixture)."""

    def test_rbc_mod_parses(self):
        model, cal, ss = load_model(FIXTURES_DIR / "rbc.mod")
        assert model.n_equations == 4
        assert model.variable_names == ["y", "c", "k", "a"]

    def test_rbc_residuals_near_zero(self):
        model, cal, ss = load_model(FIXTURES_DIR / "rbc.mod")
        residuals = model.residuals_at_steady_state(ss.values, cal.parameters)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-8)

    def test_rbc_linearizes(self):
        model, cal, ss = load_model(FIXTURES_DIR / "rbc.mod")
        lin = linearize(model, ss, cal)
        assert lin.A.shape == (4, 4)
        assert lin.B.shape == (4, 4)
        assert lin.C.shape == (4, 4)
        assert lin.D.shape == (4, 1)
        # Jacobians should have non-zero entries
        assert np.count_nonzero(lin.B) > 0

    def test_rbc_full_pipeline(self):
        """Load -> linearize -> solve for RBC.

        Regression for DYN-A04: the companion-system BK counting must
        not over-count structural zero roots.
        """
        from dsgekit.solvers import diagnose_bk, solve_linear

        model, cal, ss = load_model(FIXTURES_DIR / "rbc.mod")
        lin = linearize(model, ss, cal)
        solution = solve_linear(lin)
        diag = diagnose_bk(solution)
        assert solution.T.shape == (4, 4)
        assert solution.R.shape == (4, 1)
        assert solution.n_stable == model.n_predetermined
        assert diag.status == "determinate"


# =========================================================================
# Edge cases
# =========================================================================


class TestNonlinearEdgeCases:
    """Edge cases for non-linear expression parsing."""

    def test_reciprocal_expression(self):
        """1/x parses as division."""
        content = dedent("""\
            var c;
            varexo e;
            parameters beta;
            beta = 0.99;
            model;
                1/c = beta * (1/c(+1)) + e;
            end;
            initval; c = 1; end;
            shocks; var e; stderr 0.01; end;
        """)
        model, cal, ss = mod_to_ir(content)
        assert model.n_equations == 1

    def test_compound_power_expression(self):
        """k^(alpha-1) where alpha is a parameter."""
        content = dedent("""\
            var k y;
            varexo e;
            parameters alpha;
            alpha = 0.33;
            model;
                y = k(-1)^(alpha - 1) + e;
                k = y;
            end;
            initval; y = 1; k = 1; end;
            shocks; var e; stderr 0.01; end;
        """)
        model, cal, ss = mod_to_ir(content)
        assert model.n_equations == 2

    def test_exp_wrapping_pattern(self):
        """Common .mod pattern: variables wrapped in exp()."""
        content = dedent("""\
            var c y;
            varexo e;
            parameters beta;
            beta = 0.99;
            model;
                exp(c) = beta * exp(c(+1)) + e;
                exp(y) = exp(c);
            end;
            initval; c = 0; y = 0; end;
            shocks; var e; stderr 0.01; end;
        """)
        model, cal, ss = mod_to_ir(content)
        assert model.n_equations == 2
        # At c=0, y=0: exp(0)=1, beta*exp(0)=0.99, residual = 1 - 0.99 - 0 = 0.01
        # (not zero because this is a simplified model, but it should parse)
