"""Unit tests for `.mod` parser."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from dsgekit.exceptions import ParseError, UnsupportedFormatFeatureError
from dsgekit.io.formats.mod import (
    load_mod_file,
    mod_to_ir,
    parse_mod_file,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "models"


class TestParseModFileAST:
    def test_parses_core_blocks_comments_and_shocks(self):
        content = dedent(
            """
            // single-line comment should be ignored
            var y, c;
            varexo e, u;
            parameters rho, gamma;

            rho = 0.9;
            gamma = 2;

            model;
                [eq_y] y = rho * y(-1) + e;
                c = y^gamma + u; // inline comment
            end;

            initval;
                y = 0;
                c = 0;
            end;

            shocks;
                var e; stderr 0.01;
                var u = 0.0004;
            end;
            """
        )

        ast = parse_mod_file(content)

        assert ast.var == ["y", "c"]
        assert ast.varexo == ["e", "u"]
        assert ast.parameters == ["rho", "gamma"]
        assert ast.param_values == {"rho": "0.9", "gamma": "2"}
        assert ast.equation_names == ["eq_y", "eq_2"]
        assert len(ast.equations) == 2
        assert ast.initval == {"y": "0", "c": "0"}
        assert ast.shock_stderr == {"e": "0.01", "u": "sqrt(0.0004)"}

    def test_parses_deterministic_shocks_block(self):
        content = dedent(
            """
            var y;
            varexo e;
            parameters rho;
            rho = 0.9;

            model;
                y = rho * y(-1) + e;
            end;

            initval;
                y = 0;
            end;

            shocks;
                var e; stderr 0.01;
                var e;
                    periods 1:3 5;
                    values 0.1 0.2 0.3 0.4;
            end;
            """
        )

        ast = parse_mod_file(content)
        assert ast.shock_stderr == {"e": "0.01"}
        assert ast.deterministic_shocks == {
            ("e", 1): "0.1",
            ("e", 2): "0.2",
            ("e", 3): "0.3",
            ("e", 5): "0.4",
        }

        _model, _cal, ss = mod_to_ir(content)
        assert ss.deterministic_shocks[("e", 1)] == pytest.approx(0.1)
        assert ss.deterministic_shocks[("e", 2)] == pytest.approx(0.2)
        assert ss.deterministic_shocks[("e", 3)] == pytest.approx(0.3)
        assert ss.deterministic_shocks[("e", 5)] == pytest.approx(0.4)

    def test_rejects_mismatched_deterministic_periods_and_values(self):
        content = dedent(
            """
            var y;
            varexo e;
            parameters rho;
            rho = 0.9;
            model;
                y = rho * y(-1) + e;
            end;
            initval;
                y = 0;
            end;
            shocks;
                var e;
                    periods 1 2 3;
                    values 0.1 0.2;
            end;
            """
        )

        with pytest.raises(ParseError, match="1 value or as many values as periods"):
            parse_mod_file(content)

    def test_keywords_are_case_insensitive(self):
        content = dedent(
            """
            VAR y;
            VAREXO e;
            PARAMETERS rho;
            rho = 0.9;
            MODEL;
                y = rho * y(-1) + e;
            END;
            INITVAL;
                y = 0;
            END;
            SHOCKS;
                var e; stderr 0.01;
            END;
            """
        )

        ast = parse_mod_file(content)
        assert ast.var == ["y"]
        assert ast.varexo == ["e"]
        assert ast.parameters == ["rho"]
        assert len(ast.equations) == 1

    def test_processes_define_and_if_else_blocks(self):
        content = dedent(
            """
            @#define USE_NEWSHOCK = true
            var y;
            varexo e, u;
            parameters rho;
            rho = 0.9;
            model;
                @#if USE_NEWSHOCK
                    y = rho * y(-1) + e + u;
                @#else
                    y = rho * y(-1) + e;
                @#endif
            end;
            initval;
                y = 0;
            end;
            shocks;
                var e; stderr 0.01;
                var u; stderr 0.01;
            end;
            """
        )

        ast = parse_mod_file(content)
        assert len(ast.equations) == 1
        assert ast.equations[0] == "y = rho * y(-1) + e + u"

    def test_rejects_unsupported_macro_directive(self):
        content = dedent(
            """
            @#for i in 1:3
            var y;
            @#endfor
            """
        )

        with pytest.raises(UnsupportedFormatFeatureError, match="macro '@#for'"):
            parse_mod_file(content)

    def test_unclosed_if_macro_raises_parse_error(self):
        content = dedent(
            """
            @#define USE_A = 1
            @#if USE_A
            var y;
            """
        )

        with pytest.raises(ParseError, match="Unclosed @#if"):
            parse_mod_file(content)


class TestModToIR:
    def test_evaluates_parameter_dependencies_and_expressions(self):
        content = dedent(
            """
            var y;
            varexo e;
            parameters alpha beta rho sigma2;
            alpha = 0.3;
            beta = 1 - alpha;
            rho = beta;
            sigma2 = 0.1^2;

            model;
                y = rho * y(-1) + e;
            end;

            initval;
                y = alpha + beta;
            end;

            shocks;
                var e; stderr sqrt(sigma2);
            end;
            """
        )

        model, cal, ss = mod_to_ir(content, name="dep_params")

        assert model.name == "dep_params"
        assert cal.parameters["alpha"] == pytest.approx(0.3)
        assert cal.parameters["beta"] == pytest.approx(0.7)
        assert cal.parameters["rho"] == pytest.approx(0.7)
        assert cal.parameters["sigma2"] == pytest.approx(0.01)
        assert cal.shock_stderr["e"] == pytest.approx(0.1)
        assert ss.values["y"] == pytest.approx(1.0)

    def test_invalid_equation_raises_parse_error(self):
        content = dedent(
            """
            var y;
            varexo e;
            parameters rho;
            rho = 0.9;
            model;
                [bad_eq] y = rho * * y(-1) + e;
            end;
            initval;
                y = 0;
            end;
            shocks;
                var e; stderr 0.01;
            end;
            """
        )

        with pytest.raises(ParseError, match="bad_eq"):
            mod_to_ir(content, name="bad")


class TestLoadModFile:
    def test_uses_filename_stem_as_model_name(self, tmp_path):
        path = tmp_path / "my_reference.mod"
        path.write_text(
            dedent(
                """
                var y;
                varexo e;
                parameters rho;
                rho = 0.9;
                model;
                    y = rho * y(-1) + e;
                end;
                initval;
                    y = 0;
                end;
                shocks;
                    var e; stderr 0.01;
                end;
                """
            )
        )

        model, cal, ss = load_mod_file(str(path))
        assert model.name == "my_reference"
        assert model.variable_names == ["y"]
        assert cal.parameters["rho"] == pytest.approx(0.9)
        assert ss.values["y"] == pytest.approx(0.0)

    def test_resolves_include_relative_to_main_mod(self, tmp_path):
        inc = tmp_path / "params.inc"
        inc.write_text(
            dedent(
                """
                @#define USE_ALT = 0
                parameters rho;
                rho = 0.85;
                """
            )
        )

        main = tmp_path / "with_include.mod"
        main.write_text(
            dedent(
                """
                @#include "params.inc"
                var y;
                varexo e;
                model;
                    @#if USE_ALT
                        y = e;
                    @#else
                        y = rho * y(-1) + e;
                    @#endif
                end;
                initval;
                    y = 0;
                end;
                shocks;
                    var e; stderr 0.02;
                end;
                """
            )
        )

        model, cal, ss = load_mod_file(str(main))
        assert model.name == "with_include"
        assert cal.parameters["rho"] == pytest.approx(0.85)
        assert model.equations[0].name == "eq_1"
        assert ss.values["y"] == pytest.approx(0.0)

    def test_detects_cyclic_include(self, tmp_path):
        a = tmp_path / "a.mod"
        b = tmp_path / "b.mod"

        a.write_text(
            dedent(
                """
                @#include "b.mod"
                var y;
                varexo e;
                model;
                    y = e;
                end;
                initval;
                    y = 0;
                end;
                shocks;
                    var e; stderr 0.01;
                end;
                """
            )
        )
        b.write_text('@#include "a.mod"\n')

        with pytest.raises(ParseError, match="Cyclic @#include"):
            load_mod_file(str(a))


# ---------------------------------------------------------------------------
# New block tests (DYN-A02)
# ---------------------------------------------------------------------------

_BASE_MOD = dedent(
    """\
    var y;
    varexo e;
    parameters rho;
    rho = 0.9;
    model;
        y = rho * y(-1) + e;
    end;
    initval;
        y = 0;
    end;
    shocks;
        var e; stderr 0.01;
    end;
    """
)


class TestVarobsParsing:
    def test_parses_varobs_symbol_list(self):
        content = _BASE_MOD + "varobs y;\n"
        ast = parse_mod_file(content)
        assert ast.varobs == ["y"]

    def test_parses_multiple_varobs(self):
        content = dedent(
            """
            var x pi i;
            varexo e;
            parameters rho;
            rho = 0.9;
            model;
                x = rho * x(-1) + e;
                pi = x;
                i = pi;
            end;
            initval; x = 0; pi = 0; i = 0; end;
            shocks; var e; stderr 0.01; end;
            varobs x pi i;
            """
        )
        ast = parse_mod_file(content)
        assert ast.varobs == ["x", "pi", "i"]

    def test_varobs_populates_model_ir_observables(self):
        content = _BASE_MOD + "varobs y;\n"
        model, _cal, _ss = mod_to_ir(content)
        assert model.observables == ["y"]

    def test_missing_varobs_gives_empty_list(self):
        model, _cal, _ss = mod_to_ir(_BASE_MOD)
        assert model.observables == []

    def test_varobs_does_not_clash_with_var(self):
        content = _BASE_MOD + "varobs y;\n"
        ast = parse_mod_file(content)
        assert ast.var == ["y"]
        assert ast.varobs == ["y"]


class TestEstimatedParamsParsing:
    def test_parses_parameter_with_prior(self):
        content = _BASE_MOD + dedent(
            """
            estimated_params;
                rho, 0.9, 0.0, 0.99, normal_pdf, 0.9, 0.05;
            end;
            """
        )
        ast = parse_mod_file(content)
        assert len(ast.estimated_params_raw) == 1
        ep = ast.estimated_params_raw[0]
        assert ep["type"] == "param"
        assert ep["name"] == "rho"
        assert ep["init"] == "0.9"
        assert ep["lower"] == "0.0"
        assert ep["upper"] == "0.99"
        assert ep["prior_shape"] == "normal_pdf"
        assert ep["prior_mean"] == "0.9"
        assert ep["prior_std"] == "0.05"

    def test_parses_stderr_entry(self):
        content = _BASE_MOD + dedent(
            """
            estimated_params;
                stderr e, 0.01, 0.001, 0.1;
            end;
            """
        )
        ast = parse_mod_file(content)
        ep = ast.estimated_params_raw[0]
        assert ep["type"] == "stderr"
        assert ep["name"] == "e"
        assert ep["init"] == "0.01"

    def test_parses_corr_entry(self):
        content = dedent(
            """
            var y z;
            varexo e1 e2;
            parameters rho;
            rho = 0.9;
            model;
                y = rho * y(-1) + e1;
                z = rho * z(-1) + e2;
            end;
            initval; y = 0; z = 0; end;
            shocks;
                var e1; stderr 0.01;
                var e2; stderr 0.01;
            end;
            estimated_params;
                corr e1, e2, 0.5, -1, 1;
            end;
            """
        )
        ast = parse_mod_file(content)
        ep = ast.estimated_params_raw[0]
        assert ep["type"] == "corr"
        assert ep["name"] == "e1"
        assert ep["name2"] == "e2"
        assert ep["init"] == "0.5"

    def test_mle_only_no_priors(self):
        content = _BASE_MOD + dedent(
            """
            estimated_params;
                rho, 0.5, 0, 1;
            end;
            """
        )
        ast = parse_mod_file(content)
        ep = ast.estimated_params_raw[0]
        assert ep["type"] == "param"
        assert "prior_shape" not in ep

    def test_populates_calibration_estimated_params(self):
        content = _BASE_MOD + dedent(
            """
            estimated_params;
                rho, 0.9, 0.0, 0.99, normal_pdf, 0.9, 0.05;
                stderr e, 0.01, 0.001, 0.1;
            end;
            """
        )
        _model, cal, _ss = mod_to_ir(content)
        assert len(cal.estimated_params) == 2
        assert cal.estimated_params[0].entry_type == "param"
        assert cal.estimated_params[0].name == "rho"
        assert cal.estimated_params[0].init_value == pytest.approx(0.9)
        assert cal.estimated_params[0].prior_shape == "normal_pdf"
        assert cal.estimated_params[1].entry_type == "stderr"
        assert cal.estimated_params[1].name == "e"

    def test_empty_block_gives_empty_list(self):
        content = _BASE_MOD + "estimated_params;\nend;\n"
        _model, cal, _ss = mod_to_ir(content)
        assert cal.estimated_params == []

    def test_invalid_prior_distribution_raises(self):
        content = _BASE_MOD + dedent(
            """
            estimated_params;
                rho, 0.9, 0.0, 0.99, laplace_pdf, 0.9, 0.05;
            end;
            """
        )
        with pytest.raises(ParseError, match="Invalid estimated_params entry"):
            mod_to_ir(content)


class TestHistvalParsing:
    def test_parses_timed_assignments(self):
        content = _BASE_MOD + dedent(
            """
            histval;
                y(0) = 1.5;
                y(-1) = 1.2;
            end;
            """
        )
        ast = parse_mod_file(content)
        assert ast.histval == {("y", 0): "1.5", ("y", -1): "1.2"}

    def test_populates_steady_state_histval(self):
        content = _BASE_MOD + dedent(
            """
            histval;
                y(0) = 1.5;
                y(-1) = 1.2;
            end;
            """
        )
        _model, _cal, ss = mod_to_ir(content)
        assert ss.histval[("y", 0)] == pytest.approx(1.5)
        assert ss.histval[("y", -1)] == pytest.approx(1.2)


class TestEndvalParsing:
    def test_parses_endval_block(self):
        content = _BASE_MOD + dedent(
            """
            endval;
                y = 2.0;
            end;
            """
        )
        ast = parse_mod_file(content)
        assert ast.endval == {"y": "2.0"}

    def test_populates_steady_state_endval(self):
        content = _BASE_MOD + dedent(
            """
            endval;
                y = 2.0;
            end;
            """
        )
        _model, _cal, ss = mod_to_ir(content)
        assert ss.endval["y"] == pytest.approx(2.0)

    def test_endval_separate_from_initval(self):
        content = _BASE_MOD + dedent(
            """
            endval;
                y = 5.0;
            end;
            """
        )
        _model, _cal, ss = mod_to_ir(content)
        assert ss.values["y"] == pytest.approx(0.0)  # from initval
        assert ss.endval["y"] == pytest.approx(5.0)


class TestSteadyStateModelParsing:
    def test_parses_ordered_assignments(self):
        content = _BASE_MOD + dedent(
            """
            steady_state_model;
                y = 0;
            end;
            """
        )
        ast = parse_mod_file(content)
        assert ast.steady_state_model == {"y": "0"}

    def test_evaluates_to_steady_state_values(self):
        content = dedent(
            """
            var y;
            varexo e;
            parameters rho;
            rho = 0.9;
            model;
                y = rho * y(-1) + e;
            end;
            shocks;
                var e; stderr 0.01;
            end;
            steady_state_model;
                y = 0;
            end;
            """
        )
        _model, _cal, ss = mod_to_ir(content)
        assert ss.values["y"] == pytest.approx(0.0)

    def test_preserves_raw_expressions(self):
        content = _BASE_MOD + dedent(
            """
            steady_state_model;
                y = 1 + rho;
            end;
            """
        )
        _model, _cal, ss = mod_to_ir(content)
        assert ss.analytical_equations["y"] == "1 + rho"
        assert ss.values["y"] == pytest.approx(1.9)  # 1 + 0.9

    def test_steady_state_model_overwrites_initval(self):
        content = dedent(
            """
            var y;
            varexo e;
            parameters rho;
            rho = 0.9;
            model;
                y = rho * y(-1) + e;
            end;
            initval;
                y = 99;
            end;
            shocks;
                var e; stderr 0.01;
            end;
            steady_state_model;
                y = 0;
            end;
            """
        )
        _model, _cal, ss = mod_to_ir(content)
        assert ss.values["y"] == pytest.approx(0.0)


class TestFixtureFiles:
    def test_nk_estimation_fixture_parses(self):
        model, cal, ss = load_mod_file(str(FIXTURES_DIR / "nk_estimation.mod"))
        assert model.observables == ["x", "pi", "i"]
        assert len(cal.estimated_params) == 4
        assert ss.values["x"] == pytest.approx(0.0)
        assert ss.values["pi"] == pytest.approx(0.0)
        assert ss.values["i"] == pytest.approx(0.0)
        assert ss.analytical_equations == {"x": "0", "pi": "0", "i": "0"}
        # Check estimated_params types
        types = [ep.entry_type for ep in cal.estimated_params]
        assert types == ["param", "param", "stderr", "stderr"]

    def test_rbc_histval_fixture_parses(self):
        model, cal, ss = load_mod_file(str(FIXTURES_DIR / "rbc_histval.mod"))
        assert model.variable_names == ["y", "c", "k", "a"]
        # initval
        assert ss.values["k"] == pytest.approx(10.0)
        # endval
        assert ss.endval["k"] == pytest.approx(12.0)
        assert ss.endval["y"] == pytest.approx(1.2)
        # histval
        assert ss.histval[("k", 0)] == pytest.approx(9.5)
        assert ss.histval[("a", 0)] == pytest.approx(1.0)
        assert ss.histval[("a", -1)] == pytest.approx(0.98)
