"""core 求值、复数、精度隔离、函数全集与安全限制测试。"""

import ast
import time
import unittest
from pathlib import Path
from unittest import mock
from mpmath import mp

from core import MathError, evaluator, evaluate
from core import registry
from core.evaluator import build_symbol_table
from core.security import validate

CORE_DIR = Path(__file__).resolve().parent.parent / "core"


def expect_error(testcase, expression, error_code):
    with testcase.assertRaises(MathError) as ctx:
        evaluate(expression)
    testcase.assertEqual(ctx.exception.error_code, error_code)


class RegistryTest(unittest.TestCase):
    def test_function_counts(self):
        self.assertEqual(len(registry.MATH_FUNCTIONS), 53)
        self.assertEqual(len(registry.STATISTICS_FUNCTIONS), 10)
        self.assertEqual(len(registry.COMPLEX_FUNCTIONS), 10)
        self.assertEqual(len(registry.ALL_FUNCTION_NAMES), 73)

    def test_constants(self):
        self.assertEqual(len(registry.CONSTANT_NAMES), 7)

    def test_unit_counts(self):
        self.assertEqual(len(registry.UNIT_CONVERSIONS), 15)
        self.assertEqual(
            sum(len(units) for units in registry.UNIT_CONVERSIONS.values()), 158
        )
        self.assertEqual(len(registry.UNIT_ALIASES), 153)

    def test_no_duplicate_names(self):
        names = registry.ALL_FUNCTION_NAMES
        self.assertEqual(len(names), len(set(names)))

    def test_aliases_resolve_to_known_units(self):
        all_units = {
            unit for conversions in registry.UNIT_CONVERSIONS.values() for unit in conversions
        }
        self.assertTrue(set(registry.UNIT_ALIASES.values()) <= all_units)


class EvaluateTest(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(evaluate("1+2"), "3")
        self.assertEqual(evaluate("2**10"), "1024")
        self.assertEqual(evaluate("7//2"), "3")
        self.assertEqual(evaluate("7%3"), "1")
        self.assertEqual(evaluate("5^3"), "125")
        self.assertEqual(evaluate("1<<4"), "16")

    def test_float_result(self):
        self.assertAlmostEqual(float(evaluate("1/3")), 1 / 3, places=15)

    def test_functions(self):
        self.assertEqual(evaluate("factorial(10)"), "3628800")
        self.assertEqual(evaluate("gcd(12, 18)"), "6")
        self.assertEqual(evaluate("sqrt(4)"), "2")
        self.assertEqual(evaluate("abs(-5)"), "5")
        self.assertEqual(evaluate("trunc(-2.7)"), "-2")
        self.assertEqual(evaluate("exp2(10)"), "1024")

    def test_list_and_tuple_results(self):
        self.assertEqual(evaluate("mean([1, 2, 3])"), "2")
        self.assertEqual(evaluate("quantiles([1, 2, 3, 4])"), "[1.25, 2.5, 3.75]")
        self.assertEqual(evaluate("modf(2.5)"), "(0.5, 2)")

    def test_boolean_result(self):
        self.assertEqual(evaluate("isfinite(1)"), "True")
        self.assertEqual(evaluate("isclose(1, 1)"), "True")
        self.assertEqual(evaluate("isinf(inf)"), "True")

    def test_inf_and_nan(self):
        self.assertEqual(evaluate("inf"), "inf")
        self.assertEqual(evaluate("nan"), "nan")
        self.assertEqual(evaluate("inf*0"), "nan")

    def test_keyword_argument(self):
        self.assertEqual(evaluate("isclose(1.0, 1.0000001, rel_tol=1e-3)"), "True")

    def test_caret_is_power_not_xor(self):
        # ^ 按上游约定视为乘方而非位异或；指数上限与映射在替换后仍生效
        self.assertEqual(evaluate("2^10"), "1024")
        self.assertEqual(evaluate("(-2)^2"), "4")
        self.assertEqual(evaluate("-2^2"), "-4")
        self.assertEqual(evaluate("(1+1)^10"), "1024")
        self.assertEqual(evaluate("2^-2"), "0.25")
        expect_error(self, "2^10001", "exponent_too_large")
        expect_error(self, "'2^3'", "invalid_argument")  # 引号内不替换，字符串仍被拒绝


class ComplexTest(unittest.TestCase):
    def test_sqrt_of_negative_returns_complex_string(self):
        self.assertEqual(evaluate("sqrt(-1)"), "(0.0 + 1.0j)")

    def test_real_root_of_negative(self):
        self.assertEqual(evaluate("root(-8, 3)"), "-2")
        self.assertEqual(evaluate("root(27, 3)"), "3")

    def test_complex_functions(self):
        self.assertEqual(evaluate("csqrt(4)"), "(2.0 + 0.0j)")
        self.assertAlmostEqual(float(evaluate("phase(1+1j)")), 0.7853981633974483, places=12)

    def test_complex_arithmetic(self):
        self.assertTrue(evaluate("1j * 1j").startswith("(-1"))


class PrecisionTest(unittest.TestCase):
    def test_dps_controls_precision(self):
        self.assertTrue(evaluate("pi", dps=50).startswith("3.1415926535897932384626433832795"))
        self.assertNotEqual(evaluate("pi", dps=20), evaluate("pi", dps=50))
        self.assertTrue(evaluate("e", dps=30).startswith("2.71828182845904523536028747135"))

    def test_global_dps_untouched(self):
        before = mp.dps
        evaluate("pi", dps=123)
        evaluate("sqrt(2)", dps=3)
        self.assertEqual(mp.dps, before)

    def test_invalid_dps_rejected(self):
        for bad in (0, -1, 201, True):
            with self.assertRaises(MathError) as ctx:
                evaluate("1+1", dps=bad)
            self.assertEqual(ctx.exception.error_code, "invalid_argument")


class ErrorMappingTest(unittest.TestCase):
    def test_division_by_zero(self):
        expect_error(self, "1/0", "division_by_zero")

    def test_domain_error(self):
        expect_error(self, "log(0)", "domain_error")
        expect_error(self, "log10(0)", "domain_error")
        expect_error(self, "log1p(-1)", "domain_error")
        expect_error(self, "factorial(301)", "domain_error")
        expect_error(self, "factorial(-1)", "domain_error")

    def test_syntax_error(self):
        expect_error(self, "1+*2", "syntax_error")

    def test_unknown_function(self):
        expect_error(self, "foo(1)", "unknown_function")
        expect_error(self, "sum([1,2])", "unknown_function")

    def test_exponent_too_large(self):
        expect_error(self, "2**10001", "exponent_too_large")
        expect_error(self, "2**-10001", "exponent_too_large")

    def test_expression_too_long(self):
        expect_error(self, "1+" * 500 + "1", "expression_too_long")

    def test_expression_too_complex(self):
        expect_error(self, "1" + "+1" * 11, "expression_too_complex")

    def test_invalid_argument(self):
        expect_error(self, "1 if 2 else 3", "invalid_argument")
        expect_error(self, "(lambda: 1)()", "invalid_argument")
        expect_error(self, '"abc"', "invalid_argument")

    def test_asteval_exponent_guard_maps_to_code(self):
        # asteval 自带 safe_pow 动态指数上限，超限 RuntimeError 映射为本项目错误码
        expect_error(self, "2**(10**6)", "exponent_too_large")

    def test_timeout(self):
        def slow(expression, dps):
            time.sleep(1)

        with mock.patch.object(evaluator, "_evaluate_sync", slow):
            with self.assertRaises(MathError) as ctx:
                evaluate("1+1", timeout_seconds=0.05)
        self.assertEqual(ctx.exception.error_code, "timeout")

    def test_forbidden_patterns_rejected(self):
        expect_error(self, '__import__("os")', "invalid_argument")
        expect_error(self, "import os", "invalid_argument")
        expect_error(self, "eval(1)", "invalid_argument")
        expect_error(self, "getattr(1, 'a')", "invalid_argument")


class FullFunctionSetTest(unittest.TestCase):
    # 每个注册函数一组安全示例参数，保证符号表完整可用。
    EXAMPLE_ARGS = {
        "sin": "0", "cos": "0", "tan": "0", "asin": "0", "acos": "1",
        "atan": "0", "atan2": "1, 1", "sinh": "0", "cosh": "0", "tanh": "0",
        "asinh": "0", "acosh": "1", "atanh": "0.5", "log": "e",
        "log10": "100", "log2": "8", "exp": "0", "sqrt": "4", "pow": "2, 10",
        "abs": "-1", "fabs": "-1", "factorial": "5", "ceil": "1.5",
        "floor": "1.5", "trunc": "1.5", "degrees": "pi", "radians": "180",
        "gcd": "12, 18", "lcm": "4, 6", "isqrt": "10", "hypot": "3, 4",
        "copysign": "1, -1", "fmod": "5, 3", "remainder": "5, 3",
        "modf": "2.5", "frexp": "8", "ldexp": "1, 3", "isfinite": "1",
        "isinf": "inf", "isnan": "nan", "isclose": "1, 1", "comb": "5, 2",
        "perm": "5, 2", "erf": "0", "erfc": "0", "gamma": "5", "lgamma": "5",
        "cbrt": "8", "exp2": "10", "expm1": "0", "log1p": "0",
        "nextafter": "1, 2", "ulp": "1",
        "mean": "[1, 2, 3]", "median": "[1, 2, 3]", "mode": "[1, 2, 2]",
        "stdev": "[1, 2, 3]", "pstdev": "[1, 2, 3]", "variance": "[1, 2, 3]",
        "pvariance": "[1, 2, 3]", "harmonic_mean": "[1, 2, 4]",
        "geometric_mean": "[1, 4]", "quantiles": "[1, 2, 3, 4]",
        "phase": "1", "polar": "1", "rect": "1, 0", "csin": "0",
        "ccos": "0", "ctan": "0", "cexp": "0", "clog": "1", "clog10": "1",
        "csqrt": "4", "root": "-8, 3",
    }

    def test_every_registered_function_is_callable(self):
        self.assertEqual(set(self.EXAMPLE_ARGS), set(registry.ALL_FUNCTION_NAMES) | {"root"})
        with mp.workdps(20):
            table = build_symbol_table()
        for name in registry.ALL_FUNCTION_NAMES:
            self.assertIn(name, table, name)
            self.assertTrue(callable(table[name]), name)
            self.assertIsNotNone(evaluate(f"{name}({self.EXAMPLE_ARGS[name]})"), name)
        self.assertTrue(callable(table["root"]))

    def test_constants_injected(self):
        for name in registry.CONSTANT_NAMES:
            self.assertIsNotNone(evaluate(name))


class SecurityValidateTest(unittest.TestCase):
    def test_valid_expression_returns_ast(self):
        tree = validate("1 + 2 * sin(pi/2)")
        self.assertIsInstance(tree, ast.Expression)

    def test_empty_expression_is_syntax_error(self):
        with self.assertRaises(MathError) as ctx:
            validate("")
        self.assertEqual(ctx.exception.error_code, "syntax_error")

class NoForbiddenImportsTest(unittest.TestCase):
    """core 层不依赖 astrbot / logging，不用 print。"""

    def test_core_has_no_astrbot_or_logging(self):
        for path in sorted(CORE_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    self.assertNotIn("astrbot", name, f"{path}: {name}")
                    self.assertNotIn("logging", name, f"{path}: {name}")


if __name__ == "__main__":
    unittest.main()
