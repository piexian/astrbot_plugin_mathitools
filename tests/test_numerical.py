"""数值求根与微积分公开核心测试。"""

import asyncio
import math
import sys
import unittest
from unittest import mock

from mpmath import mp

from astrbot_plugin_mathitools.core.errors import MathError
from astrbot_plugin_mathitools.core.numerical import numerical_operation


def _run(*args, **kwargs):
    return asyncio.run(numerical_operation(*args, **kwargs))


class FindRootTest(unittest.TestCase):
    def test_findroot_normal_roots(self):
        result = _run("findroot", expression="x^2-2", a=1, b=2)
        with mp.workdps(50):
            self.assertAlmostEqual(float(mp.mpf(result)), float(mp.sqrt(2)), places=12)
        result = _run("findroot", expression="sin(x)", a=3, b=4)
        with mp.workdps(50):
            self.assertAlmostEqual(float(mp.mpf(result)), float(mp.pi), places=12)

    def test_findroot_endpoint_and_negative_root(self):
        result = _run("findroot", expression="x*(x-2)", a=2, b=3)
        self.assertEqual(mp.mpf(result), mp.mpf(2))
        result = _run("findroot", expression="x^2-2", a=-2, b=-1)
        with mp.workdps(50):
            self.assertAlmostEqual(float(mp.mpf(result)), -float(mp.sqrt(2)), places=12)

    def test_findroot_flat_function(self):
        result = _run("findroot", expression="(x-1)^99", a=0.9, b=1.1)
        with mp.workdps(50):
            self.assertLess(abs(mp.mpf(result) - mp.mpf(1)), mp.mpf(10) ** -18)

    def test_findroot_dps80_real_precision(self):
        result = _run("findroot", expression="x^2-2", a=1, b=2, dps=80)
        with mp.workdps(100):
            error = abs(mp.mpf(result) - mp.sqrt(2))
            self.assertLess(error, mp.mpf(10) ** -78)

    def test_findroot_parent_dps_isolated(self):
        before = mp.dps
        with mp.workdps(7):
            _run("findroot", expression="x^2-2", a=1, b=2)
            self.assertEqual(mp.dps, 7)
        self.assertEqual(mp.dps, before)

    def test_findroot_tiny_flat_function(self):
        result = _run("findroot", expression="(x-1e-300)^99", a=0, b=3e-300, dps=20)
        with mp.workdps(100):
            root = mp.mpf(result)
            expected = mp.mpf(1e-300)
            self.assertNotEqual(root, 0)
            self.assertLess(abs(root - expected) / abs(expected), mp.mpf(1e-18))

    def test_serialization_failure_does_not_consume_capacity(self):
        target = "astrbot_plugin_mathitools.core.numerical.json.dumps"
        with mock.patch(target, side_effect=TypeError("controlled")):
            for _ in range(2):
                with self.assertRaises(TypeError):
                    _run("findroot", expression="x-2", a=1, b=3)
        result = _run("findroot", expression="x-2", a=1, b=3)
        self.assertEqual(float(result), 2.0)


class DifferentiateTest(unittest.TestCase):
    def test_polynomial_and_exp_derivatives(self):
        self.assertEqual(float(_run("differentiate", expression="x^3", x=2)), 12.0)
        self.assertEqual(float(_run("differentiate", expression="x^3", x=2, n=2)), 12.0)
        result = _run("differentiate", expression="exp(x)", x=1, n=5)
        with mp.workdps(50):
            self.assertLess(
                abs(mp.mpf(result) - mp.e) / mp.e, mp.mpf("1e-18")
            )

    def test_derivative_at_extreme_arguments(self):
        result = _run("differentiate", expression="sin(x)", x=1e308)
        self.assertLess(abs(float(result) - math.cos(1e308)), 1e-12)
        result = _run("differentiate", expression="cos(x)", x=1e-300)
        with mp.workdps(80):
            value = mp.mpf(result)
            reference = -mp.sin(mp.mpf(1e-300))
            self.assertNotEqual(value, 0)
            self.assertLess(abs(value - reference) / abs(reference), mp.mpf(1e-18))

    def test_derivative_keeps_big_int_argument(self):
        big = 10**307
        result = _run("differentiate", expression="(x-10^307)^2", x=big + 1)
        self.assertEqual(float(result), 2.0)
        result = _run("differentiate", expression="exp(-1000)*x+x^3", x=0)
        with mp.workdps(50):
            self.assertLess(
                abs(mp.mpf(result) - mp.exp(mp.mpf(-1000))) / mp.exp(mp.mpf(-1000)),
                mp.mpf("1e-18"),
            )

    def test_derivative_dps80_and_parent_dps_isolated(self):
        before = mp.dps
        with mp.workdps(7):
            result = _run("differentiate", expression="exp(x)", x=1, dps=80)
            self.assertEqual(mp.dps, 7)
        self.assertEqual(mp.dps, before)
        with mp.workdps(100):
            self.assertLess(
                abs(mp.mpf(result) - mp.e) / mp.e, mp.mpf(10) ** -78
            )


class IntegrateTest(unittest.TestCase):
    def test_integral_reversed_and_degenerate(self):
        result = _run("integrate", expression="x^2", a=0, b=1)
        with mp.workdps(50):
            self.assertLess(abs(mp.mpf(result) - mp.mpf(1) / 3), mp.mpf("1e-20"))
        result = _run("integrate", expression="x^2", a=1, b=0)
        self.assertEqual(float(result), -1 / 3)
        result = _run("integrate", expression="x^2", a=2, b=2)
        self.assertEqual(float(result), 0.0)

    def test_integral_split_points(self):
        result = _run("integrate", expression="abs(x)", a=-1, b=1, points=[0])
        self.assertAlmostEqual(float(result), 1.0, places=12)

    def test_integral_two_narrow_peaks_at_default_dps(self):
        result = _run(
            "integrate",
            expression="exp(-100*(x-3)^2)+exp(-100*(x-7)^2)",
            a=0,
            b=10,
            points=[3, 7],
        )
        with mp.workdps(50):
            expected = mp.sqrt(mp.pi) / 5
            self.assertLess(abs(mp.mpf(result) - expected), mp.mpf("1e-19"))

    def test_integral_tiny_interval(self):
        result = _run("integrate", expression="x^2", a=0, b=1e-300)
        with mp.workdps(80):
            expected = mp.mpf(1e-300) ** 3 / 3
            value = mp.mpf(result)
            self.assertNotEqual(value, 0)
            self.assertLess(abs(value - expected) / expected, mp.mpf(1e-18))

    def test_integral_big_int_and_extreme_scale(self):
        big = 10**307
        result = _run("integrate", expression="x-10^307", a=big, b=big + 2)
        self.assertEqual(float(result), 2.0)
        result = _run("integrate", expression="1e308*x+1", a=-1, b=1)
        self.assertAlmostEqual(float(result), 2.0, places=12)
        self.assertNotEqual(float(result), 0.0)

    def test_integral_dps80_and_inputs_unchanged(self):
        before = mp.dps
        points = [0.25, 0.75]
        with mp.workdps(7):
            result = _run(
                "integrate", expression="3*x^2+2*x+1", a=0, b=1,
                points=points, dps=80,
            )
            self.assertEqual(mp.dps, 7)
        self.assertEqual(mp.dps, before)
        self.assertEqual(points, [0.25, 0.75])
        with mp.workdps(100):
            self.assertLess(abs(mp.mpf(result) - 3), mp.mpf(10) ** -78)

    def test_integral_boundary_scale(self):
        maximum = sys.float_info.max
        result = _run(
            "integrate", expression="1e-308*(x/1e308)^2", a=-maximum, b=maximum
        )
        with mp.workdps(100):
            value = mp.mpf(result)
            self.assertNotEqual(value, 0)
            reference = (
                mp.mpf(1e-308) * 2 * mp.mpf(maximum) ** 3 / (3 * mp.mpf(1e308) ** 2)
            )
            self.assertLess(abs(value - reference) / reference, mp.mpf(1e-18))

    def test_integral_narrow_peak_rejected(self):
        with self.assertRaises(MathError) as ctx:
            _run("integrate", expression="exp(-10^8*(x-0.3)^2)", a=0, b=1, points=[0.3])
        self.assertEqual(ctx.exception.error_code, "no_convergence")

    def test_integral_normal_zero_and_constant_zero_ok(self):
        result = _run("integrate", expression="x", a=-1, b=1)
        with mp.workdps(80):
            self.assertLess(abs(mp.mpf(result)), mp.mpf("1e-40"))
        result = _run("integrate", expression="0", a=0, b=1)
        self.assertEqual(float(result), 0.0)


class DifferentiateErrorTest(unittest.TestCase):
    def test_x_argument_errors(self):
        bad_x = [None, "1", True, float("nan"), float("inf"), 10**400]
        for x in bad_x:
            with self.subTest(x=x):
                with self.assertRaises(MathError) as ctx:
                    _run("differentiate", expression="x^2", x=x)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        with self.subTest(case="missing"):
            with self.assertRaises(MathError) as ctx:
                _run("differentiate", expression="x^2")
            self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_n_errors(self):
        bad_n = [None, True, 1.5, 0, -1, 6, "2"]
        for n in bad_n:
            with self.subTest(n=n):
                with self.assertRaises(MathError) as ctx:
                    _run("differentiate", expression="x^3", x=2, n=n)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_no_convergence_and_domain_errors(self):
        cases = [
            (dict(expression="abs(x)", x=0), "no_convergence"),
            (dict(expression="1e-300*abs(x)", x=0), "no_convergence"),
            (dict(expression="x*abs(x)", x=0, n=2), "no_convergence"),
            (dict(expression="sqrt(x)", x=-1), "domain_error"),
        ]
        for kwargs, code in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(MathError) as ctx:
                    _run("differentiate", **kwargs)
                self.assertEqual(ctx.exception.error_code, code)

    def test_big_constant_offset_not_zero_derivative(self):
        result = _run("differentiate", expression="exp(5000)+x", x=0)
        self.assertAlmostEqual(float(result), 1.0, places=12)


class IntegrateErrorTest(unittest.TestCase):
    def test_ab_argument_errors(self):
        bad = [None, True, "1", float("nan"), float("inf"), 10**400]
        cases = [(v, 1) for v in bad] + [(0, v) for v in bad]
        cases += [("missing", 1), (0, "missing")]
        for a, b in cases:
            kwargs = dict(expression="x", a=a, b=b)
            kwargs = {k: v for k, v in kwargs.items() if v != "missing"}
            with self.subTest(a=a, b=b):
                with self.assertRaises(MathError) as ctx:
                    _run("integrate", **kwargs)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_points_errors(self):
        bad_points = [
            None, 5, "x", [True, 0.5], [float("nan")], [float("inf")],
            [10**400], [[0.5]], [0, 0.5], [-0.5, 0.5], [0.5, 0.5],
            [0.7, 0.3],
        ]
        cases = [(p, 0, 1) for p in bad_points]
        cases.append((list(range(1, 34)), 0, 34))
        cases.append(([0.5], 1, 1))
        for points, a, b in cases:
            with self.subTest(points=points, a=a, b=b):
                with self.assertRaises(MathError) as ctx:
                    _run("integrate", expression="x", a=a, b=b, points=points)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_division_by_zero_at_singularity(self):
        with self.assertRaises(MathError) as ctx:
            _run("integrate", expression="1/x", a=0, b=1)
        self.assertEqual(ctx.exception.error_code, "division_by_zero")


class NumericalExpressionErrorTest(unittest.TestCase):
    def test_expression_argument_errors(self):
        cases = [dict(x=1), dict(expression=None, x=1), dict(expression="", x=1)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(MathError) as ctx:
                    _run("differentiate", **kwargs)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        with self.subTest(expression=7):
            with self.assertRaises(MathError) as ctx:
                _run("integrate", expression=7, a=0, b=1)
            self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_expression_too_long_and_unsafe_syntax(self):
        cases = [
            ("x" * 1001, "expression_too_long"),
            ("__import__('os')", "invalid_argument"),
            ("(lambda: 1)()", "invalid_argument"),
            ("x.real", "invalid_argument"),
        ]
        for expression, code in cases:
            with self.subTest(expression=expression):
                with self.assertRaises(MathError) as ctx:
                    _run("differentiate", expression=expression, x=1)
                self.assertEqual(ctx.exception.error_code, code)

    def test_unknown_function_names(self):
        for op, kwargs in (
            ("differentiate", dict(expression="unknown_name(x)", x=1)),
            ("integrate", dict(expression="unknown_name(x)", a=0, b=1)),
        ):
            with self.subTest(operation=op):
                with self.assertRaises(MathError) as ctx:
                    _run(op, **kwargs)
                self.assertEqual(ctx.exception.error_code, "unknown_function")


class PolyRootsTest(unittest.TestCase):
    def test_polyroots_real_roots(self):
        result = _run("polyroots", coefficients=[1, 0, -2])
        self.assertEqual(len(result), 2)
        roots = sorted(
            (complex(r.replace(" ", "")) for r in result), key=lambda z: z.real
        )
        self.assertAlmostEqual(roots[0].real, -2 ** 0.5, places=12)
        self.assertAlmostEqual(roots[1].real, 2 ** 0.5, places=12)
        for root in roots:
            self.assertAlmostEqual(root.imag, 0, places=12)

    def test_polyroots_repeated_real_roots_are_clean(self):
        result = _run("polyroots", coefficients=[1, -6, 11, -6])
        self.assertEqual(sorted(result), ["1", "2", "3"])

    def test_polyroots_small_complex_roots_are_kept(self):
        result = _run("polyroots", coefficients=[1, 0, 1e-40], dps=60)
        self.assertEqual(len(result), 2)
        for entry in result:
            root = complex(entry.replace(" ", ""))
            self.assertNotEqual(root.imag, 0)
            self.assertAlmostEqual(abs(root.imag), 1e-20, places=25)

    def test_polyroots_conjugate_roots(self):
        result = _run("polyroots", coefficients=[1, 0, 1])
        self.assertEqual(len(result), 2)
        for r in result:
            root = complex(r.replace(" ", ""))
            self.assertAlmostEqual(root.real, 0, places=12)
            self.assertAlmostEqual(abs(root.imag), 1, places=12)
        self.assertNotEqual(result[0], result[1])

    def test_polyroots_tiny_and_zero_roots(self):
        result = _run("polyroots", coefficients=[1, -1e-300])
        with mp.workdps(80):
            root = mp.mpf(result[0])
            expected = mp.mpf(1e-300)
            self.assertNotEqual(root, 0)
            self.assertLess(abs(root - expected) / abs(expected), mp.mpf(10) ** -18)
        result = _run("polyroots", coefficients=[1, 0, 0])
        self.assertEqual([mp.mpf(r) for r in result], [mp.mpf(0), mp.mpf(0)])
        self.assertEqual(_run("polyroots", coefficients=[5]), [])


class FindRootErrorTest(unittest.TestCase):
    def test_operation_and_dps_errors(self):
        cases = [
            (dict(expression="x", a=1, b=2, operation=7), "invalid_argument"),
            (dict(expression="x", a=1, b=2, operation="zz"), "unknown_operation"),
            (dict(expression="x", a=1, b=2, operation="findroot", dps=None), "invalid_argument"),
            (dict(expression="x", a=1, b=2, operation="findroot", dps=True), "invalid_argument"),
            (dict(expression="x", a=1, b=2, operation="findroot", dps=0), "invalid_argument"),
            (dict(expression="x", a=1, b=2, operation="findroot", dps=201), "invalid_argument"),
        ]
        for kwargs, code in cases:
            op = kwargs.pop("operation")
            with self.subTest(operation=op, dps=kwargs.get("dps", "default")):
                with self.assertRaises(MathError) as ctx:
                    _run(op, **kwargs)
                self.assertEqual(ctx.exception.error_code, code)

    def test_ab_errors(self):
        bad_a = [None, "1", True, float("nan"), float("inf"), 10**400]
        cases = [(v, 2) for v in bad_a] + [(1, v) for v in bad_a]
        cases += [(2, 1), (1, 1)]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                with self.assertRaises(MathError) as ctx:
                    _run("findroot", expression="x", a=a, b=b)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        for missing in ("a", "b"):
            kwargs = dict(expression="x", a=1, b=2)
            kwargs.pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaises(MathError) as ctx:
                    _run("findroot", **kwargs)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_expression_errors(self):
        cases = [
            (None, "invalid_argument"),
            ("", "invalid_argument"),
            (7, "invalid_argument"),
            ("x" * 1001, "expression_too_long"),
            ("__import__('os')", "invalid_argument"),
            ("(lambda: 1)()", "invalid_argument"),
            ("unknown_name(x)", "unknown_function"),
        ]
        for expression, code in cases:
            with self.subTest(expression=expression):
                with self.assertRaises(MathError) as ctx:
                    _run("findroot", expression=expression, a=1, b=2)
                self.assertEqual(ctx.exception.error_code, code)

    def test_no_sign_change_and_no_convergence(self):
        with self.assertRaises(MathError) as ctx:
            _run("findroot", expression="x^2+1", a=0, b=1)
        self.assertEqual(ctx.exception.error_code, "domain_error")
        with self.assertRaises(MathError) as ctx:
            _run("findroot", expression="floor(x)-0.5", a=0, b=1)
        self.assertEqual(ctx.exception.error_code, "no_convergence")


class PolyRootsErrorTest(unittest.TestCase):
    def test_coefficients_argument_errors(self):
        bad_cases = [
            None,
            [],
            7,
            [1] * 22,
            [1, "x"],
            [1, True],
            [1, float("nan")],
            [1, float("inf")],
            [1, 10**400],
            [0, 1],
        ]
        for coefficients in bad_cases:
            with self.subTest(coefficients=coefficients):
                with self.assertRaises(MathError) as ctx:
                    _run("polyroots", coefficients=coefficients)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        with self.subTest(coefficients=[0]):
            with self.assertRaises(MathError) as ctx:
                _run("polyroots", coefficients=[0])
            self.assertEqual(ctx.exception.error_code, "domain_error")
        with self.subTest(case="missing"):
            with self.assertRaises(MathError) as ctx:
                _run("polyroots")
            self.assertEqual(ctx.exception.error_code, "invalid_argument")
