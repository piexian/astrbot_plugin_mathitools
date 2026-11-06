"""数值求根 findroot/polyroots 公开核心测试。"""

import asyncio
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
