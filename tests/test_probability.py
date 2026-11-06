"""概率分布公开核心测试（normal/binomial/poisson）。"""

import asyncio
import math
import unittest

from mpmath import mp

from astrbot_plugin_mathitools.core.errors import MathError
from astrbot_plugin_mathitools.core.numerical import numerical_operation


def _run(operation, **params):
    return asyncio.run(numerical_operation(operation, **params))


class NormalProbabilityTest(unittest.TestCase):
    def test_normal_standard_normal(self):
        cases = [
            ("normal_pdf", dict(x=0), 1 / math.sqrt(2 * math.pi)),
            ("normal_cdf", dict(x=0), 0.5),
            ("normal_ppf", dict(q=0.5), 0.0),
        ]
        for operation, params, expected in cases:
            with self.subTest(operation=operation):
                self.assertAlmostEqual(
                    float(_run(operation, **params)), expected, places=12
                )

    def test_normal_mu10_sigma2(self):
        expected_pdf = 1 / (2 * math.sqrt(2 * math.pi))
        result = _run("normal_pdf", mu=10, sigma=2, x=10)
        self.assertAlmostEqual(float(result), expected_pdf, places=12)
        self.assertEqual(float(_run("normal_cdf", mu=10, sigma=2, x=10)), 0.5)
        self.assertEqual(float(_run("normal_ppf", mu=10, sigma=2, q=0.5)), 10.0)

    def test_normal_pdf_dps80_real_precision(self):
        result = _run("normal_pdf", x=0, dps=80)
        with mp.workdps(100):
            reference = 1 / mp.sqrt(2 * mp.pi)
            self.assertLess(abs(mp.mpf(result) - reference) / reference, mp.mpf(10) ** -78)

    def test_normal_ppf_tail_precision(self):
        q = 1e-300
        result = _run("normal_ppf", q=q, dps=80)
        with mp.workdps(400):
            ncdf = (1 + mp.erf(mp.mpf(result) / mp.sqrt(2))) / 2
            self.assertLess(abs(ncdf - mp.mpf(q)) / mp.mpf(q), mp.mpf(10) ** -76)

    def test_normal_pdf_big_int_mu_precision(self):
        mu = 10**307
        result = _run("normal_pdf", mu=mu, x=mu + 1, sigma=3)
        with mp.workdps(100):
            reference = mp.exp(-mp.mpf(1) / 18) / (3 * mp.sqrt(2 * mp.pi))
            value = mp.mpf(result)
            self.assertLess(abs(value - reference) / reference, mp.mpf(1e-18))

    def test_normal_parent_dps_isolated(self):
        before = mp.dps
        with mp.workdps(7):
            _run("normal_pdf", x=0)
            self.assertEqual(mp.dps, 7)
        self.assertEqual(mp.dps, before)

    def test_normal_cdf_near_one_keeps_precision(self):
        result = _run("normal_cdf", x=10, dps=80)
        with mp.workdps(120):
            value = mp.mpf(result)
            expected = 1 - mp.erfc(mp.mpf(10) / mp.sqrt(2)) / 2
            self.assertLess(value, 1)
            self.assertLess(abs(value - expected), mp.mpf("1e-78"))


class BinomialProbabilityTest(unittest.TestCase):
    def test_binomial_n10_p05(self):
        with self.subTest(case="pmf_k5"):
            self.assertEqual(float(_run("binomial_pmf", n=10, p=0.5, k=5)), 0.24609375)
        with self.subTest(case="cdf_k4"):
            self.assertEqual(float(_run("binomial_cdf", n=10, p=0.5, k=4)), 0.376953125)
        with self.subTest(case="ppf_q05"):
            self.assertEqual(float(_run("binomial_ppf", n=10, p=0.5, q=0.5)), 5.0)
        with self.subTest(case="ppf_exact_boundary"):
            self.assertEqual(
                float(_run("binomial_ppf", n=10, p=0.5, q=0.376953125)), 4.0
            )

    def test_binomial_degenerate_and_out_of_range(self):
        cases = [
            ("binomial_pmf", dict(n=10, p=0.0, k=0), 1.0),
            ("binomial_pmf", dict(n=10, p=1.0, k=10), 1.0),
            ("binomial_cdf", dict(n=10, p=0.0, k=0), 1.0),
            ("binomial_cdf", dict(n=10, p=1.0, k=10), 1.0),
            ("binomial_cdf", dict(n=10, p=1.0, k=0), 0.0),
            ("binomial_cdf", dict(n=10, p=1.0, k=9), 0.0),
            ("binomial_pmf", dict(n=0, p=0.5, k=0), 1.0),
            ("binomial_pmf", dict(n=10, p=0.5, k=-1), 0.0),
            ("binomial_cdf", dict(n=10, p=0.5, k=-1), 0.0),
            ("binomial_pmf", dict(n=10, p=0.5, k=11), 0.0),
            ("binomial_cdf", dict(n=10, p=0.5, k=11), 1.0),
        ]
        for operation, params, expected in cases:
            with self.subTest(operation=operation, **params):
                self.assertEqual(float(_run(operation, **params)), expected)

    def test_binomial_ppf_degenerate_q(self):
        cases = [
            (dict(n=10, p=0.5, q=0.0), 0.0),
            (dict(n=10, p=0.5, q=1.0), 10.0),
            (dict(n=10, p=0.0, q=1.0), 0.0),
        ]
        for params, expected in cases:
            with self.subTest(**params):
                self.assertEqual(float(_run("binomial_ppf", **params)), expected)

    def test_binomial_pmf_tiny_p_high_precision(self):
        p = 1e-300
        result = _run("binomial_pmf", n=2, p=p, k=1)
        with mp.workdps(400):
            value = mp.mpf(result)
            expected = 2 * mp.mpf(p) * (1 - mp.mpf(p))
            self.assertNotEqual(value, 0)
            self.assertLess(abs(value - expected) / expected, mp.mpf(1e-18))

    def test_binomial_pmf_subnormal_probability_at_size_limit(self):
        n = 10000
        p = math.ulp(0.0)
        result = _run("binomial_pmf", n=n, p=p, k=n, dps=200)
        with mp.workdps(250):
            value = mp.mpf(result)
            expected = mp.mpf(p) ** n
            self.assertGreater(value, 0)
            self.assertLess(abs(value / expected - 1), mp.mpf("1e-198"))


class PoissonProbabilityTest(unittest.TestCase):
    def test_poisson_rate3(self):
        with self.subTest(case="pmf_k0"):
            self.assertAlmostEqual(
                float(_run("poisson_pmf", rate=3, k=0)), math.exp(-3), places=12
            )
        with self.subTest(case="cdf_k2"):
            expected = (1 + 3 + 9 / 2) * math.exp(-3)
            self.assertAlmostEqual(
                float(_run("poisson_cdf", rate=3, k=2)), expected, places=12
            )
        with self.subTest(case="ppf_q05"):
            self.assertEqual(float(_run("poisson_ppf", rate=3, q=0.5)), 3.0)
        with self.subTest(case="ppf_q0"):
            self.assertEqual(float(_run("poisson_ppf", rate=3, q=0.0)), 0.0)

    def test_poisson_rate0_degenerate(self):
        cases = [
            ("poisson_pmf", dict(k=0), 1.0),
            ("poisson_pmf", dict(k=1), 0.0),
            ("poisson_cdf", dict(k=-1), 0.0),
            ("poisson_cdf", dict(k=0), 1.0),
            ("poisson_cdf", dict(k=1), 1.0),
            ("poisson_cdf", dict(k=10), 1.0),
            ("poisson_ppf", dict(q=1.0), 0.0),
        ]
        for operation, params, expected in cases:
            with self.subTest(operation=operation, **params):
                result = _run(operation, rate=0, **params)
                self.assertEqual(float(result), expected)

    def test_poisson_negative_k_and_tiny_rate(self):
        with self.subTest(case="negative_k_pmf"):
            self.assertEqual(float(_run("poisson_pmf", rate=3, k=-1)), 0.0)
        with self.subTest(case="negative_k_cdf"):
            self.assertEqual(float(_run("poisson_cdf", rate=3, k=-1)), 0.0)
        rate = math.ulp(0.0)
        result = _run("poisson_pmf", rate=rate, k=1)
        with mp.workdps(400):
            value = mp.mpf(result)
            expected = mp.mpf(rate) * mp.exp(-mp.mpf(rate))
            self.assertNotEqual(value, 0)
            self.assertLess(abs(value - expected) / expected, mp.mpf(1e-18))


class ProbabilityValidationTest(unittest.TestCase):
    def test_normal_param_errors(self):
        bad_sigma = [0, -1, None, True, float("nan"), float("inf")]
        for sigma in bad_sigma:
            with self.subTest(sigma=sigma):
                with self.assertRaises(MathError) as ctx:
                    _run("normal_pdf", mu=0, sigma=sigma, x=1)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        bad_mu = [None, True, float("nan"), float("inf")]
        for mu in bad_mu:
            with self.subTest(mu=mu):
                with self.assertRaises(MathError) as ctx:
                    _run("normal_cdf", mu=mu, sigma=1, x=1)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        with self.subTest(case="missing_x"):
            with self.assertRaises(MathError) as ctx:
                _run("normal_pdf", mu=0, sigma=1)
            self.assertEqual(ctx.exception.error_code, "invalid_argument")
        bad_q = [-0.1, 1.1, None, True]
        for q in bad_q:
            with self.subTest(q=q):
                with self.assertRaises(MathError) as ctx:
                    _run("normal_ppf", mu=0, sigma=1, q=q)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_binomial_param_errors(self):
        with self.subTest(case="missing"):
            for missing in ("n", "p", "k"):
                kwargs = dict(n=10, p=0.5, k=5)
                kwargs.pop(missing)
                with self.subTest(missing=missing):
                    with self.assertRaises(MathError) as ctx:
                        _run("binomial_pmf", **kwargs)
                    self.assertEqual(ctx.exception.error_code, "invalid_argument")
        bad_n = [-1, 10001, True, 1.5]
        for n in bad_n:
            with self.subTest(n=n):
                with self.assertRaises(MathError) as ctx:
                    _run("binomial_pmf", n=n, p=0.5, k=1)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        bad_p = [-0.1, 1.1, None, True]
        for p in bad_p:
            with self.subTest(p=p):
                with self.assertRaises(MathError) as ctx:
                    _run("binomial_cdf", n=10, p=p, k=5)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_k_and_q_errors(self):
        bad_k = [True, 1.5, 100001, -100001]
        for k in bad_k:
            with self.subTest(k=k):
                with self.assertRaises(MathError) as ctx:
                    _run("binomial_pmf", n=10, p=0.5, k=k)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        bad_q = [-0.1, 1.1, None, True]
        for q in bad_q:
            with self.subTest(q=q):
                with self.assertRaises(MathError) as ctx:
                    _run("binomial_ppf", n=10, p=0.5, q=q)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        with self.subTest(case="normal_ppf_q_endpoints"):
            for q in (0.0, 1.0):
                with self.assertRaises(MathError) as ctx:
                    _run("normal_ppf", mu=0, sigma=1, q=q)
                self.assertEqual(ctx.exception.error_code, "domain_error")
        with self.subTest(case="poisson_ppf_q1"):
            with self.assertRaises(MathError) as ctx:
                _run("poisson_ppf", rate=1, q=1.0)
            self.assertEqual(ctx.exception.error_code, "domain_error")

    def test_poisson_param_errors(self):
        bad_rate = [-1, 10001, None, True]
        for rate in bad_rate:
            with self.subTest(rate=rate):
                with self.assertRaises(MathError) as ctx:
                    _run("poisson_cdf", rate=rate, k=2)
                self.assertEqual(ctx.exception.error_code, "invalid_argument")
        with self.subTest(case="missing"):
            for missing in ("rate", "k"):
                kwargs = dict(rate=3, k=2)
                kwargs.pop(missing)
                with self.subTest(missing=missing):
                    with self.assertRaises(MathError) as ctx:
                        _run("poisson_cdf", **kwargs)
                    self.assertEqual(ctx.exception.error_code, "invalid_argument")
