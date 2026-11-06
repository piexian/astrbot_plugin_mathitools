"""统计、矩阵、数论、金融、单位运算测试。"""

import unittest

from astrbot_plugin_mathitools.core.errors import MathError
from astrbot_plugin_mathitools.core.finance import finance_operation
from astrbot_plugin_mathitools.core.matrix import matrix_operation
from astrbot_plugin_mathitools.core.number_theory import number_theory_operation
from astrbot_plugin_mathitools.core.statistics import calculate_statistics
from astrbot_plugin_mathitools.core.units import convert_natural_language, convert_unit


class StatisticsTest(unittest.TestCase):
    def test_central_tendency(self):
        self.assertEqual(calculate_statistics([1, 2, 3, 4, 5], "mean"), 3)
        self.assertEqual(calculate_statistics([1, 2, 3, 10], "median"), 2.5)
        self.assertEqual(calculate_statistics([1, 2, 2, 3], "mode"), 2)

    def test_dispersion(self):
        self.assertAlmostEqual(calculate_statistics([1, 2, 3], "variance"), 1.0, places=12)
        self.assertAlmostEqual(calculate_statistics([1, 2, 3], "pvariance"), 2 / 3, places=12)
        self.assertAlmostEqual(calculate_statistics([1, 2, 3], "stdev"), 1.0, places=12)
        self.assertAlmostEqual(calculate_statistics([1, 2, 3], "pstdev"), (2 / 3) ** 0.5, places=12)

    def test_other_averages(self):
        self.assertAlmostEqual(calculate_statistics([1, 2, 4], "harmonic_mean"), 12 / 7, places=12)
        self.assertEqual(calculate_statistics([1, 4], "geometric_mean"), 2.0)
        self.assertEqual(calculate_statistics([1, 2, 3, 4], "quantiles"), [1.25, 2.5, 3.75])

    def test_too_few_points_is_domain_error(self):
        with self.assertRaises(MathError) as ctx:
            calculate_statistics([1], "stdev")
        self.assertEqual(ctx.exception.error_code, "domain_error")

    def test_empty_data_rejected(self):
        with self.assertRaises(MathError) as ctx:
            calculate_statistics([], "mean")
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_unknown_operation_lists_options(self):
        with self.assertRaises(MathError) as ctx:
            calculate_statistics([1], "zz")
        self.assertEqual(ctx.exception.error_code, "unknown_operation")
        self.assertIn("mean", ctx.exception.valid_options)

    def test_non_numeric_rejected(self):
        with self.assertRaises(MathError) as ctx:
            calculate_statistics(["a"], "mean")
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_non_finite_bool_and_overflow_rejected(self):
        bad_data = [
            [1, float("nan")],
            [1, float("inf")],
            [1, float("-inf")],
            [1, True],
            [1, [2]],
            [10**400],  # 超出浮点表示范围
        ]
        for data in bad_data:
            with self.assertRaises(MathError) as ctx:
                calculate_statistics(data, "mean")
            self.assertEqual(ctx.exception.error_code, "invalid_argument")


class MatrixTest(unittest.TestCase):
    def test_multiply(self):
        result = matrix_operation([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "multiply")
        self.assertEqual(result, [[19.0, 22.0], [43.0, 50.0]])

    def test_multiply_dimension_mismatch(self):
        with self.assertRaises(MathError) as ctx:
            matrix_operation([[[1, 2], [3, 4]], [[1, 2], [3, 4], [5, 6]]], "multiply")
        self.assertEqual(ctx.exception.error_code, "matrix_dimension")

    def test_determinant(self):
        self.assertEqual(matrix_operation([[[1, 2], [3, 4]]], "determinant"), -2.0)
        self.assertEqual(matrix_operation([[[4, 7], [2, 6]]], "determinant"), 10.0)

    def test_inverse_round_trip(self):
        matrix = [[4, 7], [2, 6]]
        inverse = matrix_operation([matrix], "inverse")
        product = matrix_operation([matrix, inverse], "multiply")
        for i, row in enumerate(product):
            for j, value in enumerate(row):
                self.assertAlmostEqual(value, 1.0 if i == j else 0.0, places=9)

    def test_singular_matrix(self):
        with self.assertRaises(MathError) as ctx:
            matrix_operation([[[1, 2], [2, 4]]], "inverse")
        self.assertEqual(ctx.exception.error_code, "singular_matrix")

    def test_transpose(self):
        self.assertEqual(
            matrix_operation([[[1, 2, 3], [4, 5, 6]]], "transpose"),
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        )

    def test_rank(self):
        self.assertEqual(matrix_operation([[[1, 2], [3, 4]]], "rank"), 2)
        self.assertEqual(matrix_operation([[[1, 2], [2, 4]]], "rank"), 1)

    def test_ragged_matrix_rejected(self):
        with self.assertRaises(MathError) as ctx:
            matrix_operation([[[1, 2], [3]]], "rank")
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_non_numeric_rejected(self):
        with self.assertRaises(MathError) as ctx:
            matrix_operation([[["a"]]], "rank")
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_non_finite_and_bool_elements_rejected(self):
        for matrix in (
            [[1, float("nan")], [3, 4]],
            [[1, float("inf")], [3, 4]],
            [[1, True], [3, 4]],
        ):
            with self.assertRaises(MathError) as ctx:
                matrix_operation([matrix], "determinant")
            self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_unknown_operation(self):
        with self.assertRaises(MathError) as ctx:
            matrix_operation([[[1]]], "zz")
        self.assertEqual(ctx.exception.error_code, "unknown_operation")


class NumberTheoryTest(unittest.TestCase):
    def test_is_prime(self):
        self.assertTrue(number_theory_operation(97, "is_prime"))
        self.assertTrue(number_theory_operation(2, "is_prime"))
        self.assertFalse(number_theory_operation(1, "is_prime"))
        self.assertFalse(number_theory_operation(561, "is_prime"))  # Carmichael 数
        self.assertFalse(number_theory_operation(-7, "is_prime"))

    def test_is_prime_large(self):
        self.assertTrue(number_theory_operation(2**61 - 1, "is_prime"))
        self.assertTrue(number_theory_operation(2**64 + 13, "is_prime"))

    def test_prime_factors(self):
        self.assertEqual(number_theory_operation(360, "prime_factors"), [2, 2, 2, 3, 3, 5])
        self.assertEqual(
            number_theory_operation(10**12, "prime_factors"), [2] * 12 + [5] * 12
        )
        self.assertEqual(number_theory_operation(1, "prime_factors"), [])

    def test_divisors(self):
        self.assertEqual(number_theory_operation(12, "divisors"), [1, 2, 3, 4, 6, 12])
        self.assertEqual(number_theory_operation(1, "divisors"), [1])

    def test_totient(self):
        self.assertEqual(number_theory_operation(36, "totient"), 12)
        self.assertEqual(number_theory_operation(100, "totient"), 40)
        self.assertEqual(number_theory_operation(1, "totient"), 1)

    def test_non_integer_rejected(self):
        with self.assertRaises(MathError) as ctx:
            number_theory_operation(3.5, "is_prime")
        self.assertEqual(ctx.exception.error_code, "non_integer")

    def test_non_positive_rejected(self):
        with self.assertRaises(MathError) as ctx:
            number_theory_operation(-5, "prime_factors")
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_unknown_operation(self):
        with self.assertRaises(MathError) as ctx:
            number_theory_operation(5, "zz")
        self.assertEqual(ctx.exception.error_code, "unknown_operation")


class FinanceTest(unittest.TestCase):
    def test_percentage(self):
        self.assertAlmostEqual(
            finance_operation("percentage", value=200, percentage_value=15), 30.0, places=12
        )

    def test_percentage_change(self):
        self.assertAlmostEqual(
            finance_operation("percentage_change", old_value=100, new_value=120), 20.0, places=12
        )
        with self.assertRaises(MathError) as ctx:
            finance_operation("percentage_change", old_value=0, new_value=5)
        self.assertEqual(ctx.exception.error_code, "division_by_zero")

    def test_tip(self):
        result = finance_operation("tip", amount=100, tip_percent=15)
        self.assertAlmostEqual(result["tip_amount"], 15.0, places=12)
        self.assertAlmostEqual(result["total_amount"], 115.0, places=12)

    def test_compound_interest(self):
        result = finance_operation(
            "compound_interest", principal=1000, rate=5, time=10, compounds_per_year=4
        )
        self.assertAlmostEqual(result["amount"], 1643.6195, places=3)
        with self.assertRaises(MathError) as ctx:
            finance_operation("compound_interest", principal=1, rate=1, time=1, compounds_per_year=0)
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_simple_interest(self):
        result = finance_operation("simple_interest", principal=1000, rate=5, time=2)
        self.assertAlmostEqual(result["amount"], 1100.0, places=12)

    def test_loan_payment(self):
        result = finance_operation(
            "loan_payment", principal=300000, annual_rate=3.5, years=30, payments_per_year=12
        )
        self.assertAlmostEqual(result["payment"], 1347.134063, places=4)
        zero = finance_operation("loan_payment", principal=1200, annual_rate=0, years=1)
        self.assertAlmostEqual(zero["payment"], 100.0, places=12)
        with self.assertRaises(MathError) as ctx:
            finance_operation("loan_payment", principal=1, annual_rate=1, years=0)
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_split_bill(self):
        result = finance_operation("split_bill", total=100, people=3, tip_percent=10)
        self.assertAlmostEqual(result["per_person"], 36.666667, places=6)
        with self.assertRaises(MathError) as ctx:
            finance_operation("split_bill", total=100, people=0)
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_tax(self):
        exclusive = finance_operation("tax", amount=100, tax_rate=20)
        self.assertAlmostEqual(exclusive["tax_amount"], 20.0, places=12)
        inclusive = finance_operation("tax", amount=100, tax_rate=20, is_inclusive=True)
        self.assertAlmostEqual(inclusive["base_amount"], 83.333333, places=6)
        self.assertAlmostEqual(inclusive["total_amount"], 100.0, places=12)
        with self.assertRaises(MathError) as ctx:
            finance_operation("tax", amount=100, tax_rate=-100, is_inclusive=True)
        self.assertEqual(ctx.exception.error_code, "division_by_zero")

    def test_unknown_operation(self):
        with self.assertRaises(MathError) as ctx:
            finance_operation("zz")
        self.assertEqual(ctx.exception.error_code, "unknown_operation")

    def test_missing_params_rejected(self):
        with self.assertRaises(MathError) as ctx:
            finance_operation("tip", amount=100)
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_string_and_non_finite_params_rejected(self):
        for amount in ("100", float("nan"), float("inf"), True):
            with self.assertRaises(MathError) as ctx:
                finance_operation("tip", amount=amount, tip_percent=15)
            self.assertEqual(ctx.exception.error_code, "invalid_argument")
        with self.assertRaises(MathError) as ctx:
            finance_operation("compound_interest", principal="1000", rate=5, time=2)
        self.assertEqual(ctx.exception.error_code, "invalid_argument")

class UnitsTest(unittest.TestCase):
    def test_convert_unit_rejects_non_finite_and_bool(self):
        for value in (float("nan"), float("inf"), float("-inf"), True):
            with self.assertRaises(MathError) as ctx:
                convert_unit(value, "km", "mi", "length")
            self.assertEqual(ctx.exception.error_code, "invalid_argument")

    def test_natural_language_overflow_rejected(self):
        for query in ("1e309 km to miles", "-1e309 celsius to fahrenheit"):
            with self.assertRaises(MathError) as ctx:
                convert_natural_language(query)
            self.assertEqual(ctx.exception.error_code, "invalid_argument")

if __name__ == "__main__":
    unittest.main()
