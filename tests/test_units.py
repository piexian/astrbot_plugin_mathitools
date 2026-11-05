"""单位换算与自然语言解析测试。"""

import unittest

from astrbot_plugin_mathitools.core.errors import MathError
from astrbot_plugin_mathitools.core.units import convert_natural_language, convert_unit


class ConvertUnitTest(unittest.TestCase):
    def test_linear_conversions(self):
        self.assertAlmostEqual(convert_unit(5, "km", "mi", "length"), 3.106856, places=6)
        self.assertAlmostEqual(convert_unit(1, "kg", "g", "mass"), 1000.0, places=9)
        self.assertAlmostEqual(convert_unit(2, "GB", "MB", "data"), 2000.0, places=9)
        self.assertAlmostEqual(convert_unit(1, "atm", "kPa", "pressure"), 101.325, places=6)

    def test_same_unit_is_identity(self):
        self.assertEqual(convert_unit(42, "m", "m", "length"), 42)

    def test_aliases(self):
        direct = convert_unit(5, "km", "mi", "length")
        self.assertEqual(convert_unit(5, "kilometers", "miles", "length"), direct)
        self.assertEqual(convert_unit(1, "hour", "seconds", "time"), 3600)

    def test_case_insensitive(self):
        self.assertAlmostEqual(convert_unit(5, "KM", "Mi", "length"), 3.106856, places=6)

    def test_temperature(self):
        self.assertAlmostEqual(convert_unit(212, "F", "C", "temperature"), 100.0, places=9)
        self.assertAlmostEqual(convert_unit(0, "C", "K", "temperature"), 273.15, places=9)
        self.assertAlmostEqual(convert_unit(0, "K", "F", "temperature"), -459.67, places=6)
        self.assertAlmostEqual(
            convert_unit(convert_unit(25, "C", "F", "temperature"), "F", "C", "temperature"),
            25.0,
            places=9,
        )

    def test_fuel_economy(self):
        self.assertAlmostEqual(convert_unit(25, "mpg", "L/100km", "fuel_economy"), 9.4086, places=3)
        self.assertAlmostEqual(
            convert_unit(9.4086, "L/100km", "mpg", "fuel_economy"), 25.0, places=2
        )
        self.assertAlmostEqual(
            convert_unit(25, "mpg", "mpg_uk", "fuel_economy"), 20.8169, places=3
        )
        self.assertAlmostEqual(convert_unit(6, "gal/100mi", "mpg", "fuel_economy"), 100 / 6, places=6)

    def test_unknown_unit(self):
        with self.assertRaises(MathError) as ctx:
            convert_unit(1, "km", "zz", "length")
        self.assertEqual(ctx.exception.error_code, "unknown_unit")
        self.assertIn("km", ctx.exception.valid_options)

    def test_unknown_category(self):
        with self.assertRaises(MathError) as ctx:
            convert_unit(1, "km", "m", "zzz")
        self.assertEqual(ctx.exception.error_code, "unknown_unit")

    def test_zero_fuel_value_rejected(self):
        with self.assertRaises(MathError) as ctx:
            convert_unit(0, "L/100km", "mpg", "fuel_economy")
        self.assertEqual(ctx.exception.error_code, "division_by_zero")


class NaturalLanguageTest(unittest.TestCase):
    def test_basic_query(self):
        result = convert_natural_language("5 km to miles")
        self.assertEqual(result["category"], "length")
        self.assertEqual(result["from_unit"], "km")
        self.assertEqual(result["to_unit"], "mi")
        self.assertAlmostEqual(result["result"], 3.106856, places=6)

    def test_convert_prefix(self):
        self.assertEqual(
            convert_natural_language("convert 100 celsius to fahrenheit")["result"], 212.0
        )

    def test_degree_sign_and_tight_spacing(self):
        self.assertEqual(convert_natural_language("25°C to F")["result"], 77.0)

    def test_alternative_connectors(self):
        self.assertEqual(convert_natural_language("100 min in s")["result"], 6000.0)
        self.assertEqual(convert_natural_language("2TB to GB")["result"], 2000.0)

    def test_unparseable_query(self):
        with self.assertRaises(MathError) as ctx:
            convert_natural_language("hello world")
        self.assertEqual(ctx.exception.error_code, "parse_failed")

    def test_cross_category_rejected(self):
        with self.assertRaises(MathError) as ctx:
            convert_natural_language("5 km to kg")
        self.assertEqual(ctx.exception.error_code, "parse_failed")

    def test_missing_value_rejected(self):
        with self.assertRaises(MathError) as ctx:
            convert_natural_language("km to miles")
        self.assertEqual(ctx.exception.error_code, "parse_failed")


if __name__ == "__main__":
    unittest.main()
