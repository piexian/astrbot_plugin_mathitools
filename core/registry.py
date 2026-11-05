"""静态注册表：73 函数名、7 常量名、15 类单位、153 别名、运算符与禁用模式。

数据转录自上游 registry.json，是本插件全部函数/单位清单的唯一来源。
"""

import ast
import operator
from collections.abc import Callable

MATH_FUNCTIONS: tuple[str, ...] = (
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "log", "log10", "log2", "exp", "sqrt", "pow", "abs", "fabs",
    "factorial", "ceil", "floor", "trunc",
    "degrees", "radians", "gcd", "lcm", "isqrt", "hypot",
    "copysign", "fmod", "remainder", "modf", "frexp", "ldexp",
    "isfinite", "isinf", "isnan", "isclose",
    "comb", "perm", "erf", "erfc", "gamma", "lgamma",
    "cbrt", "exp2", "expm1", "log1p", "nextafter", "ulp",
)

STATISTICS_FUNCTIONS: tuple[str, ...] = (
    "mean", "median", "mode", "stdev", "pstdev",
    "variance", "pvariance", "harmonic_mean", "geometric_mean", "quantiles",
)

COMPLEX_FUNCTIONS: tuple[str, ...] = (
    "phase", "polar", "rect", "csin", "ccos", "ctan",
    "cexp", "clog", "clog10", "csqrt",
)

ALL_FUNCTION_NAMES: tuple[str, ...] = (
    MATH_FUNCTIONS + STATISTICS_FUNCTIONS + COMPLEX_FUNCTIONS
)

# root 不在 73 函数清单内，为满足约束卡示例 root(-8,3) 额外注入 mpmath.root。
EXTRA_FUNCTION_NAMES: tuple[str, ...] = ("root",)

CONSTANT_NAMES: tuple[str, ...] = ("pi", "e", "tau", "inf", "nan", "phi", "euler")

OPERATORS: dict[type[ast.AST], Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.BitXor: operator.xor,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"import\s+",
    r"exec\s*\(",
    r"eval\s*\(",
    r"__.*__",
    r"globals\s*\(",
    r"locals\s*\(",
    r"getattr\s*\(",
    r"setattr\s*\(",
    r"delattr\s*\(",
    r"hasattr\s*\(",
)

STATISTICS_OPERATIONS: tuple[str, ...] = STATISTICS_FUNCTIONS

MATRIX_OPERATIONS: tuple[str, ...] = (
    "multiply", "determinant", "inverse", "transpose", "rank",
)

NUMBER_THEORY_OPERATIONS: tuple[str, ...] = (
    "is_prime", "prime_factors", "divisors", "totient",
)

FINANCE_OPERATIONS: tuple[str, ...] = (
    "percentage", "percentage_change", "tip", "compound_interest",
    "simple_interest", "loan_payment", "split_bill", "tax",
)

# 线性类单位：数值为「相对基准单位的倍率」；温度为「换算到开尔文」的函数。
UNIT_CONVERSIONS: dict[str, dict[str, object]] = {
    "length": {
        "m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
        "mi": 1609.344, "yd": 0.9144, "ft": 0.3048, "in": 0.0254,
        "nmi": 1852.0, "ly": 9460700000000000.0, "AU": 149600000000.0,
        "pc": 3.0857e16, "angstrom": 1e-10, "micron": 1e-06, "nm": 1e-09,
    },
    "mass": {
        "kg": 1.0, "g": 0.001, "mg": 1e-06, "lb": 0.453592,
        "oz": 0.0283495, "ton": 1000.0, "t": 1000.0, "ton_us": 907.185,
        "ton_uk": 1016.05, "st": 6.35029, "ct": 0.0002,
        "gr": 6.47989e-05, "amu": 1.66054e-27,
    },
    "time": {
        "s": 1.0, "ms": 0.001, "us": 1e-06, "ns": 1e-09,
        "min": 60.0, "h": 3600.0, "d": 86400.0, "wk": 604800.0,
        "mo": 2592000.0, "yr": 31536000.0, "decade": 315360000.0,
        "century": 3153600000.0, "ps": 1e-12, "fortnight": 1209600.0,
        "millennium": 31536000000.0,
    },
    "temperature": {
        "K": lambda x: x,
        "C": lambda x: x + 273.15,
        "F": lambda x: (x - 32) * 5 / 9 + 273.15,
    },
    "area": {
        "m2": 1.0, "km2": 1000000.0, "cm2": 0.0001, "mm2": 1e-06,
        "ft2": 0.092903, "yd2": 0.836127, "in2": 0.00064516,
        "mi2": 2590000.0, "acre": 4046.86, "hectare": 10000.0,
        "are": 100.0, "sqch": 404.686,
    },
    "volume": {
        "L": 0.001, "mL": 1e-06, "m3": 1.0, "cm3": 1e-06,
        "gal": 0.00378541, "qt": 0.000946353, "pt": 0.000473176,
        "fl_oz": 2.95735e-05, "gal_uk": 0.00454609, "qt_uk": 0.00113652,
        "pt_uk": 0.000568261, "cup": 0.000236588, "tbsp": 1.47868e-05,
        "tsp": 4.92892e-06, "ft3": 0.0283168, "in3": 1.63871e-05,
    },
    "speed": {
        "m/s": 1.0, "km/h": 0.277778, "mph": 0.44704, "ft/s": 0.3048,
        "knot": 0.514444, "mach": 340.29, "cm/s": 0.01,
        "mi/min": 26.8224, "in/s": 0.0254, "c": 299792458.0,
    },
    "data": {
        "B": 1.0, "KB": 1000.0, "MB": 1000000.0, "GB": 1000000000.0,
        "TB": 1000000000000.0, "PB": 1000000000000000.0, "EB": 1e18,
        "ZB": 1e21, "bit": 0.125, "Kbit": 125.0, "Mbit": 125000.0,
        "Gbit": 125000000.0, "Tbit": 125000000000.0, "KiB": 1024.0,
        "MiB": 1048576.0, "GiB": 1073741824.0,
    },
    "pressure": {
        "Pa": 1.0, "kPa": 1000.0, "MPa": 1000000.0, "atm": 101325.0,
        "bar": 100000.0, "mbar": 100.0, "psi": 6894.76,
        "torr": 133.322, "mmHg": 133.322, "inHg": 3386.39,
    },
    "energy": {
        "J": 1.0, "kJ": 1000.0, "MJ": 1000000.0, "cal": 4.184,
        "kcal": 4184.0, "Wh": 3600.0, "kWh": 3600000.0, "BTU": 1055.06,
        "eV": 1.60218e-19, "ft_lb": 1.35582, "erg": 1e-07,
        "therm": 105500000.0,
    },
    "power": {
        "W": 1.0, "kW": 1000.0, "MW": 1000000.0, "hp": 745.7,
        "PS": 735.499, "BTU/h": 0.293071, "ft_lb/s": 1.35582,
        "cal/s": 4.184, "erg/s": 1e-07, "ton_refrigeration": 3516.85,
    },
    "force": {
        "N": 1.0, "kN": 1000.0, "lbf": 4.44822, "kgf": 9.80665,
        "dyne": 1e-05, "pdl": 0.138255, "ozf": 0.278014, "tonf": 9806.65,
    },
    "angle": {
        "deg": 1.0, "rad": 57.2958, "grad": 0.9, "arcmin": 0.0166667,
        "arcsec": 0.000277778, "turn": 360.0,
    },
    "frequency": {
        "Hz": 1.0, "kHz": 1000.0, "MHz": 1000000.0, "GHz": 1000000000.0,
        "rpm": 0.0166667, "rad/s": 0.159155,
    },
    "fuel_economy": {
        "mpg": 1.0, "mpg_uk": 1.20095, "L/100km": 235.215,
        "km/L": 2.35215, "mi/L": 3.78541, "gal/100mi": 100.0,
    },
}

UNIT_ALIASES: dict[str, str] = {
    "meters": "m", "meter": "m", "metres": "m", "metre": "m",
    "kilometers": "km", "kilometer": "km", "kilometres": "km", "kilometre": "km",
    "centimeters": "cm", "centimeter": "cm", "centimetres": "cm", "centimetre": "cm",
    "millimeters": "mm", "millimeter": "mm", "millimetres": "mm", "millimetre": "mm",
    "miles": "mi", "mile": "mi", "yards": "yd", "yard": "yd",
    "feet": "ft", "foot": "ft", "inches": "in", "inch": "in",
    "kilograms": "kg", "kilogram": "kg", "grams": "g", "gram": "g",
    "milligrams": "mg", "milligram": "mg", "pounds": "lb", "pound": "lb",
    "lbs": "lb", "ounces": "oz", "ounce": "oz", "tons": "ton",
    "seconds": "s", "second": "s", "sec": "s", "secs": "s",
    "milliseconds": "ms", "millisecond": "ms", "millisec": "ms", "millisecs": "ms",
    "microseconds": "us", "microsecond": "us", "microsec": "us", "microsecs": "us",
    "nanoseconds": "ns", "nanosecond": "ns", "nanosec": "ns", "nanosecs": "ns",
    "minutes": "min", "minute": "min", "mins": "min",
    "hours": "h", "hour": "h", "hrs": "h", "hr": "h",
    "days": "d", "day": "d", "weeks": "wk", "week": "wk",
    "months": "mo", "month": "mo", "years": "yr", "year": "yr",
    "yrs": "yr", "decades": "decade", "centuries": "century",
    "celsius": "C", "centigrade": "C", "fahrenheit": "F", "fahr": "F", "kelvin": "K",
    "liters": "L", "liter": "L", "litres": "L", "litre": "L",
    "milliliters": "mL", "milliliter": "mL", "millilitres": "mL", "millilitre": "mL",
    "gallons": "gal", "gallon": "gal", "quarts": "qt", "quart": "qt",
    "pints": "pt", "pint": "pt", "cups": "cup",
    "tablespoons": "tbsp", "tablespoon": "tbsp", "teaspoons": "tsp", "teaspoon": "tsp",
    "bytes": "B", "byte": "B", "kilobytes": "KB", "kilobyte": "KB",
    "megabytes": "MB", "megabyte": "MB", "gigabytes": "GB", "gigabyte": "GB",
    "terabytes": "TB", "terabyte": "TB", "petabytes": "PB", "petabyte": "PB",
    "bits": "bit", "kilobits": "Kbit", "kilobit": "Kbit",
    "megabits": "Mbit", "megabit": "Mbit", "gigabits": "Gbit", "gigabit": "Gbit",
    "pascals": "Pa", "pascal": "Pa", "kilopascals": "kPa", "kilopascal": "kPa",
    "megapascals": "MPa", "megapascal": "MPa", "atmospheres": "atm", "atmosphere": "atm",
    "bars": "bar", "millibars": "mbar", "millibar": "mbar",
    "joules": "J", "joule": "J", "kilojoules": "kJ", "kilojoule": "kJ",
    "megajoules": "MJ", "megajoule": "MJ", "calories": "cal", "calorie": "cal",
    "kilocalories": "kcal", "kilocalorie": "kcal",
    "watts": "W", "watt": "W", "kilowatts": "kW", "kilowatt": "kW",
    "megawatts": "MW", "megawatt": "MW", "horsepower": "hp",
    "newtons": "N", "newton": "N", "kilonewtons": "kN", "kilonewton": "kN",
    "degrees": "deg", "degree": "deg", "radians": "rad", "radian": "rad",
    "hertz": "Hz", "kilohertz": "kHz", "megahertz": "MHz", "gigahertz": "GHz",
}
