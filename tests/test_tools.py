"""工具契约测试：sys.modules 注入最小 astrbot.api 桩，不导入真实 AstrBot。"""

import asyncio
import json
import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _StubFunctionTool:
    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)
    handler: Any = None
    handler_module_path: Any = None
    active: bool = True
    is_background_task: bool = False


class _StubLogger:
    def __init__(self):
        self.errors = []
        self.infos = []

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))

    def info(self, message, *args, **kwargs):
        self.infos.append(str(message))

    def warning(self, message, *args, **kwargs):
        self.infos.append(str(message))


_stub_api = types.ModuleType("astrbot.api")
_stub_api.FunctionTool = _StubFunctionTool
_stub_api.logger = _StubLogger()
_stub_api.AstrBotConfig = dict
_stub_astrbot = types.ModuleType("astrbot")
_stub_astrbot.api = _stub_api
sys.modules.setdefault("astrbot", _stub_astrbot)
sys.modules.setdefault("astrbot.api", _stub_api)

from astrbot_plugin_mathitools.math_tools import (  # noqa: E402
    MathBatchTool,
    MathCompareTool,
    MathEvalTool,
    MathFinanceTool,
    MathFunctionsTool,
    MathMatrixTool,
    MathNaturalLanguageUnitsTool,
    MathNumberTheoryTool,
    MathStatisticsTool,
    MathUnitsTool,
)

LOGGER = _stub_api.logger

EXPECTED_REQUIRED = {
    "math_eval": ["expression"],
    "math_compare": ["left", "right"],
    "math_batch": ["expressions"],
    "math_functions": [],
    "math_statistics": ["operation", "data"],
    "math_matrix": ["operation", "matrices"],
    "math_units": ["value", "from_unit", "to_unit", "unit_type"],
    "math_nl_units": ["query"],
    "math_number_theory": ["operation", "number"],
    "math_finance": ["operation"],
}

FORBIDDEN_SCHEMA_KEYS = {"default", "oneOf", "allOf", "anyOf", "additionalProperties"}


def _call(tool, **kwargs):
    return json.loads(asyncio.run(tool.call(None, **kwargs)))


def _all_tools():
    return [
        MathEvalTool(),
        MathCompareTool(),
        MathBatchTool(),
        MathFunctionsTool(),
        MathStatisticsTool(),
        MathMatrixTool(),
        MathUnitsTool(),
        MathNaturalLanguageUnitsTool(),
        MathNumberTheoryTool(),
        MathFinanceTool(),
    ]


class ToolSchemaTest(unittest.TestCase):
    def test_ten_tools_names_and_required(self):
        tools = _all_tools()
        self.assertEqual({tool.name for tool in tools}, set(EXPECTED_REQUIRED))
        for tool in tools:
            self.assertEqual(tool.parameters["type"], "object")
            self.assertEqual(tool.parameters["required"], EXPECTED_REQUIRED[tool.name])
            for key in EXPECTED_REQUIRED[tool.name]:
                self.assertIn(key, tool.parameters["properties"])

    def test_schema_has_no_gateway_rejected_keys(self):
        for tool in _all_tools():
            self._assert_no_forbidden(tool.parameters, f"{tool.name}.parameters")

    def _assert_no_forbidden(self, node, path):
        if isinstance(node, dict):
            bad = set(node) & FORBIDDEN_SCHEMA_KEYS
            self.assertFalse(bad, f"{path} 含禁用关键字: {bad}")
            for value in node.values():
                self._assert_no_forbidden(value, path)
        elif isinstance(node, list):
            for value in node:
                self._assert_no_forbidden(value, path)


class ToolBehaviorTest(unittest.TestCase):
    def setUp(self):
        LOGGER.errors.clear()

    def test_math_eval_ok(self):
        payload = _call(MathEvalTool(), expression="1+2")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"], "3")

    def test_math_eval_caret_is_power(self):
        payload = _call(MathEvalTool(), expression="2^10")
        self.assertEqual(payload["result"], "1024")

    def test_math_eval_invalid_dps(self):
        payload = _call(MathEvalTool(), expression="1+1", dps=500)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "invalid_argument")

    def test_math_batch_single_failure_does_not_interrupt(self):
        payload = _call(MathBatchTool(), expressions=["1+1", "foo(1)", "2*3"])
        self.assertTrue(payload["ok"])
        results = payload["results"]
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0]["ok"])
        self.assertEqual(results[0]["result"], "2")
        self.assertFalse(results[1]["ok"])
        self.assertEqual(results[1]["error_code"], "unknown_function")
        self.assertTrue(results[2]["ok"])
        self.assertEqual(results[2]["result"], "6")

    def test_math_functions_counts(self):
        payload = _call(MathFunctionsTool())
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertEqual(len(result["functions"]), 73)
        self.assertEqual(len(result["constants"]), 7)
        self.assertEqual(len(result["unit_categories"]), 15)
        self.assertEqual(sum(len(units) for units in result["unit_categories"].values()), 158)
        self.assertEqual(len(result["unit_aliases"]), 153)
        self.assertIn("Add", result["operators"])
        self.assertIn("mean", result["statistics_operations"])
        self.assertIn("determinant", result["matrix_operations"])
        self.assertIn("is_prime", result["number_theory_operations"])
        self.assertIn("tax", result["finance_operations"])

    def test_math_compare_ok(self):
        payload = _call(MathCompareTool(), left="2^10", right="1000")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["comparison"], ">")

    def test_math_compare_complex_rejected(self):
        payload = _call(MathCompareTool(), left="sqrt(-1)", right="1")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "complex_result")

    def test_math_compare_syntax_error(self):
        payload = _call(MathCompareTool(), left="1+*2", right="1")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "syntax_error")

    def test_math_compare_non_numeric_rejected(self):
        payload = _call(MathCompareTool(), left="isfinite(1)", right="1")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "invalid_argument")

    def test_math_statistics_unknown_operation_has_valid_options(self):
        payload = _call(MathStatisticsTool(), operation="zz", data=[1, 2, 3])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "unknown_operation")
        self.assertIn("mean", payload["valid_options"])

    def test_math_statistics_ok(self):
        payload = _call(MathStatisticsTool(), operation="mean", data=[1, 2, 3, 4])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"], 2.5)

    def test_math_matrix_dimension_error(self):
        payload = _call(
            MathMatrixTool(),
            operation="multiply",
            matrices=[[[1, 2], [3, 4]], [[1, 2], [3, 4], [5, 6]]],
        )
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "matrix_dimension")

    def test_math_matrix_ok(self):
        payload = _call(MathMatrixTool(), operation="determinant", matrices=[[[1, 2], [3, 4]]])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"], -2.0)

    def test_math_units_unknown_unit(self):
        payload = _call(MathUnitsTool(), value=1, from_unit="km", to_unit="zz", unit_type="length")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "unknown_unit")
        self.assertIn("km", payload["valid_options"])

    def test_math_units_cross_category_unit_rejected(self):
        payload = _call(MathUnitsTool(), value=1, from_unit="km", to_unit="kg", unit_type="length")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "unknown_unit")

    def test_math_units_ok(self):
        payload = _call(MathUnitsTool(), value=5, from_unit="km", to_unit="mi", unit_type="length")
        self.assertTrue(payload["ok"])
        self.assertAlmostEqual(payload["result"], 3.106856, places=6)

    def test_math_nl_units_unparseable(self):
        payload = _call(MathNaturalLanguageUnitsTool(), query="hello world")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "parse_failed")

    def test_math_nl_units_ok(self):
        payload = _call(MathNaturalLanguageUnitsTool(), query="5 km to miles")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["category"], "length")

    def test_math_number_theory_non_integer(self):
        payload = _call(MathNumberTheoryTool(), operation="is_prime", number=3.5)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "non_integer")

    def test_math_number_theory_unknown_operation(self):
        payload = _call(MathNumberTheoryTool(), operation="zz", number=5)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "unknown_operation")

    def test_math_finance_unknown_operation(self):
        payload = _call(MathFinanceTool(), operation="zz")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "unknown_operation")
        self.assertIn("tip", payload["valid_options"])

    def test_math_finance_missing_params_has_example(self):
        payload = _call(MathFinanceTool(), operation="tip", amount=100)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "invalid_argument")
        self.assertIn("tip_percent", payload["example"])

    def test_math_finance_ok(self):
        payload = _call(MathFinanceTool(), operation="tip", amount=100, tip_percent=15)
        self.assertTrue(payload["ok"])
        self.assertAlmostEqual(payload["result"]["total_amount"], 115.0, places=12)

    def test_logger_error_called_on_failure(self):
        _call(MathStatisticsTool(), operation="zz", data=[1])
        self.assertTrue(LOGGER.errors)
        self.assertIn("math_statistics", LOGGER.errors[0])


if __name__ == "__main__":
    unittest.main()
