"""11 个数学 FunctionTool：统一 JSON 字符串返回与 AI 友好错误契约。"""

from dataclasses import dataclass, field
from typing import Any, Callable

from astrbot.api import FunctionTool, logger

from .core.errors import MathError
from .core.evaluator import evaluate
from .core.finance import finance_operation
from .core.matrix import matrix_operation
from .core.numerical import numerical_operation
from .core.number_theory import number_theory_operation
from .core.registry import (
    ALL_FUNCTION_NAMES,
    CONSTANT_NAMES,
    FINANCE_OPERATIONS,
    MATRIX_OPERATIONS,
    NUMBER_THEORY_OPERATIONS,
    NUMERICAL_OPERATIONS,
    OPERATORS,
    STATISTICS_OPERATIONS,
    UNIT_ALIASES,
    UNIT_CONVERSIONS,
)
from .core.statistics import calculate_statistics, valid_number
from .core.units import convert_natural_language, convert_unit
from .result import error_result, ok_result
from .schema import (
    array_parameter,
    boolean_parameter,
    integer_parameter,
    number_parameter,
    object_parameters,
    string_parameter,
)

DEFAULT_DPS = 20

_FINANCE_EXAMPLE = 'operation="tip", amount=100, tip_percent=15'


def _math_error_response(tool_name: str, exc: MathError) -> str:
    logger.error(f"{tool_name} 失败 error_code={exc.error_code}: {exc}")
    return error_result(
        str(exc),
        exc.error_code,
        hint=exc.hint,
        example=exc.example,
        valid_options=exc.valid_options,
    )


def _unexpected_error_response(tool_name: str, exc: Exception) -> str:
    logger.error(f"{tool_name} 未预期异常: {type(exc).__name__}: {exc}")
    return error_result(f"{type(exc).__name__}: {exc}", "internal_error", hint="请简化输入后重试")


def _run(tool_name: str, work: Callable[[], Any]) -> str:
    """执行核心计算并统一错误契约。"""
    try:
        return ok_result(result=work())
    except MathError as exc:
        return _math_error_response(tool_name, exc)
    except Exception as exc:
        return _unexpected_error_response(tool_name, exc)


def _require_str(kwargs: dict, key: str, example: str) -> str:
    value = kwargs.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MathError(
            f"{key} 必须是非空字符串",
            "invalid_argument",
            hint=f"缺少或非法参数 {key}",
            example=example,
        )
    return value


def _require_number(kwargs: dict, key: str, example: str) -> float:
    value = kwargs.get(key)
    if not valid_number(value):
        raise MathError(
            f"{key} 必须是有限数字",
            "invalid_argument",
            hint=f"缺少或非法参数 {key}",
            example=example,
        )
    return value


def _dps(kwargs: dict, default: int = DEFAULT_DPS) -> int:
    return kwargs.get("dps", default)


def _as_comparable(text: str, label: str) -> float:
    try:
        value = complex(text.replace(" ", ""))
    except ValueError as exc:
        raise MathError(
            f"{label} 不是可比较的数值: {text}",
            "invalid_argument",
            hint="比较两边必须是求值为实数的表达式",
            example='left="2^10", right="1000"',
        ) from exc
    if value.imag != 0:
        raise MathError(
            f"{label} 是复数，无法比较大小: {text}",
            "complex_result",
            hint="复数没有大小关系，可分别求模或取实部再比较",
        )
    return value.real


@dataclass
class MathEvalTool(FunctionTool):
    name: str = "math_eval"
    description: str = "求值数学表达式；支持 78 个函数、7 个常量、列表与复数输出，^ 按乘方处理"
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {
                "expression": string_parameter("数学表达式，如 sqrt(2)、mean([1,2,3])；^ 按乘方处理"),
                "dps": integer_parameter("输出精度位数，1..200，默认 20"),
            },
            required=["expression"],
        )
    )
    default_dps: int = DEFAULT_DPS

    async def call(self, context: Any, **kwargs: Any) -> str:
        def work() -> Any:
            expression = _require_str(kwargs, "expression", 'expression="sqrt(2)"')
            return evaluate(expression, dps=_dps(kwargs, self.default_dps))

        return _run("math_eval", work)


@dataclass
class MathCompareTool(FunctionTool):
    name: str = "math_compare"
    description: str = "比较两个实数表达式的大小；复数或非数值结果返回明确错误"
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {
                "left": string_parameter("左侧表达式，求值为实数"),
                "right": string_parameter("右侧表达式，求值为实数"),
                "dps": integer_parameter("求值精度位数，1..200，默认 20"),
            },
            required=["left", "right"],
        )
    )
    default_dps: int = DEFAULT_DPS

    async def call(self, context: Any, **kwargs: Any) -> str:
        def work() -> dict:
            dps = _dps(kwargs, self.default_dps)
            left_text = _require_str(kwargs, "left", 'left="2^10", right="1000"')
            right_text = _require_str(kwargs, "right", 'left="2^10", right="1000"')
            left = _as_comparable(evaluate(left_text, dps=dps), "left")
            right = _as_comparable(evaluate(right_text, dps=dps), "right")
            if left < right:
                comparison = "<"
            elif left > right:
                comparison = ">"
            else:
                comparison = "=="
            return {"left": left, "right": right, "comparison": comparison}

        return _run("math_compare", work)


@dataclass
class MathBatchTool(FunctionTool):
    name: str = "math_batch"
    description: str = "批量求值数学表达式；单项失败落在该项 ok=false，不中断整批"
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {
                "expressions": array_parameter(
                    "表达式字符串数组，如 [\"1+1\", \"sqrt(2)\"]",
                    items=string_parameter("数学表达式"),
                ),
                "dps": integer_parameter("输出精度位数，1..200，默认 20"),
            },
            required=["expressions"],
        )
    )
    default_dps: int = DEFAULT_DPS

    async def call(self, context: Any, **kwargs: Any) -> str:
        expressions = kwargs.get("expressions")
        if not isinstance(expressions, list) or not expressions:
            return error_result(
                "expressions 必须是非空字符串数组",
                "invalid_argument",
                hint="如 expressions=[\"1+1\", \"sqrt(2)\"]",
            )
        dps = _dps(kwargs, self.default_dps)
        results = []
        for expression in expressions:
            if not isinstance(expression, str) or not expression.strip():
                logger.error(f"math_batch 单项非法: {expression!r}")
                results.append(
                    {
                        "expression": expression,
                        "ok": False,
                        "error": "表达式必须是非空字符串",
                        "error_code": "invalid_argument",
                    }
                )
                continue
            try:
                results.append(
                    {
                        "expression": expression,
                        "ok": True,
                        "result": evaluate(expression, dps=dps),
                    }
                )
            except MathError as exc:
                logger.error(f"math_batch 单项失败 error_code={exc.error_code}: {expression}")
                results.append(
                    {
                        "expression": expression,
                        "ok": False,
                        "error": str(exc),
                        "error_code": exc.error_code,
                        "hint": exc.hint,
                    }
                )
            except Exception as exc:
                logger.error(f"math_batch 单项未预期异常: {expression}")
                results.append(
                    {
                        "expression": expression,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "error_code": "internal_error",
                    }
                )
        return ok_result(results=results)


@dataclass
class MathFunctionsTool(FunctionTool):
    name: str = "math_functions"
    description: str = "返回可用函数、常量、单位、别名与各类 operation 清单"
    parameters: dict = field(default_factory=lambda: object_parameters({}, required=[]))

    async def call(self, context: Any, **kwargs: Any) -> str:
        payload = {
            "functions": list(ALL_FUNCTION_NAMES),
            "constants": list(CONSTANT_NAMES),
            "unit_categories": {
                category: list(units) for category, units in UNIT_CONVERSIONS.items()
            },
            "unit_aliases": dict(UNIT_ALIASES),
            "operators": sorted(op.__name__ for op in OPERATORS),
            "statistics_operations": list(STATISTICS_OPERATIONS),
            "matrix_operations": list(MATRIX_OPERATIONS),
            "numerical_operations": list(NUMERICAL_OPERATIONS),
            "number_theory_operations": list(NUMBER_THEORY_OPERATIONS),
            "finance_operations": list(FINANCE_OPERATIONS),
        }
        return ok_result(result=payload)


@dataclass
class MathStatisticsTool(FunctionTool):
    name: str = "math_statistics"
    description: str = "对数字列表执行 15 种统计运算，含加权平均、百分位、样本协方差、Pearson 相关与线性回归"
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {
                "operation": string_parameter(
                    "统计运算: " + ", ".join(STATISTICS_OPERATIONS)
                ),
                "data": array_parameter(
                    "数字列表，最多 10000 个有限数字，如 [1, 2, 3]（协方差/相关/回归中为 x）",
                    items=number_parameter("数据点"),
                ),
                "weights": array_parameter(
                    "与 data 等长的权重，允许负数但权重和不能为 0（仅 weighted_mean）",
                    items=number_parameter("权重"),
                ),
                "data2": array_parameter(
                    "y 数据，与 data（x）等长且至少两对（仅 covariance/correlation/linear_regression）",
                    items=number_parameter("数据点"),
                ),
                "p": number_parameter(
                    "percentile 的百分位，0 到 100，包含端点线性插值"
                ),
            },
            required=["operation", "data"],
        )
    )

    async def call(self, context: Any, **kwargs: Any) -> str:
        def work() -> Any:
            operation = _require_str(kwargs, "operation", 'operation="mean", data=[1, 2, 3]')
            return calculate_statistics(
                kwargs.get("data"),
                operation,
                weights=kwargs.get("weights"),
                data2=kwargs.get("data2"),
                p=kwargs.get("p"),
            )

        return _run("math_statistics", work)


@dataclass
class MathMatrixTool(FunctionTool):
    name: str = "math_matrix"
    description: str = "矩阵运算；multiply 传 [A, B]、其余传 [A]；solve 传 [A, b 列矩阵] 解 Ax=b 返回 x 列矩阵，仅 solve 限 A 至多 50 阶"
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {
                "operation": string_parameter(
                    "矩阵运算: " + ", ".join(MATRIX_OPERATIONS)
                ),
                "matrices": array_parameter(
                    "矩阵列表；multiply 传 [A, B]，solve 传 [A, b 列矩阵]（A 至多 50 阶，返回 x 列矩阵），其余传 [A]",
                    items=array_parameter(
                        "矩阵的一行，如 [1, 2]",
                        items=array_parameter("元素", items=number_parameter("数值")),
                    ),
                ),
            },
            required=["operation", "matrices"],
        )
    )

    async def call(self, context: Any, **kwargs: Any) -> str:
        def work() -> Any:
            operation = _require_str(
                kwargs, "operation", 'operation="determinant", matrices=[[[1, 2], [3, 4]]]'
            )
            return matrix_operation(kwargs.get("matrices"), operation)

        return _run("math_matrix", work)


@dataclass
class MathNumericalTool(FunctionTool):
    name: str = "math_numerical"
    description: str = (
        "数值求根；findroot 对连续实函数表达式在区间二分求根，polyroots 对降幂系数"
        "（至多 20 次）求全部根，根以字符串/字符串数组返回"
    )
    default_dps: int = DEFAULT_DPS
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {
                "operation": string_parameter(
                    "数值求根运算: " + ", ".join(NUMERICAL_OPERATIONS)
                ),
                "expression": string_parameter(
                    "findroot 的连续实函数表达式，唯一自变量 x"
                ),
                "a": number_parameter("findroot 区间下端点，需 a<b 且端点异号或端点为根"),
                "b": number_parameter("findroot 区间上端点，需 a<b 且端点异号或端点为根"),
                "coefficients": array_parameter(
                    "polyroots 的降幂系数，至多 21 项且前导项非零",
                    items=number_parameter("系数"),
                ),
                "dps": integer_parameter("计算精度 1..200，默认采用插件配置"),
            },
            required=["operation"],
        )
    )

    async def call(self, context: Any, **kwargs: Any) -> str:
        try:
            operation = _require_str(
                kwargs, "operation", 'operation="findroot", expression="x^2-2", a=1, b=2'
            )
            params = {
                key: kwargs[key]
                for key in ("expression", "a", "b", "coefficients")
                if key in kwargs
            }
            result = await numerical_operation(
                operation, dps=_dps(kwargs, self.default_dps), **params
            )
            return ok_result(result=result)
        except MathError as exc:
            return _math_error_response("math_numerical", exc)
        except Exception as exc:
            return _unexpected_error_response("math_numerical", exc)


@dataclass
class MathUnitsTool(FunctionTool):
    name: str = "math_units"
    description: str = "单位换算；支持 15 类 158 单位与常见英文别名，温度为仿射换算"
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {
                "value": number_parameter("数值"),
                "from_unit": string_parameter("源单位，如 km、celsius"),
                "to_unit": string_parameter("目标单位，如 miles、fahrenheit"),
                "unit_type": string_parameter(
                    "单位类别: length, mass, time, temperature, area, volume, speed, "
                    "data, pressure, energy, power, force, angle, frequency, fuel_economy"
                ),
            },
            required=["value", "from_unit", "to_unit", "unit_type"],
        )
    )

    async def call(self, context: Any, **kwargs: Any) -> str:
        def work() -> float:
            value = _require_number(kwargs, "value", 'value=5, from_unit="km", to_unit="miles", unit_type="length"')
            from_unit = _require_str(kwargs, "from_unit", 'value=5, from_unit="km", to_unit="miles", unit_type="length"')
            to_unit = _require_str(kwargs, "to_unit", 'value=5, from_unit="km", to_unit="miles", unit_type="length"')
            unit_type = _require_str(kwargs, "unit_type", 'value=5, from_unit="km", to_unit="miles", unit_type="length"')
            return convert_unit(value, from_unit, to_unit, unit_type)

        return _run("math_units", work)


@dataclass
class MathNaturalLanguageUnitsTool(FunctionTool):
    name: str = "math_nl_units"
    description: str = "自然语言单位换算，如 \"5 km to miles\"；无法确定时不猜测"
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {"query": string_parameter('换算请求，如 "5 km to miles"、"100 celsius to fahrenheit"')},
            required=["query"],
        )
    )

    async def call(self, context: Any, **kwargs: Any) -> str:
        def work() -> dict:
            query = _require_str(kwargs, "query", 'query="5 km to miles"')
            return convert_natural_language(query)

        return _run("math_nl_units", work)


@dataclass
class MathNumberTheoryTool(FunctionTool):
    name: str = "math_number_theory"
    description: str = "数论运算：素性判定、因式分解、因子列表、欧拉函数"
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {
                "operation": string_parameter(
                    "数论运算: is_prime, prime_factors, divisors, totient"
                ),
                "number": integer_parameter("整数"),
            },
            required=["operation", "number"],
        )
    )

    async def call(self, context: Any, **kwargs: Any) -> str:
        def work() -> Any:
            operation = _require_str(kwargs, "operation", 'operation="prime_factors", number=360')
            return number_theory_operation(kwargs.get("number"), operation)

        return _run("math_number_theory", work)


@dataclass
class MathFinanceTool(FunctionTool):
    name: str = "math_finance"
    description: str = "金融数学计算；按 operation 传入对应参数"
    parameters: dict = field(
        default_factory=lambda: object_parameters(
            {
                "operation": string_parameter(
                    "金融运算: percentage, percentage_change, tip, compound_interest, "
                    "simple_interest, loan_payment, split_bill, tax"
                ),
                "value": number_parameter("percentage 用：数值"),
                "percentage_value": number_parameter("percentage 用：百分比数值"),
                "old_value": number_parameter("percentage_change 用：旧值"),
                "new_value": number_parameter("percentage_change 用：新值"),
                "amount": number_parameter("tip/tax 用：金额"),
                "tip_percent": number_parameter("tip/split_bill 用：小费百分比"),
                "principal": number_parameter("利息/贷款/split_bill 用：本金或总额"),
                "rate": number_parameter("利息类用：年利率百分比"),
                "time": number_parameter("利息类用：年数"),
                "compounds_per_year": integer_parameter("compound_interest 用：年计息次数，默认 1"),
                "years": number_parameter("loan_payment 用：期限年数"),
                "payments_per_year": integer_parameter("loan_payment 用：年还款次数，默认 12"),
                "people": integer_parameter("split_bill 用：人数"),
                "tax_rate": number_parameter("tax 用：税率百分比"),
                "is_inclusive": boolean_parameter("tax 用：amount 是否含税，默认 false"),
            },
            required=["operation"],
        )
    )

    async def call(self, context: Any, **kwargs: Any) -> str:
        def work() -> Any:
            operation = _require_str(kwargs, "operation", _FINANCE_EXAMPLE)
            params = {key: value for key, value in kwargs.items() if key != "operation"}
            try:
                return finance_operation(operation, **params)
            except MathError as exc:
                if exc.error_code == "invalid_argument" and exc.example is None:
                    exc.example = _FINANCE_EXAMPLE
                raise

        return _run("math_finance", work)


ALL_TOOLS: tuple[type[FunctionTool], ...] = (
    MathEvalTool,
    MathCompareTool,
    MathBatchTool,
    MathFunctionsTool,
    MathStatisticsTool,
    MathMatrixTool,
    MathNumericalTool,
    MathUnitsTool,
    MathNaturalLanguageUnitsTool,
    MathNumberTheoryTool,
    MathFinanceTool,
)
