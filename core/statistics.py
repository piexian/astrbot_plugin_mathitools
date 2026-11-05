"""10 个统计运算，基于标准库 statistics。"""

import statistics

from .errors import MathError
from .registry import STATISTICS_OPERATIONS

_OPS: dict = {
    "mean": statistics.mean,
    "median": statistics.median,
    "mode": statistics.mode,
    "stdev": statistics.stdev,
    "pstdev": statistics.pstdev,
    "variance": statistics.variance,
    "pvariance": statistics.pvariance,
    "harmonic_mean": statistics.harmonic_mean,
    "geometric_mean": statistics.geometric_mean,
    "quantiles": lambda data: statistics.quantiles(data, n=4),
}


def calculate_statistics(data: list, operation: str):
    """对数字列表执行指定统计运算。"""
    if operation not in _OPS:
        raise MathError(
            f"未知统计运算: {operation}",
            "unknown_operation",
            hint=f"可用运算: {', '.join(STATISTICS_OPERATIONS)}",
            valid_options=list(STATISTICS_OPERATIONS),
            example='calculate_statistics([1, 2, 3], "mean")',
        )
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        raise MathError(
            "data 必须是非空数字列表",
            "invalid_argument",
            hint="至少需要一个数据点",
            example='calculate_statistics([1, 2, 3], "mean")',
        )
    try:
        return _OPS[operation](data)
    except statistics.StatisticsError as exc:
        raise MathError(
            str(exc),
            "domain_error",
            hint="数据不满足该运算的要求（如方差需要至少两个数据点）",
        ) from exc
    except TypeError as exc:
        raise MathError(
            str(exc), "invalid_argument", hint="data 必须全部为数字"
        ) from exc
