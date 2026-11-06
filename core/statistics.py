"""15 个统计运算，标准库 statistics 加精确分数辅助。"""

import math
import statistics
from decimal import Context, Decimal, localcontext
from fractions import Fraction

from .errors import MathError
from .registry import STATISTICS_OPERATIONS
from .security import MAX_LIST_ELEMENTS


def valid_number(value) -> bool:
    """有限实数且非布尔值；超出浮点表示范围的整数同样视为非法。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_data(data, name: str) -> None:
    """校验一组非空、规模受限的有限数字列表。"""
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        raise MathError(
            f"{name} 必须是非空数字列表",
            "invalid_argument",
            hint=f"至少需要一个数据点，且不超过 {MAX_LIST_ELEMENTS} 个元素",
            example='calculate_statistics([1, 2, 3], "mean")',
        )
    if len(data) > MAX_LIST_ELEMENTS:
        raise MathError(
            f"{name} 元素数量超过上限 {MAX_LIST_ELEMENTS}",
            "invalid_argument",
            hint=f"每个列表最多 {MAX_LIST_ELEMENTS} 个元素",
        )
    for entry in data:
        if not valid_number(entry):
            raise MathError(
                f"{name} 元素必须是有限数字",
                "invalid_argument",
                hint="不接受字符串、布尔值、NaN、Infinity 或嵌套列表",
                example='calculate_statistics([1, 2, 3], "mean")',
            )


def _bivariate_moments(data, data2):
    """精确均值与中心积和 (sxx, syy, sxy)，输入已校验且 n>=2。"""
    fx = [Fraction(v) for v in data]
    fy = [Fraction(v) for v in data2]
    n = len(fx)
    x_mean = sum(fx) / n
    y_mean = sum(fy) / n
    sxx = syy = sxy = Fraction(0)
    for x, y in zip(fx, fy):
        dx = x - x_mean
        dy = y - y_mean
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    return x_mean, y_mean, sxx, syy, sxy


def _weighted_mean(data, weights) -> float:
    """精确加权和除以权重和，允许负权重。"""
    w = [Fraction(v) for v in weights]
    total_w = sum(w)
    if total_w == 0:
        raise statistics.StatisticsError("权重之和为 0")
    return float(sum(Fraction(v) * wi for v, wi in zip(data, w)) / total_w)


def _percentile(data, p) -> int | float:
    """R7 包含端点线性插值的百分位数。"""
    ordered = sorted(data)
    pos = Fraction(p) * (len(ordered) - 1) / 100
    idx, frac = divmod(pos, 1)
    if frac == 0:
        return ordered[idx]
    lo, hi = Fraction(ordered[idx]), Fraction(ordered[idx + 1])
    return float(lo + (hi - lo) * frac)


def _paired_statistics(data, data2, operation):
    """样本协方差、Pearson 相关系数与线性回归。"""
    x_mean, y_mean, sxx, syy, sxy = _bivariate_moments(data, data2)
    n = len(data)
    if operation == "covariance":
        return float(sxy / (n - 1))
    if operation == "correlation":
        if sxx == 0 or syy == 0:
            raise statistics.StatisticsError("常量数据无法计算相关系数")
        q = sxy * sxy / (sxx * syy)
        with localcontext(Context(prec=40)):
            r = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
            r = float(r)
        return r if sxy >= 0 else -r
    if sxx == 0:
        raise statistics.StatisticsError("x 数据全为常量，无法计算回归")
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    return {"slope": float(slope), "intercept": float(intercept)}


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


def calculate_statistics(
    data: list,
    operation: str,
    *,
    weights=None,
    data2=None,
    p=None,
):
    """对数字列表执行指定统计运算，新操作使用精确分数。"""
    if operation not in STATISTICS_OPERATIONS:
        raise MathError(
            f"未知统计运算: {operation}",
            "unknown_operation",
            hint=f"可用运算: {', '.join(STATISTICS_OPERATIONS)}",
            valid_options=list(STATISTICS_OPERATIONS),
            example='calculate_statistics([1, 2, 3], "mean")',
        )
    _validate_data(data, "data")
    if operation == "weighted_mean":
        _validate_data(weights, "weights")
        if len(weights) != len(data):
            raise MathError(
                "weights 与 data 长度不同", "invalid_argument", hint="两组数据必须等长"
            )
    elif operation == "percentile":
        if not valid_number(p) or not 0 <= p <= 100:
            raise MathError(
                "p 必须是 0 到 100 之间的有限数字",
                "invalid_argument",
                hint="不接受布尔值、NaN、Infinity 或范围外的数字",
            )
    elif operation in ("covariance", "correlation", "linear_regression"):
        _validate_data(data2, "data2")
        if len(data2) != len(data):
            raise MathError(
                "data2 与 data 长度不同", "invalid_argument", hint="两组数据必须等长"
            )
        if len(data) < 2:
            raise MathError(
                "至少需要两个数据点",
                "domain_error",
                hint="协方差、相关系数和回归需要至少两对数据",
            )
    try:
        if operation == "weighted_mean":
            result = _weighted_mean(data, weights)
        elif operation == "percentile":
            result = _percentile(data, p)
        elif operation in ("covariance", "correlation", "linear_regression"):
            result = _paired_statistics(data, data2, operation)
        else:
            result = _OPS[operation](data)
    except statistics.StatisticsError as exc:
        raise MathError(
            str(exc),
            "domain_error",
            hint="数据不满足该运算的要求",
        ) from exc
    except (OverflowError, ZeroDivisionError) as exc:
        raise MathError(
            str(exc), "domain_error", hint="运算结果数学未定义或超出有限范围"
        ) from exc
    except TypeError as exc:
        raise MathError(
            str(exc), "invalid_argument", hint="data 必须全部为数字"
        ) from exc
    return _check_result(result)


def _check_result(result):
    """检查统计结果中的数值均为有限数。"""
    if isinstance(result, dict):
        values = result.values()
    elif isinstance(result, (list, tuple)):
        values = result
    else:
        values = (result,)
    for value in values:
        if not valid_number(value):
            raise MathError(
                "运算结果超出有限数值范围",
                "domain_error",
                hint="结果不能是 NaN 或 Infinity",
            )
    return result
