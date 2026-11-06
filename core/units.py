"""单位换算：15 类 158 单位 + 153 别名解析 + 自然语言换算。"""

import re

from .errors import MathError
from .registry import UNIT_ALIASES, UNIT_CONVERSIONS
from .statistics import valid_number

# 反向族：距离/体积（mpg、mpg_uk、km/L、mi/L），系数含义为「多少 mpg 当量」。
_FUEL_INVERSE: frozenset[str] = frozenset({"mpg", "mpg_uk", "km/L", "mi/L"})

_QUERY_RE = re.compile(
    r"^\s*(?:convert\s+|换算\s*)?"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*°?\s*"
    r"(\S+?)\s*"
    r"(?:to|in|as|into|->|→|=)\s*"
    r"(\S+?)"
    r"\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def convert_unit(
    value: float, from_unit: str, to_unit: str, unit_type: str
) -> float:
    """把 value 从 from_unit 换算到 to_unit；别名自动解析。"""
    if not valid_number(value):
        raise MathError(
            "value 必须是有限数字",
            "invalid_argument",
            hint="不接受布尔值、NaN 或无穷大",
        )
    conversions = UNIT_CONVERSIONS.get(unit_type)
    if conversions is None:
        raise MathError(
            f"未知单位类别: {unit_type}",
            "unknown_unit",
            hint=f"可用类别: {', '.join(sorted(UNIT_CONVERSIONS))}",
            valid_options=sorted(UNIT_CONVERSIONS),
        )
    src = _resolve(from_unit, conversions)
    dst = _resolve(to_unit, conversions)

    if unit_type == "temperature":
        kelvin = conversions[src](value)
        if dst == "K":
            return kelvin
        if dst == "C":
            return kelvin - 273.15
        return (kelvin - 273.15) * 9 / 5 + 32
    if unit_type == "fuel_economy":
        return _convert_fuel_economy(value, src, dst, conversions)
    return value * conversions[src] / conversions[dst]


def convert_natural_language(query: str) -> dict:
    """解析「5 km to miles」等自然语言换算请求；无法确定时不猜测。"""
    match = _QUERY_RE.match(query or "")
    if match is None:
        raise MathError(
            "无法解析换算请求",
            "parse_failed",
            hint='格式: "<数值> <源单位> to <目标单位>"，如 "5 km to miles"',
            example="5 km to miles",
        )
    value = float(match.group(1))
    if not valid_number(value):
        raise MathError(
            f"换算数值不是有限数字: {match.group(1)}",
            "invalid_argument",
            hint="数值必须是有限实数，拒绝 1e309 这类溢出",
            example="5 km to miles",
        )
    src_candidates = _candidates(match.group(2))
    dst_candidates = _candidates(match.group(3))

    pairs = [
        (src, dst)
        for src in src_candidates
        for dst in dst_candidates
        if src[0] == dst[0]
    ]
    if not pairs:
        raise MathError(
            "无法确定单位类别或不支持跨类别换算",
            "parse_failed",
            hint="源单位与目标单位必须属于同一类别",
            example="5 km to miles",
        )
    src_hit, dst_hit = pairs[0]
    category, src = src_hit
    dst = dst_hit[1]
    result = convert_unit(value, src, dst, category)
    return {
        "value": value,
        "from_unit": src,
        "to_unit": dst,
        "category": category,
        "result": result,
    }


def _resolve(unit: str, conversions: dict) -> str:
    canonical = _lookup(unit, conversions)
    if canonical is None:
        raise MathError(
            f"未知单位: {unit}",
            "unknown_unit",
            hint=f"该类别可用单位: {', '.join(sorted(conversions))}，另支持常见英文别名",
            valid_options=sorted(conversions),
        )
    return canonical


def _lookup(unit: str, conversions: dict) -> str | None:
    if unit in conversions:
        return unit
    canonical = UNIT_ALIASES.get(unit) or UNIT_ALIASES.get(unit.lower())
    if canonical in conversions:
        return canonical
    for key in conversions:
        if key.lower() == unit.lower():
            return key
    return None


def _candidates(token: str) -> list[tuple[str, str]]:
    found = []
    for category, conversions in UNIT_CONVERSIONS.items():
        canonical = _lookup(token, conversions)
        if canonical is not None:
            found.append((category, canonical))
    return found


def _convert_fuel_economy(
    value: float, src: str, dst: str, conversions: dict
) -> float:
    """油耗换算：同族线性缩放；反向族与正向族（体积/距离）之间经 mpg 当量换算。"""
    if src == dst:
        return value
    src_inverse, dst_inverse = src in _FUEL_INVERSE, dst in _FUEL_INVERSE
    if src_inverse and dst_inverse:
        return value * conversions[src] / conversions[dst]
    if src_inverse:
        base_mpg = value * conversions[src]
        if base_mpg == 0:
            raise MathError(
                "油耗为 0 时无法换算为每距离油耗",
                "division_by_zero",
                hint="反向族与正向族之间换算需要除以 mpg 当量",
            )
        return conversions[dst] / base_mpg
    if value == 0:
        raise MathError(
            "每距离油耗为 0 时无法换算为油耗",
            "division_by_zero",
            hint="L/100km 或 gal/100mi 不能为 0",
        )
    base_mpg = conversions[src] / value
    return base_mpg / conversions[dst]
