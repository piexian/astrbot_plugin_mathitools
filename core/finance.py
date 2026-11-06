"""金融数学：percentage、percentage_change、tip、compound_interest、
simple_interest、loan_payment、split_bill、tax。"""

from .errors import MathError
from .registry import FINANCE_OPERATIONS
from .statistics import valid_number


_NUMERIC_PARAMS: dict[str, tuple[str, ...]] = {
    "percentage": ("value", "percentage_value"),
    "percentage_change": ("old_value", "new_value"),
    "tip": ("amount", "tip_percent"),
    "compound_interest": ("principal", "rate", "time"),
    "simple_interest": ("principal", "rate", "time"),
    "loan_payment": ("principal", "annual_rate", "years"),
    "split_bill": ("total", "tip_percent"),
    "tax": ("amount", "tax_rate"),
}

def finance_operation(operation: str, **params) -> dict | float:
    """按 operation 分发金融计算。"""
    if operation not in FINANCE_OPERATIONS:
        raise MathError(
            f"未知金融运算: {operation}",
            "unknown_operation",
            hint=f"可用运算: {', '.join(FINANCE_OPERATIONS)}",
            valid_options=list(FINANCE_OPERATIONS),
            example='finance_operation("tip", amount=100, tip_percent=15)',
        )
    for name in _NUMERIC_PARAMS[operation]:
        value = params.get(name)
        if value is not None and not valid_number(value):
            raise MathError(
                f"{name} 必须是有限数字",
                "invalid_argument",
                hint="不接受字符串、布尔值、NaN 或无穷大",
            )
    try:
        return _OPS[operation](**params)
    except TypeError as exc:
        raise MathError(
            str(exc), "invalid_argument", hint="检查该运算所需的参数"
        ) from exc


def percentage(value: float, percentage_value: float) -> float:
    """value 的百分之 percentage_value 是多少。"""
    return value * (percentage_value / 100)


def percentage_change(old_value: float, new_value: float) -> float:
    """从 old_value 到 new_value 的百分比变化。"""
    if old_value == 0:
        raise MathError(
            "无法计算从 0 开始的百分比变化",
            "division_by_zero",
            hint="old_value 不能为 0",
            example="percentage_change(old_value=100, new_value=120)",
        )
    return (new_value - old_value) / old_value * 100


def tip(amount: float, tip_percent: float) -> dict:
    """按百分比计算小费。"""
    tip_amount = amount * (tip_percent / 100)
    return {
        "base_amount": amount,
        "tip_percent": tip_percent,
        "tip_amount": tip_amount,
        "total_amount": amount + tip_amount,
    }


def compound_interest(
    principal: float, rate: float, time: float, compounds_per_year: int = 1
) -> dict:
    """复利本息。"""
    _require_positive_int(compounds_per_year, "compounds_per_year")
    rate_decimal = rate / 100
    amount = principal * (1 + rate_decimal / compounds_per_year) ** (
        compounds_per_year * time
    )
    return {
        "principal": principal,
        "rate": rate,
        "time": time,
        "compounds_per_year": compounds_per_year,
        "interest": amount - principal,
        "amount": amount,
    }


def simple_interest(principal: float, rate: float, time: float) -> dict:
    """单利本息。"""
    interest = principal * (rate / 100) * time
    return {
        "principal": principal,
        "rate": rate,
        "time": time,
        "interest": interest,
        "amount": principal + interest,
    }


def loan_payment(
    principal: float, annual_rate: float, years: float, payments_per_year: int = 12
) -> dict:
    """等额分期还款（年金公式）。"""
    _require_positive_int(payments_per_year, "payments_per_year")
    if years <= 0:
        raise MathError(
            "years 必须大于 0",
            "invalid_argument",
            hint="贷款期限必须为正",
        )
    if annual_rate == 0:
        payment = principal / (years * payments_per_year)
        return {
            "principal": principal,
            "annual_rate": annual_rate,
            "years": years,
            "payments_per_year": payments_per_year,
            "payment": payment,
            "total_paid": principal,
            "interest_paid": 0.0,
        }
    rate = annual_rate / 100 / payments_per_year
    n = years * payments_per_year
    payment = principal * (rate * (1 + rate) ** n) / ((1 + rate) ** n - 1)
    total_paid = payment * n
    return {
        "principal": principal,
        "annual_rate": annual_rate,
        "years": years,
        "payments_per_year": payments_per_year,
        "payment": payment,
        "total_paid": total_paid,
        "interest_paid": total_paid - principal,
    }


def split_bill(total: float, people: int, tip_percent: float = 0) -> dict:
    """账单加小费后按人数均摊。"""
    _require_positive_int(people, "people")
    tip_amount = total * (tip_percent / 100)
    total_with_tip = total + tip_amount
    return {
        "total": total,
        "tip_amount": tip_amount,
        "total_with_tip": total_with_tip,
        "per_person": total_with_tip / people,
        "people": people,
    }


def tax(amount: float, tax_rate: float, is_inclusive: bool = False) -> dict:
    """计算税费；is_inclusive 时 amount 为含税价。"""
    if is_inclusive:
        if tax_rate == -100:
            raise MathError(
                "含税价税率不能为 -100",
                "division_by_zero",
                hint="base_amount = amount / (1 + rate/100) 要求分母非零",
            )
        base_amount = amount / (1 + tax_rate / 100)
        tax_amount = amount - base_amount
    else:
        tax_amount = amount * (tax_rate / 100)
        base_amount = amount
    return {
        "base_amount": base_amount,
        "tax_rate": tax_rate,
        "tax_amount": tax_amount,
        "total_amount": base_amount + tax_amount,
    }


def _require_positive_int(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MathError(
            f"{name} 必须是不小于 1 的整数",
            "invalid_argument",
            hint=f"{name} 用于均摊或计息周期，必须为正整数",
        )


_OPS: dict = {
    "percentage": percentage,
    "percentage_change": percentage_change,
    "tip": tip,
    "compound_interest": compound_interest,
    "simple_interest": simple_interest,
    "loan_payment": loan_payment,
    "split_bill": split_bill,
    "tax": tax,
}
