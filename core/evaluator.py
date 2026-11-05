"""asteval 白名单求值：符号表只含注册表，精度按调用隔离，复数以字符串输出。"""

import cmath
import io
import math
import statistics
import threading
from collections.abc import Callable
from typing import Any

import mpmath
from asteval import Interpreter
from mpmath import mp

from .errors import MathError
from .registry import (
    CONSTANT_NAMES,
    MATH_FUNCTIONS,
    STATISTICS_FUNCTIONS,
)
from . import security

DEFAULT_DPS = 20
MIN_DPS = 1
MAX_DPS = 200
EVAL_TIMEOUT_SECONDS = 15.0
MAX_RESULT_LENGTH = 10000
MAX_FACTORIAL_ARGUMENT = 300
_MAX_INT_DIGITS = 10000


def _safe_factorial(x: Any) -> Any:
    if x > MAX_FACTORIAL_ARGUMENT:
        raise ValueError(f"阶乘参数超过上限 {MAX_FACTORIAL_ARGUMENT}")
    return mpmath.factorial(x)


def _strict_log(name: str, fn: Callable) -> Callable:
    """对数真数为 0 时抛 ValueError（映射 domain_error），不做 -inf 静默。"""

    def wrapper(x: Any, *args: Any) -> Any:
        if x == 0:
            raise ValueError(f"{name}(0) 无定义")
        return fn(x, *args)

    return wrapper


def _strict_log1p(fn: Callable) -> Callable:
    def wrapper(x: Any) -> Any:
        if x == -1:
            raise ValueError("log1p(-1) 无定义")
        return fn(x)

    return wrapper


def _real_root(x: Any, n: Any) -> Any:
    """负实数的奇数次方根返回实根（root(-8,3) == -2），其余走 mpmath.nthroot。"""
    if x < 0 and n % 2 == 1:
        return -mpmath.nthroot(-x, n)
    return mpmath.nthroot(x, n)


# mpmath 缺失或需要固定语义的函数，显式指明实现。
_MATH_OVERRIDES: dict[str, Callable] = {
    "pow": pow,
    "abs": abs,
    "trunc": lambda x: int(x),
    "gcd": math.gcd,
    "lcm": math.lcm,
    "isqrt": math.isqrt,
    "copysign": math.copysign,
    "remainder": math.remainder,
    "modf": math.modf,
    "isclose": math.isclose,
    "comb": math.comb,
    "perm": math.perm,
    "lgamma": mpmath.loggamma,
    "nextafter": math.nextafter,
    "log": _strict_log("log", mpmath.log),
    "log10": _strict_log("log10", mpmath.log10),
    "log2": _strict_log("log2", mpmath.log2),
    "log1p": _strict_log1p(mpmath.log1p),
    "ulp": math.ulp,
    "factorial": _safe_factorial,
}

_COMPLEX_TABLE: dict[str, Callable] = {
    "phase": cmath.phase,
    "polar": cmath.polar,
    "rect": cmath.rect,
    "csin": cmath.sin,
    "ccos": cmath.cos,
    "ctan": cmath.tan,
    "cexp": cmath.exp,
    "clog": cmath.log,
    "clog10": cmath.log10,
    "csqrt": cmath.sqrt,
}

_STATS_TABLE: dict[str, Callable] = {
    name: getattr(statistics, name) for name in STATISTICS_FUNCTIONS
}

_ERROR_CODE_BY_EXC: dict[type[BaseException], str] = {
    ZeroDivisionError: "division_by_zero",
    NameError: "unknown_function",
    TypeError: "invalid_argument",
    AttributeError: "invalid_argument",
    IndexError: "invalid_argument",
    KeyError: "invalid_argument",
    ValueError: "domain_error",
    OverflowError: "domain_error",
}

_HINT_BY_CODE: dict[str, str] = {
    "division_by_zero": "分母不能为 0",
    "unknown_function": "该名称不在可用函数清单中",
    "invalid_argument": "检查参数类型与数量",
    "domain_error": "参数超出该函数的定义域",
    "internal_error": "求值出现未预期错误，请简化表达式重试",
    "exponent_too_large": "幂运算指数绝对值不能超过 10000",
}


def build_symbol_table() -> dict[str, Any]:
    """构造求值符号表：73+root 函数与 7 常量；需在 mp.workdps 内调用。"""
    table: dict[str, Any] = {}
    for name in MATH_FUNCTIONS:
        table[name] = _MATH_OVERRIDES.get(name) or getattr(mpmath, name)
    table.update(_COMPLEX_TABLE)
    table.update(_STATS_TABLE)
    table["root"] = _real_root
    table.update(
        {
            "pi": mp.pi,
            "e": mp.e,
            "tau": 2 * mp.pi,
            "inf": mp.inf,
            "nan": mp.nan,
            "phi": mp.phi,
            "euler": mp.euler,
        }
    )
    assert set(CONSTANT_NAMES) <= set(table)
    return table


def evaluate(
    expression: str, dps: int = DEFAULT_DPS, timeout_seconds: float = EVAL_TIMEOUT_SECONDS
) -> str:
    """校验并在隔离精度下求值表达式，返回结果字符串。"""
    if not isinstance(dps, int) or isinstance(dps, bool) or not (MIN_DPS <= dps <= MAX_DPS):
        raise MathError(
            f"dps 必须是 {MIN_DPS}..{MAX_DPS} 的整数",
            "invalid_argument",
            hint=f"dps 默认 {DEFAULT_DPS}，范围 {MIN_DPS}..{MAX_DPS}",
            example='evaluate("sqrt(2)", dps=30)',
        )
    security.validate(expression)

    outcome: dict[str, Any] = {}
    worker = threading.Thread(
        target=_evaluate_in_thread, args=(expression, dps, outcome), daemon=True
    )
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise MathError(
            f"表达式求值超过 {timeout_seconds:g} 秒",
            "timeout",
            hint="简化表达式后重试",
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _evaluate_in_thread(expression: str, dps: int, outcome: dict[str, Any]) -> None:
    try:
        outcome["value"] = _evaluate_sync(expression, dps)
    except MathError as exc:
        outcome["error"] = exc
    except Exception as exc:  # 兜底：底层异常不裸抛，统一映射错误码
        outcome["error"] = MathError(f"{type(exc).__name__}: {exc}", "internal_error")



def _evaluate_sync(expression: str, dps: int) -> str:
    with mp.workdps(dps):
        interpreter = Interpreter(writer=io.StringIO())
        interpreter.symtable.clear()
        interpreter.symtable.update(build_symbol_table())
        value = interpreter.eval(security.preprocess_caret(expression))
        if interpreter.error:
            raise _map_asteval_error(interpreter.error[0])
        return format_result(value, dps)


def _map_asteval_error(holder: Any) -> MathError:
    exc_type = getattr(holder, "exc", None) or type(holder)
    if not isinstance(exc_type, type):
        exc_type = type(exc_type)
    message = getattr(holder, "msg", "") or getattr(holder, "message", "") or str(holder)
    error_code = _ERROR_CODE_BY_EXC.get(exc_type, "internal_error")
    # asteval 自带 safe_pow 指数上限（10000），超限 RuntimeError 按本项目错误码映射
    if exc_type is RuntimeError and "exponent" in message:
        error_code = "exponent_too_large"
    return MathError(
        f"{exc_type.__name__}: {message}",
        error_code,
        hint=_HINT_BY_CODE.get(error_code),
    )


def format_result(value: Any, dps: int) -> str:
    """把求值结果渲染为字符串；复数输出形如 (0.0 + 1.0j)。"""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        if abs(value) >= 10**_MAX_INT_DIGITS:
            raise MathError(
                "结果过大，超出输出上限",
                "result_too_large",
                hint=f"整数结果位数不能超过 {_MAX_INT_DIGITS}",
            )
        text = str(value)
    elif isinstance(value, (mpmath.mpc, complex)):
        text = mpmath.nstr(mpmath.mpc(value), dps)
    elif isinstance(value, (mpmath.mpf, float)):
        mpf_value = mpmath.mpf(value)
        if mpmath.isfinite(mpf_value) and mpf_value == int(mpf_value):
            return format_result(int(mpf_value), dps)
        text = mpmath.nstr(value, dps)
    elif isinstance(value, (list, tuple)):
        opener, closer = ("[", "]") if isinstance(value, list) else ("(", ")")
        text = opener + ", ".join(format_result(item, dps) for item in value) + closer
    else:
        # mpmath 常量对象（pi/e/phi/euler）等先归一为 mpf
        try:
            return format_result(mpmath.mpf(value), dps)
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise MathError(
                f"不支持的结果类型: {type(value).__name__}", "internal_error"
            ) from exc
    if len(text) > MAX_RESULT_LENGTH:
        raise MathError(
            f"结果长度 {len(text)} 超过上限 {MAX_RESULT_LENGTH}",
            "result_too_large",
            hint="结果过大，请缩小计算规模",
        )
    return text
