"""asteval 白名单求值：符号表只含注册表，精度按调用隔离，复数以字符串输出。"""

import ast
import cmath
import io
import math
import operator
import statistics
import threading
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import mpmath
from asteval import Interpreter
from asteval.astutils import op2func
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
_MAX_INTEGER_MAGNITUDE = 10**_MAX_INT_DIGITS


def _safe_factorial(x: Any) -> Any:
    if x > MAX_FACTORIAL_ARGUMENT:
        raise ValueError(f"阶乘参数超过上限 {MAX_FACTORIAL_ARGUMENT}")
    return mpmath.factorial(x)


def _collection_size(value: Any) -> int:
    """统计嵌套元素总数并检查上限。"""
    if not isinstance(value, (list, tuple)):
        return 0
    total = len(value)
    if total > security.MAX_LIST_ELEMENTS:
        raise MathError(
            f"集合元素数量超过上限 {security.MAX_LIST_ELEMENTS}",
            "invalid_argument",
            hint=f"列表最多 {security.MAX_LIST_ELEMENTS} 个元素",
        )
    for item in value:
        if isinstance(item, (list, tuple)):
            total += _collection_size(item)
        if total > security.MAX_LIST_ELEMENTS:
            raise MathError(
                f"集合元素数量超过上限 {security.MAX_LIST_ELEMENTS}",
                "invalid_argument",
                hint=f"列表最多 {security.MAX_LIST_ELEMENTS} 个元素",
            )
    return total


def _check_integer_budget(value: Any) -> None:
    """结果数值超过整数位数上限时抛 result_too_large。"""
    if isinstance(value, int) and abs(value) >= _MAX_INTEGER_MAGNITUDE:
        raise MathError(
            "结果过大，超出输出上限",
            "result_too_large",
            hint=f"整数结果位数不能超过 {_MAX_INT_DIGITS}",
        )
    if isinstance(value, (float, mpmath.mpf)) and mpmath.isfinite(value):
        if mpmath.mpf(abs(value)) >= _MAX_INTEGER_MAGNITUDE:
            raise MathError(
                "结果过大，超出输出上限",
                "result_too_large",
                hint=f"整数结果位数不能超过 {_MAX_INT_DIGITS}",
            )


def _bounded_multiply(left: Any, right: Any) -> Any:
    """序列乘整数先做预算检查，再走原 safe_mult，最后复核结果预算。"""
    for sequence, multiplier in ((left, right), (right, left)):
        if isinstance(sequence, (list, tuple)) and isinstance(multiplier, (int, bool)):
            count = _collection_size(sequence) * max(int(multiplier), 0)
            if count > security.MAX_LIST_ELEMENTS:
                raise MathError(
                    f"乘法结果元素数量超过上限 {security.MAX_LIST_ELEMENTS}",
                    "invalid_argument",
                    hint=f"列表最多 {security.MAX_LIST_ELEMENTS} 个元素",
                )
    result = op2func(ast.Mult())(left, right)
    if isinstance(result, (list, tuple)):
        _collection_size(result)
    else:
        _check_integer_budget(result)
    return result


def _bounded_add(left: Any, right: Any) -> Any:
    """序列拼接先做预算检查，再走原 safe_add，最后复核结果预算。"""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        count = _collection_size(left) + _collection_size(right)
        if count > security.MAX_LIST_ELEMENTS:
            raise MathError(
                f"拼接结果元素数量超过上限 {security.MAX_LIST_ELEMENTS}",
                "invalid_argument",
                hint=f"列表最多 {security.MAX_LIST_ELEMENTS} 个元素",
            )
    result = op2func(ast.Add())(left, right)
    if isinstance(result, (list, tuple)):
        _collection_size(result)
    else:
        _check_integer_budget(result)
    return result


def _safe_sum(iterable: Any, /, start: Any = 0) -> Any:
    """保留内建 sum 语义，先检查容器与数值预算。"""
    if isinstance(iterable, (list, tuple)):
        _collection_size(iterable)
    if isinstance(start, (list, tuple)):
        total = _collection_size(start)
        for item in iterable:
            total += _collection_size(item)
            if total > security.MAX_LIST_ELEMENTS:
                raise MathError(
                    f"求和结果元素数量超过上限 {security.MAX_LIST_ELEMENTS}",
                    "invalid_argument",
                    hint=f"列表最多 {security.MAX_LIST_ELEMENTS} 个元素",
                )
    _check_integer_budget(start)
    result = sum(iterable, start)
    if isinstance(result, (list, tuple)):
        _collection_size(result)
    else:
        _check_integer_budget(result)
    return result


def _safe_prod(iterable: Any, /, *, start: Any = 1) -> Any:
    """逐项 _bounded_multiply 累乘，空输入保留 start 类型。"""
    _collection_size(iterable)
    if isinstance(start, (list, tuple)):
        _collection_size(start)
    _check_integer_budget(start)
    result = start
    for item in iterable:
        # 序列乘 1 内容不变，跳过 _bounded_multiply 避免每次全量重扫
        if isinstance(result, (list, tuple)) and isinstance(item, int) and item == 1:
            continue
        result = _bounded_multiply(result, item)
    if isinstance(result, (list, tuple)):
        _collection_size(result)
    else:
        _check_integer_budget(result)
    return result


def _safe_round(number: Any, ndigits: Any = None) -> Any:
    """保留原生 round 语义，先做整数预算检查避免大数分配。"""
    if ndigits is not None:
        ndigits = operator.index(ndigits)
        if abs(ndigits) > _MAX_INT_DIGITS:
            raise MathError(
                "round 的 ndigits 超过上限",
                "invalid_argument",
                hint=f"ndigits 绝对值不能超过 {_MAX_INT_DIGITS}",
            )
    _check_integer_budget(number)
    if ndigits is None:
        result = round(number)
    else:
        result = round(number, ndigits)
    _check_integer_budget(result)
    return result


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
    "sum": _safe_sum,
    "prod": _safe_prod,
    "min": min,
    "max": max,
    "round": _safe_round,
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
    """构造求值符号表：78+root 函数与 7 常量；需在 mp.workdps 内调用。"""
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



class _BudgetInterpreter(Interpreter):
    """带集合/数值预算检查的局部 Interpreter，避免超限分配。"""

    def on_list(self, node: Any) -> Any:
        items = []
        total = 0
        for item in node.elts:
            value = self.run(item)
            total += 1 + _collection_size(value)
            if total > security.MAX_LIST_ELEMENTS:
                raise MathError(
                    f"列表元素数量超过上限 {security.MAX_LIST_ELEMENTS}",
                    "invalid_argument",
                    hint=f"每个列表最多 {security.MAX_LIST_ELEMENTS} 个元素",
                )
            items.append(value)
        return items

    def on_binop(self, node: Any) -> Any:
        left = self.run(node.left)
        right = self.run(node.right)
        if isinstance(node.op, ast.Add):
            result = _bounded_add(left, right)
        elif isinstance(node.op, ast.Mult):
            result = _bounded_multiply(left, right)
        else:
            result = op2func(node.op)(left, right)
        if isinstance(result, (list, tuple)):
            _collection_size(result)
        else:
            _check_integer_budget(result)
        return result

    def on_call(self, node: Any) -> Any:
        func = self.run(node.func)
        args = [self.run(arg) for arg in node.args]
        kwargs: dict[str, Any] = {}
        for arg in args:
            _collection_size(arg)
        for kw in node.keywords:
            if kw.arg is None:
                raise MathError("不支持 ** 展开参数", "invalid_argument")
            if kw.arg in kwargs:
                raise MathError(f"重复的关键字参数: {kw.arg}", "invalid_argument")
            value = self.run(kw.value)
            _collection_size(value)
            kwargs[kw.arg] = value
        result = func(*args, **kwargs)
        if isinstance(result, (list, tuple)):
            _collection_size(result)
        else:
            _check_integer_budget(result)
        return result


def _evaluate_sync(expression: str, dps: int) -> str:
    with mp.workdps(dps):
        interpreter = _BudgetInterpreter(writer=io.StringIO())
        interpreter.symtable.clear()
        interpreter.symtable.update(build_symbol_table())
        value = interpreter.eval(security.preprocess_caret(expression))
        if interpreter.error:
            raise _map_asteval_error(interpreter.error[0])
        return format_result(value, dps)


def _map_asteval_error(holder: Any) -> MathError:
    original = getattr(holder, "exc_info", None)
    if original:
        exc = original[1]
        if isinstance(exc, MathError):
            return exc
    exc_type = getattr(holder, "exc", None) or type(holder)
    if not isinstance(exc_type, type):
        exc_type = type(exc_type)
    message = getattr(holder, "msg", "") or getattr(holder, "message", "") or str(holder)
    message = message[:200]
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
        if abs(value) >= _MAX_INTEGER_MAGNITUDE:
            raise MathError(
                "结果过大，超出输出上限",
                "result_too_large",
                hint=f"整数结果位数不能超过 {_MAX_INT_DIGITS}",
            )
        text = str(Decimal(value))
    elif isinstance(value, (mpmath.mpc, complex)):
        text = mpmath.nstr(mpmath.mpc(value), dps)
    elif isinstance(value, (mpmath.mpf, float)):
        mpf_value = mpmath.mpf(value)
        _check_integer_budget(mpf_value)
        if mpmath.isfinite(mpf_value) and mpf_value == int(mpf_value):
            return format_result(int(mpf_value), dps)
        text = mpmath.nstr(value, dps)
    elif isinstance(value, (list, tuple)):
        opener, closer = ("[", "]") if isinstance(value, list) else ("(", ")")
        pieces = []
        total = len(opener) + len(closer)
        for item in value:
            piece = format_result(item, dps)
            total += len(piece) + (2 if pieces else 0)
            if total > MAX_RESULT_LENGTH:
                raise MathError(
                    f"结果长度超过上限 {MAX_RESULT_LENGTH}",
                    "result_too_large",
                    hint="结果过大，请缩小计算规模",
                )
            pieces.append(piece)
        text = opener + ", ".join(pieces) + closer
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
