"""数值求根：findroot 二分与 polyroots 的同步基础。"""

import asyncio
import ast
import io
import json
import os
import sys
import threading

from mpmath import mp

from .errors import MathError
from .evaluator import (
    DEFAULT_DPS,
    EVAL_TIMEOUT_SECONDS,
    MAX_DPS,
    MAX_RESULT_LENGTH,
    MIN_DPS,
    _BudgetInterpreter,
    _map_asteval_error,
    build_symbol_table,
    format_result,
)
from .registry import NUMERICAL_OPERATIONS
from .security import validate
from .statistics import valid_number


def _expression_function(expression: str):
    """把已校验表达式编译为反复绑定 mp.mpf(x) 的回调。"""
    node = validate(expression)
    interpreter = _BudgetInterpreter(writer=io.StringIO())
    interpreter.symtable.clear()
    interpreter.symtable.update(build_symbol_table())
    allowed = set(interpreter.symtable) | {"x"}
    for name_node in ast.walk(node):
        if isinstance(name_node, ast.Name) and name_node.id not in allowed:
            raise MathError(
                f"未知名称: {name_node.id}",
                "unknown_function",
                hint="只支持注册的函数、常量与自变量 x",
            )

    def callback(x_value):
        interpreter.symtable["x"] = mp.mpf(x_value)
        try:
            result = interpreter.eval(node.body, raise_errors=True)
        except MemoryError:
            raise
        except Exception:
            if interpreter.error:
                holder = interpreter.error[0]
                if holder.exc is MemoryError:
                    raise MemoryError("数值计算超出工作进程内存限额") from None
                raise _map_asteval_error(holder)
            raise
        if isinstance(result, bool) or not isinstance(result, (int, float, mp.mpf)):
            raise MathError(
                "表达式结果不是有限实数",
                "domain_error",
                hint="不接受复数、容器或布尔结果",
            )
        value = mp.mpf(result)
        if not mp.isfinite(value):
            raise MathError(
                "表达式结果不是有限实数",
                "domain_error",
                hint="结果不能是 NaN 或 Infinity",
            )
        return value

    return callback


def _validate_request(operation, dps, params) -> dict:
    """校验 operation/dps 与各 operation 参数，返回规范化 packet。"""
    if not isinstance(operation, str) or not operation:
        raise MathError(
            "operation 必须是非空字符串",
            "invalid_argument",
            example='numerical_operation("findroot", expression="x^2-2", a=0, b=2)',
        )
    if operation not in NUMERICAL_OPERATIONS:
        raise MathError(
            f"未知数值运算: {operation}",
            "unknown_operation",
            hint=f"可用运算: {', '.join(NUMERICAL_OPERATIONS)}",
            valid_options=list(NUMERICAL_OPERATIONS),
            example='numerical_operation("findroot", expression="x^2-2", a=0, b=2)',
        )
    if isinstance(dps, bool) or not isinstance(dps, int) or not MIN_DPS <= dps <= MAX_DPS:
        raise MathError(
            f"dps 必须是 {MIN_DPS}..{MAX_DPS} 的整数",
            "invalid_argument",
            hint="dps 为结果的有效十进制位数",
        )
    packet: dict = {"operation": operation, "dps": dps}
    if operation == "findroot":
        _validate_findroot_params(params, packet)
    else:
        _validate_polyroots_params(params, packet)
    return packet


def _validate_findroot_params(params, packet) -> None:
    """校验 findroot 的 expression 与区间端点 a/b。"""
    expression = params.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise MathError(
            "expression 必须是非空字符串",
            "invalid_argument",
            example='findroot(expression="x^2 - 2", a=0, b=2)',
        )
    validate(expression)
    a, b = params.get("a"), params.get("b")
    if not valid_number(a) or not valid_number(b):
        raise MathError(
            "a 和 b 必须是有限数字",
            "invalid_argument",
            hint="不接受布尔值、字符串、NaN 或 Infinity",
        )
    if not a < b:
        raise MathError(
            "必须满足 a < b",
            "invalid_argument",
            hint="a 是区间左端点，b 是右端点",
        )
    packet["expression"] = expression
    packet["a"] = a
    packet["b"] = b


def _validate_polyroots_params(params, packet) -> None:
    """校验 polyroots 的降幂系数列表，只做参数验证。"""
    coeffs = params.get("coefficients")
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) == 0:
        raise MathError(
            "coefficients 必须是非空数字列表",
            "invalid_argument",
            hint="按降幂顺序给出，如 [2, -3, 1] 表示 2x^2 - 3x + 1",
        )
    if len(coeffs) > 21:
        raise MathError(
            "coefficients 最多 21 项（20 次多项式）",
            "invalid_argument",
            hint="次数上限 20",
        )
    if any(not valid_number(entry) for entry in coeffs):
        raise MathError(
            "coefficients 元素必须是有限数字",
            "invalid_argument",
            hint="不接受布尔值、字符串、NaN 或 Infinity",
        )
    if all(entry == 0 for entry in coeffs):
        raise MathError(
            "零多项式没有确定的有限根集",
            "domain_error",
            hint="coefficients 不能全为 0",
        )
    if coeffs[0] == 0:
        raise MathError("前导系数不能为 0", "invalid_argument", hint="请先去掉前导零项")
    packet["coefficients"] = list(coeffs)


def _findroot(packet) -> str:
    """在连续区间 [a, b] 上用二分法求根，返回格式化字符串。"""
    dps = packet["dps"]
    with mp.workdps(dps + 30):
        f = _expression_function(packet["expression"])
        a, b = mp.mpf(packet["a"]), mp.mpf(packet["b"])
        fa, fb = f(a), f(b)
        if fa == 0:
            return format_result(a, dps)
        if fb == 0:
            return format_result(b, dps)
        if (fa > 0) == (fb > 0):
            raise MathError(
                "区间端点函数值同号",
                "domain_error",
                hint="要求函数在区间上连续且端点异号",
            )
        f_init = max(abs(fa), abs(fb))
        tol = mp.mpf(10) ** (-dps)
        for _ in range(4096):
            mid = (a + b) / 2
            fm = f(mid)
            if fm == 0:
                return format_result(mid, dps)
            # 微小根需要相对宽度：max(1,...) 会提前满足并错报
            accepted_width = abs(b - a) <= tol * abs(mid)
            accepted_residual = abs(fm) <= tol * f_init
            if accepted_width and accepted_residual:
                return format_result(mid, dps)
            if mid == a or mid == b:
                raise MathError(
                    "二分区间宽度无法继续推进",
                    "no_convergence",
                    hint="尝试放宽 dps 或调整区间",
                )
            if (fa > 0) != (fm > 0):
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        raise MathError(
            "达到最大迭代步数仍未收敛",
            "no_convergence",
            hint="尝试放宽 dps 或调整区间",
        )


def _polyroots(packet) -> list[str]:
    """求多项式全部根，返回字符串数组。"""
    dps = packet["dps"]
    coeffs = list(packet["coefficients"])
    zero_count = 0
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
        zero_count += 1
    if len(coeffs) == 1:
        return [format_result(mp.mpf(0), dps) for _ in range(zero_count)]
    with mp.workdps(dps + 30):
        try:
            roots = mp.polyroots(
                [mp.mpf(entry) for entry in coeffs],
                asc=False,
                cleanup=False,
                maxsteps=200,
                extraprec=100,
            )
        except mp.NoConvergence as exc:
            raise MathError(
                "多项式求根未收敛",
                "no_convergence",
                hint="多项式可能有重根或病态分布的根",
            ) from exc
        out: list[str] = []
        for root in roots:
            if not mp.isfinite(root):
                raise MathError(
                    "多项式存在非有限根",
                    "domain_error",
                    hint="结果不能是 NaN 或 Infinity",
                )
            if mp.im(root) == 0:
                root = mp.re(root)
            out.append(format_result(root, dps))
        out.extend(format_result(mp.mpf(0), dps) for _ in range(zero_count))
    if len(json.dumps(out, ensure_ascii=False)) > MAX_RESULT_LENGTH:
        raise MathError(
            "根列表超过输出上限",
            "result_too_large",
            hint=f"总长度不能超过 {MAX_RESULT_LENGTH} 字符",
        )
    return out


def _calculate(packet):
    """按 packet 分派到对应同步算法。"""
    if packet["operation"] == "findroot":
        return _findroot(packet)
    return _polyroots(packet)


_WORKER_SLOTS = threading.BoundedSemaphore(2)
_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_numerical_worker.py")


def _recover_error(payload: dict) -> MathError:
    """把 worker 的错误协议转回 MathError。"""
    return MathError(
        payload.get("error") or "数值运算失败",
        payload.get("error_code") or "internal_error",
        hint=payload.get("hint"),
        example=payload.get("example"),
        valid_options=payload.get("valid_options"),
    )


async def numerical_operation(
    operation: str, *, dps: int = DEFAULT_DPS, timeout_seconds: float = 15.0, **params
) -> str | list[str]:
    """在受限独立进程中执行数值求根，成功返回结果。"""
    packet = _validate_request(operation, dps, params)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < timeout_seconds <= EVAL_TIMEOUT_SECONDS
    ):
        raise MathError(
            f"timeout_seconds 必须是 (0, {EVAL_TIMEOUT_SECONDS:g}] 的有限数字",
            "invalid_argument",
            hint=f"默认 {EVAL_TIMEOUT_SECONDS:g} 秒",
        )
    encoded = json.dumps(packet, allow_nan=False).encode()
    if not _WORKER_SLOTS.acquire(blocking=False):
        raise MathError(
            "数值运算并发已达上限",
            "timeout",
            hint="请稍后重试",
        )
    proc = None
    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-B",
                _WORKER_PATH,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
        except Exception as exc:
            raise MathError(
                f"无法启动数值工作进程: {exc}"[:200], "internal_error"
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(encoded), timeout_seconds
            )
        except asyncio.TimeoutError:
            raise MathError(
                f"数值运算超过 {timeout_seconds:g} 秒",
                "timeout",
                hint="简化问题、降低 dps 或重试",
            )
        if proc.returncode in (-24, -9):
            raise MathError(
                "数值运算超时或被系统终止",
                "timeout",
                hint="问题规模超出资源限额",
            )
        if proc.returncode != 0:
            raise MathError(
                f"工作进程异常退出 code={proc.returncode} {stderr[:200].decode(errors='replace')}",
                "internal_error",
            )
        try:
            payload = json.loads(stdout.decode())
        except (ValueError, UnicodeDecodeError) as exc:
            raise MathError(
                f"工作进程输出不是合法 JSON: {exc}"[:200], "internal_error"
            ) from exc
        if not payload.get("ok"):
            raise _recover_error(payload)
        return payload["value"]
    finally:
        try:
            if proc is not None:
                if proc.returncode is None:
                    proc.kill()
                await proc.communicate()
        finally:
            _WORKER_SLOTS.release()
