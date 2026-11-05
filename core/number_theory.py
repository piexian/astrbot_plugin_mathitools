"""数论运算：is_prime、prime_factors、divisors、totient。

素性用确定性 Miller-Rabin 基（< 3.3e24 内确定，更高位为强概率判定），
分解用小素数试除 + Pollard rho，避免对大整数做 O(n) 全表扫描。
"""

import math

from .errors import MathError
from .registry import NUMBER_THEORY_OPERATIONS

_MR_BASES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_MAX_DIVISOR_COUNT = 10000


def number_theory_operation(number: int, operation: str):
    """执行指定数论运算。"""
    if operation not in NUMBER_THEORY_OPERATIONS:
        raise MathError(
            f"未知数论运算: {operation}",
            "unknown_operation",
            hint=f"可用运算: {', '.join(NUMBER_THEORY_OPERATIONS)}",
            valid_options=list(NUMBER_THEORY_OPERATIONS),
            example='number_theory_operation(360, "prime_factors")',
        )
    _require_int(number)
    if operation == "is_prime":
        return is_prime(number)
    if number < 1:
        raise MathError(
            f"{operation} 要求正整数，收到 {number}",
            "invalid_argument",
            hint="prime_factors/divisors/totient 只接受 n >= 1",
            example='number_theory_operation(360, "prime_factors")',
        )
    if operation == "prime_factors":
        return prime_factors(number)
    if operation == "divisors":
        return divisors(number)
    return totient(number)


def is_prime(n: int) -> bool:
    """Miller-Rabin 素性判定；n < 2 一律为 False。"""
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    if n < 2:
        return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_factors(n: int) -> list[int]:
    """返回 n 的素因子列表（升序，含重复）。"""
    return _factorize(n)


def divisors(n: int) -> list[int]:
    """返回 n 的全部正因子（升序）；因子个数超过上限时报结果过大。"""
    if n == 1:
        return [1]
    factors = _factorize(n)
    groups: dict[int, int] = {}
    for p in factors:
        groups[p] = groups.get(p, 0) + 1
    count = math.prod(exp + 1 for exp in groups.values())
    if count > _MAX_DIVISOR_COUNT:
        raise MathError(
            f"{n} 有 {count} 个因子，超过上限 {_MAX_DIVISOR_COUNT}",
            "result_too_large",
            hint="因子过多的整数请缩小规模",
        )
    divs = [1]
    for p, exp in groups.items():
        divs = [d * p**k for d in divs for k in range(exp + 1)]
    return sorted(divs)


def totient(n: int) -> int:
    """欧拉函数 φ(n)，基于素因式分解。"""
    if n == 1:
        return 1
    result = n
    for p in set(_factorize(n)):
        result = result // p * (p - 1)
    return result


def _require_int(number) -> None:
    if isinstance(number, bool) or not isinstance(number, int):
        raise MathError(
            f"参数必须是整数，收到 {type(number).__name__}",
            "non_integer",
            hint="数论运算只接受整数",
            example='number_theory_operation(360, "prime_factors")',
        )


def _factorize(n: int) -> list[int]:
    factors: list[int] = []
    for p in _MR_BASES:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    stack = [n]
    while stack:
        m = stack.pop()
        if is_prime(m):
            factors.append(m)
            continue
        d = _pollard_rho(m)
        stack.append(d)
        stack.append(m // d)
    return sorted(factors)


def _pollard_rho(n: int) -> int:
    """Pollard rho（Floyd 判圈），返回 n 的一个非平凡因子。"""
    if n % 2 == 0:
        return 2
    c = 1
    while True:
        x = y = 2
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d
        c += 1
