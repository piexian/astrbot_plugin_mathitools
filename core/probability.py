"""正态、二项、泊松分布数值计算。"""

import math
from fractions import Fraction

from mpmath import mp

from .errors import MathError
from .evaluator import format_result
from .statistics import valid_number


MAX_TRIALS = 10000
MAX_RATE = 10000
MAX_K = 100000


def _integer(value) -> bool:
    """非布尔整数。"""
    return not isinstance(value, bool) and isinstance(value, int)


def _proportion(value) -> bool:
    """0..1 的有限比例。"""
    return valid_number(value) and 0 <= value <= 1


def validate_probability_params(params, packet) -> None:
    """校验 probability 的分布参数，写入本动作所需字段。"""
    distribution, _, action = packet["operation"].partition("_")
    if distribution == "normal":
        _validate_normal(params, packet, action)
    elif distribution == "binomial":
        _validate_binomial(params, packet, action)
    else:
        _validate_poisson(params, packet, action)


def _validate_normal(params, packet, action) -> None:
    """mu/sigma 默认 0/1，按动作写 x 或 q。"""
    mu = params.get("mu", 0)
    sigma = params.get("sigma", 1)
    if not valid_number(mu):
        raise MathError("mu 必须是有限数字", "invalid_argument")
    if not valid_number(sigma) or sigma <= 0:
        raise MathError("sigma 必须是正的有限数字", "invalid_argument")
    if action == "ppf":
        q = params.get("q")
        if not _proportion(q):
            raise MathError("q 必须是 0..1 的有限数字", "invalid_argument")
        packet["q"] = q
    else:
        x = params.get("x")
        if not valid_number(x):
            raise MathError("x 必须是有限数字", "invalid_argument")
        packet["x"] = x
    packet["mu"] = mu
    packet["sigma"] = sigma


def _validate_binomial(params, packet, action) -> None:
    """n 0..MAX_TRIALS，p 0..1；k 允许负，q 0..1。"""
    n = params.get("n")
    p = params.get("p")
    if not _integer(n) or not 0 <= n <= MAX_TRIALS:
        raise MathError(f"n 必须是 0..{MAX_TRIALS} 的整数", "invalid_argument")
    if not _proportion(p):
        raise MathError("p 必须是 0..1 的有限数字", "invalid_argument")
    packet["n"] = n
    packet["p"] = p
    if action == "ppf":
        q = params.get("q")
        if not _proportion(q):
            raise MathError("q 必须是 0..1 的有限数字", "invalid_argument")
        packet["q"] = q
    else:
        k = params.get("k")
        if not _integer(k) or abs(k) > MAX_K:
            raise MathError(f"k 必须是整数且绝对值不超过 {MAX_K}", "invalid_argument")
        packet["k"] = k


def _validate_poisson(params, packet, action) -> None:
    """rate 0..MAX_RATE；k 允许负，q 0..1。"""
    rate = params.get("rate")
    if not valid_number(rate) or not 0 <= rate <= MAX_RATE:
        raise MathError(f"rate 必须是 0..{MAX_RATE} 的有限数字", "invalid_argument")
    packet["rate"] = rate
    if action == "ppf":
        q = params.get("q")
        if not _proportion(q):
            raise MathError("q 必须是 0..1 的有限数字", "invalid_argument")
        packet["q"] = q
    else:
        k = params.get("k")
        if not _integer(k) or abs(k) > MAX_K:
            raise MathError(f"k 必须是整数且绝对值不超过 {MAX_K}", "invalid_argument")
        packet["k"] = k


def _binomial_terms(n, a, b):
    """惰性精确质量序列 term_j = comb(n, j) * a^j * (b-a)^(n-j)。"""
    term = (b - a) ** n
    for index in range(n + 1):
        yield term
        if index < n:
            term = term * (n - index) * a // ((index + 1) * (b - a))


def _binomial(packet) -> mp.mpf:
    """二项分布 pmf/cdf/ppf 精确计算，返回 mp.mpf。"""
    action = packet["operation"].partition("_")[2]
    n, p = packet["n"], packet["p"]
    frac = Fraction(p)
    a, b = frac.numerator, frac.denominator
    denominator_power = (b.bit_length() - 1) * n
    denominator = 1 << denominator_power
    dps = packet["dps"]
    if n == 0 or p == 0:
        if action == "ppf":
            return mp.mpf(0)
        k = packet["k"]
        if action == "pmf":
            return mp.mpf(1 if k == 0 else 0)
        return mp.mpf(0 if k < 0 else 1)
    if p == 1:
        if action == "ppf":
            return mp.mpf(0 if packet["q"] == 0 else n)
        k = packet["k"]
        if action == "cdf":
            return mp.mpf(1 if k >= n else 0)
    if action == "pmf":
        k = packet["k"]
        if k < 0 or k > n:
            return mp.mpf(0)
        numerator = math.comb(n, k) * a**k * (b - a) ** (n - k)
        with mp.workdps(dps + 30):
            # 分母为 2 的幂，ldexp 直接缩放，避免 mpmath 逐字节归一化巨分母
            return mp.ldexp(mp.mpf(numerator), -denominator_power)
    if action == "cdf":
        k = packet["k"]
        if k < 0:
            return mp.mpf(0)
        if k >= n:
            return mp.mpf(1)
        # 提前终止有严格上界：剩余质量存在时真 CDF 仍在 [累计/分母, 1]
        cutoff = denominator - denominator // 10 ** (dps + 10)
        cumulative = 0
        for index, term in enumerate(_binomial_terms(n, a, b)):
            cumulative += term
            if cumulative >= cutoff:
                return mp.mpf(1)
            if index == k:
                with mp.workdps(dps + 30):
                    return mp.ldexp(mp.mpf(cumulative), -denominator_power)
    q = packet["q"]
    if q == 0:
        return mp.mpf(0)
    if q == 1:
        return mp.mpf(n)
    target = math.ceil(Fraction(q) * denominator)
    cumulative = 0
    for index, term in enumerate(_binomial_terms(n, a, b)):
        cumulative += term
        if cumulative >= target:
            return mp.mpf(index)
    return mp.mpf(n)


def _normal(packet) -> mp.mpf:
    """正态分布 pdf/cdf/ppf 数值，返回 mp.mpf。"""
    action = packet["operation"].partition("_")[2]
    with mp.workdps(packet["dps"] + 30):
        span = max(
            [0]
            + [abs(int(mp.mag(v))) for v in (packet["mu"], packet["sigma"]) if v]
        )
        key = packet["q"] if action == "ppf" else packet["x"]
        if key:
            span = max(span, abs(int(mp.mag(key))))
        # 四倍保护位：极端 x/sigma 比例的平方跨上千个量级，q≈1e-300 防舍入
        with mp.extraprec(4 * span):
            mu = mp.mpf(packet["mu"])
            sigma = mp.mpf(packet["sigma"])
            if action == "pdf":
                return mp.npdf((mp.mpf(packet["x"]) - mu) / sigma) / sigma
            if action == "cdf":
                # 上不完全伽马避开 erfc 的大整数转 float 溢出
                z = (mp.mpf(packet["x"]) - mu) / sigma
                tail = mp.gammainc(mp.mpf(1) / 2, z * z / 2, mp.inf, regularized=True) / 2
                return tail if z < 0 else 1 - tail
            q = mp.mpf(packet["q"])
            if q == 0 or q == 1:
                raise MathError(
                    "q=0 或 q=1 无有限分位数",
                    "domain_error",
                    hint="q 必须严格介于 0 与 1 之间",
                )
            return mu + sigma * mp.sqrt(2) * mp.erfinv(2 * q - 1)


def _poisson(packet) -> mp.mpf:
    """泊松分布 pmf/cdf/ppf 数值计算，返回 mp.mpf。"""
    action = packet["operation"].partition("_")[2]
    dps = packet["dps"]
    k = packet.get("k")
    q = packet.get("q")
    if action != "ppf" and k < 0:
        return mp.mpf(0)
    with mp.workdps(dps + 30):
        rate = mp.mpf(packet["rate"])
        if rate == 0:
            if action == "ppf":
                return mp.mpf(0)
            if action == "cdf":
                return mp.mpf(1)
            return mp.mpf(1 if k == 0 else 0)
        if action == "pmf":
            return mp.exp(k * mp.log(rate) - rate - mp.loggamma(k + 1))
        if action == "cdf":
            return mp.gammainc(k + 1, rate, mp.inf, regularized=True)
        if q == 0:
            return mp.mpf(0)
        if q == 1:
            raise MathError(
                "泊松分布的 q=1 无有限分位数",
                "domain_error",
                hint="泊松分布的支撑无上界",
            )
        low, high = -1, max(1, math.ceil(rate))
        while _poisson(packet | {"operation": "poisson_cdf", "k": high}) < mp.mpf(q):
            if high >= MAX_K:
                raise MathError(
                    "泊松分位数搜索超过上限",
                    "no_convergence",
                    hint=f"分位数索引不超过 {MAX_K}",
                )
            high = min(high * 2, MAX_K)
        while low + 1 < high:
            middle = (low + high) // 2
            if _poisson(packet | {"operation": "poisson_cdf", "k": middle}) >= mp.mpf(q):
                high = middle
            else:
                low = middle
        return mp.mpf(high)


def calculate_probability(packet) -> str:
    """按分布分派计算并统一格式化输出。"""
    with mp.workdps(packet["dps"] + 30):
        distribution = packet["operation"].partition("_")[0]
        try:
            if distribution == "normal":
                value = _normal(packet)
            elif distribution == "binomial":
                value = _binomial(packet)
            else:
                value = _poisson(packet)
        except mp.NoConvergence as exc:
            raise MathError("概率计算未收敛", "no_convergence") from exc
        if not mp.isfinite(value):
            raise MathError("结果非有限", "domain_error", hint="结果不能是 NaN 或 Infinity")
        action = packet["operation"].partition("_")[2]
        if action in ("cdf", "pmf") and not 0 <= value <= 1:
            raise MathError("概率越界", "no_convergence", hint="pmf/cdf 必须在 0..1 内")
        return format_result(value, packet["dps"])
