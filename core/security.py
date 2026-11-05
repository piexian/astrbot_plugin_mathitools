"""表达式静态校验：长度、控制字符、禁用模式、AST 白名单、深度/列表/指数上限。"""

import ast
import re

from .errors import MathError
from .registry import FORBIDDEN_PATTERNS, OPERATORS

MAX_EXPRESSION_LENGTH = 1000
MAX_AST_DEPTH = 10
MAX_LIST_ELEMENTS = 10000
MAX_EXPONENT = 10000

_ALLOWED_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Call,
        ast.keyword,
        ast.BinOp,
        ast.UnaryOp,
        ast.List,
        *OPERATORS,
    }
)

_ALLOWED_CONSTANT_TYPES: tuple[type, ...] = (int, float, complex, bool)

_FORBIDDEN_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in FORBIDDEN_PATTERNS
)

def preprocess_caret(expression: str) -> str:
    """上游约定 ^ 为乘方：把 ^ 替换为 **；引号内文本跳过，不动字符串字面量。"""
    chars: list[str] = []
    quote = ""
    for ch in expression:
        if quote:
            chars.append(ch)
            if ch == quote:
                quote = ""
        elif ch in ("'", '"'):
            quote = ch
            chars.append(ch)
        elif ch == "^":
            chars.append("**")
        else:
            chars.append(ch)
    return "".join(chars)



def validate(expression: str) -> ast.Expression:
    """校验表达式并返回 AST；违规时抛出带错误码的 MathError。"""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise MathError(
            f"表达式长度 {len(expression)} 超过上限 {MAX_EXPRESSION_LENGTH}",
            "expression_too_long",
            hint=f"表达式长度不能超过 {MAX_EXPRESSION_LENGTH} 个字符",
            example="sin(pi/4) * 2",
        )
    if any(ord(ch) < 32 for ch in expression):
        raise MathError(
            "表达式包含控制字符",
            "invalid_argument",
            hint="只允许可见字符与空格",
        )
    for pattern in _FORBIDDEN_RES:
        if pattern.search(expression):
            raise MathError(
                "表达式包含被禁止的构造",
                "invalid_argument",
                hint="不允许 import、exec、eval、双下划线属性等构造",
                example="sqrt(2) + log(e)",
            )
    expression = preprocess_caret(expression)

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise MathError(
            f"表达式语法错误: {exc.msg}",
            "syntax_error",
            hint="只支持算术、函数调用与列表字面量",
            example="sqrt(2) + log(e)",
        ) from exc
    except (ValueError, RecursionError, MemoryError) as exc:
        raise MathError(
            "表达式过于复杂，无法解析",
            "expression_too_complex",
            hint=f"AST 深度不能超过 {MAX_AST_DEPTH}",
        ) from exc

    _walk(tree)
    return tree


def _walk(tree: ast.Expression) -> None:
    if _node_depth(tree) > MAX_AST_DEPTH:
        raise MathError(
            f"表达式 AST 深度超过上限 {MAX_AST_DEPTH}",
            "expression_too_complex",
            hint="减少嵌套层级，分步计算",
            example="(1+2) * (3+4)",
        )
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise MathError(
                f"表达式包含不支持的语法: {type(node).__name__}",
                "invalid_argument",
                hint="只支持算术运算、函数调用、名称引用与列表字面量",
            )
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, _ALLOWED_CONSTANT_TYPES
        ):
            raise MathError(
                "表达式包含不支持的常量类型",
                "invalid_argument",
                hint="只支持数字字面量",
            )
        if isinstance(node, ast.List) and len(node.elts) > MAX_LIST_ELEMENTS:
            raise MathError(
                f"列表元素数量超过上限 {MAX_LIST_ELEMENTS}",
                "invalid_argument",
                hint=f"每个列表最多 {MAX_LIST_ELEMENTS} 个元素",
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            _check_exponent(node.right)


def _check_exponent(node: ast.expr) -> None:
    """静态检查幂指数常数（含一元正负号包裹），超限即拒绝。"""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float, complex))
        and abs(node.value) > MAX_EXPONENT
    ):
        raise MathError(
            f"指数超过上限 {MAX_EXPONENT}",
            "exponent_too_large",
            hint=f"幂运算指数的绝对值不能超过 {MAX_EXPONENT}",
            example="2**10",
        )


def _node_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + max((_node_depth(child) for child in children), default=0)
