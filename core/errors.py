"""结构化错误对象：MathError 携带稳定 error_code 与 AI 友好提示。"""

ERROR_CODES: frozenset[str] = frozenset(
    {
        "invalid_argument",
        "unknown_function",
        "unknown_operation",
        "unknown_unit",
        "syntax_error",
        "division_by_zero",
        "domain_error",
        "complex_result",
        "expression_too_long",
        "expression_too_complex",
        "exponent_too_large",
        "result_too_large",
        "timeout",
        "matrix_dimension",
        "singular_matrix",
        "non_integer",
        "parse_failed",
        "internal_error",
    }
)


class MathError(ValueError):
    """计算层唯一异常类型，字段直接映射到错误 JSON。"""

    def __init__(
        self,
        message: str,
        error_code: str,
        *,
        hint: str | None = None,
        example: str | None = None,
        valid_options: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.hint = hint
        self.example = example
        self.valid_options = list(valid_options) if valid_options is not None else None
