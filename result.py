"""JSON 字符串结果封装：ok=true 成功、ok=false 失败；不用 CallToolResult。"""

import json
from typing import Any


def ok_result(**fields: Any) -> str:
    """成功结果：{"ok": true, ...fields}。"""
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, allow_nan=False)


def error_result(
    error: str,
    error_code: str,
    *,
    hint: str | None = None,
    example: str | None = None,
    valid_options: list | None = None,
) -> str:
    """失败结果：至少 ok/error/error_code/hint，可选 example 与 valid_options。"""
    payload: dict[str, Any] = {
        "ok": False,
        "error": error,
        "error_code": error_code,
        "hint": hint,
    }
    if example is not None:
        payload["example"] = example
    if valid_options is not None:
        payload["valid_options"] = valid_options
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)
