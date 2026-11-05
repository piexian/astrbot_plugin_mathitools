"""schema 辅助：构造 JSON Schema object 参数块，required 显式；默认值写进 description。"""

from typing import Any


def object_parameters(properties: dict[str, Any], required: list[str] | None = None) -> dict:
    """输出 {"type":"object","properties":...,"required":[...]}；required 必须显式存在。"""
    return {"type": "object", "properties": properties, "required": list(required or [])}


def string_parameter(description: str) -> dict:
    return {"type": "string", "description": description}


def number_parameter(description: str) -> dict:
    return {"type": "number", "description": description}


def integer_parameter(description: str) -> dict:
    return {"type": "integer", "description": description}


def boolean_parameter(description: str) -> dict:
    return {"type": "boolean", "description": description}


def array_parameter(description: str, items: dict) -> dict:
    return {"type": "array", "description": description, "items": items}
