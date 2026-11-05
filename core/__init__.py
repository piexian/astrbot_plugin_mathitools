"""astrbot_plugin_mathitools 纯计算层：无 astrbot 依赖，可离线单测。"""

from .errors import ERROR_CODES, MathError
from .evaluator import evaluate

__all__ = ["ERROR_CODES", "MathError", "evaluate"]
