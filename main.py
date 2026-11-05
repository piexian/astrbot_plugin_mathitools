"""astrbot_plugin_mathitools 入口：注册 10 个数学 FunctionTool，计算逻辑在 core/。"""

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context, Star

from .math_tools import (
    MathBatchTool,
    MathCompareTool,
    MathEvalTool,
    MathFinanceTool,
    MathFunctionsTool,
    MathMatrixTool,
    MathNaturalLanguageUnitsTool,
    MathNumberTheoryTool,
    MathStatisticsTool,
    MathUnitsTool,
)

DEFAULT_DPS = 20
MIN_DPS = 1
MAX_DPS = 200


def _sanitize_dps(config: AstrBotConfig) -> int:
    dps = config.get("dps", DEFAULT_DPS)
    if isinstance(dps, bool) or not isinstance(dps, int) or not MIN_DPS <= dps <= MAX_DPS:
        logger.error(f"配置 dps={dps!r} 非法，应为 {MIN_DPS}..{MAX_DPS} 的整数，回落为 {DEFAULT_DPS}")
        return DEFAULT_DPS
    return dps


class MathitoolsPlugin(Star):
    """数学工具集插件。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self._dps = _sanitize_dps(config)
        self._tools = [
            MathEvalTool(default_dps=self._dps),
            MathCompareTool(default_dps=self._dps),
            MathBatchTool(default_dps=self._dps),
            MathFunctionsTool(),
            MathStatisticsTool(),
            MathMatrixTool(),
            MathUnitsTool(),
            MathNaturalLanguageUnitsTool(),
            MathNumberTheoryTool(),
            MathFinanceTool(),
        ]
        context.add_llm_tools(*self._tools)
