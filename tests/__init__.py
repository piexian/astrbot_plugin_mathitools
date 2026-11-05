"""测试引导：把插件父目录加入 sys.path，使插件能以包名 astrbot_plugin_mathitools 导入。"""

import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parents[2]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
