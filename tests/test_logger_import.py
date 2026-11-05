"""生产代码 logger 白名单测试：AST 静态扫描，不导入真实 astrbot。"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PRODUCTION_FILES = [
    ROOT / "main.py",
    ROOT / "schema.py",
    ROOT / "result.py",
    ROOT / "math_tools.py",
    *sorted((ROOT / "core").glob("*.py")),
]

# logger 唯一允许的导入来源
ALLOWED_LOGGER_IMPORTS = {("astrbot.api", "logger")}

# 需要写日志的生产文件必须显式 from astrbot.api import logger；其余文件禁止导入 logger
FILES_REQUIRING_LOGGER = {"main.py", "math_tools.py"}

FORBIDDEN_LOGGER_NAMES = {"logger", "log", "getlogger"}


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class LoggerImportWhitelistTest(unittest.TestCase):
    def test_production_files_exist(self):
        self.assertTrue(PRODUCTION_FILES, "未找到生产文件")
        for path in PRODUCTION_FILES:
            self.assertTrue(path.is_file(), f"缺少生产文件: {path}")

    def test_logger_import_whitelist(self):
        for path in PRODUCTION_FILES:
            for node in ast.walk(_parse(path)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        banned = (
                            module == "logging"
                            or module.startswith("logging.")
                            or module == "astrbot.core.log"
                            or module.startswith("astrbot.core.log.")
                        )
                        self.assertFalse(banned, f"{path}: 禁止 import {module}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self._check_import_from(path, module, node)

    def _check_import_from(self, path: Path, module: str, node: ast.ImportFrom) -> None:
        self.assertFalse(
            module == "logging" or module.startswith("logging."),
            f"{path}: 禁止 from {module} import ...",
        )
        if module != "astrbot" and not module.startswith("astrbot."):
            return
        for alias in node.names:
            if alias.name.lower() in FORBIDDEN_LOGGER_NAMES:
                self.assertEqual(
                    (module, alias.name),
                    ("astrbot.api", "logger"),
                    f"{path}: logger 只允许 from astrbot.api import logger，"
                    f"发现 from {module} import {alias.name}",
                )
                self.assertIn(
                    path.name,
                    FILES_REQUIRING_LOGGER,
                    f"{path}: 不在日志白名单的文件不得导入 logger",
                )

    def test_required_files_import_logger_from_astrbot_api(self):
        for path in PRODUCTION_FILES:
            if path.name not in FILES_REQUIRING_LOGGER:
                continue
            found = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "astrbot.api"
                and any(alias.name == "logger" for alias in node.names)
                for node in ast.walk(_parse(path))
            )
            self.assertTrue(found, f"{path}: 必须 from astrbot.api import logger")

    def test_no_get_logger_or_dynamic_logging_import(self):
        for path in PRODUCTION_FILES:
            for node in ast.walk(_parse(path)):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "getLogger":
                    self.fail(f"{path}: 禁止 getLogger 调用")
                if (
                    isinstance(func, ast.Name)
                    and func.id == "__import__"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and "logging" in node.args[0].value
                ):
                    self.fail(f'{path}: 禁止 __import__("logging")')

    def test_no_module_level_print(self):
        for path in PRODUCTION_FILES:
            tree = _parse(path)
            for statement in tree.body:
                for node in ast.walk(statement):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                        self.fail(f"{path}: 顶层 print(...) 调用")


if __name__ == "__main__":
    unittest.main()
