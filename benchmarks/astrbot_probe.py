"""真实加载 AstrBot、注册 11 个数学工具并执行冒烟调用。"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

EXPECTED_TOOLS = [
    "math_eval", "math_compare", "math_batch", "math_functions", "math_matrix",
    "math_numerical", "math_units", "math_nl_units", "math_number_theory",
    "math_statistics", "math_finance",
]

TOOL_FILES = [
    "main.py", "math_tools.py", "schema.py", "result.py",
    "metadata.yaml", "_conf_schema.json", "requirements.txt",
]


def prepare_root(repo: Path, work_root: Path) -> tuple[Path, dict[str, str]]:
    work_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="probe_", dir=work_root))
    for child in ("data", "tmp", "home"):
        (root / child).mkdir()
    os.environ.update(
        ASTRBOT_ROOT=str(root),
        ASTRBOT_DISABLE_METRICS="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TMPDIR=str(root / "tmp"),
        TEMP=str(root / "tmp"),
        TMP=str(root / "tmp"),
    )
    tempfile.tempdir = None
    os.chdir(root)
    sys.path.insert(0, str(root))
    config = {
        "t2i_endpoint": "http://127.0.0.1:9/t2i",
        "disable_metrics": True,
        "dashboard": {"enable": False},
        "provider": [],
        "provider_sources": [],
        "platform": [],
    }
    (root / "data" / "cmd_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    target = root / "data" / "plugins" / "astrbot_plugin_mathitools"
    relative_paths = [Path(name) for name in TOOL_FILES]
    relative_paths.extend(p.relative_to(repo) for p in (repo / "core").rglob("*.py"))
    manifest = {}
    for relative in relative_paths:
        source = repo / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"复制期间源码发生变化: {relative}")
        manifest[relative.as_posix()] = digest
    return root, manifest


async def run_probe(args: argparse.Namespace) -> dict:
    repo = args.repo.resolve()
    work_root = args.work_root.resolve()
    root, manifest = prepare_root(repo, work_root)

    from astrbot.api import FunctionTool
    from astrbot.core import LogBroker, db_helper
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.astr_agent_context import AstrAgentContext
    from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
    from astrbot.core.core_lifecycle import AstrBotCoreLifecycle
    from astrbot.core.message.components import Plain
    from astrbot.core.platform.astr_message_event import AstrMessageEvent
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata

    core = AstrBotCoreLifecycle(LogBroker(), db_helper)
    await core.initialize()
    try:
        metadata_tasks = [
            task for task in asyncio.all_tasks()
            if getattr(task.get_coro(), "__qualname__", "") == "update_llm_metadata"
        ]
        for task in metadata_tasks:
            task.cancel()
        if metadata_tasks:
            await asyncio.gather(*metadata_tasks, return_exceptions=True)

        tools = {
            tool.name: tool
            for tool in core.provider_manager.llm_tools.func_list
            if tool.name.startswith("math_")
        }
        if sorted(tools) != sorted(EXPECTED_TOOLS):
            raise AssertionError(f"工具注册清单不匹配: {sorted(tools)}")
        if not all(isinstance(tool, FunctionTool) for tool in tools.values()):
            raise AssertionError("注册项并非全部为真实 FunctionTool")

        message = AstrBotMessage()
        message.type = MessageType.FRIEND_MESSAGE
        message.self_id = "mathitools-probe"
        message.session_id = "native-probe"
        message.message_id = "probe"
        message.sender = MessageMember("probe-user", "probe")
        message.message = [Plain("数学工具原生预检")]
        message.message_str = "数学工具原生预检"
        message.raw_message = {}
        event = AstrMessageEvent(
            message.message_str,
            message,
            PlatformMetadata("webchat", "isolated probe", "mathitools-probe"),
            message.session_id,
        )
        context = ContextWrapper(
            context=AstrAgentContext(context=core.star_context, event=event),
            tool_call_timeout=20,
        )

        async def invoke(tool_name, params):
            payloads = []
            tool = tools[tool_name]
            async for response in FunctionToolExecutor.execute(tool, context, **params):
                for content in response.content:
                    if content.type == "text":
                        payloads.append(json.loads(content.text))
            if len(payloads) != 1:
                raise AssertionError(f"预期一项JSON工具返回，实际 {len(payloads)}")
            return payloads[0]

        cases = [
            (
                "math_eval", {"expression": "1+2"},
                lambda p: p.get("ok") and p.get("result") == "3",
            ),
            (
                "math_numerical",
                {"operation": "findroot", "expression": "x-2", "a": 1, "b": 3},
                lambda p: p.get("ok") and p.get("result") == "2",
            ),
            (
                "math_numerical",
                {"operation": "normal_cdf", "x": 0},
                lambda p: p.get("ok") and p.get("result") == "0.5",
            ),
            (
                "math_numerical",
                {"operation": "normal_pdf", "sigma": 0, "x": 0},
                lambda p: not p.get("ok") and p.get("error_code") == "invalid_argument",
            ),
            (
                "math_eval", {"expression": "2+3"},
                lambda p: p.get("ok") and p.get("result") == "5",
            ),
        ]
        results = []
        for tool_name, params, check in cases:
            started = time.perf_counter()
            payload = await invoke(tool_name, params)
            elapsed = time.perf_counter() - started
            results.append({
                "tool": tool_name,
                "params": params,
                "actual": payload,
                "elapsed_seconds": round(elapsed, 6),
                "passed": bool(check(payload)),
            })

        import psutil

        worker_pids = []
        for child in psutil.Process().children(recursive=True):
            cmdline = child.cmdline()
            if cmdline and cmdline[-1].endswith("_numerical_worker.py"):
                worker_pids.append(child.pid)

        passed = all(item["passed"] for item in results) and not worker_pids
        first_tool = next(iter(tools.values()))
        versions = {
            package: importlib.metadata.version(package)
            for package in ("astrbot", "mpmath", "asteval", "psutil")
        }
        return {
            "stage": "ok",
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "dependency_versions": versions,
            "registered_tools": sorted(tools),
            "tool_base_class": f"{FunctionTool.__module__}.{FunctionTool.__qualname__}",
            "tool_class": f"{type(first_tool).__module__}.{type(first_tool).__qualname__}",
            "executor_class": (
                f"{FunctionToolExecutor.__module__}.{FunctionToolExecutor.__qualname__}"
            ),
            "source_manifest": manifest,
            "runtime_root": str(root),
            "cases": results,
            "worker_residue_pids": worker_pids,
            "passed": passed,
        }
    finally:
        await core.stop()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="AstrBot 数学工具原生预检")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.work_root = args.work_root.resolve()
    args.output = args.output.resolve()
    try:
        report = asyncio.run(run_probe(args))
    except Exception as exc:
        report = {
            "stage": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "passed": False,
        }
    write_report(args.output, report)
    sys.stdout.write(
        f"PROBE {'PASS' if report.get('passed') else 'FAIL'} -> {args.output}\n"
    )
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
