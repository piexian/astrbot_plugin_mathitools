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


async def observe_worker_cpu(min_user_seconds: float, deadline_seconds: float):
    """观察本进程后代中的 _numerical_worker，等其用户态 CPU 达到阈值。"""
    import psutil

    observed = []
    started = time.perf_counter()
    while time.perf_counter() - started < deadline_seconds:
        for child in psutil.Process().children(recursive=True):
            cmdline = child.cmdline()
            if cmdline and cmdline[-1].endswith("_numerical_worker.py"):
                if child.pid not in {item.pid for item in observed}:
                    observed.append(child)
        for worker in observed:
            try:
                if worker.cpu_times().user >= min_user_seconds:
                    return observed
            except psutil.NoSuchProcess:
                pass
        await asyncio.sleep(0.05)
    return observed


async def run_cancellation_case(invoke) -> dict:
    """取消重计算并验证本 case 启动的工作进程被回收。"""
    import psutil

    started = time.perf_counter()
    observed = []
    create_times = {}
    cancelled = False
    remaining = []
    task = asyncio.create_task(invoke(
        "math_numerical",
        {
            "operation": "differentiate",
            "expression": "comb(1000000000,500000000)+x",
            "x": 0,
        },
    ))

    async def ensure_task_cancelled():
        nonlocal cancelled
        if task.done():
            return
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), 5)
        except asyncio.CancelledError:
            cancelled = True

    async def wait_workers_exit(workers, timeout_seconds):
        deadline = time.perf_counter() + timeout_seconds
        while time.perf_counter() < deadline:
            if not any(worker.is_running() for worker in workers):
                return
            await asyncio.sleep(0.05)

    try:
        observed = await observe_worker_cpu(1.0, 8.0)
        for worker in observed:
            create_times[worker.pid] = worker.create_time()
        await ensure_task_cancelled()
        workers = list(observed)
        await wait_workers_exit(workers, 5)
        for worker in workers:
            try:
                if (
                    worker.create_time() == create_times[worker.pid]
                    and worker.is_running()
                ):
                    worker.kill()
                    await wait_workers_exit([worker], 2)
                    if worker.is_running():
                        remaining.append(worker.pid)
            except psutil.NoSuchProcess:
                pass
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    return {
        "tool": "math_numerical",
        "params": {"operation": "differentiate-cancellation"},
        "actual": None,
        "observed_worker_pids": [item.pid for item in observed],
        "observed_worker_create_times": create_times,
        "cancelled": cancelled,
        "remaining_worker_pids": remaining,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "passed": bool(observed) and cancelled and not remaining,
    }


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
                "math_numerical",
                {"operation": "findroot", "expression": "sqrt(x)", "a": -2, "b": -1},
                lambda p: not p.get("ok") and p.get("error_code") == "domain_error",
            ),
            (
                "math_numerical",
                {"operation": "normal_cdf", "x": 0},
                lambda p: p.get("ok") and p.get("result") == "0.5",
            ),
            (
                "math_eval", {"expression": "2+3"},
                lambda p: p.get("ok") and p.get("result") == "5",
            ),
            (
                "math_finance",
                {"operation": "loan_payment", "principal": 1200, "annual_rate": 0, "years": 1},
                lambda p: p.get("ok") and p.get("result", {}).get("payment") == 100,
            ),
            (
                "math_finance",
                {"operation": "split_bill", "total": 100, "people": 4, "tip_percent": 10},
                lambda p: p.get("ok") and p.get("result", {}).get("per_person") == 27.5,
            ),
            (
                "math_compare", {"left": "nan", "right": "1"},
                lambda p: not p.get("ok") and p.get("error_code") == "domain_error",
            ),
            (
                "math_compare", {"left": "pi", "right": "3.1416", "dps": 5},
                lambda p: p.get("ok") and p.get("result", {}).get("comparison") == "<",
            ),
            (
                "math_compare", {"left": "pi + 1e-20", "right": "pi", "dps": 30},
                lambda p: p.get("ok") and p.get("result", {}).get("comparison") == ">",
            ),
            (
                "math_numerical", {"operation": "polyroots", "coefficients": [1, -6, 11, -6]},
                lambda p: p.get("ok") and sorted(p.get("result", [])) == ["1", "2", "3"],
            ),
            (
                "math_numerical",
                {
                    "operation": "integrate",
                    "expression": "exp(-100*(x-3)^2)+exp(-100*(x-7)^2)",
                    "a": 0, "b": 10, "points": [3, 7],
                },
                lambda p: p.get("ok") and abs(float(p["result"]) - 0.3544907701811032) < 1e-18,
            ),
            (
                "math_units",
                {"value": "bad", "from_unit": "hp", "to_unit": "kW", "unit_type": "power"},
                lambda p: not p.get("ok") and p.get("error_code") == "invalid_argument",
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

        parallel_cases = [
            ("math_eval", {"expression": "1+2"}, "3"),
            ("math_compare", {"left": "pi", "right": "3.1416", "dps": 5}, None),
            ("math_batch", {"expressions": ["1+1", "sqrt(2)"]}, None),
            ("math_functions", {}, None),
            ("math_statistics", {"operation": "mean", "data": [1, 2, 3]}, 2),
            ("math_matrix", {"operation": "determinant", "matrices": [[[1, 2], [3, 4]]]}, -2),
            ("math_numerical", {"operation": "normal_cdf", "x": 0}, "0.5"),
            ("math_units", {"value": 1, "from_unit": "hp", "to_unit": "kW", "unit_type": "power"}, 0.7457),
            ("math_units", {"value": 1, "from_unit": "kgf", "to_unit": "N", "unit_type": "force"}, 9.80665),
            ("math_nl_units", {"query": "5 km to miles"}, None),
            ("math_number_theory", {"operation": "is_prime", "number": 17}, True),
            ("math_finance", {"operation": "split_bill", "total": 100, "people": 4}, None),
            ("math_eval", {"expression": "gamma(0.5)"}, None),
            ("math_statistics", {"operation": "variance", "data": [1, 2, 3]}, 1),
            ("math_units", {"value": 0, "from_unit": "C", "to_unit": "K", "unit_type": "temperature"}, 273.15),
        ]
        started = time.perf_counter()
        parallel_payloads = await asyncio.gather(
            *(invoke(tool_name, params) for tool_name, params, _ in parallel_cases)
        )
        elapsed = time.perf_counter() - started
        for (tool_name, params, expected), payload in zip(parallel_cases, parallel_payloads):
            results.append({
                "tool": tool_name,
                "params": params,
                "actual": payload,
                "parallel": True,
                "elapsed_seconds": round(elapsed, 6),
                "passed": bool(
                    payload.get("ok")
                    and (expected is None or payload.get("result") == expected)
                ),
            })

        cancellation = await run_cancellation_case(invoke)
        results.append(cancellation)
        recovery = await invoke("math_numerical", {"operation": "normal_cdf", "x": 0})
        results.append({
            "tool": "math_numerical",
            "params": {"operation": "normal_cdf", "x": 0},
            "actual": recovery,
            "elapsed_seconds": None,
            "passed": bool(recovery.get("ok") and recovery.get("result") == "0.5"),
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
