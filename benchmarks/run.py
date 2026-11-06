"""父调度预检：依次运行原生资源探针与 SDK 工具探针并汇总 JSON 报告。"""

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
SDK_PROBE = BENCH_DIR / "astrbot_probe.py"
NATIVE_PROBE = BENCH_DIR / "native_probe.py"
NATIVE_PHASES = ("baseline", "fixed-memory", "memory", "cpu")

# Windows 退出码使用原生 NTSTATUS。
WINDOWS_CPU_STATUS_QUOTA_EXCEEDED = 0xC0000044
EXPECTED_SDK_TOOLS = [
    "math_eval", "math_compare", "math_batch", "math_functions", "math_matrix",
    "math_numerical", "math_units", "math_nl_units", "math_number_theory",
    "math_statistics", "math_finance",
]

CI_ENV_WHITELIST = (
    "RUNNER_OS", "RUNNER_ARCH", "ImageOS", "ImageVersion",
    "GITHUB_SHA", "GITHUB_RUN_ID",
)


class ProbeRun:
    def __init__(self, name: str):
        self.name = name
        self.pid = None
        self.returncode = None
        self.elapsed_seconds = None
        self.timed_out = False
        self.reaped = False
        self.cleanup_needed = False
        self.payload = None
        self.stdout_path = None
        self.stderr_path = None

    def as_dict(self) -> dict:
        return {
            "probe": self.name,
            "pid": self.pid,
            "returncode": self.returncode,
            "elapsed_seconds": self.elapsed_seconds,
            "timed_out": self.timed_out,
            "reaped": self.reaped,
            "cleanup_needed": self.cleanup_needed,
            "payload": self.payload,
            "stdout_log": self.stdout_path,
            "stderr_log": self.stderr_path,
        }


def start_probe(script: Path, arguments: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-X", "utf8", "-I", "-B", str(script), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def reap(
    proc: subprocess.Popen, run: ProbeRun, logs: dict, timeout_seconds: int = 8
) -> None:
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        run.timed_out = True
        run.cleanup_needed = True
        proc.kill()
        stdout, stderr = proc.communicate()
    run.reaped = proc.returncode is not None
    run.returncode = proc.returncode
    run.pid = proc.pid
    for key, text in (("stdout", stdout), ("stderr", stderr)):
        path = logs[key]
        path.write_text(text or "", encoding="utf-8")
        setattr(run, f"{key}_path", str(path))


def parse_jsonl(text: str):
    payload = None
    for line in text.splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
    return payload


def native_pass(phase: str, run: ProbeRun) -> bool:
    payload = run.payload
    if not isinstance(payload, dict):
        return False
    if phase == "cpu":
        if run.timed_out or run.returncode == 0 or payload.get("event") != "ready":
            return False
        if sys.platform == "win32":
            return run.returncode == WINDOWS_CPU_STATUS_QUOTA_EXCEEDED
        return run.returncode in (-signal.SIGXCPU, -signal.SIGKILL)
    if run.timed_out or run.returncode != 0:
        return False
    if phase == "baseline":
        memory = payload.get("memory") or {}
        if "rss_bytes" not in memory or "vms_bytes" not in memory:
            return False
        if sys.platform == "win32" and "private_bytes" not in memory:
            return False
        return True
    if phase == "memory":
        return payload.get("enforced") is True
    if phase == "fixed-memory":
        if not isinstance(payload.get("result"), dict):
            return False
        return isinstance(payload["result"].get("accepted"), bool)
    return True


def run_native_phases(log_dir: Path) -> list[dict]:
    kinds = {
        "baseline": "measurement",
        "fixed-memory": "observation",
        "memory": "enforcement",
        "cpu": "enforcement",
    }
    runs = []
    for phase in NATIVE_PHASES:
        run = ProbeRun(phase)
        run_dict = run.as_dict()
        run_dict["kind"] = kinds[phase]
        logs = {
            key: log_dir / f"native_{phase}_{key}.log" for key in ("stdout", "stderr")
        }
        started = time.perf_counter()
        proc = None
        try:
            proc = start_probe(NATIVE_PROBE, ["--probe", phase])
            try:
                reap(proc, run, logs, timeout_seconds=20 if phase == "cpu" else 8)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()
                    run.reaped = True
                    run.cleanup_needed = True
            run.payload = parse_jsonl(logs["stdout"].read_text(encoding="utf-8"))
            run_dict["passed"] = native_pass(phase, run)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            run_dict["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            run_dict["passed"] = False
        run_dict.update(run.as_dict())
        run_dict["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        if phase == "fixed-memory" and isinstance(run.payload, dict):
            accepted = run.payload.get("result", {}).get("accepted")
            run_dict["limit_supported"] = accepted
        runs.append(run_dict)
    return runs


def terminate_descendants(root_pid: int) -> list[int]:
    import psutil

    terminated = []
    try:
        parent = psutil.Process(root_pid)
        for child in parent.children(recursive=True):
            child.kill()
            terminated.append(child.pid)
    except psutil.NoSuchProcess:
        pass
    return terminated


def run_sdk_probe(repo: Path, log_dir: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="sdk_probe_work_") as work_root:
        output = log_dir / "sdk_report.json"
        arguments = [
            "--repo", str(repo),
            "--work-root", work_root,
            "--output", str(output),
        ]
        proc = None
        logs = {
            "stdout": log_dir / "sdk_stdout.log",
            "stderr": log_dir / "sdk_stderr.log",
        }
        stdout_text = stderr_text = ""
        timed_out = False
        descendant_pids = []
        started = time.perf_counter()
        try:
            proc = start_probe(SDK_PROBE, arguments)
            try:
                stdout_text, stderr_text = proc.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                timed_out = True
                descendant_pids = terminate_descendants(proc.pid)
                proc.kill()
                stdout_text, stderr_text = proc.communicate()
        except OSError as exc:
            return {
                "probe": "sdk",
                "pid": None,
                "returncode": None,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "timed_out": False,
                "reaped": False,
                "cleanup_needed": False,
                "terminated_descendant_pids": [],
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "stdout_log": str(logs["stdout"]),
                "stderr_log": str(logs["stderr"]),
                "report": {},
                "passed": False,
            }
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.communicate()
        elapsed = round(time.perf_counter() - started, 6)
        if proc is not None:
            (logs["stdout"]).write_text(stdout_text or "", encoding="utf-8")
            (logs["stderr"]).write_text(stderr_text or "", encoding="utf-8")
        report = {}
        error = None
        try:
            if output.is_file():
                report = json.loads(output.read_text(encoding="utf-8"))
            cases = report.get("cases") or []
            passed = (
                not timed_out
                and proc.returncode == 0
                and report.get("passed") is True
                and report.get("dependency_versions", {}).get("astrbot") == "4.29.1"
                and report.get("worker_residue_pids") == []
                and len(cases) == 5
                and all(item.get("passed") is True for item in cases)
                and sorted(report.get("registered_tools", []))
                == sorted(EXPECTED_SDK_TOOLS)
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            passed = False
        return {
            "probe": "sdk",
            "pid": proc.pid,
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "timed_out": timed_out,
            "reaped": proc.returncode is not None,
            "cleanup_needed": timed_out,
            "terminated_descendant_pids": descendant_pids,
            "error": error,
            "stdout_log": str(logs["stdout"]),
            "stderr_log": str(logs["stderr"]),
            "report": report,
            "passed": bool(passed),
        }


def source_commit(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def dependency_versions() -> dict:
    import importlib.metadata

    return {
        package: importlib.metadata.version(package)
        for package in ("mpmath", "asteval", "psutil")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="数学插件预检父调度")
    parser.add_argument("--repo", type=Path, default=REPO_DIR)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output.parent / "preflight_logs"
    log_dir.mkdir(exist_ok=True)
    sys.stdout.write(f"PREFLIGHT repo={repo}\n")
    script_hashes = {}
    for name in ("run.py", "native_probe.py", "astrbot_probe.py"):
        script_hashes[name] = hashlib.sha256(
            (BENCH_DIR / name).read_bytes()
        ).hexdigest()
    native_results = run_native_phases(log_dir)
    sdk_result = run_sdk_probe(repo, log_dir)
    ci_info = {key: os.environ.get(key) for key in CI_ENV_WHITELIST}
    source_commit_error = None
    try:
        commit = source_commit(repo)
    except subprocess.CalledProcessError as exc:
        commit = None
        source_commit_error = {
            "type": "CalledProcessError",
            "message": exc.stderr.strip(),
        }
    report = {
        "phase": "preflight",
        "production_limits_changed": False,
        "source_commit": commit,
        "source_commit_error": source_commit_error,
        "platform": platform.platform(),
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "ci_env": ci_info,
        "dependency_versions": dependency_versions(),
        "script_sha256": script_hashes,
        "native_probes": native_results,
        "sdk_probe": sdk_result,
    }
    passed = (
        all(item["passed"] for item in native_results)
        and sdk_result["passed"]
        and source_commit_error is None
    )
    report["passed"] = passed
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(
        f"PREFLIGHT {'PASS' if passed else 'FAIL'} -> {output}\n"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
