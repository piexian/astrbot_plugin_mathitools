"""三平台原生限额预检脚本：观察内核对内存/CPU 限额的执行。"""

import argparse
import json
import os
import sys

import psutil

MiB = 1024 * 1024
WINDOWS = sys.platform.startswith("win")


def _emit(payload) -> None:
    """单行 JSON 输出并立即 flush。"""
    sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
    sys.stdout.flush()


def _memory_snapshot() -> dict:
    """RSS/VMS 快照字段，Windows 另含提交内存。"""
    info = psutil.Process().memory_info()
    payload = {"rss_bytes": info.rss, "vms_bytes": info.vms}
    if WINDOWS:
        payload["private_bytes"] = info.private
    return payload


def _setrlimit_observed(kind, limits):
    """尝试设置并返回实际 rlim 或 OSError/ValueError 观测。"""
    import resource

    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(kind, limits)
        return {"accepted": True, "actual": list(resource.getrlimit(kind))}
    except (OSError, ValueError) as exc:
        return {
            "accepted": False,
            "oserror": {
                "type": type(exc).__name__,
                "errno": getattr(exc, "errno", None),
                "message": str(exc),
            },
        }


def _win_setup(process_memory=None, job_time_ticks=None) -> dict:
    """经由共享 core/_resource_limits 模块设置 Job 限额并读回报告。"""
    from pathlib import Path

    core_dir = Path(__file__).resolve().parents[1] / "core"
    sys.path.insert(0, str(core_dir))
    from _resource_limits import set_windows_limits

    return set_windows_limits(
        process_memory=process_memory, job_time_ticks=job_time_ticks
    )


def _probe_fixed_memory() -> dict:
    """观测内核是否接受 256MiB 地址空间限额。"""
    if WINDOWS:
        try:
            report = _win_setup(process_memory=256 * MiB)
        except OSError as exc:
            report = {
                "accepted": False,
                "oserror": {
                    "type": type(exc).__name__,
                    "errno": getattr(exc, "winerror", exc.errno),
                    "message": str(exc),
                },
            }
        return {
            "probe": "fixed-memory",
            "limit_bytes": 256 * MiB,
            "result": report,
        }
    import resource

    return {
        "probe": "fixed-memory",
        "limit_bytes": 256 * MiB,
        "result": _setrlimit_observed(resource.RLIMIT_AS, (256 * MiB, 256 * MiB)),
    }


def _probe_memory() -> dict:
    """按当前内存基线+32MiB 设限，观察 64MiB 分配是否被内核拒绝。"""
    if WINDOWS:
        baseline = psutil.Process().memory_info().private
        metric = "committed_memory"
        setup = _win_setup(process_memory=baseline + 32 * MiB)
    else:
        import resource

        baseline = psutil.Process().memory_info().vms
        metric = "virtual_address_space"
        setup = _setrlimit_observed(resource.RLIMIT_AS, (baseline + 32 * MiB,) * 2)
    limit_bytes = baseline + 32 * MiB
    if not setup["accepted"]:
        return {
            "probe": "memory",
            "baseline": baseline,
            "limit_bytes": limit_bytes,
            "allocation_bytes": 64 * MiB,
            "metric": metric,
            "enforced": False,
            "setrlimit": setup,
        }
    try:
        bytearray(64 * MiB)
        enforced = False
    except MemoryError:
        enforced = True
    return {
        "probe": "memory",
        "baseline": baseline,
        "limit_bytes": limit_bytes,
        "allocation_bytes": 64 * MiB,
        "metric": metric,
        "enforced": enforced,
        "setrlimit": setup,
    }


def _busy_loop() -> None:
    """常量内存忙循环，等待内核 CPU 限额终止。"""
    value = 1
    while True:
        value = (value * 1103515245 + 12345) % 2**31


def _probe_cpu() -> dict:
    """设 1 秒用户态 CPU 限额后输出 ready 并忙循环等待内核终止。"""
    if WINDOWS:
        report = _win_setup(job_time_ticks=10_000_000)
        return {
            "event": "ready",
            "metric": report["metric"],
            "requested": report["requested"],
            "actual": report["actual"],
            "struct_size": report["struct_size"],
        }
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (1, 2))
    return {
        "event": "ready",
        "actual_limits": list(resource.getrlimit(resource.RLIMIT_CPU)),
    }


def main(argv=None) -> int:
    """CLI 入口：分发探针，输出观测 JSON，可恢复异常返回 1。"""
    emergency = bytearray(MiB)
    parser = argparse.ArgumentParser(description="原生限额预检")
    parser.add_argument(
        "--probe",
        required=True,
        choices=["baseline", "fixed-memory", "memory", "cpu"],
    )
    args = parser.parse_args(argv)
    payload = {
        "probe": args.probe,
        "pid": os.getpid(),
        "platform": sys.platform,
    }
    try:
        if not WINDOWS and sys.platform not in ("linux", "darwin"):
            raise NotImplementedError(
                f"平台 {sys.platform} 的限额预检尚未接入"
            )
        if args.probe == "cpu":
            payload.update(_probe_cpu())
            _emit(payload)
            _busy_loop()
            return 0
        if args.probe == "baseline":
            payload["memory"] = _memory_snapshot()
        elif args.probe == "fixed-memory":
            payload["memory"] = _memory_snapshot()
            payload.update(_probe_fixed_memory())
        else:
            payload.update(_probe_memory())
        _emit(payload)
        return 0
    except Exception as exc:
        del emergency
        _emit({
            "probe": args.probe,
            "pid": os.getpid(),
            "platform": sys.platform,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:200],
            },
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
