"""三平台原生限额预检脚本：观察内核对内存/CPU 限额的执行。"""

import argparse
import json
import os
import sys

import psutil

MiB = 1024 * 1024
WINDOWS = sys.platform.startswith("win")
# 匿名 Job 句柄保留到进程退出，避免提前关闭
_JOB_HANDLE = None


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
    """Windows 匿名 Job 自限：设提交内存/用户态时间并读回实际值。"""
    import ctypes
    from ctypes import wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMIT),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.QueryInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = _EXTENDED_LIMIT()
        flags = 0
        if process_memory is not None:
            flags |= 0x100
            info.ProcessMemoryLimit = process_memory
        if job_time_ticks is not None:
            flags |= 0x4
            info.BasicLimitInformation.PerJobUserTimeLimit = job_time_ticks
        info.BasicLimitInformation.LimitFlags = flags
        if not kernel32.SetInformationJobObject(
            ctypes.c_void_p(handle), 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(
            ctypes.c_void_p(handle), ctypes.c_void_p(kernel32.GetCurrentProcess())
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        read = _EXTENDED_LIMIT()
        returned = wintypes.DWORD(0)
        if not kernel32.QueryInformationJobObject(
            ctypes.c_void_p(handle), 9, ctypes.byref(read),
            ctypes.sizeof(read), ctypes.byref(returned),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        raise
    global _JOB_HANDLE
    _JOB_HANDLE = handle
    return {
        "metric": "committed_memory" if process_memory is not None else "job_user_time",
        "requested": {
            "process_memory_bytes": process_memory,
            "job_user_time_ticks": job_time_ticks,
        },
        "actual": {
            "process_memory_bytes": read.ProcessMemoryLimit,
            "job_user_time_ticks": read.BasicLimitInformation.PerJobUserTimeLimit,
            "limit_flags": read.BasicLimitInformation.LimitFlags,
        },
        "struct_size": ctypes.sizeof(info),
        "accepted": True,
    }


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
