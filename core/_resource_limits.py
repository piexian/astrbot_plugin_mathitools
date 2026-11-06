"""跨平台原生资源限额：Windows Job Objects 与 macOS RLIMIT。"""

# 任务对象用户态 CPU 超时的原生终止码（STATUS_QUOTA_EXCEEDED）
WINDOWS_CPU_LIMIT_EXIT_CODE = 0xC0000044

MiB = 1024 * 1024
# 匿名 Job 句柄保留到进程退出，避免提前关闭
_JOB_HANDLE = None


def set_windows_limits(*, process_memory=None, job_time_ticks=None) -> dict:
    """设置当前进程的提交内存/用户态时间限额并读回实际值。"""
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


def set_macos_limits(*, address_space_headroom: int, cpu_seconds: int) -> dict:
    """按 XNU 基线测量地址空间并设置 macOS 限额，返回观测报告。"""
    import ctypes
    import os
    import resource

    class _PROC_TASKINFO(ctypes.Structure):
        _fields_ = [
            (f"pti_uint64_{index}", ctypes.c_uint64) for index in range(6)
        ] + [
            (f"pti_int32_{index}", ctypes.c_int32) for index in range(12)
        ]

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_pidinfo.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ctypes.c_void_p, ctypes.c_int,
    ]
    libproc.proc_pidinfo.restype = ctypes.c_int

    info = _PROC_TASKINFO()
    written = libproc.proc_pidinfo(
        os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info)
    )
    if written < ctypes.sizeof(info):
        raise OSError(
            ctypes.get_errno(),
            f"proc_pidinfo 只返回 {written} 字节，需要 {ctypes.sizeof(info)}",
        )
    baseline = info.pti_uint64_0
    limit = baseline + address_space_headroom
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    return {
        "metric": "virtual_address_space",
        "baseline_bytes": baseline,
        "limit_bytes": limit,
        "actual_as": list(resource.getrlimit(resource.RLIMIT_AS)),
        "actual_cpu": list(resource.getrlimit(resource.RLIMIT_CPU)),
        "accepted": True,
    }
