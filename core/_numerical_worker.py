"""数值求根独立工作进程：限额内运行 core.numerical._calculate。"""

import json
import sys
from pathlib import Path


def main() -> None:
    emergency_memory = bytearray(1024 * 1024)
    try:
        if sys.platform == "linux":
            import resource

            limit = 256 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from core.numerical import _calculate

        packet = json.loads(sys.stdin.readline())
        value = _calculate(packet)
        sys.stdout.write(json.dumps({"ok": True, "value": value}, allow_nan=False) + "\n")
    except Exception as exc:
        # 先释放预留余量，否则构造错误 JSON 时可能再次 MemoryError
        del emergency_memory
        error_code = getattr(exc, "error_code", None)
        if error_code is None:
            if isinstance(exc, MemoryError):
                error_code = "result_too_large"
            elif isinstance(exc, (ValueError, OverflowError, ZeroDivisionError)):
                error_code = "domain_error"
            else:
                error_code = "internal_error"
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "error_code": error_code,
            "hint": getattr(exc, "hint", None),
            "example": getattr(exc, "example", None),
            "valid_options": getattr(exc, "valid_options", None),
        }
        sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
