from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
MODEL_REPO = "LiquidAI/LFM2.5-2.6B"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class RunLog:
    """Quiet terminal, one compact log per run, only the latest failure retained."""
    def __init__(self, root: Path, preserve: bool = False):
        self.root = root
        self.preserve = preserve
        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / ".lfm-managed").touch(exist_ok=True)
        self.path = logs / f"running-{time.time_ns()}.log"
        self.started = time.monotonic()

    def __enter__(self):
        self.stream = self.path.open("w", encoding="utf-8")
        return self

    def say(self, message: str) -> None:
        print(message, flush=True)
        self.stream.write(message + "\n")
        self.stream.flush()

    def command(self, args: list[str], *, cwd: Path | None = None, env=None) -> None:
        self.stream.write("$ " + " ".join(args) + "\n")
        self.stream.flush()
        result = subprocess.run(args, cwd=cwd, env=env, stdout=self.stream,
                                stderr=subprocess.STDOUT, check=False)
        self.stream.flush()
        if result.returncode:
            raise RuntimeError(f"Command failed (exit {result.returncode}): {args[0]}")

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.stream.close()
            self.path.unlink(missing_ok=True)
            return False
        traceback_module = sys.modules.get("traceback")
        if traceback_module:
            traceback_module.print_exception(kind, value, traceback, file=self.stream)
            self.stream.flush()
        self.stream.close()
        logs = self.path.parent
        if self.preserve:
            destination = logs / f"failure-{time.time_ns()}.log"
        else:
            destination = logs / "failure.log"
        self.path.replace(destination)
        excerpt = ""
        try:
            lines = destination.read_text(encoding="utf-8", errors="replace").splitlines()
            excerpt = "\n".join(lines[-12:])
        except OSError:
            pass
        print(f"Failed after {time.monotonic() - self.started:.1f}s: {value}", file=sys.stderr)
        print(f"Details: {destination}", file=sys.stderr)
        if excerpt:
            print(excerpt, file=sys.stderr)
        return False


def require_profile(path: Path) -> dict:
    profile_path = path / "profile.json" if path.is_dir() else path
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "context_length": 131072,
        "tile_size": 2048,
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError(f"Invalid profile {key}: expected {value!r}")
    kv = profile.get("kv", {})
    if kv.get("dtype") not in {"fp32", "bf16"}:
        raise ValueError("profile kv.dtype must be fp32 or bf16")
    if kv.get("storage") not in {"disk", "cmm"}:
        raise ValueError("profile kv.storage must be disk or cmm")
    if kv.get("storage") not in kv.get("supported_storage", []):
        raise ValueError("profile kv.storage is not listed in supported_storage")
    if kv["dtype"] == "fp32" and kv["storage"] != "disk":
        raise ValueError("FP32 profile storage must be disk")
    if profile.get("name") not in {"LFM2.5-2.6B-FP32", "LFM2.5-2.6B-BF16"}:
        raise ValueError("Unexpected model profile name")
    if profile["name"] != f"LFM2.5-2.6B-{kv['dtype'].upper()}":
        raise ValueError("profile name does not match its KV precision")
    return profile


def disk_available(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def memory_available() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return 0


def require_marker(target: Path, root: Path) -> None:
    marker = target / ".lfm-managed"
    if not target.exists() or not marker.is_file() or target.resolve().parent != root.resolve():
        raise ValueError(f"Refusing to remove unmarked or out-of-scope path: {target}")
