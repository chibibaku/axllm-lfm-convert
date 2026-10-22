from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .util import (MODEL_REVISION, MODEL_REPO, RunLog, atomic_json,
                   disk_available, memory_available, require_marker, require_profile)

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = json.loads((ROOT / "defaults.json").read_text(encoding="utf-8"))


def parser():
    p = argparse.ArgumentParser(prog="lfm", description="LFM-to-AX model pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="install pinned CPU/CUDA dependencies in a private venv")
    setup.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULTS["device"])
    setup.add_argument("--python", default=sys.executable)
    setup.add_argument("--with-pulsar2", action="store_true", default=True,
                       help=argparse.SUPPRESS)
    setup.add_argument("--keep-failure-logs", action="store_true")
    setup.add_argument("--cache", type=Path, default=ROOT / DEFAULTS["cache"])
    setup.add_argument("--pulsar-archive", type=Path)
    for command, help_text in [("build", "fetch, calibrate, export, compile and package"),
                               ("calibrate", "fetch the pinned teacher and save reference calibration samples"),
                               ("verify", "verify a generated package"),
                               ("package", "package existing compiled models"),
                               ("clean", "remove only convert-managed work products")]:
        item = sub.add_parser(command, help=help_text)
        if command != "clean":
            item.add_argument("--profile", type=Path, required=True)
            item.add_argument("--output", type=Path)
        if command == "package":
            item.add_argument("--assets", type=Path, required=True)
            item.add_argument("--source", type=Path,
                              help="pinned model snapshot, used for tokenizer and generation metadata")
        if command == "build":
            item.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULTS["device"])
            item.add_argument("--source", type=Path, help="reuse an existing, model snapshot")
            item.add_argument("--cache", type=Path, default=ROOT / DEFAULTS["cache"])
            item.add_argument("--jobs", type=int, choices=[1], default=DEFAULTS["jobs"],
                              help="AX650 compile is sequential; currently only 1 is supported")
            item.add_argument("--pulsar-archive", type=Path,
                              help="local vendor image archive, e.g. Pulsar2 7.0-patch1")
            item.add_argument("--assets", type=Path,
                              help="precompiled payload root")
            item.add_argument("--keep-failure-logs", action="store_true")
            item.add_argument("--min-free-gib", type=float, default=DEFAULTS["min_free_gib"])
        if command == "calibrate":
            item.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULTS["device"])
            item.add_argument("--source", type=Path, help="reuse a model snapshot")
            item.add_argument("--cache", type=Path, default=ROOT / DEFAULTS["cache"])
            item.add_argument("--keep-failure-logs", action="store_true")
            item.add_argument("--min-free-gib", type=float, default=DEFAULTS["min_free_gib"])
        if command in {"verify", "package"}:
            item.add_argument("--keep-failure-logs", action="store_true")
        if command == "clean":
            item.add_argument("--all", action="store_true", help="also remove cached model and checkpoints")
            item.add_argument("--yes", action="store_true")
            item.add_argument("--cache", type=Path, default=ROOT / DEFAULTS["cache"])
            item.add_argument("--output", type=Path, help="remove this marker-owned package output")
    return p


def ensure_venv(device: str, keep_failure_logs: bool = False):
    python = ROOT / ".venv" / "bin" / "python"
    mode_file = ROOT / ".venv" / ".lfm-device"
    requirements = ROOT / f"requirements-{device}.txt"
    environment_identity = device + "\n" + requirements.read_text(encoding="utf-8")
    if sys.platform == "linux" and (not python.exists() or not mode_file.is_file() or
                                     mode_file.read_text(encoding="utf-8") != environment_identity):
        print("Setting up the isolated environment first…", flush=True)
        command = [sys.executable, str(ROOT / "lfm_convert/bootstrap.py"), "setup", "--device", device]
        if keep_failure_logs:
            command.append("--keep-failure-logs")
        result = subprocess.run(command)
        if result.returncode:
            return result.returncode
        os.execv(str(python), [str(python), "-m", "lfm_convert", *sys.argv[1:]])
    return 0


def ensure_compiler(cache: Path, archive: Path | None, keep_failure_logs: bool = False):
    command = [sys.executable, str(ROOT / "lfm_convert/bootstrap.py"), "compiler",
               "--cache", str(cache.resolve())]
    if archive:
        command.extend(["--pulsar-archive", str(archive.resolve())])
    if keep_failure_logs:
        command.append("--keep-failure-logs")
    return subprocess.call(command)


def preflight_setup(cache: Path, min_free_gib: float):
    if sys.platform != "linux" or not Path("/etc/os-release").is_file():
        raise RuntimeError("Setup requires Ubuntu 22.04+ (Ubuntu 24.04 recommended).")
    release = Path("/etc/os-release").read_text(encoding="utf-8", errors="replace")
    if "ID=ubuntu" not in release.splitlines():
        raise RuntimeError("Setup requires Ubuntu 22.04+ (Ubuntu 24.04 recommended).")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("Pulsar2 7.0-patch1 requires Ubuntu x86_64.")
    free = disk_available(cache) / (1024 ** 3)
    ram = memory_available() / (1024 ** 3)
    if not ram:
        try:
            fields = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                          if line.startswith("MemAvailable:")).split()
            ram = int(fields[1]) / (1024 ** 2)
        except (OSError, StopIteration, ValueError, IndexError):
            pass
    print(f"Preflight: Ubuntu x86_64; {free:.1f} GiB free disk; {ram:.1f} GiB available RAM.")
    if free < min_free_gib:
        raise RuntimeError(f"Need {min_free_gib:.1f} GiB free before setup; found {free:.1f} GiB.")
    if ram and ram < DEFAULTS["min_memory_gib"]:
        raise RuntimeError(f"Full-precision teacher needs {DEFAULTS['min_memory_gib']} GiB available RAM; found {ram:.1f} GiB.")


def check_environment(device: str, min_free_gib: float, cache: Path,
                      pulsar_archive: Path | None):
    if sys.platform != "linux":
        raise RuntimeError("Model conversion requires Ubuntu/Linux; WSL Ubuntu is supported.")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("The verified Pulsar2 7.0-patch1 toolchain requires Ubuntu x86_64.")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
    if not shutil.which("docker"):
        raise RuntimeError("Docker CLI is unavailable after Pulsar2 setup.")
    if pulsar_archive:
        if not pulsar_archive.is_file():
            raise FileNotFoundError(f"Pulsar2 archive not found: {pulsar_archive}")
    docker = shutil.which("docker")
    if docker:
        command_path = cache / "docker-command.json"
        docker_command = json.loads(command_path.read_text()) if command_path.is_file() else [docker]
        daemon = subprocess.run(docker_command + ["info"], stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, timeout=20)
        if daemon.returncode:
            hint = "Start Docker (`sudo service docker start`) or enable Docker Desktop's Ubuntu integration."
            raise RuntimeError(f"Docker daemon is not available. {hint}")
    available = disk_available(cache)
    gib = available / (1024 ** 3)
    ram_gib = memory_available() / (1024 ** 3)
    print(f"Resources: {gib:.1f} GiB free disk; {ram_gib:.1f} GiB available RAM")
    if gib < min_free_gib:
        raise RuntimeError(f"Need at least {min_free_gib:.1f} GiB free before model conversion.")
    if ram_gib and ram_gib < DEFAULTS["min_memory_gib"]:
        raise RuntimeError(f"At least {DEFAULTS['min_memory_gib']} GiB available RAM is required for the full-precision teacher.")


def peak_rss_gib() -> float:
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 2)
    except (ImportError, AttributeError):
        return 0.0


class ResourceMonitor:
    """Sample host memory and process RSS while a potentially long stage runs."""
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.minimum_available_ram = memory_available()
        self.peak_process_rss = peak_rss_gib()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(self.interval):
            available = memory_available()
            if available:
                self.minimum_available_ram = min(self.minimum_available_ram, available) if self.minimum_available_ram else available
            self.peak_process_rss = max(self.peak_process_rss, peak_rss_gib())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=self.interval + 1)


def build(args, profile, out, log):
    status_path = out / ".lfm-managed"
    if out.exists() and any(out.iterdir()) and not status_path.exists():
        raise RuntimeError(f"Output contains user files and is not marked as convert-managed: {out}")
    out.mkdir(parents=True, exist_ok=True)
    status_path.touch(exist_ok=True)
    cache = args.cache.resolve()
    if cache.exists() and any(cache.iterdir()) and not (cache / ".lfm-managed").is_file():
        raise RuntimeError(f"Cache contains user files and is not convert-managed: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    (cache / ".lfm-managed").touch(exist_ok=True)
    check_environment(args.device, args.min_free_gib, cache, args.pulsar_archive)
    source = acquire_source(args.source, cache, log)
    fingerprint = {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                   "context_length": profile["context_length"], "tile_size": profile["tile_size"],
                   "device": args.device, "pipeline_version": 3}
    state_path = cache / "checkpoints" / "run.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        prior = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if not isinstance(prior, dict):
            prior = {}
    except (OSError, ValueError):
        prior = {}
    if prior.get("fingerprint") != fingerprint:
        if prior:
            # The marked cache owns these generated directories. Recipe/device changes
            # need fresh results; source downloads can be reused.
            for name in ("calibration", "onnx", "compiled"):
                directory = cache / name
                if directory.is_symlink():
                    raise RuntimeError(f"Generated cache directory must not be a symlink: {directory}")
                if directory.exists():
                    shutil.rmtree(directory)
        prior = {"fingerprint": fingerprint, "completed": []}
    atomic_json(state_path, prior)
    log.say("Model snapshot ready.")

    # The implementation is imported here so setup/verify/clean remain lightweight.
    if args.assets:
        from .pipeline import package
        profile_path = args.profile.resolve()
        profile_dir = profile_path if profile_path.is_dir() else profile_path.parent
        package(args.assets.resolve(), out, profile, source, profile_dir)
        print(f"Package ready: {out}")
        return 0
    from .pipeline import calibrate, export_graphs, compile_graphs, package
    profile_path = args.profile.resolve()
    profile_dir = profile_path if profile_path.is_dir() else profile_path.parent
    def update_run_state(stage, graphs=None):
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        state["current"] = {"stage": stage, "graphs": graphs or [], "started_at": time.time()}
        atomic_json(state_path, state)

    def compile_exported_batch(graph_paths):
        update_run_state("compile", graph_paths)
        free_before = require_stage_capacity(cache, args.min_free_gib,
                                             f"compiling {', '.join(graph_paths)}")
        started = time.monotonic()
        with ResourceMonitor() as monitor:
            compile_graphs(cache, profile, None, args.jobs, log,
                           graph_paths=graph_paths, finalize=False)
        telemetry_path = cache / "checkpoints" / "batch-telemetry.json"
        telemetry = json.loads(telemetry_path.read_text()) if telemetry_path.is_file() else []
        telemetry.append({"graphs": graph_paths, "seconds": round(time.monotonic() - started, 2),
                          "free_disk_gib_before": round(free_before, 2),
                          "free_disk_gib_after": round(disk_available(cache) / (1024 ** 3), 2),
                          "minimum_available_ram_gib": round(monitor.minimum_available_ram / (1024 ** 3), 2),
                          "python_peak_rss_gib": monitor.peak_process_rss})
        atomic_json(telemetry_path, telemetry[-8:])
        update_run_state("export", [])
    stages = [
        ("calibrate", lambda: calibrate(source, cache, profile, args.device)),
        ("export", lambda: export_graphs(source, cache, profile, on_batch=compile_exported_batch)),
        ("compile", lambda: compile_graphs(cache, profile, None, args.jobs, log)),
        ("package", lambda: package(cache, out, profile, source, profile_dir)),
    ]
    receipts = cache / "checkpoints"
    receipts.mkdir(exist_ok=True)
    for name, action in stages:
        receipt_path = receipts / f"{name}.json"
        stage_fingerprint = fingerprint
        if name != "package" and _stage_valid(cache, name, stage_fingerprint, receipt_path):
            log.say(f"{name}: reusing completed stage")
            continue
        if name == "calibrate":
            # New calibration samples invalidate downstream generated results.
            for directory in (cache / "onnx", cache / "compiled"):
                if directory.is_symlink():
                    raise RuntimeError(f"Generated cache directory must not be a symlink: {directory}")
                if directory.exists():
                    shutil.rmtree(directory)
        free = require_stage_capacity(cache, args.min_free_gib, name)
        log.say(f"{name}: starting ({free:.1f} GiB free)")
        update_run_state(name)
        started = time.monotonic()
        with ResourceMonitor() as monitor:
            action()
        prior.setdefault("completed", []).append(name)
        prior["completed"] = list(dict.fromkeys(prior["completed"]))
        prior.setdefault("telemetry", {})[name] = {
            "seconds": round(time.monotonic() - started, 2),
            "free_disk_gib_after": round(disk_available(cache) / (1024 ** 3), 2),
            "minimum_available_ram_gib": round(monitor.minimum_available_ram / (1024 ** 3), 2),
            "python_peak_rss_gib": monitor.peak_process_rss,
        }
        prior.pop("current", None)
        atomic_json(state_path, prior)
        if name != "package":
            artifact_path = cache / {"calibrate": "calibration", "export": "onnx",
                                     "compile": "compiled"}[name]
            atomic_json(receipt_path, {
                "fingerprint": stage_fingerprint,
                "files": [p.relative_to(artifact_path).as_posix()
                          for p in sorted(artifact_path.rglob("*")) if p.is_file()],
            })
        log.say(f"{name}: complete in {prior['telemetry'][name]['seconds']:.1f}s")
    return 0


def require_stage_capacity(cache: Path, min_free_gib: float, stage: str) -> float:
    """Refuse another large step if disk or available RAM has fallen short."""
    free_gib = disk_available(cache) / (1024 ** 3)
    if free_gib < min_free_gib:
        raise RuntimeError(f"Only {free_gib:.1f} GiB remains before {stage}; refusing to begin the next large step.")
    ram_gib = memory_available() / (1024 ** 3)
    if ram_gib and ram_gib < DEFAULTS["min_memory_gib"]:
        raise RuntimeError(f"Only {ram_gib:.1f} GiB RAM is available before {stage}; "
                           f"need {DEFAULTS['min_memory_gib']} GiB before starting another large step.")
    return free_gib


def acquire_source(source_arg: Path | None, cache: Path, log: RunLog) -> Path:
    source = source_arg.resolve() if source_arg else cache / "source"
    if not source_arg:
        source.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        log.say(f"Fetching {MODEL_REPO}…")
        snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=source,
                          allow_patterns=["*.json", "*.jinja", "*.safetensors", "LICENSE", "README.md"],
                          max_workers=3)
    config_file = source / "config.json"
    index_file = source / "model.safetensors.index.json"
    if not config_file.is_file() or not index_file.is_file():
        raise RuntimeError("Pinned source is incomplete: config.json or safetensors index is missing.")
    index = json.loads(index_file.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if any(not isinstance(shard, str) or Path(shard).name != shard for shard in shards):
        raise RuntimeError("Model shard paths must be file names inside the source directory.")
    if not shards or any(not (source / shard).is_file() or (source / shard).stat().st_size == 0 for shard in shards):
        raise RuntimeError("Pinned source has missing or empty safetensors shards; rerun to resume download.")
    return source


def _stage_valid(cache: Path, stage: str, fingerprint: dict, receipt_path: Path) -> bool:
    """Resume completed stages by recipe and required files, without hashing payloads."""
    folder = cache / {"calibrate": "calibration", "export": "onnx", "compile": "compiled"}[stage]
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("fingerprint") != fingerprint:
            return False
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if stage in {"export", "compile"}:
            graphs = manifest.get("graphs", [])
            if len(graphs) != 48:
                return False
            for entry in graphs:
                relative = entry["path"]
                if stage == "export":
                    graph = folder / relative
                    model = cache / "compiled" / Path(relative).parent / "model.axmodel"
                    if not (_nonempty(graph) and _nonempty(graph.parent / "config.json")) and not (_nonempty(model) and (model.parent / "complete").is_file()):
                        return False
                elif not (_nonempty(folder / relative / "model.axmodel") and
                          (folder / relative / "complete").is_file()):
                    return False
            return True
        files = receipt.get("files", [])
        return bool(files) and all(_nonempty(folder / relative) for relative in files)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def clean(all_data: bool, cache: Path, output: Path | None):
    targets = [output.resolve()] if output else [ROOT / "dist", ROOT / "logs"]
    if not output:
        targets.extend(ROOT.parent.glob("axllm-lfm-*/dist"))
        targets.extend((ROOT.parent / "axllm-lfm-model" / "profiles").glob("*/dist"))
    for target in targets:
        if target.exists():
            require_marker(target, target.parent)
            shutil.rmtree(target)
    if all_data:
        cache = cache.resolve()
        if cache.exists():
            require_marker(cache, ROOT)
            shutil.rmtree(cache)
    print("Managed work products removed.")
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "setup":
        cache = args.cache.resolve()
        if cache.exists() and any(cache.iterdir()) and not (cache / ".lfm-managed").is_file():
            raise RuntimeError(f"Cache contains user files and is not convert-managed: {cache}")
        command = [sys.executable, str(ROOT / "lfm_convert/bootstrap.py"), "setup",
                   "--device", args.device, "--python", args.python, "--cache", str(args.cache)]
        if args.with_pulsar2:
            command.append("--with-pulsar2")
        if args.keep_failure_logs:
            command.append("--keep-failure-logs")
        if args.pulsar_archive:
            command.extend(["--pulsar-archive", str(args.pulsar_archive)])
        return subprocess.call(command)
    if args.command == "clean":
        if not args.yes:
            print("Pass --yes to remove convert-managed files.", file=sys.stderr)
            return 2
        return clean(args.all, args.cache, args.output)
    profile = require_profile(args.profile)
    profile_path = args.profile.resolve()
    profile_dir = profile_path if profile_path.is_dir() else profile_path.parent
    out = (args.output or profile_dir / DEFAULTS["output_directory"]).resolve()
    if args.command == "verify":
        from .pipeline import verify_package
        return verify_package(out, profile)
    if args.command == "package":
        from .pipeline import package
        profile_path = args.profile.resolve()
        profile_dir = profile_path if profile_path.is_dir() else profile_path.parent
        with RunLog(ROOT, args.keep_failure_logs) as log:
            log.say(f"package: preparing {profile['name']}")
            package(args.assets.resolve(), out, profile, args.source.resolve() if args.source else None,
                    profile_dir)
            log.say(f"Package ready: {out}")
        return 0
    if args.command == "calibrate":
        cache = args.cache.resolve()
        if cache.exists() and any(cache.iterdir()) and not (cache / ".lfm-managed").is_file():
            raise RuntimeError(f"Cache contains user files and is not convert-managed: {cache}")
        preflight_setup(cache, args.min_free_gib)
        result = ensure_venv(args.device, args.keep_failure_logs)
        if result:
            return result
        cache.mkdir(parents=True, exist_ok=True)
        (cache / ".lfm-managed").touch(exist_ok=True)
        with RunLog(ROOT, args.keep_failure_logs) as log:
            if sys.platform != "linux":
                raise RuntimeError("Calibration requires Ubuntu/Linux; WSL Ubuntu is supported.")
            import torch
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
            source = acquire_source(args.source, cache, log)
            log.say("Model snapshot ready.")
            from .pipeline import calibrate
            calibrate(source, cache, profile, args.device)
            # A standalone calibration replaces samples; the next build must
            # regenerate downstream graphs instead of using old checkpoints.
            for stage in ("calibrate", "export", "compile"):
                (cache / "checkpoints" / f"{stage}.json").unlink(missing_ok=True)
            log.say(f"Calibration ready: {cache / 'calibration'}")
        return 0
    if args.command == "build":
        cache = args.cache.resolve()
        out = (args.output or (args.profile.resolve() if args.profile.resolve().is_dir()
                               else args.profile.resolve().parent) / DEFAULTS["output_directory"]).resolve()
        if out.exists() and any(out.iterdir()) and not (out / ".lfm-managed").is_file():
            raise RuntimeError(f"Output contains user files and is not convert-managed: {out}")
        if cache.exists() and any(cache.iterdir()) and not (cache / ".lfm-managed").is_file():
            raise RuntimeError(f"Cache contains user files and is not convert-managed: {cache}")
        preflight_setup(cache, args.min_free_gib)
        if args.jobs < 1:
            raise ValueError("--jobs must be at least 1")
        result = ensure_venv(args.device, args.keep_failure_logs)
        if result:
            return result
        result = ensure_compiler(args.cache, args.pulsar_archive, args.keep_failure_logs)
        if result:
            return result
        log_root = ROOT
        with RunLog(log_root, args.keep_failure_logs) as log:
            return build(args, profile, out, log)
    return 2
