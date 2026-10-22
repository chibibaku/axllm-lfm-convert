"""Dependency-free entrypoint used only until the project venv exists."""
import argparse
import os
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PULSAR_REVISION = "03a419e01007883c4fff744584c3acd034e4a2c9"
PULSAR_URL = (f"https://huggingface.co/AXERA-TECH/Pulsar2/resolve/{PULSAR_REVISION}/7.0-patch1/"
              "ax_pulsar2_7.0_patch1.tar.gz?download=true")
PULSAR_IMAGE = "pulsar2:7.0-patch1"
DEFAULTS = json.loads((ROOT / "defaults.json").read_text(encoding="utf-8"))
KEEP_FAILURE_LOGS = False


def record_failure(command, output):
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / ".lfm-managed").touch(exist_ok=True)
    if not KEEP_FAILURE_LOGS:
        for old in logs.glob("failure*.log"):
            old.unlink(missing_ok=True)
    path = logs / (f"failure-{time.time_ns()}.log" if KEEP_FAILURE_LOGS else "failure.log")
    path.write_text("$ " + " ".join(command) + "\n" + output, encoding="utf-8")
    excerpt = "\n".join(output.splitlines()[-12:])
    print(f"Details: {path}", file=sys.stderr)
    if excerpt:
        print(excerpt, file=sys.stderr)


def run_quiet(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        record_failure(command, result.stdout)
        raise RuntimeError(f"Failed: {command[0]}")


def sudo_ready(sudo: str | None):
    if os.geteuid() == 0:
        return []
    if not sudo:
        raise RuntimeError("This setup step needs root or sudo privileges.")
    # Leave stdin/stderr attached so sudo can ask for a password once, visibly.
    result = subprocess.run([sudo, "-v"])
    if result.returncode:
        raise RuntimeError("sudo authorization failed; rerun from an Ubuntu user with sudo access.")
    return [sudo]


def preflight(cache: Path):
    if sys.platform != "linux" or not Path("/etc/os-release").is_file():
        raise RuntimeError("Setup requires Ubuntu 22.04+ (Ubuntu 24.04 recommended).")
    release = Path("/etc/os-release").read_text(encoding="utf-8", errors="replace")
    if not any(line == "ID=ubuntu" for line in release.splitlines()):
        raise RuntimeError("Setup requires Ubuntu 22.04+ (Ubuntu 24.04 recommended).")
    import platform
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("Pulsar2 7.0-patch1 requires Ubuntu x86_64.")
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free / (1024 ** 3)
    try:
        fields = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:")).split()
        available = int(fields[1]) / (1024 ** 2)
    except (OSError, StopIteration, ValueError, IndexError):
        available = 0
    print(f"Preflight: Ubuntu x86_64; {free:.1f} GiB free disk; {available:.1f} GiB available RAM.",
          flush=True)
    if free < DEFAULTS["min_free_gib"]:
        raise RuntimeError(f"Model conversion requires {DEFAULTS['min_free_gib']} GiB free disk before setup; found {free:.1f} GiB.")
    if available and available < DEFAULTS["min_memory_gib"]:
        raise RuntimeError(f"Full-precision teacher requires at least {DEFAULTS['min_memory_gib']} GiB available RAM; found {available:.1f} GiB.")


def docker_command(cache: Path):
    docker = shutil.which("docker")
    if not docker:
        return []
    if subprocess.run([docker, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return [docker]
    sudo = shutil.which("sudo")
    if sudo:
        prefix = sudo_ready(sudo)
        if subprocess.run(prefix + [docker, "info"], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0:
            return prefix + [docker]
    return [docker]


def check_pulsar_archive(archive: Path):
    if not archive.is_file() or archive.stat().st_size == 0:
        raise FileNotFoundError(f"Pulsar2 archive is missing or empty: {archive}")


def download_pulsar_archive(cache: Path) -> Path:
    """Download the compiler archive into the managed cache."""
    tools_dir = cache / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    archive = tools_dir / "ax_pulsar2_7.0_patch1.tar.gz"
    if archive.is_file():
        try:
            check_pulsar_archive(archive)
            return archive
        except (OSError, RuntimeError):
            archive.unlink(missing_ok=True)
    free = shutil.disk_usage(cache).free / (1024 ** 3)
    if free < 12:
        raise RuntimeError(f"Need 12 GiB free to fetch and load Pulsar2; only {free:.1f} GiB is available.")
    partial = archive.with_name(archive.name + ".partial")
    print("Downloading Pulsar2 7.0-patch1 (5.63 GB)…", flush=True)
    try:
        run_quiet(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
                   "--speed-limit", "1024", "--speed-time", "120", "--silent", "--show-error",
                   "--output", str(partial), PULSAR_URL])
        check_pulsar_archive(partial)
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    return archive


def install_pulsar2(cache: Path, archive: Path | None = None):
    docker = shutil.which("docker")
    sudo = shutil.which("sudo")
    privileged = [] if os.geteuid() == 0 else ([sudo] if sudo else [])
    if not docker:
        if not shutil.which("apt-get") or (os.geteuid() != 0 and not sudo):
            raise RuntimeError("Docker is missing. Install Docker Engine (Ubuntu package docker.io) with sudo/root.")
        print("Installing Docker Engine (requires sudo/root)…", flush=True)
        privileged = sudo_ready(sudo)
        run_quiet(privileged + ["apt-get", "update"])
        run_quiet(privileged + ["apt-get", "install", "-y", "docker.io", "curl"])
        docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker package installation completed but the docker CLI is still missing.")
    if not shutil.which("curl"):
        if not shutil.which("apt-get") or (os.geteuid() != 0 and not sudo):
            raise RuntimeError("curl is required to download Pulsar2; install curl with sudo/root.")
        privileged = sudo_ready(sudo) if os.geteuid() != 0 else []
        run_quiet(privileged + ["apt-get", "install", "-y", "curl"])
    command = docker_command(cache) or [docker]
    info = subprocess.run(command + ["info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if info.returncode:
        if not shutil.which("service"):
            raise RuntimeError("Docker daemon is unavailable; start Docker Engine or enable Docker Desktop integration.")
        print("Starting Docker Engine…", flush=True)
        privileged = sudo_ready(sudo) if os.geteuid() != 0 else []
        run_quiet(privileged + ["service", "docker", "start"])
        command = docker_command(cache) or [docker]
        info = subprocess.run(command + ["info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if info.returncode:
            raise RuntimeError("Docker daemon did not start; ensure the current user can access Docker.")
    image = subprocess.run(command + ["image", "inspect", PULSAR_IMAGE],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / ".lfm-managed").touch(exist_ok=True)
    (cache / "docker-command.json").write_text(json.dumps(command), encoding="utf-8")
    if not image.returncode:
        # A prior interrupted load may leave this disposable download behind.
        if archive is None:
            managed_path = cache / "tools" / "ax_pulsar2_7.0_patch1.tar.gz"
            managed_path.unlink(missing_ok=True)
            managed_path.with_name(managed_path.name + ".partial").unlink(missing_ok=True)
        print(f"Pulsar2 {PULSAR_IMAGE} is ready.", flush=True)
        return
    managed_archive = archive is None
    if managed_archive:
        archive = download_pulsar_archive(cache)
    else:
        check_pulsar_archive(archive)
    print("Loading Pulsar2 container image…", flush=True)
    run_quiet(command + ["load", "--input", str(archive)])
    image = subprocess.run(command + ["image", "inspect", PULSAR_IMAGE],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if image.returncode:
        raise RuntimeError(f"Loaded archive does not contain {PULSAR_IMAGE}.")
    (cache / "docker-command.json").write_text(json.dumps(command), encoding="utf-8")
    if managed_archive:
        archive.unlink(missing_ok=True)
    print(f"Pulsar2 {PULSAR_IMAGE} is ready.", flush=True)


def main():
    parser = argparse.ArgumentParser(prog="lfm")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="create the project environment")
    setup.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULTS["device"])
    setup.add_argument("--python", default=sys.executable)
    setup.add_argument("--with-pulsar2", action="store_true")
    setup.add_argument("--cache", type=Path, default=ROOT / DEFAULTS["cache"])
    setup.add_argument("--pulsar-archive", type=Path)
    setup.add_argument("--keep-failure-logs", action="store_true")
    compiler = sub.add_parser("compiler", help="ensure Docker and the pinned Pulsar2 image are ready")
    compiler.add_argument("--cache", type=Path, default=ROOT / DEFAULTS["cache"])
    compiler.add_argument("--pulsar-archive", type=Path)
    compiler.add_argument("--keep-failure-logs", action="store_true")
    args, _ = parser.parse_known_args()
    global KEEP_FAILURE_LOGS
    KEEP_FAILURE_LOGS = args.keep_failure_logs
    if args.command == "compiler":
        try:
            install_pulsar2(args.cache.resolve(), args.pulsar_archive.resolve() if args.pulsar_archive else None)
            return 0
        except Exception as error:
            print(str(error), file=sys.stderr)
            return 2
    if args.command != "setup":
        print("The project environment is not installed. Run: ./lfm setup --device cpu", file=sys.stderr)
        return 2
    if sys.platform != "linux":
        print("Setup requires Ubuntu/Linux (WSL Ubuntu is supported).", file=sys.stderr)
        return 2
    try:
        preflight(args.cache.resolve())
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    venv_dir = ROOT / ".venv"
    print("Creating isolated environment…", flush=True)
    python_cmd = shutil.which(args.python) or args.python
    version = subprocess.run([python_cmd, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True)
    if version.returncode or tuple(map(int, version.stdout.strip().split("."))) < (3, 10):
        print("Python 3.10 or newer is required.", file=sys.stderr)
        return 2
    create = subprocess.run([python_cmd, "-m", "venv", str(venv_dir)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if create.returncode and shutil.which("apt-get") and (os.geteuid() == 0 or shutil.which("sudo")):
        print("Installing Ubuntu's python3-venv package (requires sudo/root)…", flush=True)
        sudo = shutil.which("sudo")
        privileged = sudo_ready(sudo) if os.geteuid() != 0 else []
        run_quiet(privileged + ["apt-get", "update"])
        run_quiet(privileged + ["apt-get", "install", "-y", "python3-venv"])
        create = subprocess.run([python_cmd, "-m", "venv", str(venv_dir)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if create.returncode:
        print(create.stderr[-2500:], file=sys.stderr)
        return create.returncode
    python = venv_dir / "bin" / "python"
    requirements = ROOT / ("requirements-cuda.txt" if args.device == "cuda" else "requirements-cpu.txt")
    commands = [[str(python), "-m", "pip", "install", "--upgrade", "pip==26.0"],
                [str(python), "-m", "pip", "uninstall", "-y", "torch"]]
    if args.device == "cuda":
        if not (shutil.which("nvidia-smi") or Path("/usr/lib/wsl/lib/nvidia-smi").is_file()):
            print("CUDA setup requested, but nvidia-smi is unavailable.", file=sys.stderr)
            return 2
        commands.append([str(python), "-m", "pip", "install", "torch==2.10.0",
                            "--index-url", "https://download.pytorch.org/whl/cu128"])
    else:
        commands.append([str(python), "-m", "pip", "install", "torch==2.10.0",
                            "--index-url", "https://download.pytorch.org/whl/cpu"])
    commands.append([str(python), "-m", "pip", "install", "-r", str(requirements)])
    commands.append([str(python), "-m", "pip", "check"])
    for command in commands:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            record_failure(command, result.stdout)
            print("Setup failed; the next setup run can safely reuse the venv.", file=sys.stderr)
            return result.returncode
    verify = subprocess.run([str(python), "-c",
                             "import torch,sys; d=sys.argv[1]; "
                             "sys.exit(0 if ((torch.version.cuda is None) == (d == 'cpu')) else 1)",
                             args.device], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if verify.returncode:
        print("Installed PyTorch does not match requested device mode.", file=sys.stderr)
        return 2
    if args.device == "cuda":
        probe_code = (
            "import torch; "
            "assert torch.cuda.is_available(), 'torch.cuda.is_available() is false'; "
            "x = torch.tensor([1.0], dtype=torch.float32, device='cuda'); "
            "y = x + 1.0; torch.cuda.synchronize(); "
            "assert y.item() == 2.0, 'CUDA FP32 probe produced an unexpected result'"
        )
        probe_command = [str(python), "-c", probe_code]
        probe = subprocess.run(probe_command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
        if probe.returncode:
            record_failure(probe_command, probe.stdout)
            print("CUDA runtime probe failed; environment was not marked ready.", file=sys.stderr)
            return probe.returncode or 2
    environment_identity = args.device + "\n" + requirements.read_text(encoding="utf-8")
    (venv_dir / ".lfm-device").write_text(environment_identity, encoding="utf-8")
    print("Environment ready. Try: ./lfm build --profile ../axllm-lfm-model/profiles/FP32 --device cpu", flush=True)
    if args.with_pulsar2:
        try:
            install_pulsar2(args.cache.resolve(), args.pulsar_archive.resolve() if args.pulsar_archive else None)
        except Exception as error:
            print(str(error), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
