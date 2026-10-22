import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lfm_convert import bootstrap


GOOD_ARCHIVE = b"tiny Pulsar fixture"


class BootstrapArchiveTests(unittest.TestCase):
    @contextmanager
    def install_harness(self, root, mode):
        state = {"download_count": 0, "load_count": 0, "loaded": False}

        def run(command, **kwargs):
            if command[0] == "curl":
                self.assertEqual(command[-1], bootstrap.PULSAR_URL)
                self.assertIn(f"/resolve/{bootstrap.PULSAR_REVISION}/", command[-1])
                state["download_count"] += 1
                output = Path(command[command.index("--output") + 1])
                output.write_bytes(GOOD_ARCHIVE)
                if mode == "curl_failure":
                    return subprocess.CompletedProcess(command, 22, stdout="network failure")
                if mode == "interrupt":
                    raise KeyboardInterrupt
                return subprocess.CompletedProcess(command, 0, stdout="")
            if command[0] == "docker" and command[1] == "info":
                return subprocess.CompletedProcess(command, 0, stdout="")
            if command[0] == "docker" and command[1] == "image":
                return subprocess.CompletedProcess(command, 0 if state["loaded"] else 1, stdout="")
            if command[0] == "docker" and command[1] == "load":
                state["load_count"] += 1
                if mode == "load_failure_once" and state["load_count"] == 1:
                    return subprocess.CompletedProcess(command, 1, stdout="load failure")
                if mode in {"load_failure", "explicit_load_failure"}:
                    return subprocess.CompletedProcess(command, 1, stdout="load failure")
                state["loaded"] = True
                return subprocess.CompletedProcess(command, 0, stdout="loaded")
            raise AssertionError(f"Unexpected subprocess command: {command!r}")

        def which(name):
            return {"docker": "docker", "curl": "curl"}.get(name)

        with patch.object(bootstrap, "ROOT", root), \
                patch.object(bootstrap, "KEEP_FAILURE_LOGS", False), \
                patch.object(bootstrap, "docker_command", return_value=["docker"]), \
                patch.object(bootstrap.shutil, "which", side_effect=which), \
                patch.object(bootstrap.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)), \
                patch.object(bootstrap.os, "geteuid", return_value=0, create=True), \
                patch.object(bootstrap.subprocess, "run", side_effect=run):
            yield state

    def test_curl_failure_removes_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            with self.install_harness(root, "curl_failure") as state:
                with self.assertRaisesRegex(RuntimeError, "Failed: curl"):
                    bootstrap.install_pulsar2(cache)
            archive = cache / "tools" / "ax_pulsar2_7.0_patch1.tar.gz"
            self.assertFalse(Path(str(archive) + ".partial").exists())
            self.assertFalse(archive.exists())
            self.assertEqual(state["download_count"], 1)

    def test_interrupt_removes_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            with self.install_harness(root, "interrupt") as state:
                with self.assertRaises(KeyboardInterrupt):
                    bootstrap.install_pulsar2(cache)
            archive = cache / "tools" / "ax_pulsar2_7.0_patch1.tar.gz"
            self.assertFalse(Path(str(archive) + ".partial").exists())
            self.assertFalse(archive.exists())
            self.assertEqual(state["download_count"], 1)


    def test_downloaded_archive_is_reused_after_load_failure_then_removed_on_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            archive = cache / "tools" / "ax_pulsar2_7.0_patch1.tar.gz"
            with self.install_harness(root, "load_failure_once") as state:
                with self.assertRaisesRegex(RuntimeError, "Failed: docker"):
                    bootstrap.install_pulsar2(cache)
                self.assertEqual(archive.read_bytes(), GOOD_ARCHIVE)
                self.assertEqual(state["download_count"], 1)
                bootstrap.install_pulsar2(cache)
                self.assertEqual(state["download_count"], 1)
                self.assertFalse(archive.exists())
                self.assertEqual(state["load_count"], 2)

    def test_success_removes_managed_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            archive = cache / "tools" / "ax_pulsar2_7.0_patch1.tar.gz"
            with self.install_harness(root, "success") as state:
                bootstrap.install_pulsar2(cache)
            self.assertFalse(archive.exists())
            self.assertEqual(state["download_count"], 1)

    def test_ready_image_removes_leftover_managed_download_without_refetch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            archive = cache / "tools" / "ax_pulsar2_7.0_patch1.tar.gz"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(GOOD_ARCHIVE)
            partial = archive.with_name(archive.name + ".partial")
            partial.write_bytes(b"interrupted")
            with self.install_harness(root, "success") as state:
                state["loaded"] = True
                bootstrap.install_pulsar2(cache)
            self.assertFalse(archive.exists())
            self.assertFalse(partial.exists())
            self.assertEqual(state["download_count"], 0)
            self.assertEqual(state["load_count"], 0)


    def test_explicit_archive_is_preserved_on_docker_load_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            explicit = root / "user-selected.tar.gz"
            explicit.write_bytes(GOOD_ARCHIVE)
            with self.install_harness(root, "explicit_load_failure"):
                with self.assertRaisesRegex(RuntimeError, "Failed: docker"):
                    bootstrap.install_pulsar2(cache, explicit)
            self.assertEqual(explicit.read_bytes(), GOOD_ARCHIVE)

    def test_cuda_probe_failure_does_not_mark_environment_ready_or_load_compiler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "requirements-cuda.txt").write_text("# fixture\n", encoding="utf-8")
            cache = root / "cache"
            calls = []

            def which(name):
                return "nvidia-smi" if name == "nvidia-smi" else "python"

            def run(command, **kwargs):
                calls.append(command)
                if command[0] == "python" and command[1:2] == ["-c"]:
                    code = command[2]
                    if "sys.version_info" in code:
                        return subprocess.CompletedProcess(command, 0, stdout="3.11")
                if command[0] == "python" and command[1:3] == ["-m", "venv"]:
                    Path(command[-1]).mkdir(parents=True)
                    return subprocess.CompletedProcess(command, 0, stdout="")
                if "-m" in command and "pip" in command:
                    return subprocess.CompletedProcess(command, 0, stdout="")
                if command[0].endswith("python") and command[1:2] == ["-c"]:
                    code = command[2]
                    if "torch.version.cuda" in code:
                        return subprocess.CompletedProcess(command, 0, stdout="")
                    if "torch.cuda.is_available()" in code:
                        return subprocess.CompletedProcess(command, 1, stdout="CUDA driver unavailable")
                raise AssertionError(f"Unexpected setup command: {command!r}")

            with patch.object(bootstrap, "ROOT", root), \
                    patch.object(bootstrap, "DEFAULTS", dict(bootstrap.DEFAULTS)), \
                    patch.object(bootstrap, "KEEP_FAILURE_LOGS", False), \
                    patch.object(bootstrap.sys, "platform", "linux"), \
                    patch.object(bootstrap.sys, "argv", [
                        "lfm", "setup", "--device", "cuda", "--with-pulsar2",
                        "--cache", str(cache),
                    ]), \
                    patch.object(bootstrap, "preflight"), \
                    patch.object(bootstrap.shutil, "which", side_effect=which), \
                    patch.object(bootstrap.subprocess, "run", side_effect=run), \
                    patch.object(bootstrap, "install_pulsar2") as install_pulsar:
                result = bootstrap.main()

            self.assertNotEqual(result, 0)
            self.assertFalse((root / ".venv" / ".lfm-device").exists())
            install_pulsar.assert_not_called()
            self.assertTrue(any(
                command[0].endswith("python") and command[1:2] == ["-c"]
                and "torch.cuda.is_available()" in command[2]
                for command in calls
            ))
            failure_logs = list((root / "logs").glob("failure*.log"))
            self.assertEqual(len(failure_logs), 1)
            self.assertIn("CUDA driver unavailable", failure_logs[0].read_text(encoding="utf-8"))

    def test_cpu_setup_skips_cuda_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "requirements-cpu.txt").write_text("# fixture\n", encoding="utf-8")
            cache = root / "cache"
            calls = []

            def which(name):
                return "python"

            def run(command, **kwargs):
                calls.append(command)
                if command[0] == "python" and command[1:2] == ["-c"]:
                    return subprocess.CompletedProcess(command, 0, stdout="3.11")
                if command[0] == "python" and command[1:3] == ["-m", "venv"]:
                    Path(command[-1]).mkdir(parents=True)
                    return subprocess.CompletedProcess(command, 0, stdout="")
                if "-m" in command and "pip" in command:
                    return subprocess.CompletedProcess(command, 0, stdout="")
                if command[0].endswith("python") and command[1:2] == ["-c"]:
                    return subprocess.CompletedProcess(command, 0, stdout="")
                raise AssertionError(f"Unexpected setup command: {command!r}")

            with patch.object(bootstrap, "ROOT", root), \
                    patch.object(bootstrap, "KEEP_FAILURE_LOGS", False), \
                    patch.object(bootstrap.sys, "platform", "linux"), \
                    patch.object(bootstrap.sys, "argv", [
                        "lfm", "setup", "--device", "cpu", "--with-pulsar2",
                        "--cache", str(cache),
                    ]), \
                    patch.object(bootstrap, "preflight"), \
                    patch.object(bootstrap.shutil, "which", side_effect=which), \
                    patch.object(bootstrap.subprocess, "run", side_effect=run), \
                    patch.object(bootstrap, "install_pulsar2") as install_pulsar:
                result = bootstrap.main()

            self.assertEqual(result, 0)
            self.assertTrue((root / ".venv" / ".lfm-device").is_file())
            install_pulsar.assert_called_once()
            self.assertFalse(any(
                len(command) > 2 and "torch.cuda.is_available()" in command[2]
                for command in calls
            ))


if __name__ == "__main__":
    unittest.main()
