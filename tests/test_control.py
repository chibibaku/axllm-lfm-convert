"""Control-plane regressions: resumption, isolation, failure logs and profiles."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from lfm_convert.util import (MODEL_REPO, MODEL_REVISION, RunLog, atomic_json,
                              require_marker, require_profile)


def recipe(dtype="fp32", storage="disk"):
    return {"schema_version": 1, "name": f"LFM2.5-2.6B-{dtype.upper()}",
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "context_length": 131072, "tile_size": 2048,
            "eos_token_id": 124900,
            "generation": {"max_new_tokens": 384, "repetition_penalty": 1.1},
            "kv": {"dtype": dtype, "storage": storage,
                   "supported_storage": ["disk"] if dtype == "fp32" else ["cmm", "disk"]},
            "compatibility": {"runtime_commit": None, "convert_commit": None}}


class ControlTests(unittest.TestCase):
    def test_local_source_needs_files_but_no_provenance_record(self):
        from lfm_convert.cli import acquire_source
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            atomic_json(source / "config.json", {})
            atomic_json(source / "model.safetensors.index.json", {
                "weight_map": {"weight": "weights.safetensors"}})
            weight = source / "weights.safetensors"
            weight.write_bytes(b"weights")
            self.assertEqual(acquire_source(source, Path(tmp), None), source)
            weight.write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                acquire_source(source, Path(tmp), None)

    def test_resume_checks_recipe_and_missing_files(self):
        from lfm_convert.cli import _stage_valid
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            folder = cache / "calibration"
            atomic_json(folder / "manifest.json", {})
            (folder / "input.tar").write_bytes(b"samples")
            receipt = cache / "checkpoints/calibrate.json"
            recipe = {"device": "cpu"}
            atomic_json(receipt, {"fingerprint": recipe, "files": ["manifest.json", "input.tar"]})
            self.assertTrue(_stage_valid(cache, "calibrate", recipe, receipt))
            self.assertFalse(_stage_valid(cache, "calibrate", {"device": "cuda"}, receipt))
            (folder / "input.tar").unlink()
            self.assertFalse(_stage_valid(cache, "calibrate", recipe, receipt))

    def test_compiler_reuses_complete_models_and_retries_failed_graph(self):
        from lfm_convert.pipeline import compile_graphs
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            calls = []
            fail = False

            class Compiler:
                def command(self, args):
                    calls.append(args)
                    target = Path(args[args.index("--output_dir") + 1])
                    (target / "model.axmodel").write_bytes(b"partial" if fail else b"model")
                    (target / "scratch.txt").write_text("compiler scratch")
                    if fail:
                        raise RuntimeError("compiler failed")

            def compile():
                graph = cache / "onnx/generic/layer_00/model.onnx"
                graph.parent.mkdir(parents=True, exist_ok=True)
                graph.write_bytes(b"graph")
                atomic_json(graph.parent / "config.json", {})
                with patch("lfm_convert.pipeline.shutil.which", side_effect=lambda name: None if name == "docker" else "pulsar2"):
                    compile_graphs(cache, recipe(), "pulsar2", 1, Compiler(),
                                   graph_paths=["generic/layer_00"], finalize=False)

            compile()
            compile()
            self.assertEqual(len(calls), 1)
            target = cache / "compiled/generic/layer_00"
            self.assertEqual({p.name for p in target.iterdir()}, {"model.axmodel", "complete"})
            (target / "model.axmodel").unlink()
            fail = True
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                compile()
            self.assertFalse(target.exists())
            target.mkdir(parents=True)
            (target / "model.axmodel").write_bytes(b"interrupted output")
            fail = False
            compile()
            self.assertEqual(len(calls), 3)

    def test_build_resumes_missing_graph_and_rebuilds_when_device_changes(self):
        from lfm_convert.cli import build
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, output = root / "cache", root / "dist"
            profile_dir = root / "profile"
            profile_dir.mkdir()
            args = SimpleNamespace(cache=cache, device="cpu", min_free_gib=0,
                                   pulsar_archive=None, source=None, assets=None,
                                   profile=profile_dir, jobs=1)
            calls = {"calibrate": 0, "export": 0, "graphs": 0, "package": 0}
            paths = [f"generic/graph_{i:02d}" for i in range(48)]

            def calibrate(source, cache, profile, device):
                calls["calibrate"] += 1
                atomic_json(cache / "calibration/manifest.json", {"device": device})
                (cache / "calibration/input.tar").write_bytes(b"samples")

            def compile_graphs(cache, profile, compiler, jobs, log, graph_paths=None, finalize=True):
                for relative in graph_paths or []:
                    target = cache / "compiled" / relative
                    if not (target / "model.axmodel").exists():
                        calls["graphs"] += 1
                        target.mkdir(parents=True, exist_ok=True)
                        (target / "model.axmodel").write_bytes(b"model")
                        (target / "complete").touch()
                if finalize:
                    atomic_json(cache / "compiled/manifest.json", {"graphs": [{"path": p} for p in paths]})

            def export_graphs(source, cache, profile, on_batch):
                calls["export"] += 1
                for relative in paths:
                    graph = cache / "onnx" / relative / "model.onnx"
                    graph.parent.mkdir(parents=True, exist_ok=True)
                    graph.write_bytes(b"graph")
                    atomic_json(graph.parent / "config.json", {})
                on_batch(paths)
                for relative in paths:
                    (cache / "onnx" / relative / "model.onnx").unlink()
                    (cache / "onnx" / relative / "config.json").unlink()
                atomic_json(cache / "onnx/manifest.json", {
                    "graphs": [{"path": f"{p}/model.onnx"} for p in paths]})

            def package(*args):
                calls["package"] += 1

            with patch("lfm_convert.cli.check_environment"), \
                    patch("lfm_convert.cli.acquire_source", return_value=root), \
                    patch("lfm_convert.cli.require_stage_capacity", return_value=40), \
                    patch("lfm_convert.pipeline.calibrate", side_effect=calibrate), \
                    patch("lfm_convert.pipeline.export_graphs", side_effect=export_graphs), \
                    patch("lfm_convert.pipeline.compile_graphs", side_effect=compile_graphs), \
                    patch("lfm_convert.pipeline.package", side_effect=package), \
                    contextlib.redirect_stdout(io.StringIO()):
                with RunLog(root) as log:
                    self.assertEqual(build(args, recipe(), output, log), 0)
                    self.assertEqual(build(args, recipe(), output, log), 0)
                    self.assertEqual(calls, {"calibrate": 1, "export": 1, "graphs": 48, "package": 2})
                    (cache / "compiled" / paths[0] / "model.axmodel").unlink()
                    build(args, recipe(), output, log)
                    self.assertEqual(calls["graphs"], 49)
                    self.assertEqual(calls["calibrate"], 1)
                    args.device = "cuda"
                    build(args, recipe(), output, log)
                    self.assertEqual(calls["calibrate"], 2)
                    self.assertEqual(calls["graphs"], 97)

    def test_large_stage_rechecks_ram_and_disk_before_starting(self):
        from lfm_convert.cli import require_stage_capacity
        gib = 1024 ** 3
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            with patch("lfm_convert.cli.disk_available", return_value=40 * gib), \
                    patch("lfm_convert.cli.memory_available", return_value=17 * gib):
                with self.assertRaisesRegex(RuntimeError, "RAM.*before export"):
                    require_stage_capacity(cache, 20, "export")
            with patch("lfm_convert.cli.disk_available", return_value=19 * gib), \
                    patch("lfm_convert.cli.memory_available", return_value=30 * gib):
                with self.assertRaisesRegex(RuntimeError, "before compiling"):
                    require_stage_capacity(cache, 20, "compiling layer_00")
            with patch("lfm_convert.cli.disk_available", return_value=40 * gib), \
                    patch("lfm_convert.cli.memory_available", return_value=30 * gib):
                self.assertEqual(require_stage_capacity(cache, 20, "export"), 40)

    def test_public_setup_prepares_compiler_without_an_extra_flag(self):
        from lfm_convert.cli import main
        with tempfile.TemporaryDirectory() as temporary, \
                patch("lfm_convert.cli.subprocess.call", return_value=0) as setup_call:
            self.assertEqual(main(["setup", "--device", "cpu", "--cache", temporary]), 0)
        command = setup_call.call_args.args[0]
        self.assertIn("--with-pulsar2", command)
        self.assertEqual(command[command.index("--device") + 1], "cpu")

    def test_setup_does_not_claim_a_cache_containing_user_files(self):
        from lfm_convert.cli import main
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            user_file = cache / "important.txt"
            user_file.write_text("user input", encoding="utf-8")
            with patch("lfm_convert.cli.subprocess.call") as setup_call:
                with self.assertRaisesRegex(RuntimeError, "Cache contains user files"):
                    main(["setup", "--device", "cpu", "--cache", str(cache)])
                setup_call.assert_not_called()
            self.assertEqual(user_file.read_text(encoding="utf-8"), "user input")
            self.assertFalse((cache / ".lfm-managed").exists())

    def test_preflight_rejects_low_disk_before_environment_setup(self):
        from lfm_convert.cli import preflight_setup
        with patch("lfm_convert.cli.sys.platform", "linux"), \
                patch("lfm_convert.cli.platform.machine", return_value="x86_64"), \
                patch("lfm_convert.cli.Path.is_file", return_value=True), \
                patch("lfm_convert.cli.Path.read_text", return_value="ID=ubuntu\n"), \
                patch("lfm_convert.cli.disk_available", return_value=19 * 1024 ** 3), \
                patch("lfm_convert.cli.memory_available", return_value=32 * 1024 ** 3), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Need 20.0 GiB free before setup"):
                preflight_setup(Path("cache"), 20)

    def test_missing_docker_stops_before_model_work(self):
        from lfm_convert.cli import check_environment
        from types import SimpleNamespace
        with patch("lfm_convert.cli.sys.platform", "linux"), \
                patch("lfm_convert.cli.platform.machine", return_value="x86_64"), \
                patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
                patch("lfm_convert.cli.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Docker CLI is unavailable"):
                check_environment("cpu", 20, Path("cache"), None)

    def test_cuda_request_fails_without_cuda_before_compiler_setup(self):
        from lfm_convert.cli import check_environment
        from types import SimpleNamespace
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch("lfm_convert.cli.sys.platform", "linux"), \
                patch("lfm_convert.cli.platform.machine", return_value="x86_64"), \
                patch.dict(sys.modules, {"torch": torch}), \
                patch("lfm_convert.cli.shutil.which") as dependency_lookup:
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                check_environment("cuda", 20, Path("cache"), None)
            dependency_lookup.assert_not_called()

    def test_preflight_uses_configured_memory_limit(self):
        from lfm_convert.cli import preflight_setup
        with patch("lfm_convert.cli.sys.platform", "linux"), \
                patch("lfm_convert.cli.platform.machine", return_value="x86_64"), \
                patch("lfm_convert.cli.Path.is_file", return_value=True), \
                patch("lfm_convert.cli.Path.read_text", return_value="ID=ubuntu\n"), \
                patch("lfm_convert.cli.disk_available", return_value=100 * 1024 ** 3), \
                patch("lfm_convert.cli.memory_available", return_value=17 * 1024 ** 3), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "18 GiB available RAM"):
                preflight_setup(Path("cache"), 20)


    def test_success_keeps_the_last_failure_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    with RunLog(root):
                        raise RuntimeError("retain this failure")
            before = list((root / "logs").glob("*.log"))
            with RunLog(root):
                pass
            self.assertEqual(list((root / "logs").glob("*.log")), before)
            self.assertIn("retain this failure", before[0].read_text())

    def test_interrupted_child_preserves_failure_log_without_masking_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("lfm_convert.util.subprocess.run", side_effect=KeyboardInterrupt), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    with RunLog(root) as log:
                        log.command(["interrupted-child"])
            logs = list((root / "logs").glob("*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("KeyboardInterrupt", logs[0].read_text())
            self.assertFalse(list((root / "logs").glob("running-*.log")))

    def test_setup_failure_replaces_prior_pipeline_failure(self):
        from lfm_convert import bootstrap
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    with RunLog(root):
                        raise RuntimeError("pipeline failure")
                with patch.object(bootstrap, "ROOT", root), \
                        patch.object(bootstrap, "KEEP_FAILURE_LOGS", False):
                    bootstrap.record_failure(["installer"], "setup failure")
            logs = list((root / "logs").glob("*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("setup failure", logs[0].read_text())


    def test_unsupported_parallelism_is_rejected_before_setup(self):
        from lfm_convert.cli import main
        with patch("lfm_convert.cli.ensure_venv") as setup, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["build", "--profile", "unused", "--jobs", "2"])
            setup.assert_not_called()


    def test_profile_directory_and_file_resolve_same_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            atomic_json(path / "profile.json", recipe())
            self.assertEqual(require_profile(path), require_profile(path / "profile.json"))

    def test_model_outputs_are_isolated_for_directory_profiles(self):
        from lfm_convert.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            for precision, storage in [("fp32", "disk"), ("bf16", "cmm")]:
                directory = Path(tmp) / precision
                directory.mkdir()
                atomic_json(directory / "profile.json", recipe(precision, storage))
                with patch("lfm_convert.pipeline.verify_package", return_value=0) as verify:
                    self.assertEqual(main(["verify", "--profile", str(directory)]), 0)
                    self.assertEqual(verify.call_args.args[0], directory / "dist")
                with patch("lfm_convert.pipeline.verify_package", return_value=0) as verify:
                    self.assertEqual(main(["verify", "--profile", str(directory / "profile.json")]), 0)
                    self.assertEqual(verify.call_args.args[0], directory / "dist")

    def test_packaging_does_not_replace_unmanaged_user_files(self):
        from lfm_convert.pipeline import package
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "assets", root / "user-output"
            source.mkdir()
            output.mkdir()
            keep = output / "important.txt"
            keep.write_text("user data")
            with self.assertRaises((RuntimeError, ValueError)):
                package(source, output, recipe())
            self.assertEqual(keep.read_text(), "user data")
            self.assertFalse((output / ".lfm-managed").exists())

    def test_profile_rejects_wrong_revision_and_fp32_cmm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            for profile in [recipe("fp32", "cmm"), recipe()]:
                if profile["kv"]["storage"] == "disk":
                    profile["model"]["revision"] = "main"
                atomic_json(path, profile)
                with self.assertRaises(ValueError):
                    require_profile(path)

    def test_profile_rejects_unsupported_storage_and_mislabeled_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            unsupported = recipe("bf16", "remote")
            unsupported["kv"]["supported_storage"] = ["remote"]
            mislabeled = recipe("bf16", "cmm")
            mislabeled["name"] = "LFM2.5-2.6B-FP32"
            for profile in [unsupported, mislabeled]:
                atomic_json(path, profile)
                with self.assertRaises(ValueError):
                    require_profile(path)

    def test_explicit_log_history_keeps_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ("first failure", "second failure"):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        with RunLog(root, preserve=True) as log:
                            log.say(label)
                            raise RuntimeError(label)
            logs = list((root / "logs").glob("*.log"))
            self.assertEqual(len(logs), 2)
            self.assertTrue(any("first failure" in item.read_text() for item in logs))
            self.assertTrue(any("second failure" in item.read_text() for item in logs))

    def test_default_failures_preserve_explicitly_saved_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, preserve in (("saved history", True),
                                    ("older automatic failure", False),
                                    ("latest automatic failure", False)):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        with RunLog(root, preserve=preserve) as log:
                            log.say(label)
                            raise RuntimeError(label)
            history = list((root / "logs").glob("failure-*.log"))
            self.assertEqual(len(history), 1)
            self.assertIn("saved history", history[0].read_text())
            automatic = (root / "logs" / "failure.log").read_text()
            self.assertIn("latest automatic failure", automatic)
            self.assertNotIn("older automatic failure", automatic)
            self.assertEqual(len(list((root / "logs").glob("*.log"))), 2)

    def test_cleanup_finds_unified_model_profiles_and_preserves_recipes(self):
        from lfm_convert import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "axllm-lfm-convert"
            root.mkdir()
            profiles = root.parent / "axllm-lfm-model" / "profiles"
            for precision in ("FP32", "BF16"):
                directory = profiles / precision
                output = directory / "dist"
                output.mkdir(parents=True)
                (output / ".lfm-managed").touch()
                (output / "model.axmodel").write_bytes(b"generated")
                (directory / "profile.json").write_text("{}")
            with patch.object(cli, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                cli.clean(False, root / ".cache", None)
            for precision in ("FP32", "BF16"):
                self.assertFalse((profiles / precision / "dist").exists())
                self.assertTrue((profiles / precision / "profile.json").is_file())

    def test_cleanup_rejects_unmarked_and_external_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            target = root / ".cache"
            target.mkdir()
            with self.assertRaises(ValueError):
                require_marker(target, root)
            (target / ".lfm-managed").touch()
            require_marker(target, root)
            outside = Path(tmp) / "user-data"
            outside.mkdir()
            (outside / ".lfm-managed").touch()
            with self.assertRaises(ValueError):
                require_marker(outside, root)

    def test_success_removes_verbose_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                with RunLog(root) as log:
                    log.command([sys.executable, "-c", "print('verbose installer output')"])
            self.assertFalse(list((root / "logs").glob("*.log")))

    def test_only_latest_failure_is_retained_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ("first failure", "latest failure"):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        with RunLog(root) as log:
                            log.say(label)
                            raise RuntimeError(label)
            logs = list((root / "logs").glob("*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("latest failure", logs[0].read_text())
            self.assertNotIn("first failure", logs[0].read_text())


if __name__ == "__main__":
    unittest.main()
