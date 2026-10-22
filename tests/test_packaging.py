"""Packaging regressions for standalone profile documentation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lfm_convert.pipeline import package
from lfm_convert.util import MODEL_REPO, MODEL_REVISION


def profile():
    return {
        "schema_version": 1,
        "name": "LFM2.5-2.6B-FP32",
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "context_length": 131072,
        "tile_size": 2048,
        "eos_token_id": 124900,
        "generation": {"max_new_tokens": 1024, "repetition_penalty": 1.1},
        "kv": {"dtype": "fp32", "storage": "disk", "supported_storage": ["disk"]},
        "compatibility": {"runtime_commit": None, "convert_commit": None},
    }


class PackageDocumentationTests(unittest.TestCase):
    def test_package_checks_required_payload_without_file_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            source = root / "source"
            profile_root = root / "profile"
            output = root / "standalone-package"

            compiled = assets / "compiled"
            generic_layers = [i for i in range(30) if i not in {2, 5, 9, 13, 17, 21, 24, 27}]
            for name in [*(f"layer_{i:02d}" for i in generic_layers), "head"]:
                path = compiled / "generic" / name / "model.axmodel"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"graph:{name}".encode())
            for index in (2, 5, 9, 13, 17, 21, 24, 27):
                for part in ("prepare", "tile", "finish"):
                    path = compiled / "parts" / f"layer_{index:02d}" / part / "model.axmodel"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"graph:{index}:{part}".encode())
            merge = compiled / "parts" / "merge" / "model.axmodel"
            merge.parent.mkdir(parents=True, exist_ok=True)
            merge.write_bytes(b"graph:merge")

            embedding = assets / "calibration" / "embedding.bf16"
            embedding.parent.mkdir(parents=True)
            embedding.write_bytes(b"fake-embedding")
            rope = assets / "onnx" / "parts" / "rope_inv_freq.fp32"
            rope.parent.mkdir(parents=True)
            rope.write_bytes(bytes(128))

            source.mkdir()
            (source / "config.json").write_text(json.dumps({"eos_token_id": 124900}))
            for filename, content in {
                "tokenizer.json": "{}",
                "tokenizer_config.json": "{}",
                "generation_config.json": "{}",
                "special_tokens_map.json": "{}",
                "chat_template.jinja": "{{ messages }}",
                "LICENSE": "model license",
            }.items():
                (source / filename).write_text(content)

            profile_root.mkdir()
            (profile_root / "README.md").write_text(
                "See [validation summary](docs/validation-summary.md).")
            (profile_root / "LICENSE").write_text("profile license")
            summary = "Standalone validation summary.\n"
            summary_path = profile_root / "docs" / "validation-summary.md"
            summary_path.parent.mkdir()
            summary_path.write_text(summary)

            with patch("lfm_convert.pipeline.EMBEDDING_BYTES", len(b"fake-embedding")):
                package(assets, output, profile(), model_source=source, profile_root=profile_root)

            packaged_summary = output / "docs" / "validation-summary.md"
            self.assertEqual(packaged_summary.read_text(), summary)
            self.assertFalse((output / "manifest.json").exists())
            # User-added documentation is allowed; missing runtime files are not.
            (output / "notes.txt").write_text("local notes")
            from lfm_convert.pipeline import verify_package
            with patch("lfm_convert.pipeline.EMBEDDING_BYTES", len(b"fake-embedding")):
                self.assertEqual(verify_package(output, profile()), 0)
                (output / "generic/head.axmodel").unlink()
                with self.assertRaisesRegex(RuntimeError, "AX model set mismatch"):
                    verify_package(output, profile())



if __name__ == "__main__":
    unittest.main()
