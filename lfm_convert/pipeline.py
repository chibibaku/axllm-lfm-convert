"""The single, revision-pinned calibration/export/compile/package pipeline."""
from __future__ import annotations

import io
import json
import math
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from .util import MODEL_REVISION, MODEL_REPO, atomic_json

PROMPTS = ["日本の首都はどこですか。一文で答えてください。",
           "What is 2 + 3? Answer briefly.", "猫について短く説明してください。",
           "Write a short greeting in English."]
PINNED_IMAGE = "pulsar2:7.0-patch1"
EMBEDDING_BYTES = 524288000
LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
               "conv", "conv", "conv", "full_attention", "conv", "conv", "conv",
               "full_attention", "conv", "conv", "conv", "full_attention", "conv",
               "conv", "conv", "full_attention", "conv", "conv", "full_attention",
               "conv", "conv", "full_attention", "conv", "conv"]


def _archive(path: Path, arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for i, tensor in enumerate(arrays):
            buffer = io.BytesIO()
            np.save(buffer, tensor.detach().float().cpu().numpy())
            content = buffer.getvalue()
            member = tarfile.TarInfo(f"{i:04d}.npy")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def _mask(page_length: int, tile_size: int, device):
    import torch
    return torch.tensor([[[[0.0] * page_length + [-10000.0] * (tile_size - page_length)]]],
                        dtype=torch.float32, device=device)


def calibrate(source: Path, cache: Path, profile: dict, device: str):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.lfm2.modeling_lfm2 import Lfm2RMSNorm
    from .validation import NativeParityGate

    if profile["context_length"] != 131072 or profile["tile_size"] != 2048:
        raise ValueError("Only the verified 131072/2048 model contract is supported.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    target = torch.device("cuda" if device == "cuda" else "cpu")
    load_options = {"dtype": torch.float32, "attn_implementation": "eager"}
    if device == "cuda":
        # Dispatch directly from the safetensors shards into CUDA. Loading on
        # CPU and then calling Module.to(cuda) creates a large transfer-time
        # peak for this FP32 teacher even when its steady-state weights fit.
        load_options["device_map"] = {"": str(target)}
        model = AutoModelForCausalLM.from_pretrained(source, **load_options).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(source, **load_options).to(target).eval()
    tokenizer = AutoTokenizer.from_pretrained(source)
    if model.config.layer_types != LAYER_TYPES or model.config.max_position_embeddings != 131072:
        raise RuntimeError("Model architecture differs from the pinned 30-layer LFM2.5 contract.")
    captured: dict[int, dict[str, list]] = {i: {} for i in range(31)}
    capture = False
    position = 0

    def save(index, **tensors):
        bucket = captured[index]
        for name, value in tensors.items():
            bucket.setdefault(name, []).append(value.detach().float().cpu().clone())

    def layer_hook(index):
        layer = model.model.layers[index]
        def callback(_module, args, kwargs):
            if not capture:
                return
            hidden = args[0] if args else kwargs["hidden_states"]
            cache_obj = kwargs["past_key_values"]
            length = hidden.shape[1]
            if layer.is_attention_layer:
                cos, sin = kwargs["position_embeddings"]
                normalized = layer.operator_norm(hidden)
                attn = layer.self_attn
                q = attn.q_layernorm(attn.q_proj(normalized).view(1, length, 32, 64)).transpose(1, 2)
                k = attn.k_layernorm(attn.k_proj(normalized).view(1, length, 8, 64)).transpose(1, 2)
                v = attn.v_proj(normalized).view(1, length, 8, 64).transpose(1, 2)
                half_q = torch.cat((-q[..., 32:], q[..., :32]), dim=-1)
                half_k = torch.cat((-k[..., 32:], k[..., :32]), dim=-1)
                q = q * cos.unsqueeze(1) + half_q * sin.unsqueeze(1)
                k = k * cos.unsqueeze(1) + half_k * sin.unsqueeze(1)
                for local in range(length):
                    if length > 1:
                        pk, pv = k[..., :local, :], v[..., :local, :]
                    else:
                        pk, pv = cache_obj.key_cache[index], cache_obj.value_cache[index]
                    key_now, value_now = k[..., local:local + 1, :], v[..., local:local + 1, :]
                    keys = torch.cat((pk, key_now), dim=2)
                    values = torch.cat((pv, value_now), dim=2)
                    hidden_now = hidden[:, local:local + 1]
                    cos_now, sin_now = cos[:, local:local + 1], sin[:, local:local + 1]
                    save(index, prep_hidden=hidden_now, cos=cos_now, sin=sin_now)
                    # Cache a real active 2048-token page, left aligned and zero padded.
                    page_count = keys.shape[2]
                    page_count = min(page_count, 2048)
                    page_k = torch.zeros((1, 8, 2048, 64), device=target)
                    page_v = torch.zeros_like(page_k)
                    page_k[:, :, :page_count] = keys[:, :, :page_count]
                    page_v[:, :, :page_count] = values[:, :, :page_count]
                    tile_q = q[:, :, local:local + 1]
                    tile_mask = _mask(page_count, 2048, target)
                    repeated_k = page_k[:, :, None].expand(1, 8, 4, 2048, 64).reshape(1, 32, 2048, 64)
                    repeated_v = page_v[:, :, None].expand(1, 8, 4, 2048, 64).reshape(1, 32, 2048, 64)
                    scores = tile_q @ repeated_k.transpose(-1, -2) * attn.scaling + tile_mask
                    maximum = scores.amax(dim=-1, keepdim=True)
                    total = torch.exp(scores - maximum).sum(dim=-1, keepdim=True)
                    probs = torch.softmax(scores, dim=-1)
                    tile_out = probs @ repeated_v
                    save(index, tile_query=tile_q, tile_keys=page_k, tile_values=page_v,
                         tile_mask=tile_mask, finish_hidden=hidden_now, finish_attention=tile_out,
                         merge_out=tile_out, merge_max=maximum, merge_total=total)
            else:
                if length > 1:
                    b, c, x = layer.conv.in_proj(layer.operator_norm(hidden)).transpose(1, 2).chunk(3, dim=1)
                    bx = b * x
                    for local in range(length):
                        state = torch.zeros((1, model.config.hidden_size, 3), device=target)
                        if local:
                            count = min(local, 3)
                            state[:, :, -count:] = bx[:, :, local - count:local]
                        save(index, conv_hidden=hidden[:, local:local + 1], conv_state=state)
                else:
                    save(index, conv_hidden=hidden, conv_state=cache_obj.conv_cache[index].clone())
        return callback

    handles = [layer.register_forward_pre_hook(layer_hook(i), with_kwargs=True)
               for i, layer in enumerate(model.model.layers)]
    parity = NativeParityGate(model, target)
    parity.install()
    def head_hook(_module, args):
        if capture:
            for token_hidden in args[0].split(1, dim=1):
                save(30, head_hidden=token_hidden)
    handles.append(model.model.embedding_norm.register_forward_pre_hook(head_hook))
    prompt_meta = []
    try:
        with torch.inference_mode():
            for prompt_index, prompt in enumerate(PROMPTS):
                ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                    tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").input_ids[0].tolist()
                prompt_meta.append({"text": prompt, "input_ids": ids})
                capture = True
                position = 0
                if prompt_index == 0:
                    parity.begin("initial4")
                result = model(torch.tensor([ids[:4]], device=target), use_cache=True)
                if prompt_index == 0:
                    parity.end("initial4")
                past = result.past_key_values
                next_token = int(result.logits[0, -1].argmax())
                selected = {len(ids) - 1, len(ids), len(ids) + 15, len(ids) + 31}
                for position in range(4, len(ids) + 32):
                    capture = position in selected
                    token = ids[position] if position < len(ids) else next_token
                    if prompt_index == 0 and position == 4:
                        parity.begin("decode1")
                    result = model(torch.tensor([[token]], device=target), past_key_values=past,
                                   use_cache=True, cache_position=torch.tensor([position], device=target))
                    if prompt_index == 0 and position == 4:
                        parity.end("decode1")
                    past = result.past_key_values
                    next_token = int(result.logits[0, -1].argmax())
        parity.assert_complete()
    finally:
        for handle in handles:
            handle.remove()
        parity.remove()

    calibration = cache / "calibration"
    if calibration.exists():
        shutil.rmtree(calibration)
    calibration.mkdir(parents=True)
    atomic_json(calibration / "native-parity.json", parity.to_json())
    expected = {
        **{i: ({"conv_hidden", "conv_state"} if LAYER_TYPES[i] == "conv" else
               {"prep_hidden", "cos", "sin", "tile_query", "tile_keys", "tile_values",
                "tile_mask", "finish_hidden", "finish_attention", "merge_out", "merge_max", "merge_total"})
           for i in range(30)},
        30: {"head_hidden"},
    }
    manifest_layers = []
    for index in range(31):
        folder = calibration / (f"layer_{index:02d}" if index < 30 else "head")
        folder.mkdir()
        values = captured[index]
        missing = expected[index] - values.keys()
        if missing:
            raise RuntimeError(f"Layer {index} missing calibration tensors: {sorted(missing)}")
        for name in expected[index]:
            _archive(folder / f"{name}.tar", values[name][:32])
            np.save(folder / f"{name}.npy", values[name][-1].detach().float().cpu().numpy())
        manifest_layers.append({"layer": index, "samples": min(len(x) for x in values.values()),
                                "inputs": sorted(expected[index])})
    # Keep all compiler calibration inputs in the immutable calibration checkpoint.
    merge_samples = calibration / "merge"
    merge_samples.mkdir()
    merge_values = {"old_output": "merge_out", "old_maximum": "merge_max",
                    "old_total": "merge_total", "tile_output": "merge_out",
                    "tile_maximum": "merge_max", "tile_total": "merge_total"}
    for name, source_name in merge_values.items():
        tensors = captured[2][source_name][:32]
        if name == "old_output":
            tensors = [tensor * 0.7 for tensor in tensors]
        elif name == "old_maximum":
            tensors = [tensor - 0.5 for tensor in tensors]
        elif name == "old_total":
            tensors = [tensor * 1.2 for tensor in tensors]
        _archive(merge_samples / f"{name}.tar", tensors)
        np.save(merge_samples / f"{name}.npy", tensors[-1].detach().float().cpu().numpy())
    emb = model.model.embed_tokens.weight.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy()
    emb.tofile(calibration / "embedding.bf16")
    from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding
    rotary = Lfm2RotaryEmbedding(config=model.config)
    rotary.inv_freq.detach().to(device="cpu", dtype=torch.float32).numpy().tofile(
        calibration / "rope_inv_freq.fp32")
    atomic_json(calibration / "manifest.json", {
        "revision": MODEL_REVISION, "repo": MODEL_REPO, "context_length": 131072,
        "tile_size": 2048, "calibration_dtype": "FP32", "calibration_device": target.type,
        "teacher_initial_chunk": 4,
        "reason": "avoids the pinned Transformers 5.2 early one-token Conv cache defect",
        "sample_count": 32, "positions": "prefill 0..3; prompt final; generated 0/15/31",
        "prompts": prompt_meta, "layers": manifest_layers,
        "native_parity": {"path": "native-parity.json", "tensor_checks": len(parity.results),
                          "cases": ["initial4", "decode1"]},
    })
    del model
    if target.type == "cuda":
        torch.cuda.empty_cache()


def _read_samples(path: Path):
    import numpy as np
    with tarfile.open(path) as archive:
        for member in sorted((m for m in archive.getmembers() if m.isfile()), key=lambda m: m.name):
            data = archive.extractfile(member).read()
            yield np.load(io.BytesIO(data), allow_pickle=False)


def _export(module, sample_names, output_names, sample_values, folder: Path,
            sample_keys=None, float_only=False):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    module.eval()
    sample_keys = sample_keys or sample_names
    sample = tuple(torch.from_numpy(np.load(sample_values / f"{name}.npy")) for name in sample_keys)
    tmp = folder / "model.partial.onnx"
    try:
        with torch.inference_mode():
            expected = module(*sample)
            if isinstance(expected, torch.Tensor):
                expected = (expected,)
            torch.onnx.export(module, sample, tmp, input_names=sample_names,
                              output_names=output_names, opset_version=17, dynamo=False)
        onnx.checker.check_model(str(tmp))
        tmp.replace(folder / "model.onnx")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    session = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
    actual = session.run(None, {name: tensor.numpy() for name, tensor in zip(sample_names, sample)})
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b.numpy(), rtol=3e-4, atol=3e-5)
    calibration_root = sample_values.relative_to(sample_values.parents[1]).as_posix()
    selectors = [{"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"],
                  "data_type": "FP32" if float_only else "U16"}]
    if not float_only:
        selectors.append({"op_types": ["Pow", "ReduceMean", "Sqrt", "RMSNormalization", "Add",
                                        "Mul", "Div", "Sub", "Sigmoid", "Concat", "Softmax",
                                        "ReduceMax", "ReduceSum", "Exp"], "data_type": "FP32"})
    config = {
        "model_type": "ONNX", "target_hardware": "AX650", "npu_mode": "NPU3",
        "quant": {"input_configs": [
            {"tensor_name": name, "calibration_dataset": f"/data/{calibration_root}/{key}.tar",
             "calibration_format": "Numpy", "calibration_size": 32}
            for name, key in zip(sample_names, sample_keys)], "calibration_method": "MinMax",
            "layer_configs": selectors},
        "input_processors": [{"tensor_name": "DEFAULT", "src_dtype": "FP32"}],
        "output_processors": [{"tensor_name": "DEFAULT", "output_dtype": "FP32"}],
        "compiler": {"check": 0, "check_cosine_simularity": 0.999},
    }
    atomic_json(folder / "config.json", config)


def export_graphs(source: Path, cache: Path, profile: dict, on_batch=None):
    import numpy as np
    import onnx
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.lfm2.modeling_lfm2 import Lfm2DecoderLayer, Lfm2RMSNorm
    from .models import ExplicitConvDecoder, Prepare, Tile, Finish, Merge, Head

    calibration = cache / "calibration"
    out = cache / "onnx"
    if out.exists():
        shutil.rmtree(out)
    config = AutoConfig.from_pretrained(source, trust_remote_code=False)
    config._attn_implementation = "eager"
    if config.layer_types != LAYER_TYPES:
        raise RuntimeError("Layer layout changed from the pinned recipe.")
    graphs = []

    def complete_batch(folders):
        batch = []
        for folder in folders:
            graph = folder / "model.onnx"
            config_path = folder / "config.json"
            relative = folder.relative_to(out).as_posix()
            graphs.append({"path": f"{relative}/model.onnx"})
            batch.append(relative)
        if on_batch:
            on_batch(batch)
            for folder in folders:
                (folder / "model.onnx").unlink(missing_ok=True)
                (folder / "config.json").unlink(missing_ok=True)

    for index in range(30):
        layer = Lfm2DecoderLayer(config, layer_idx=index).float().eval()
        weights = {}
        prefix = f"model.layers.{index}."
        for shard in sorted(source.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if key.startswith(prefix):
                        weights[key[len(prefix):]] = sf.get_tensor(key).float()
        layer.load_state_dict(weights, strict=True)
        samples = calibration / f"layer_{index:02d}"
        if config.layer_types[index] == "conv":
            _export(ExplicitConvDecoder(layer), ["hidden", "conv_state"],
                    ["output", "new_conv_state"], samples, out / "generic" / f"layer_{index:02d}",
                    ["conv_hidden", "conv_state"])
            complete_batch([out / "generic" / f"layer_{index:02d}"])
        else:
            base = out / "parts" / f"layer_{index:02d}"
            _export(Prepare(layer), ["hidden", "cos", "sin"], ["query", "key", "value"],
                    samples, base / "prepare", ["prep_hidden", "cos", "sin"])
            _export(Tile(layer.self_attn.scaling), ["query", "keys", "values", "mask"],
                    ["tile_output", "tile_maximum", "tile_total"], samples, base / "tile",
                    ["tile_query", "tile_keys", "tile_values", "tile_mask"])
            _export(Finish(layer), ["hidden", "attention_output"], ["output"],
                    samples, base / "finish", ["finish_hidden", "finish_attention"])
            complete_batch([base / "prepare", base / "tile", base / "finish"])
        del layer, weights
    head_weights = {}
    for shard in sorted(source.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key in {"model.embedding_norm.weight", "model.embed_tokens.weight"}:
                    head_weights[key] = sf.get_tensor(key).float()
    norm = Lfm2RMSNorm(2048, eps=config.norm_eps)
    norm.weight.data.copy_(head_weights["model.embedding_norm.weight"])
    head = Head(norm, head_weights["model.embed_tokens.weight"])
    _export(head, ["hidden"], ["logits"], calibration / "head",
            out / "generic" / "head", ["head_hidden"])
    complete_batch([out / "generic" / "head"])
    merge_folder = out / "parts" / "merge"
    merge_sample_root = calibration / "merge"
    _export(Merge(), ["old_output", "old_maximum", "old_total", "tile_output", "tile_maximum", "tile_total"],
            ["output", "maximum", "total"], merge_sample_root, merge_folder, float_only=True)
    complete_batch([merge_folder])
    inv = np.fromfile(calibration / "rope_inv_freq.fp32", dtype=np.float32)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    inv.tofile(out / "parts" / "rope_inv_freq.fp32")
    if len(graphs) != 48:
        raise RuntimeError(f"Export produced {len(graphs)} graphs; expected 48.")
    atomic_json(out / "manifest.json", {"revision": MODEL_REVISION, "context_length": 131072,
                                         "tile_size": 2048, "graphs": graphs})


def compile_graphs(cache: Path, profile: dict, pulsar2: str | None, jobs: int, log,
                   graph_paths: list[str] | None = None, finalize: bool = True):
    if jobs != 1:
        raise ValueError("Only --jobs 1 is supported; AX model compilation is sequential to cap memory.")
    onnx_root = cache / "onnx"
    compiled = cache / "compiled"
    command_file = cache / "docker-command.json"
    docker_binary = shutil.which("docker")
    docker_cmd = json.loads(command_file.read_text(encoding="utf-8")) if command_file.is_file() else (
        [docker_binary] if docker_binary else [])
    cli = shutil.which(pulsar2) if pulsar2 else None
    if not docker_cmd and not cli:
        raise RuntimeError("Pulsar2 compiler not found. Run ./lfm setup first.")
    graphs = (sorted(onnx_root.rglob("model.onnx")) if graph_paths is None else
              [onnx_root / relative / "model.onnx" for relative in graph_paths])
    for graph in graphs:
        relative = graph.parent.relative_to(onnx_root)
        target = compiled / relative
        model = target / "model.axmodel"
        # A separate completion marker distinguishes success from interrupted output.
        complete = target / "complete"
        if not complete.is_file() or not model.is_file() or not model.stat().st_size:
            if not graph.is_file():
                raise FileNotFoundError(f"Exported graph is missing: {graph}")
            _prune_compiler_outputs(target, keep_model=False)
            if docker_cmd and not cli:
                command = docker_cmd + ["run", "--rm", "-v", f"{cache.resolve()}:/data", PINNED_IMAGE,
                           "pulsar2", "build", "--config", f"/data/onnx/{relative.as_posix()}/config.json",
                           "--input", f"/data/onnx/{relative.as_posix()}/model.onnx",
                           "--output_dir", f"/data/compiled/{relative.as_posix()}",
                           "--output_name", "model.axmodel"]
            else:
                command = [cli, "build", "--config", str(graph.parent / "config.json"), "--input", str(graph),
                           "--output_dir", str(target), "--output_name", "model.axmodel"]
            try:
                log.command(command)
                if not model.is_file() or not model.stat().st_size:
                    raise RuntimeError(f"Pulsar2 produced no model for {relative}.")
                complete.touch()
            except BaseException:
                if target.exists():
                    shutil.rmtree(target)
                raise
        _prune_compiler_outputs(target, keep_model=True)
        graph.unlink(missing_ok=True)
        (graph.parent / "config.json").unlink(missing_ok=True)
    if finalize:
        models = sorted(compiled.rglob("model.axmodel"))
        if len(models) != 48 or any(not model.stat().st_size or not (model.parent / "complete").is_file() for model in models):
            raise RuntimeError(f"Pulsar2 compiled {len(models)} graphs; expected 48 nonempty models.")
        atomic_json(compiled / "manifest.json", {
            "image": PINNED_IMAGE,
            "graphs": [{"path": model.parent.relative_to(compiled).as_posix()} for model in models],
        })


def _prune_compiler_outputs(target: Path, *, keep_model: bool):
    """Discard Pulsar2 scratch after a successful compile."""
    target.mkdir(parents=True, exist_ok=True)
    for item in target.iterdir():
        if keep_model and item.name in {"model.axmodel", "complete"} and item.is_file() and not item.is_symlink():
            continue
        if item.is_symlink() or item.is_file():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            shutil.rmtree(item)


def package(assets: Path, output: Path, profile: dict, model_source: Path | None = None,
           profile_root: Path | None = None):
    """Copy only deployable files into the runtime package; leave compiler scratch behind."""
    if output.exists() and any(output.iterdir()) and not (output / ".lfm-managed").is_file():
        raise ValueError(f"Refusing to replace unmarked package directory: {output}")
    staged = output.with_name(output.name + ".staging")
    if staged.exists():
        if not (staged / ".lfm-managed").is_file():
            raise ValueError(f"Refusing to remove unmarked staging directory: {staged}")
        shutil.rmtree(staged)
    try:
        return _package_into(assets, output, profile, model_source, profile_root, staged)
    except BaseException:
        if staged.exists() and (staged / ".lfm-managed").is_file():
            shutil.rmtree(staged)
        raise


def _package_into(assets: Path, output: Path, profile: dict, model_source: Path | None,
                  profile_root: Path | None, staged: Path):
    cache_layout = (assets / "compiled").is_dir()
    compiled = assets / "compiled" if cache_layout else assets
    source = model_source or (assets / "source" if (assets / "source").is_dir() else
                              (assets / "tokenizer" if (assets / "tokenizer").is_dir() else assets))
    calibration = assets / "calibration"
    staged.mkdir(parents=True)
    (staged / ".lfm-managed").touch()
    generic = staged / "generic"
    parts = staged / "parts"
    tokenizer = staged / "tokenizer"
    generic.mkdir(); parts.mkdir(); tokenizer.mkdir()

    if cache_layout:
        for item in sorted((compiled / "generic").glob("*/model.axmodel")):
            shutil.copy2(item, generic / f"{item.parent.name}.axmodel")
        for item in sorted((compiled / "parts").rglob("model.axmodel")):
            relative = item.parent.relative_to(compiled / "parts")
            target = parts / relative / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
        embedding = calibration / "embedding.bf16"
        rope = assets / "onnx" / "parts" / "rope_inv_freq.fp32"
    else:
        for item in sorted((compiled / "generic").glob("*.axmodel")):
            shutil.copy2(item, generic / item.name)
        for item in sorted((compiled / "parts").rglob("*.axmodel")):
            relative = item.relative_to(compiled / "parts")
            target = parts / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
        embedding = assets / "generic" / "embedding.bf16"
        if not embedding.is_file():
            embedding = assets / "embedding.bf16"
        rope = assets / "parts" / "rope_inv_freq.fp32"
    head_candidates = [generic / "head.axmodel", generic / "head" / "model.axmodel"]
    head = next((item for item in head_candidates if item.is_file()), None)
    if head and head.name == "model.axmodel":
        head.rename(generic / "head.axmodel")
    if not embedding.is_file():
        raise FileNotFoundError("Missing FP32-produced tied embedding; expected embedding.bf16.")
    shutil.copy2(embedding, generic / "embedding.bf16")
    if rope.is_file():
        shutil.copy2(rope, parts / "rope_inv_freq.fp32")
    else:
        raise FileNotFoundError("Missing rotary inverse frequencies for the 131072-token model.")
    if len(list(generic.glob("*.axmodel"))) != 23:
        raise RuntimeError("Generic payload requires 22 Conv models and one head model.")
    if len(list(parts.rglob("*.axmodel"))) != 25:
        raise RuntimeError("Attention payload requires 24 layer parts and one merge model.")

    if not (source / "config.json").is_file():
        raise FileNotFoundError("Pinned model config is required to package EOS/tokenizer metadata.")
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    source_generation = {}
    generation_file = source / "generation_config.json"
    if generation_file.is_file():
        source_generation = json.loads(generation_file.read_text(encoding="utf-8"))
    for filename in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                     "special_tokens_map.json", "chat_template.jinja", "LICENSE"]:
        path = source / filename
        if path.is_file():
            shutil.copy2(path, tokenizer / filename)
    config = {"model_type": "lfm", "tokenizer_type": "LFM2", "tokenizer_dir": "tokenizer",
              "lfm": {"format_version": 1, "context_length": 131072, "tile_size": 2048,
                      "eos_token_id": profile.get("eos_token_id", source_config.get("eos_token_id", 124900)),
                      "generation": profile.get("generation", {
                          "max_new_tokens": 384,
                          "repetition_penalty": source_generation.get("repetition_penalty", 1.1)}),
                      "kv_cache": {"dtype": profile["kv"]["dtype"], "storage": profile["kv"]["storage"]},
                      "paths": {"generic": "generic", "parts": "parts"}}}
    atomic_json(staged / "config.json", config)
    for filename in ["README.md", "LICENSE"]:
        source_file = (profile_root / filename) if profile_root else (source / filename)
        if not source_file.is_file() and filename == "LICENSE":
            source_file = source / filename
        if not source_file.is_file():
            shutil.rmtree(staged)
            raise FileNotFoundError(f"Profile package requires {filename}; supply --source and a model profile.")
        shutil.copy2(source_file, staged / filename)
    validation_summary = (profile_root / "docs" / "validation-summary.md") if profile_root else None
    if validation_summary is not None and validation_summary.is_file():
        docs = staged / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(validation_summary, docs / "validation-summary.md")
    verify_package(staged, profile)
    if output.exists():
        marker = output / ".lfm-managed"
        if not marker.is_file():
            shutil.rmtree(staged)
            raise ValueError(f"Refusing to replace unmarked package directory: {output}")
        shutil.rmtree(output)
    staged.replace(output)


def verify_package(output: Path, profile: dict):
    physical = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    attention = {2, 5, 9, 13, 17, 21, 24, 27}
    conv = set(range(30)) - attention
    expected_graphs = {f"generic/layer_{i:02d}.axmodel" for i in conv}
    expected_graphs.add("generic/head.axmodel")
    expected_graphs.add("parts/merge/model.axmodel")
    expected_graphs.update(f"parts/layer_{i:02d}/{part}/model.axmodel"
                           for i in attention for part in ["prepare", "tile", "finish"])
    actual_graphs = {path for path in physical if path.endswith(".axmodel")}
    if actual_graphs != expected_graphs:
        missing, extra = expected_graphs - actual_graphs, actual_graphs - expected_graphs
        raise RuntimeError(f"AX model set mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    required = {"generic/embedding.bf16", "parts/rope_inv_freq.fp32", "tokenizer/config.json",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                "tokenizer/generation_config.json", "tokenizer/chat_template.jinja", "tokenizer/LICENSE",
                "README.md", "LICENSE", "config.json"}
    if not required.issubset(physical):
        raise RuntimeError(f"Required package files missing: {sorted(required - physical)}")
    if any((output / path).stat().st_size == 0 for path in actual_graphs | required):
        raise RuntimeError("Package contains empty required files.")
    if (output / "generic/embedding.bf16").stat().st_size != EMBEDDING_BYTES:
        raise RuntimeError("Tied BF16 embedding has an unexpected size.")
    if (output / "parts/rope_inv_freq.fp32").stat().st_size != 128:
        raise RuntimeError("Rotary inverse frequency table has an unexpected size.")
    config = json.loads((output / "config.json").read_text())
    lfm = config.get("lfm", {})
    expected_eos = profile.get("eos_token_id", 124900)
    expected_gen = profile.get("generation", {"max_new_tokens": 384, "repetition_penalty": 1.1})
    if (lfm.get("context_length") != 131072 or lfm.get("tile_size") != 2048 or
            lfm.get("kv_cache") != {"dtype": profile["kv"]["dtype"], "storage": profile["kv"]["storage"]} or
            lfm.get("eos_token_id") != expected_eos or lfm.get("generation") != expected_gen):
        raise RuntimeError("Runtime config does not match the selected profile.")
    print(f"Verified {len(actual_graphs)} AX models and required tokenizer/configuration files.")
    return 0
