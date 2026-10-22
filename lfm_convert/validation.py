"""Native HF layer parity checks for the pinned LFM export wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _PendingCheck:
    phase: str
    layer_index: int
    layer_type: str
    expected_output: Any
    cache: Any
    expected_state: Any = None
    expected_key: Any = None
    expected_value: Any = None
    key_length_before: int = 0


class NativeParityError(RuntimeError):
    """Raised when a wrapper tensor diverges from the native teacher result."""


def _cache_sequence_length(cache_tensor) -> int:
    """Return zero for HF's uninitialized scalar cache sentinel, else its sequence length."""
    if cache_tensor.numel() == 0:
        return 0
    if cache_tensor.ndim < 2:
        raise RuntimeError(
            f"Expected an initialized attention cache with a sequence axis, got "
            f"shape {tuple(cache_tensor.shape)}.")
    return int(cache_tensor.shape[-2])


def compare_tensors(actual, expected, *, rtol: float = 1e-4, atol: float = 1e-5) -> dict[str, Any]:
    """Assert FP32 tensor parity and return compact numerical diagnostics."""
    import torch

    if actual.shape != expected.shape:
        raise NativeParityError(
            f"Shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}.")
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    except AssertionError as exc:
        raise NativeParityError(str(exc)) from exc
    actual_fp32 = actual.detach().float()
    expected_fp32 = expected.detach().float()
    delta = actual_fp32 - expected_fp32
    norm = torch.linalg.vector_norm(expected_fp32)
    actual_norm = torch.linalg.vector_norm(actual_fp32)
    cosine = (torch.dot(actual_fp32.reshape(-1), expected_fp32.reshape(-1)) /
              (actual_norm * norm)) if actual_norm.item() and norm.item() else torch.tensor(1.0)
    return {
        "shape": list(actual.shape),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(delta) / norm.clamp_min(1e-30)).item()),
        "cosine": float(cosine.item()),
        "rtol": rtol,
        "atol": atol,
        "passed": True,
    }


class NativeParityGate:
    """Compare the exported wrappers with live native HF layers during calibration.

    The gate is armed around the initial four-token teacher call and the next
    single-token call. Wrappers share the teacher layer weights and run before
    the native forward pass; post-hooks compare outputs and the mutated HF cache.
    """

    VERSION = 1
    PHASES = {"initial4": 4, "decode1": 1}

    def __init__(self, model, device, *, rtol: float = 1e-4, atol: float = 1e-5):
        import torch
        from .models import ExplicitConvDecoder, Finish, Prepare, Tile

        self.model = model
        self.torch = torch
        self.device = torch.device(device)
        self.rtol = rtol
        self.atol = atol
        self.phase: str | None = None
        self.pending: dict[int, _PendingCheck] = {}
        self.handles = []
        self.results: list[dict[str, Any]] = []
        self.wrappers = {}
        for index, layer in enumerate(model.model.layers):
            if layer.is_attention_layer:
                self.wrappers[index] = (
                    Prepare(layer), Tile(layer.self_attn.scaling), Finish(layer))
            else:
                self.wrappers[index] = ExplicitConvDecoder(layer)

    def install(self) -> None:
        """Register pre/post hooks on the already-loaded HF teacher layers."""
        if self.handles:
            raise RuntimeError("Native parity hooks are already installed.")
        for index, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(
                self._before(index), with_kwargs=True))
            self.handles.append(layer.register_forward_hook(
                self._after(index), with_kwargs=True))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def begin(self, phase: str) -> None:
        if phase not in self.PHASES:
            raise ValueError(f"Unsupported native parity phase: {phase}")
        if self.phase is not None or self.pending:
            raise RuntimeError("Cannot arm parity while another phase is active.")
        self.phase = phase

    def end(self, phase: str) -> None:
        if self.phase != phase:
            raise RuntimeError(f"Expected parity phase {self.phase!r}, got {phase!r}.")
        if self.pending:
            raise RuntimeError(f"HF layer hooks did not complete: {sorted(self.pending)}")
        self.phase = None

    def assert_complete(self) -> None:
        if self.phase is not None or self.pending:
            raise RuntimeError("Native parity ended with an active or incomplete phase.")
        expected = set()
        for phase in self.PHASES:
            for index, layer in enumerate(self.model.model.layers):
                names = ("output", "key_cache", "value_cache") if layer.is_attention_layer else (
                    "output", "conv_state")
                expected.update((phase, index, name) for name in names)
        observed = [(item["phase"], item["layer"], item["tensor"])
                    for item in self.results]
        observed_set = set(observed)
        duplicates = sorted(key for key in observed_set if observed.count(key) != 1)
        missing = sorted(expected - observed_set)
        unexpected = sorted(observed_set - expected)
        if duplicates or missing or unexpected or len(observed) != len(expected):
            raise RuntimeError(
                "Native parity tensor checks are incomplete or duplicated: "
                f"expected={len(expected)} observed={len(observed)} "
                f"missing={missing[:5]} duplicates={duplicates[:5]} "
                f"unexpected={unexpected[:5]}.")

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "status": "passed",
            "reference": "live Transformers HF native layer output/cache",
            "cases": ["initial4", "decode1"],
            "rtol": self.rtol,
            "atol": self.atol,
            "device": str(self.device),
            "layer_count": len(self.model.model.layers),
            "tensor_checks": len(self.results),
            "results": self.results,
        }

    def _before(self, index: int):
        torch = self.torch
        layer = self.model.model.layers[index]

        def callback(_module, args, kwargs):
            if self.phase is None:
                return
            if index in self.pending:
                raise RuntimeError(f"Layer {index} was called twice in parity phase {self.phase}.")
            hidden = args[0] if args else kwargs["hidden_states"]
            expected_length = self.PHASES[self.phase]
            if hidden.shape[1] != expected_length:
                raise RuntimeError(
                    f"Layer {index} parity phase {self.phase} expected sequence length "
                    f"{expected_length}, got {hidden.shape[1]}.")
            cache = kwargs.get("past_key_values")
            if cache is None:
                raise RuntimeError(f"Layer {index} native parity requires the HF cache.")

            if not layer.is_attention_layer:
                wrapper = self.wrappers[index]
                state = cache.conv_cache[index].detach().clone()
                expected_output, expected_state = wrapper(hidden, state)
                self.pending[index] = _PendingCheck(
                    self.phase, index, "conv", expected_output, cache,
                    expected_state=expected_state)
                return

            prepare, tile, finish = self.wrappers[index]
            cos, sin = kwargs["position_embeddings"]
            query, key, value = prepare(hidden, cos, sin)
            old_key = cache.key_cache[index]
            old_value = cache.value_cache[index]
            key_length_before = _cache_sequence_length(old_key)
            value_length_before = _cache_sequence_length(old_value)
            if key_length_before != value_length_before:
                raise RuntimeError(
                    f"Native parity found unequal pre-call attention cache lengths at "
                    f"layer {index}: key={key_length_before}, value={value_length_before}.")
            if self.phase == "initial4":
                if old_key.numel() or old_value.numel():
                    raise RuntimeError(f"Attention layer {index} initial cache is not empty.")
                past_key = key[:, :, :0, :]
                past_value = value[:, :, :0, :]
            else:
                past_key, past_value = old_key.detach().clone(), old_value.detach().clone()
            outputs = []
            for local in range(expected_length):
                keys = torch.cat((past_key, key[:, :, :local + 1, :]), dim=2)
                values = torch.cat((past_value, value[:, :, :local + 1, :]), dim=2)
                page_count = keys.shape[2]
                if page_count > 2048:
                    raise RuntimeError("Native parity page exceeds the pinned 2048-token tile.")
                page_key = torch.zeros((1, 8, 2048, 64), dtype=key.dtype, device=key.device)
                page_value = torch.zeros_like(page_key)
                page_key[:, :, :page_count, :] = keys
                page_value[:, :, :page_count, :] = values
                mask = torch.full((1, 1, 1, 2048), -10000.0,
                                  dtype=key.dtype, device=key.device)
                mask[:, :, :, :page_count] = 0.0
                tile_output, _, _ = tile(
                    query[:, :, local:local + 1, :], page_key, page_value, mask)
                outputs.append(finish(hidden[:, local:local + 1, :], tile_output))
            self.pending[index] = _PendingCheck(
                self.phase, index, "attention", torch.cat(outputs, dim=1), cache,
                expected_key=key, expected_value=value,
                key_length_before=key_length_before)

        return callback

    def _after(self, index: int):
        def callback(_module, _args, _kwargs, output):
            check = self.pending.pop(index, None)
            if check is None:
                return
            self._record(check, "output", output, check.expected_output)
            if check.layer_type == "conv":
                self._record(check, "conv_state", check.cache.conv_cache[index],
                             check.expected_state)
            else:
                length = check.expected_key.shape[2]
                expected_length = check.key_length_before + length
                key_cache = check.cache.key_cache[index]
                value_cache = check.cache.value_cache[index]
                if key_cache.shape[-2] != expected_length:
                    raise RuntimeError(
                        f"Native parity cache length mismatch: phase={check.phase} "
                        f"layer={index} key_cache expected {expected_length}, "
                        f"got {key_cache.shape[-2]}.")
                if value_cache.shape[-2] != expected_length:
                    raise RuntimeError(
                        f"Native parity cache length mismatch: phase={check.phase} "
                        f"layer={index} value_cache expected {expected_length}, "
                        f"got {value_cache.shape[-2]}.")
                self._record(check, "key_cache",
                             key_cache[:, :, -length:, :], check.expected_key)
                self._record(check, "value_cache",
                             value_cache[:, :, -length:, :], check.expected_value)

        return callback

    def _record(self, check: _PendingCheck, tensor_name: str, actual, expected) -> None:
        try:
            result = compare_tensors(actual, expected, rtol=self.rtol, atol=self.atol)
        except NativeParityError as exc:
            raise RuntimeError(
                f"Native parity failed: phase={check.phase} layer={check.layer_index} "
                f"type={check.layer_type} tensor={tensor_name}: {exc}") from exc
        self.results.append({
            "phase": check.phase,
            "layer": check.layer_index,
            "layer_type": check.layer_type,
            "tensor": tensor_name,
            **result,
        })
