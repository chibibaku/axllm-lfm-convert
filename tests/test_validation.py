"""Focused tests for the fail-closed native parity comparison helper."""
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:  # The lightweight Windows test host need not install torch.
    torch = None

from lfm_convert.validation import (
    NativeParityError,
    NativeParityGate,
    _cache_sequence_length,
    compare_tensors,
)


@unittest.skipIf(torch is None, "torch is installed by the Ubuntu conversion environment")
class NativeParityComparisonTests(unittest.TestCase):
    def test_within_tolerance_returns_compact_metrics(self):
        expected = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
        actual = expected + torch.tensor([[1e-6, -1e-6, 0.0]])

        result = compare_tensors(actual, expected, rtol=1e-4, atol=1e-5)

        self.assertTrue(result["passed"])
        self.assertEqual(result["shape"], [1, 3])
        self.assertLessEqual(result["max_abs"], 1e-5)
        self.assertGreater(result["cosine"], 0.999999)

    def test_material_drift_fails_closed(self):
        expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
        actual = torch.tensor([1.0, 2.01], dtype=torch.float32)

        with self.assertRaises(NativeParityError):
            compare_tensors(actual, expected, rtol=1e-4, atol=1e-5)

    def test_shape_mismatch_fails_closed(self):
        with self.assertRaisesRegex(NativeParityError, "Shape mismatch"):
            compare_tensors(torch.zeros(2), torch.zeros(1))

    def test_nan_fails_closed(self):
        with self.assertRaises(NativeParityError):
            compare_tensors(torch.tensor([float("nan")]), torch.tensor([0.0]))

    def test_empty_scalar_cache_sentinel_has_zero_sequence_length(self):
        self.assertEqual(_cache_sequence_length(torch.empty((0,))), 0)
        self.assertEqual(_cache_sequence_length(torch.empty((1, 8, 0, 64))), 0)
        self.assertEqual(_cache_sequence_length(torch.empty((1, 8, 4, 64))), 4)

    def test_hook_registration_uses_kwargs_and_removes_all_handles(self):
        layers = [torch.nn.Identity(), torch.nn.Identity()]
        gate = object.__new__(NativeParityGate)
        gate.model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        gate.torch = torch
        gate.handles = []

        gate.install()

        self.assertEqual(len(gate.handles), 4)
        for layer in layers:
            self.assertEqual(list(layer._forward_pre_hooks_with_kwargs.values()), [True])
            self.assertEqual(list(layer._forward_hooks_with_kwargs.values()), [True])
        gate.remove()
        self.assertFalse(gate.handles)
        for layer in layers:
            self.assertFalse(layer._forward_pre_hooks)
            self.assertFalse(layer._forward_hooks)
            self.assertFalse(layer._forward_pre_hooks_with_kwargs)
            self.assertFalse(layer._forward_hooks_with_kwargs)

    def test_complete_requires_unique_phase_layer_tensor_checks(self):
        layers = [SimpleNamespace(is_attention_layer=False),
                  SimpleNamespace(is_attention_layer=True)]
        gate = object.__new__(NativeParityGate)
        gate.model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        gate.phase = None
        gate.pending = {}
        gate.results = [
            {"phase": phase, "layer": index, "tensor": name}
            for phase in ("initial4", "decode1")
            for index, layer in enumerate(layers)
            for name in (("output", "key_cache", "value_cache") if layer.is_attention_layer
                         else ("output", "conv_state"))
        ]

        gate.assert_complete()
        gate.results.append(gate.results[0].copy())
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            gate.assert_complete()


if __name__ == "__main__":
    unittest.main()
