"""Optional CPU numerical checks for native PyTorch references.

These tests do not validate generated Triton code or establish GPU throughput.
"""

import unittest

import numpy as np

from kernels.op_registry import OP_REGISTRY
from kernels.torch_references import TORCH_REFERENCES, convert_numpy_inputs_to_torch
from kernels.triton_baselines import TRITON_BASELINES

try:
    import torch
except ImportError:
    torch = None


def _additional_stress_inputs(name, rng):
    """Odd dimensions for the three references without registry stress cases."""
    if name == "fused_add_rmsnorm_2048":
        return [(rng.standard_normal((2, 3, 771), dtype=np.float32),
                 rng.standard_normal((2, 3, 771), dtype=np.float32),
                 rng.standard_normal((771,), dtype=np.float32))]
    if name == "int8_weight_only_gemm_512":
        return [(rng.standard_normal((7, 65), dtype=np.float32),
                 rng.integers(-127, 128, size=(65, 33), dtype=np.int8),
                 rng.uniform(.005, .05, size=(33,)).astype(np.float32))]
    if name == "scaled_dot_product_attn_128":
        return [tuple(rng.standard_normal((2, 2, 37, 48), dtype=np.float32) for _ in range(3))]
    return []


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class TestTorchReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def test_references_cover_exactly_the_eleven_baseline_entries(self):
        self.assertEqual(len(TORCH_REFERENCES), 11)
        self.assertEqual(set(TORCH_REFERENCES), set(TRITON_BASELINES))

    def _check_all_references(self, dtype_name):
        target_dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
        for name in sorted(TRITON_BASELINES):
            spec = OP_REGISTRY[name]
            rng = np.random.default_rng(42)
            cases = [spec.make_inputs(rng)]
            stress = spec.make_stress_inputs(np.random.default_rng(43)) if spec.make_stress_inputs else []
            stress.extend(_additional_stress_inputs(name, np.random.default_rng(44)))
            self.assertTrue(stress, msg=f"{name}: expected at least one stress case")
            cases.extend(stress)
            for case_index, numpy_inputs in enumerate(cases):
                with self.subTest(op=name, dtype=dtype_name, case=case_index):
                    tensors = convert_numpy_inputs_to_torch(numpy_inputs, device="cpu", float_dtype=dtype_name)
                    for original, tensor in zip(numpy_inputs, tensors):
                        self.assertEqual(tuple(tensor.shape), original.shape)
                        self.assertEqual(tensor.device.type, "cpu")
                        self.assertTrue(tensor.is_contiguous())
                        if np.issubdtype(original.dtype, np.floating):
                            self.assertEqual(tensor.dtype, target_dtype)
                        else:
                            # Quantization weights and indices are data, not
                            # floating activations, even for the bf16 case.
                            self.assertEqual(tensor.numpy().dtype, original.dtype)
                            np.testing.assert_array_equal(tensor.numpy(), original)

                    # Compare the same effective inputs: bf16 conversion changes
                    # values, so NumPy must receive those rounded values in fp32.
                    effective_numpy_inputs = tuple(
                        tensor.float().numpy() if tensor.is_floating_point() else tensor.numpy()
                        for tensor in tensors
                    )
                    expected = spec.reference_fn(*effective_numpy_inputs)
                    with torch.no_grad():
                        actual = TORCH_REFERENCES[name](*tensors)
                    self.assertEqual(actual.dtype, target_dtype)
                    self.assertEqual(tuple(actual.shape), expected.shape)
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertTrue(bool(torch.isfinite(actual).all()), msg=f"{name}: non-finite output")
                    rtol, atol = spec.rtol, spec.atol
                    if dtype_name == "bf16":
                        rtol, atol = max(rtol, 2e-2), max(atol, 2e-2)
                    np.testing.assert_allclose(
                        actual.float().numpy(), expected, rtol=rtol, atol=atol,
                        err_msg=f"{name}, {dtype_name}, case {case_index}",
                    )

    def test_fp32_canonical_and_stress_match_numpy(self):
        self._check_all_references("fp32")

    def test_bf16_canonical_and_stress_match_quantized_input_numpy(self):
        self._check_all_references("bf16")

    def test_quantized_reference_rejects_floating_weights(self):
        reference = TORCH_REFERENCES["int8_weight_only_gemm_512"]
        with self.assertRaisesRegex(TypeError, "integer tensor"):
            reference(torch.ones((2, 3)), torch.ones((3, 4)), torch.ones((4,)))

    def test_input_conversion_rejects_unknown_dtype(self):
        with self.assertRaisesRegex(ValueError, "Unsupported float_dtype"):
            convert_numpy_inputs_to_torch((np.ones((2,), dtype=np.float32),), device="cpu", float_dtype="fp16")


if __name__ == "__main__":
    unittest.main()
