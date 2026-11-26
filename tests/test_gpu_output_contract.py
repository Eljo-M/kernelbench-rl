"""CPU tensor checks for the GPU worker's untimed validation boundaries."""
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from sandbox_runner.gpu_worker import _benchmark_checked_candidate, _validate_candidate_output


@unittest.skipIf(torch is None, "optional Torch dependency is unavailable")
class GpuOutputContractTests(unittest.TestCase):
    def test_dtype_shape_finiteness_and_values_use_real_cpu_tensors(self):
        reference = torch.tensor([1.0, 2.0], dtype=torch.float16)
        self.assertEqual(_validate_candidate_output(reference.clone(), reference, torch, 1e-4, 1e-4), ("", 0.0))
        for output, message in (
            (reference.float(), "dtype"),
            (reference.reshape(1, 2), "shape"),
            (torch.tensor([1.0, float("nan")], dtype=torch.float16), "non-finite"),
            (reference + 1, "tolerance"),
            ([1.0, 2.0], "not a tensor"),
        ):
            error, _ = _validate_candidate_output(output, reference, torch, 1e-4, 1e-4)
            self.assertIn(message, error)

    def test_exact_device_is_checked_before_value_comparison(self):
        # Device-only stand-ins let a CPU-only CI job exercise CUDA index checks.
        reference = SimpleNamespace(device=torch.device("cuda:0"))
        output = SimpleNamespace(device=torch.device("cuda:1"))
        with mock.patch.object(torch, "is_tensor", return_value=True):
            error, err = _validate_candidate_output(output, reference, torch, 1e-4, 1e-4)
        self.assertIn("device cuda:1 does not match cuda:0", error)
        self.assertEqual(err, 0.0)

    @staticmethod
    def repeated_cpu_timer(torch_mod, fn, inputs, warmup, iters):
        for _ in range(warmup + iters):
            fn(*inputs)
        return 0.25

    def test_valid_benchmark_preserves_measured_time_and_inputs(self):
        x = torch.tensor([1.0, 2.0])
        result = _benchmark_checked_candidate(torch, lambda a: a + 1, (x,), x + 1,
            1e-4, 1e-4, 2, 3, self.repeated_cpu_timer)
        self.assertEqual(result, (0.25, "PASS", "", 0.0))
        self.assertTrue(torch.equal(x, torch.tensor([1.0, 2.0])))

    def test_delayed_mutation_cannot_receive_timing_or_reward(self):
        x = torch.tensor([1.0, 2.0])
        calls = 0

        def candidate(a):
            nonlocal calls
            calls += 1
            output = a + 1
            if calls > 2:
                a.zero_()
            return output

        result = _benchmark_checked_candidate(torch, candidate, (x,), x + 1,
            1e-4, 1e-4, 2, 3, self.repeated_cpu_timer)
        self.assertEqual(result[0:2], (0.0, "CHEAT"))
        self.assertIn("mutated input during timing", result[2])

    def test_last_captured_output_is_checked_after_replays(self):
        x = torch.tensor([1.0, 2.0])

        def replaying_timer(torch_mod, fn, inputs, warmup, iters):
            captured = fn(*inputs)
            captured.fill_(99)  # Stand in for an incorrect final CUDA graph replay.
            return 0.25

        result = _benchmark_checked_candidate(torch, lambda a: a + 1, (x,), x + 1,
            1e-4, 1e-4, 2, 3, replaying_timer)
        self.assertEqual(result[0:2], (0.0, "NUMERICAL_MISMATCH"))
        self.assertIn("after timing", result[2])


if __name__ == "__main__":
    unittest.main()
