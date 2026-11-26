import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

from rl_trainer.frozen_base_integrity import FrozenBaseVerification, hash_frozen_base


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch unavailable")
class TestFrozenBaseIntegrity(unittest.TestCase):
    def fixture(self):
        import torch
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.uint8).reshape(3, 4), requires_grad=False)
        model.adapter = torch.nn.Parameter(torch.ones(2, 2))
        model.register_buffer("runtime_buffer", torch.tensor([1.0, 2.0]), persistent=False)
        state = types.SimpleNamespace(
            absmax=torch.tensor([0.25, 0.5]), code=torch.tensor([-1.0, 0.0, 1.0]),
            blocksize=64, shape=torch.Size([3, 8]), dtype=torch.float16, quant_type="nf4",
            nested=True, offset=torch.tensor(0.125),
            state2=types.SimpleNamespace(absmax=torch.tensor([0.1]), code=torch.tensor([0.0, 1.0]),
                                         dtype=torch.float32, blocksize=256),
        )
        model.weight.quant_state = state
        model.quant_state = state  # The same NF4 state must not be hashed twice by root traversal.
        return model

    def test_hash_excludes_trainable_but_covers_buffers_and_nested_quantization(self):
        import torch
        model = self.fixture()
        before = hash_frozen_base(model, chunk_bytes=3)
        self.assertEqual(before["frozen_parameter_tensors"], 1)
        self.assertEqual(before["frozen_parameter_bytes"], 12)
        self.assertEqual(before["buffer_tensors"], 1)
        self.assertEqual(before["quantization_state_roots"], 1)
        self.assertEqual(before["quantization_state_tensors"], 5)
        with torch.no_grad():
            model.adapter.add_(10)
        self.assertEqual(hash_frozen_base(model, chunk_bytes=100), before)
        for change in (lambda: model.weight.add_(1),
                       lambda: model.runtime_buffer.add_(1),
                       lambda: model.quant_state.state2.absmax.add_(1)):
            model = self.fixture()
            baseline = hash_frozen_base(model)
            with torch.no_grad():
                change()
            self.assertNotEqual(hash_frozen_base(model)["sha256"], baseline["sha256"])
        model = self.fixture()
        baseline = hash_frozen_base(model)
        model.quant_state.blocksize = 128
        self.assertNotEqual(hash_frozen_base(model)["sha256"], baseline["sha256"])

    def test_shape_dtype_and_bfloat16_stored_bytes_are_hashed(self):
        import torch
        first = torch.nn.Module()
        first.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16), requires_grad=False)
        base = hash_frozen_base(first, chunk_bytes=2)
        self.assertEqual(base["frozen_parameter_bytes"], 4)
        first.weight = torch.nn.Parameter(first.weight.reshape(1, 2), requires_grad=False)
        self.assertNotEqual(hash_frozen_base(first)["sha256"], base["sha256"])
        first.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float16), requires_grad=False)
        self.assertNotEqual(hash_frozen_base(first)["sha256"], base["sha256"])

    def test_noncontiguous_payload_is_streamed_in_logical_order(self):
        import torch
        model = torch.nn.Module()
        tensor = torch.arange(120, dtype=torch.int32).reshape(10, 12).t()
        model.weight = torch.nn.Parameter(tensor, requires_grad=False)
        expected = hash_frozen_base(model, chunk_bytes=13)
        model.weight = torch.nn.Parameter(tensor.contiguous(), requires_grad=False)
        self.assertEqual(hash_frozen_base(model, chunk_bytes=7), expected)

    def test_meta_and_unavailable_frozen_state_fail_closed(self):
        import torch
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "at least one frozen"):
            hash_frozen_base(model)
        model.weight = torch.nn.Parameter(torch.empty((2, 2), device="meta"), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "meta tensor"):
            hash_frozen_base(model)

    def test_completed_proof_and_resume_preserve_original_evidence(self):
        model = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            proof = Path(directory) / "frozen_base_verified.json"
            identity = {"model_revision": "pinned"}
            session = FrozenBaseVerification(model, proof, config_hash="config", model_identity=identity)
            original_initial = session.initial_path.read_bytes()
            session.finish(training_completed=True)
            original_proof = proof.read_bytes()
            resumed = FrozenBaseVerification(model, proof, config_hash="config", model_identity=identity, is_resuming=True)
            payload = resumed.finish(training_completed=True)
            self.assertTrue(payload["verified"] and payload["original_proof_preserved"])
            self.assertEqual(proof.read_bytes(), original_proof)
            self.assertEqual(session.initial_path.read_bytes(), original_initial)
            self.assertEqual(len(list(Path(directory).glob("frozen_base_verified.resume-*.json"))), 1)
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                FrozenBaseVerification(model, proof, config_hash="config", model_identity=identity)

    def test_resume_rejects_changed_base_and_mismatched_identity(self):
        import torch
        model = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            proof = Path(directory) / "frozen_base_verified.json"
            session = FrozenBaseVerification(model, proof, config_hash="config", model_identity={"model": "a"})
            session.finish(training_completed=True)
            with self.assertRaisesRegex(ValueError, "different model/configuration"):
                FrozenBaseVerification(model, proof, config_hash="other", model_identity={"model": "a"}, is_resuming=True)
            with torch.no_grad():
                model.runtime_buffer.add_(1)
            with self.assertRaisesRegex(ValueError, "differs from the original"):
                FrozenBaseVerification(model, proof, config_hash="config", model_identity={"model": "a"}, is_resuming=True)

    def test_failed_integrity_is_written_and_then_raised(self):
        import torch
        model = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            proof = Path(directory) / "frozen_base_verified.json"
            session = FrozenBaseVerification(model, proof, config_hash="config", model_identity={})
            with torch.no_grad():
                model.weight.add_(1)
            with self.assertRaisesRegex(RuntimeError, "changed during training"):
                session.finish(training_completed=False)
            saved = json.loads(proof.read_text())
            self.assertFalse(saved["verified"])
            self.assertFalse(saved["training_completed"])
            self.assertNotEqual(saved["before"]["sha256"], saved["after"]["sha256"])


if __name__ == "__main__":
    unittest.main()
