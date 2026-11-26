import json
from pathlib import Path
import unittest

import jsonschema

from sandbox_runner.verifier import verify_candidate_source


class EvaluationContractTests(unittest.TestCase):
    def test_verifier_emits_schema_valid_strict_json_for_all_failure_paths(self):
        schema = json.loads((Path(__file__).parents[1] / "contracts/kernel_eval.schema.json").read_text())
        candidates = {
            "COMPILE_ERROR": "def kernel_forward(: pass",
            "CHEAT": "import os",
            "RUNTIME_ERROR": "def kernel_forward(x,w): return 1/0",
            "NUMERICAL_MISMATCH": "def kernel_forward(x,w): return x*0",
            "TIMEOUT": "def kernel_forward(x,w):\n    while True: pass",
            "PASS": "import numpy as np\ndef kernel_forward(x,w): return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*w",
        }
        for status, source in candidates.items():
            with self.subTest(status=status):
                row = verify_candidate_source("rmsnorm_2048", source, timeout_secs=1 if status == "TIMEOUT" else 30)
                self.assertEqual(row.status, status, row.error_message)
                parsed = json.loads(json.dumps(row.to_dict(), allow_nan=False))
                jsonschema.validate(parsed, schema)
                self.assertEqual(parsed["backend"], "numpy_dev")
                self.assertEqual(parsed["compile_ms"], 0)
