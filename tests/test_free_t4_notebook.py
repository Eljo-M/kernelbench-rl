"""Static validation tests for notebooks/free_t4.ipynb."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "free_t4.ipynb"


class TestFreeT4Notebook(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(NOTEBOOK_PATH.is_file(), f"Missing notebook: {NOTEBOOK_PATH}")
        self.nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.code_cells = [c for c in self.nb["cells"] if c.get("cell_type") == "code"]
        self.md_cells = [c for c in self.nb["cells"] if c.get("cell_type") == "markdown"]
        self.all_code = "\n\n".join("".join(c.get("source", [])) for c in self.code_cells)
        self.all_text = "\n\n".join("".join(c.get("source", [])) for c in self.nb["cells"])

    def test_nbformat_structure_and_static_python_compile(self) -> None:
        self.assertEqual(self.nb.get("nbformat"), 4)
        self.assertGreaterEqual(len(self.code_cells), 6)
        self.assertGreaterEqual(len(self.md_cells), 5)

        seen_ids: set[str] = set()
        for idx, cell in enumerate(self.nb["cells"]):
            cid = cell.get("id", "")
            self.assertTrue(cid, f"Cell {idx} missing id")
            self.assertNotIn(cid, seen_ids, f"Duplicate cell id: {cid}")
            seen_ids.add(cid)

        for idx, cell in enumerate(self.code_cells):
            src = "".join(cell.get("source", []))
            # Every code cell must compile as pure Python (no !shell or %magic commands)
            for line_no, line in enumerate(src.splitlines(), start=1):
                stripped = line.lstrip()
                self.assertFalse(
                    stripped.startswith("!") or stripped.startswith("%"),
                    f"Code cell {idx} line {line_no} uses shell/magic syntax: {line!r}",
                )
            compile(src, f"free_t4.ipynb:cell_{idx}", "exec")
            tree = ast.parse(src)
            # Forbid shell=True in subprocess calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            self.fail(f"Code cell {idx} uses shell=True in subprocess call")

    def test_safe_defaults_pinned_versions_and_zero_paid_or_drive_dependencies(self) -> None:
        # Safe smoke default: must NOT automatically download 14 GB or run full training
        self.assertIn("RUN_FULL_TRAINING_AND_EVAL: bool = False", self.all_code)
        self.assertIn("TRIGGER_COLAB_DOWNLOAD: bool = False", self.all_code)

        # Pinned public repo, commit, model, and revision
        self.assertIn("https://github.com/Eljo-M/kernelbench-rl.git", self.all_code)
        self.assertIn("e72c42e794643ed4c198937e86427c3640b977c5", self.all_code)
        self.assertIn("Qwen/Qwen2.5-Coder-7B-Instruct", self.all_code)
        self.assertIn("c03e6d358207e414f1eca0bb1891e29f1db0e242", self.all_code)
        self.assertIn("token=False", self.all_code)

        # Isolated venv and pinned T4-compatible stack
        self.assertIn("/content/kernelbench-venv", self.all_code)
        self.assertIn("torch==2.6.0+cu124", self.all_code)
        self.assertIn("https://download.pytorch.org/whl/cu124", self.all_code)
        self.assertIn("triton==3.2.0", self.all_code)
        self.assertIn("transformers==4.57.6", self.all_code)
        self.assertIn("peft==0.18.0", self.all_code)
        self.assertIn("bitsandbytes>=0.48,<0.50", self.all_code)

        # Config preset, bounded budget, output dir, overwrite refusal, optimizer check
        self.assertIn("preset_t4_7b_qlora", self.all_code)
        self.assertIn("MAX_NEW_TOKENS: int = 1024", self.all_code)
        self.assertIn("MAX_STEPS: int = 8", self.all_code)
        self.assertIn("GROUP_SIZE: int = 2", self.all_code)
        self.assertIn("CHECKPOINT_EVERY: int = 2", self.all_code)
        self.assertIn("WARMUP_ITERS: int = 5", self.all_code)
        self.assertIn("BENCH_ITERS: int = 30", self.all_code)
        self.assertIn("WORKER_TIMEOUT_SECS: float = 120.0", self.all_code)
        self.assertIn("/content/t4-results", self.all_code)
        self.assertIn("Refusing to overwrite existing non-empty run directory", self.all_code)
        self.assertIn("optimizer_step_count", self.all_code)
        self.assertIn("heldout_environment.json", self.all_code)
        self.assertIn("rl_trainer.evaluate_heldout", self.all_code)
        self.assertIn("validation_seed_regrade", self.all_code)
        self.assertIn("rmsnorm_2048", self.all_code)

        # No Google Drive mount or API keys
        self.assertNotIn("drive.mount", self.all_text)
        self.assertNotIn("HF_TOKEN", self.all_text)
        self.assertNotIn("OPENAI_API_KEY", self.all_text)

    def test_honest_disclaimers_and_hygiene(self) -> None:
        self.assertIn("5,046,272", self.all_text)
        self.assertIn("not full-parameter 7b", self.all_text.lower())
        self.assertIn("quotas are not guaranteed", self.all_text.lower())
        self.assertIn("no quality or speedup claims", self.all_text.lower())



    def test_embedded_venv_scripts_compile_cleanly(self) -> None:
        script_names = {"PREFLIGHT_SCRIPT", "DOWNLOAD_SCRIPT", "CFG_SCRIPT", "VERIFY_CKPT_SCRIPT"}
        found: set[str] = set()
        for idx, cell in enumerate(self.code_cells):
            src = "".join(cell.get("source", []))
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name) and tgt.id in script_names:
                            self.assertIsInstance(node.value, ast.Constant)
                            self.assertIsInstance(node.value.value, str)
                            compile(node.value.value, f"free_t4.ipynb:{tgt.id}", "exec")
                            found.add(tgt.id)
        self.assertEqual(found, script_names)


if __name__ == "__main__":
    unittest.main()
