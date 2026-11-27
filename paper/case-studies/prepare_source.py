"""Extract two existing PASS records and author a separate selected-case manuscript.

No model execution, grading, original-artifact writes, or publication actions.
"""
from __future__ import annotations
import copy
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTED = ("fused_bias_gelu_4096", "matmul_bias_silu_512")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def find_repo():
    for parent in (ROOT, *ROOT.parents):
        for candidate in (parent, parent / "outputs/kernelbench-rl"):
            if (candidate / "experiments/retry_v3/laptop_7b/heldout.jsonl").is_file():
                return candidate
    raise FileNotFoundError("Supply --repo with the saved KernelBench-RL source repository")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve() if args.repo else find_repo()
    ROOT.mkdir(parents=True, exist_ok=True)
    source = repo / "experiments/retry_v3/laptop_7b/heldout.jsonl"
    source_bytes = source.read_bytes()
    saved_lines = []
    cases = []
    for number, line in enumerate(source_bytes.splitlines(keepends=True), 1):
        row = json.loads(line)
        if row.get("method") != "trained_checkpoint" or row.get("eval_phase") != "heldout_test" or row.get("op_name") not in SELECTED:
            continue
        assert row["status"] == "PASS" and row["compiled"] is True and row["correct"] is True
        assert row["cheat_detected"] is False and row["fixture"] is False and row["budget_k"] == 1
        assert sha(row["candidate_source"].encode("utf-8")) == row["candidate_sha256"]
        saved_lines.append(line)
        case = copy.deepcopy(row)
        case.update(source_record_path="experiments/retry_v3/laptop_7b/heldout.jsonl",
                    source_record_line=number, source_record_file_sha256=sha(source_bytes),
                    source_record_line_sha256=sha(line),
                    canonical_record_sha256=sha(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")))
        cases.append(case)
    assert [row["op_name"] for row in cases] == list(SELECTED)
    (ROOT / "selected-programs.jsonl").write_bytes(b"".join(saved_lines))
    program_dir = ROOT / "programs"
    program_dir.mkdir(exist_ok=True)
    for case in cases:
        (program_dir / (case["op_name"] + ".py.txt")).write_bytes(case["candidate_source"].encode("utf-8"))
    config = json.loads((source.parent / "config.json").read_text())
    env = json.loads((source.parent / "environment.json").read_text())
    evidence = dict(schema_version="selected_verified_case_studies_v1",
        selection_scope="Two selected positive examples, chosen after verification; not an aggregate accuracy estimate.",
        featured_case_count=2, verified_featured_count=2, complete_evaluation_accuracy_claim=False,
        source_commit=env["source_commit"], config_hash=env["config_hash"],
        model_name="Qwen2.5-Coder-7B-Instruct", model_revision=env["model_revision"],
        checkpoint_identity=cases[0]["checkpoint_identity"],
        checkpoint_weights_sha256=env["trainable_weights_sha256"],
        gpu=env["gpu"], nominal_gpu_memory_gb=8, dtype=env["dtype"],
        torch=env["torch"], triton=env["triton"], cuda=env["cuda"], transformers=env["transformers"],
        generation=dict(engine="hf_eager", chat_template_applied=True, sampling=config["sampling_config"] if "sampling_config" in config else cases[0]["sampling_config"],
                        samples_per_featured_operation=1),
        training_context=dict(sft_optimizer_steps=env["sft_optimizer_steps"], grpo_optimizer_steps=env["optimizer_steps"],
                              lora_r=config["lora_r"], lora_alpha=config["lora_alpha"], qlora_cpu_offload=True),
        verification=dict(fp16_rtol=0.01, fp16_atol=0.01, warmup_iters=config["warmup_iters"],
                          benchmark_replays=config["bench_iters"], captured_calls_per_replay=16,
                          timing_mode="cuda_graph", worker_timeout_secs=config["worker_timeout_secs"],
                          canonical_and_independent_cases=True, registered_stress_cases=True,
                          output_shape_dtype_device_finiteness=True, input_mutation_and_post_timing_checks=True,
                          timing_summary="Median CUDA-event time per captured invocation; host dispatch amortized.",
                          not_application_latency=True),
        new_generation_performed=False, new_grading_performed=False,
        canonical_record_hash_format="SHA-256 of UTF-8 JSON with sorted keys, separators=(',', ':'), ensure_ascii=False, no trailing newline.",
        cases=cases)
    write_json(ROOT / "evidence.json", evidence)
    bibliography = json.loads((repo / "paper/bibliography.json").read_text())
    wanted = ["kernelbench2025", "triton2019", "qwen25coder2024", "qlora2023", "deepseekmath2024"]
    refs = [copy.deepcopy(next(ref for ref in bibliography["references"] if ref["id"] == key)) for key in wanted]
    for number, ref in enumerate(refs, 1):
        ref["case_study_number"] = number
    write_json(ROOT / "bibliography.json", dict(schema_version="selected_primary_references_v1",
        metadata_source="paper/bibliography.json, preserved original reference metadata", references=refs))

    manuscript = dict(schema_version="kernelbench_selected_case_manuscript_v1",
        title="KernelBench-RL: Two Verified GPU Kernel Case Studies on an 8 GB Laptop",
        author={"display_name": "Eljo Mustafaj"}, publication_date="2026-11-27", date_display="November 27, 2026",
        document_type="Selected technical case studies", featured_case_count=2, verified_featured_count=2,
        complete_evaluation_accuracy_claim=False,
        abstract="This paper documents two selected positive examples of GPU kernel generation on an 8 GB laptop: fused bias plus GELU and matrix multiplication plus bias plus SiLU. Both saved programs were generated by a Qwen2.5-Coder-7B-Instruct adapter checkpoint and passed the recorded FP16 verification procedure. The paper explains each computation, the generated dataflow, correctness checks, and CUDA-graph timing boundaries. The result is 2/2 verified featured programs, selected after verification; it is not a complete-evaluation accuracy estimate. These case studies make no claim of incremental reinforcement-learning benefit or application-level acceleration.",
        scope_note="2/2 refers only to the two featured programs, selected after verification. It is not an estimate of generation accuracy across an evaluation suite.",
        opening="The useful unit of evidence here is an executable program tied to an exact source hash, numerical checks, and a recorded timing procedure. The two examples illustrate different implementation patterns: a pointwise activation over a row, and a tiled matrix reduction with a fused epilogue. The project uses custom registered workloads in the spirit of executable kernel evaluation [1]; this paper does not report the full KernelBench benchmark.",
        pipeline_caption="A saved contract-to-program-to-verification chain. Only the two selected PASS records are featured; the diagram does not denote a new experiment.",
        gelu=dict(title="Bias plus GELU", operation="fused_bias_gelu_4096",
            purpose="A bias addition followed by a smooth activation is a common boundary in neural-network feed-forward computation. This case examines that boundary as one isolated GPU operation: add the channel bias, evaluate the registered tanh-approximate GELU, and write an FP16 result.",
            implementation="The generated host function flattens leading dimensions into rows and launches one Triton program per row. Each program loads input and bias values with a column-tail mask, converts them to FP32, evaluates the activation, and casts at the final store. The code expresses tanh through the equivalent identity 2 sigmoid(2u) - 1. No intermediate activation tensor is written to global memory by this program.",
            equation="z = x + b     |     GELU(z) = 0.5 z [1 + tanh(0.79788456 (z + 0.044715 z^3))]",
            canonical="X: 512 x 4096; bias: 4096; output: 512 x 4096.",
            stress="Registered tail case: X 3 x 19 with a 19-element bias.",
            measured_note="The candidate and both references were measured on the canonical shape with the same CUDA-graph timing routine. Ratios are reference time divided by candidate time for this saved run.",
            takeaway="This selected program combines a broadcast addition and nonlinear activation in a compact row-wise dataflow. Its saved PASS includes the non-power-of-two tail check; it is not a claim about all possible tensor layouts or sizes."),
        matmul=dict(title="Matrix multiplication plus bias plus SiLU", operation="matmul_bias_silu_512",
            purpose="A linear projection followed by a channel bias and SiLU activation is a useful fused-computation pattern. The operation maps A and B to an output matrix while applying the activation before the final write. This case focuses on the generated tile structure and the verified result, rather than on a larger model's runtime.",
            implementation="The generated program covers a 64 x 64 output tile. It traverses the reduction dimension in blocks of 32, loads masked A and B tiles, and accumulates a tl.dot result in FP32 with input_precision=\"ieee\". After the K loop, it adds the column bias, computes z sigmoid(z), and stores the FP16 tile with row and column masks. The epilogue remains inside the same device program [2].",
            equation="Z = A B + b     |     Y = SiLU(Z) = Z sigmoid(Z)",
            canonical="A: 256 x 512; B: 512 x 512; bias: 512; output: 256 x 512.",
            stress="Registered tail case: (17 x 23) times (23 x 19), plus a 19-element bias.",
            measured_note="The recorded comparison is at the kernel-operation boundary. All three timing columns use the same canonical inputs and graph procedure; none measures a complete inference application.",
            takeaway="This selected program demonstrates full tiled reduction coverage and an in-program activation epilogue. The registered tail shape exercises partial M, N, and K tiles rather than only the aligned canonical dimensions."),
        verification_steps=[
            {"title":"Bind the saved program", "body":"Use the exact generated source, prompt hash, model revision, configuration hash, and checkpoint identity. Both featured records use one sampled completion for their operation; no new generation or regrading was performed for this paper."},
            {"title":"Check the implementation contract", "body":"Apply source-policy checks and execute in a sanitized child process with a wall-clock timeout. The policy requires actual Triton device computation and checks for prohibited reference fallbacks. This is a development guard for trusted candidates, not a formal security sandbox."},
            {"title":"Compare numerical outputs", "body":"Check canonical inputs, the registered non-power-of-two stress shape, and independent-seed inputs against native PyTorch and quantized-input NumPy references. Require the expected output shape, FP16 dtype, CUDA device, finite values, and unchanged input tensors."},
            {"title":"Apply the declared FP16 criterion", "body":"Use rtol = 0.01 and atol = 0.01. The recorded maximum absolute errors summarize the checked cases; PASS does not mean bitwise equality. The relative term matters when outputs have larger magnitudes."},
            {"title":"Validate timing and retain evidence", "body":"Check candidate behavior during timing and check the output again afterward. Validate the compiled reference before timing it. Retain raw numerical errors, latencies, seeds, source hashes and PASS flags instead of relying only on a plotted ratio."}],
        timing_boundary="Timings are steady-state CUDA-graph measurements: 5 warmup iterations, 16 captured calls per replay, and a median over 30 replay samples, normalized per captured invocation. They amortize Python dispatch and exclude compilation time from the steady-state latency. The compile_ms field names the torch.compile reference's measured execution latency, not compiler wall time. No confidence interval or application-level benchmark accompanies these saved measurements.",
        training_context="The generator is Qwen2.5-Coder-7B-Instruct [3], with NF4 quantization and rank-16 low-rank adapters in a QLoRA-style setup [4]. The generating checkpoint records 88 supervised optimizer updates followed by 5 applied GRPO optimizer updates [5]. The decoder uses GPU computation while embedding and vocabulary-head computation are placed on CPU. These facts identify the checkpoint; these two selected cases do not isolate the effect of either training phase.",
        provenance_note="Every featured code hash identifies the complete generated module, not a manually edited excerpt. Individual saved records and exact program text accompany this manuscript source. The original study artifacts remain separate and unchanged. This case-study edition adds explanation and presentation, not new experimental observations.",
        interpretation="Read the timing tables as observations about two chosen implementations on one recorded laptop setup. A favorable isolated-operation ratio does not establish an end-to-end inference speedup, statistical significance, broad workload coverage, or a new kernel-generation method. The evidence supports the narrower statement that these two saved programs satisfied the recorded verifier and have the listed steady-state timing measurements.",
        references=[{"number":i+1,"id":r["id"]} for i,r in enumerate(refs)],
        evidence_file="evidence.json", evidence_file_sha256=sha((ROOT / "evidence.json").read_bytes()),
        selected_records_file="selected-programs.jsonl", selected_records_file_sha256=sha((ROOT / "selected-programs.jsonl").read_bytes()))
    write_json(ROOT / "manuscript.json", manuscript)
    print(json.dumps(dict(source_dir=str(ROOT), selected_cases=SELECTED,
        manuscript_sha256=sha((ROOT / "manuscript.json").read_bytes()),
        evidence_sha256=sha((ROOT / "evidence.json").read_bytes())), indent=2))


if __name__ == "__main__":
    main()
