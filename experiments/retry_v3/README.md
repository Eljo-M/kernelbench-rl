# Prospective expanded-adapter 7B follow-up

This independent follow-up starts from the same pinned original Qwen 7B base
with fresh rank-16 adapters on attention and MLP projections. It does not resume
the version 2 adapters. The frozen configuration has 40,370,176 trainable
parameters, eleven supervised target programs, eight SFT epochs (88 example
updates), and sixteen GRPO groups of two candidates. Performed GRPO updates
must be read from the saved records; equal-reward groups do not imply an update.

The new prospective tests are `cosine_similarity_1024`,
`fused_bias_gelu_4096`, and `matmul_bias_silu_512`. The repository-local identity
audit found no prior generated-candidate or supervised-target records for these
programs. Public fixed implementations and contracts already existed. This is
shared-family compositional transfer, with no claim about unseen families,
external interactions, or pretraining contamination.

All three version 2 fresh targets become training-seen in this follow-up. Their
version 2 scores remain unchanged, and they receive no fresh or development test
credit in version 3. The original inspected three-task development cohort stays
separate. Different splits, seed, rank, module coverage and training schedules
make version 2 and version 3 an unpaired comparison.

`protocol.json` and `laptop_7b/config.json` define the exact schedule and candidate
budgets. `exposure_inventory.json` records the prospective audit. Readiness
records distinguish the actual longest training example from a synthetic maximum
context probe; neither is a quality result. Probe weights are discarded. A
reporting-only failure in the first readiness harness is retained separately.

After source publication, `source_provenance.json` binds the executed source
commit to the exact protocol bytes and configuration hash. Study environment
files, checkpoint inventories and output hashes subsequently record actual
execution. Checkpoints are distributed through versioned release assets, not
stored as large Git blobs.

The stopping rule permits one frozen schedule and preserves every test failure.
There is no promised 3/3 result. Final reporting requires base pass@1, base
best-of-eight, the SFT-only checkpoint, and the final hybrid checkpoint, each with
its own budget, identity and denominator.
