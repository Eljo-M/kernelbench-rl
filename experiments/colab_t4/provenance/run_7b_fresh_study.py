import datetime, gc, json, os, subprocess, sys, time, zipfile
from pathlib import Path

root = Path('/content/kernelbench-rl')
sys.path.insert(0, str(root))
os.chdir(root)
import torch, triton
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.qlora_grpo import run_qlora_7b_grpo
from rl_trainer.evaluate_heldout import main as evaluate

torch.set_num_threads(2)
out = Path('/content/t4-results/qlora7b_fresh')
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()):
    raise FileExistsError('Refusing to overwrite a previous fresh study')
cfg = ExperimentConfig.from_json(Path('/content/t4-results/qlora7b/config.json'))
cfg.output_jsonl = str(out / 'train.jsonl')
cfg.checkpoint_dir = str(out / 'checkpoints')
cfg.resume_from = ''
cfg.to_json(out / 'config.json')
manifest = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    model_id='Qwen/Qwen2.5-Coder-7B-Instruct', model_revision=cfg.model_revision,
    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    config_hash=cfg.config_hash(), torch=torch.__version__, triton=triton.__version__,
    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), python=sys.version,
    training_mode=cfg.training_mode, compute_dtype=cfg.dtype, training_performed=False,
    schedule='fresh on-policy groups with immediate per-group updates')
start = time.monotonic()
try:
    state = run_qlora_7b_grpo(cfg)
    manifest.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        steps=state.step, optimizer_steps=state.optimizer_step_count,
        training_performed=state.optimizer_step_count > 0,
        trainable_weights_sha256=state.trainable_weights_sha256,
        seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (out/'environment.json').write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile('/content/kernelbench-7b-fresh-checkpoint.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in out.rglob('*'):
            if p.is_file(): z.write(p, p.relative_to(out))
    print('FRESH_TRAINING_COMPLETE '+json.dumps(manifest), flush=True)
    if state.optimizer_step_count == 0:
        raise RuntimeError('No optimizer updates occurred; refusing to label checkpoint as trained')
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    heldout = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=manifest['source_commit'], model_id=manifest['model_id'],
        model_revision=cfg.model_revision, dtype=cfg.dtype, gpu=manifest['gpu'],
        torch=torch.__version__, config_hash=cfg.config_hash(), generation_engine='hf_eager',
        train_ops=cfg.train_ops, heldout_ops=cfg.heldout_ops, best_of_k=cfg.heldout_best_of_k,
        sampling=cfg.sampling_config_dict(), checkpoint_step=state.step,
        checkpoint_weights_sha256=state.trainable_weights_sha256)
    (out/'heldout_environment.json').write_text(json.dumps(heldout, indent=2))
    eval_start=time.monotonic()
    evaluate(['--config',str(out/'config.json'),'--checkpoint-dir',str(out/'checkpoints'),
              '--out',str(out/'heldout.jsonl')])
    heldout.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seconds=time.monotonic()-eval_start, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (out/'heldout_environment.json').write_text(json.dumps(heldout, indent=2))
    print('HELDOUT_COMPLETE '+json.dumps(heldout), flush=True)
except Exception as exc:
    manifest.update(error=type(exc).__name__+': '+str(exc),seconds=time.monotonic()-start)
    (out/'environment.json').write_text(json.dumps(manifest, indent=2))
    raise
