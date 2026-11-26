import datetime, json, os, subprocess, sys, time
from pathlib import Path
root=Path('/content/kernelbench-rl'); sys.path.insert(0,str(root)); os.chdir(root)
import torch, triton
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.qlora_grpo import run_qlora_7b_grpo
torch.set_num_threads(2)
model=Path('/content/models/Qwen2.5-Coder-7B-Instruct')
revision=(model/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]
out=Path('/content/t4-results/qlora7b'); out.mkdir(parents=True,exist_ok=True)
cfg=ExperimentConfig.preset_t4_7b_qlora(model_name_or_path=str(model),model_revision=revision,max_new_tokens=1024,max_steps=8,checkpoint_every=2,worker_timeout_secs=120,warmup_iters=5,bench_iters=30,checkpoint_dir=str(out/'checkpoints'),output_jsonl=str(out/'train.jsonl'))
cfg.to_json(out/'config.json')
manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_id':'Qwen/Qwen2.5-Coder-7B-Instruct','model_revision':revision,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'config_hash':cfg.config_hash(),'torch':torch.__version__,'triton':triton.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'python':sys.version,'training_mode':cfg.training_mode,'compute_dtype':cfg.dtype,'training_performed':False}
(out/'environment.json').write_text(json.dumps(manifest,indent=2))
start=time.monotonic()
try:
 state=run_qlora_7b_grpo(cfg)
 manifest.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),steps=state.step,optimizer_steps=state.optimizer_step_count,training_performed=state.optimizer_step_count>0,trainable_weights_sha256=state.trainable_weights_sha256)
except Exception as exc:
 manifest.update(error=type(exc).__name__+': '+str(exc))
 raise
finally:
 manifest.update(seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
 (out/'environment.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(manifest,indent=2),flush=True)
