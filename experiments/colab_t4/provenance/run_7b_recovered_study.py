import datetime,gc,json,os,subprocess,sys,time,zipfile
from pathlib import Path
root=Path('/content/kernelbench-rl'); os.chdir(root); sys.path.insert(0,str(root))
import torch,triton
from rl_trainer.recover_initial_batch import recover_initial_batch
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.evaluate_heldout import main as evaluate_main
torch.set_num_threads(2)
source=Path('/content/t4-results/qlora7b')
out=Path('/content/t4-results/qlora7b_recovered')
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
start=time.monotonic()
try:
 recovery=recover_initial_batch(input_jsonl=source/'train.jsonl',config_path=source/'config.json',out_dir=out,initial_checkpoint=source/'checkpoints',logp_verify_atol=0.001)
 env=dict(model_id='Qwen/Qwen2.5-Coder-7B-Instruct',model_revision=recovery['model_revision'],training_mode='qlora_7b_nf4',dtype='fp16',gpu=torch.cuda.get_device_name(0),torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,source_commit=commit,training_performed=recovery['policy_updated'],optimizer_steps=recovery['optimizer_step_count'],weight_l2_delta=recovery['weight_l2_delta'],seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (out/'environment.json').write_text(json.dumps(env,indent=2))
 with zipfile.ZipFile('/content/kernelbench-7b-recovery-checkpoint.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for p in out.rglob('*'):
   if p.is_file(): z.write(p,p.relative_to(out))
 print('RECOVERY_TRAINING_COMPLETE',json.dumps(env),flush=True)
 gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
 cfg=ExperimentConfig.from_json(out/'config.json')
 ev=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_id=env['model_id'],model_revision=cfg.model_revision,training_mode=cfg.training_mode,dtype=cfg.dtype,gpu=env['gpu'],torch=torch.__version__,triton=triton.__version__,source_commit=commit,config_hash=cfg.config_hash(),generation_engine='hf_eager',train_ops=cfg.train_ops,heldout_ops=cfg.heldout_ops,best_of_k=8,sampling=cfg.sampling_config_dict(),checkpoint_weights_sha256=recovery['trainable_weights_sha256'])
 (out/'heldout_environment.json').write_text(json.dumps(ev,indent=2))
 evaluate_main(['--config',str(out/'config.json'),'--checkpoint-dir',str(out/'checkpoints'),'--out',str(out/'heldout.jsonl')])
 ev.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
 (out/'heldout_environment.json').write_text(json.dumps(ev,indent=2))
 print('HELDOUT_COMPLETE',json.dumps(ev),flush=True)
except Exception as exc:
 if out.exists(): (out/'study_error.json').write_text(json.dumps({'error':type(exc).__name__+': '+str(exc),'source_commit':commit,'seconds':time.monotonic()-start},indent=2))
 raise
