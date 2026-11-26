import sys, json, datetime, subprocess
from pathlib import Path
import torch, triton
root=Path("/content/kernelbench-rl")
out=Path("/content/t4-results/baselines"); out.mkdir(parents=True,exist_ok=True)
manifest={"torch":torch.__version__,"triton":triton.__version__,"gpu":torch.cuda.get_device_name(0),"cuda":torch.version.cuda,"python":sys.version,"source_commit":"76d7b18de4208f4ab66b2e56c9a6ba5949e86a9a","timing_mode":"cuda_graph","seeds":[42,43,44],"dtype":"fp32"}
(out/"environment.json").write_text(json.dumps(manifest,indent=2))
rows=[]
for seed in manifest["seeds"]:
 p=out/f"seed{seed}.jsonl"
 ret=subprocess.run([sys.executable,"-u","-m","kernels.validate_cuda","--dtypes","fp32","--seed",str(seed),"--warmup","10","--iters","50","--timing-mode","cuda_graph","--out",str(p)],cwd=root)
 if p.exists(): rows.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
 print("SEED_FINISHED",seed,"exit",ret.returncode,flush=True)
(out/"baselines.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+chr(10) for r in rows))
manifest.update(records=len(rows),passed=sum(r["status"]=="PASS" for r in rows),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/"environment.json").write_text(json.dumps(manifest,indent=2))
print("T4_BASELINES_COMPLETE",json.dumps(manifest),flush=True)
