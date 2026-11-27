"""Versioned offline retry export. Run only after both studies and final docs complete.
Historical exports and DELIVERY_MANIFEST.json are never written by this builder.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile

def discover_layout(script):
    """Recognize workspace/work and repo/tools/publication without Git calls."""
    for ancestor in script.resolve().parents:
        if ((ancestor/'rl_trainer/experiment_config.py').is_file()
                and (ancestor/'kernels/op_registry.py').is_file()):
            # The documented exact-resume layout is workspace/outputs/repo.
            workspace=ancestor.parent.parent if ancestor.parent.name=='outputs' else ancestor
            return workspace,ancestor
    # Preserve the live workspace/work/export_retry.py defaults.
    workspace=script.resolve().parent.parent
    return workspace,workspace/'outputs/kernelbench-rl'


ROOT,DEFAULT_REPO=discover_layout(Path(__file__))
BUILDER='kernelbench_retry_export_v1'
SCHEMA='kernelbench_retry_delivery_v1'
MAX_ASSET_BYTES=2*1024**3-1
STUDY_COMMIT='cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e'
REPOSITORY='https://github.com/Eljo-M/kernelbench-rl'
PROTECTED={'DELIVERY_MANIFEST.json','FINAL_REPORT.html','kernelbench-rl-source.zip','kernelbench-t4-final-results.zip','kernelbench-7b-qlora-checkpoints.zip','kernelbench-05b-fullparam-checkpoint.zip','kernelbench-rl.zip'}
BAD_PARTS={'.git','.ssh','.aws','.codex','.venv','venv','__pycache__','node_modules','.cache','model_cache','models','huggingface','triton_cache','torchinductor_cache'}
TEXT_SUFFIXES={'.py','.json','.jsonl','.md','.txt','.html','.css','.js','.yml','.yaml','.toml','.sh','.ps1','.xml','.csv'}
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{28,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def require(condition,message):
    if not condition: raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def read_jsonl(path):
    return [json.loads(line,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def metadata(path,purpose,**extra):
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha256(path),purpose=purpose,**extra)


def safe_name(name):
    require('\\' not in name and ':' not in name and not name.startswith('/'),'Nonportable or absolute archive member: '+name)
    parts=PurePosixPath(name).parts
    require(parts and '..' not in parts and '.' not in parts,'Archive traversal: '+name)
    require(not ({part.lower() for part in parts}&BAD_PARTS),'Cache/credential/model-cache path refused: '+name)
    require(not any(part.lower().startswith('.env') or part.lower() in {'credentials','credentials.json','token','token.txt','id_rsa','id_ed25519'} for part in parts),'Credential filename refused: '+name)


def relative_file(root,name):
    require(isinstance(name,str) and bool(name),'Missing referenced checkpoint file')
    safe_name(name)
    path=root/name
    require(path.resolve().is_relative_to(root.resolve()),'Referenced file escapes checkpoint directory')
    require(path.is_file() and not path.is_symlink(),'Missing/symlinked checkpoint file: '+str(path))
    return path


def scan_content(name,content):
    if Path(name).suffix.lower() in TEXT_SUFFIXES or '.json.attempt-' in name:
        require(SECRET.search(content) is None,'Credential-like content refused in '+name)


def verify_zip(path,source=False):
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None,'ZIP CRC failed: '+path.name)
        names=archive.namelist()
        require(len(names)==len(set(names)),'Duplicate ZIP entries: '+path.name)
        for entry in archive.infolist():
            safe_name(entry.filename)
            require(not stat.S_ISLNK(entry.external_attr>>16),'Symlink ZIP member refused: '+entry.filename)
            if source: require('checkpoints' not in PurePosixPath(entry.filename).parts and not entry.filename.endswith(('.pt','.safetensors','.bin')),'Tensor in source ZIP')
            if not entry.is_dir() and (Path(entry.filename).suffix.lower() in TEXT_SUFFIXES or '.json.attempt-' in entry.filename): scan_content(entry.filename,archive.read(entry))
    print('CRC/path verification passed: '+path.name,flush=True)


def git(repo,*arguments):
    return subprocess.check_output(['git','-C',str(repo),*arguments],text=True).strip()


def check_source(repo,commit,study_commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Full lowercase release source SHA required')
    require(re.fullmatch('[0-9a-f]{40}',study_commit) is not None,'Full study source SHA required')
    require(git(repo,'rev-parse','HEAD')==commit,'Final HEAD differs from --source-commit')
    require(not git(repo,'status','--porcelain=v1','--untracked-files=normal'),'Commit final source/results before exporting')
    subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',study_commit,commit],check=True)


def completed(path,cfg_hash,source_commit):
    manifest=read_json(path)
    require(isinstance(manifest,dict) and manifest.get('completed_utc') and not manifest.get('error'),'Incomplete/failed stage: '+str(path))
    require(manifest.get('config_hash')==cfg_hash and manifest.get('source_commit')==source_commit,'Source/config mismatch: '+str(path))
    require(manifest.get('source_code_dirty') is False,'Stage does not attest clean reviewed source: '+str(path))
    return manifest


def checkpoint(root,cfg,manifest,sft=False):
    latest=relative_file(root,'latest.json'); pointer=read_json(latest)
    dirname=pointer.get('latest_step_dir','')
    require(re.fullmatch(r'step_[0-9]{5}',dirname) is not None,'Invalid latest checkpoint pointer')
    step_root=root/dirname
    require(step_root.is_dir() and not step_root.is_symlink(),'Missing/symlinked checkpoint step')
    meta_path=relative_file(step_root,'checkpoint_state.json'); state=read_json(meta_path)
    expected_step=0 if sft else cfg.max_steps
    expected_grpo=0 if sft else manifest['optimizer_steps']
    expected_sft=cfg.sft_epochs*len(cfg.train_ops)
    fields=dict(step=expected_step,op_cursor=expected_step,seed=cfg.seed,config_hash=cfg.config_hash(),training_mode=cfg.training_mode,optimizer_step_count=expected_grpo,sft_optimizer_steps=expected_sft,total_rollouts=0 if sft else cfg.max_steps*cfg.group_size)
    for key,expected in fields.items():
        require(type(state.get(key)) is type(expected) and state[key]==expected,'Checkpoint '+key+' mismatch: '+str(meta_path))
    require(state.get('is_fixture') is False,'Fixture checkpoint refused')
    weights_sha=state.get('trainable_weights_sha256','')
    require(re.fullmatch('[0-9a-f]{64}',weights_sha) is not None,'Missing checkpoint checksum')
    require(manifest.get('checkpoint_step')==expected_step and manifest.get('checkpoint_weights_sha256')==weights_sha,'Evaluation/checkpoint mismatch')
    require(manifest.get('checkpoint_identity')==f'checkpoint:{expected_step}:{weights_sha}','Checkpoint identity mismatch')
    require(pointer.get('step')==expected_step,'Latest pointer step mismatch')
    allowed={meta_path,relative_file(step_root,state.get('optimizer_state_file',''))}
    rng_path=relative_file(step_root,state.get('rng_state_file','')); allowed.add(rng_path)
    rng=read_json(rng_path); allowed.add(relative_file(step_root,rng.get('torch_rng_file','')))
    if cfg.training_mode=='qlora_7b_nf4':
        require(not state.get('model_weights_file') and state.get('adapter_dir'),'QLoRA export must contain adapters, not base tensors')
        safe_name(state['adapter_dir'])
        adapter=step_root/state['adapter_dir']
        require(adapter.resolve().is_relative_to(step_root.resolve()) and adapter.is_dir() and not adapter.is_symlink(),'Invalid adapter directory')
        require((adapter/'adapter_config.json').is_file() and any((adapter/name).is_file() for name in ('adapter_model.safetensors','adapter_model.bin')),'Adapter tensors/config missing')
        for path in adapter.rglob('*'):
            if path.is_file():
                require(path.parent==adapter and path.name in {'adapter_config.json','adapter_model.safetensors','adapter_model.bin','README.md'},'Unexpected adapter/cache file '+str(path))
                allowed.add(path)
    else: allowed.add(relative_file(step_root,state.get('model_weights_file','')))
    files=[latest]
    for path in sorted(step_root.rglob('*')):
        if path.is_file():
            require(not path.is_symlink(),'Checkpoint symlink refused'); safe_name(path.relative_to(root).as_posix())
            require(path in allowed,'Unreferenced file/cache in checkpoint step: '+str(path))
            require(path.suffix.lower() in {'.pt','.json','.safetensors','.bin','.md','.txt'},'Unexpected checkpoint artifact '+str(path))
            if path.suffix.lower() in {'.bin','.safetensors'}: require(path.name in {'adapter_model.bin','adapter_model.safetensors'},'Base-model cache refused')
            files.append(path)
    return dict(root=root,step_root=step_root,state=state,files=files)


def validate_training(run_dir,cfg,env,sft_state):
    gpu=env.get('gpu')
    require(isinstance(gpu,str) and bool(gpu.strip()) and gpu.lower()!='unavailable',
            'Completed training environment requires the actual GPU identity')
    require(env.get('dtype')==cfg.dtype,'Training environment precision differs from configuration')
    expected_sft=cfg.sft_epochs*len(cfg.train_ops); summary=read_json(run_dir/'sft_summary.json')
    require(summary.get('ran') is True and summary.get('optimizer_step_count')==expected_sft,'SFT warmup incomplete')
    require(summary.get('weight_sha256_after')==sft_state['trainable_weights_sha256'],'SFT summary/checkpoint checksum mismatch')
    sft_rows=read_jsonl(run_dir/'sft.jsonl')
    require(len(sft_rows)==expected_sft and [row.get('step') for row in sft_rows]==list(range(1,expected_sft+1)),'SFT step inventory mismatch')
    require(Counter((row.get('epoch'),row.get('op_name')) for row in sft_rows)==Counter((epoch,op) for epoch in range(1,cfg.sft_epochs+1) for op in cfg.train_ops),'SFT operator/epoch inventory mismatch')
    require(all(isinstance(row.get('loss'),(int,float)) and math.isfinite(row['loss']) and math.isfinite(row.get('grad_norm',float('nan'))) for row in sft_rows),'Nonfinite SFT evidence')
    rows=read_jsonl(run_dir/'train.jsonl'); require(len(rows)==cfg.max_steps*cfg.group_size,'Incomplete GRPO rollout inventory'); groups={}
    for row in rows:
        step,index=row.get('step'),row.get('group_index')
        require(type(step) is int and 1<=step<=cfg.max_steps and type(index) is int and 0<=index<cfg.group_size,'Invalid rollout identity')
        require((step,index) not in groups,'Duplicate rollout identity')
        require(row.get('op_name')==cfg.train_ops[(step-1)%len(cfg.train_ops)] and row.get('config_hash')==cfg.config_hash(),'Training operator/config mismatch')
        require(row.get('backend')==cfg.backend and row.get('dtype')==cfg.dtype,
                'Training rollout backend/precision differs from configuration')
        status=row.get('status')
        require(status in {'PASS','CHEAT','COMPILE_ERROR','NUMERICAL_MISMATCH','TIMEOUT','OOM','RUNTIME_ERROR'},
                'Invalid training rollout status')
        row_gpu=row.get('gpu_model',row.get('gpu'))
        unavailable_failure=(row_gpu=='unavailable' and row.get('compiled') is False
                             and row.get('correct') is False and status in
                             {'CHEAT','COMPILE_ERROR','TIMEOUT','OOM','RUNTIME_ERROR'})
        require(row_gpu==gpu or unavailable_failure,'Training rollout GPU differs from completed training environment')
        if status=='PASS':
            require(row_gpu==gpu,'Successful training rollout requires the actual recorded GPU')
        require(row.get('sft_optimizer_steps')==expected_sft and row.get('training_reward_mode')=='correctness_gated','Training phase/reward mismatch')
        require(type(row.get('policy_updated')) is bool and type(row.get('optimizer_step_count')) is int,'Missing optimizer evidence')
        if row.get('status')!='PASS': require(row.get('reward')==0.0,'Invalid candidate received nonzero retry reward')
        groups[(step,index)]=row
    updated=0
    for step in range(1,cfg.max_steps+1):
        current=[groups[(step,index)] for index in range(cfg.group_size)]
        require(len({row['policy_updated'] for row in current})==1,'Mixed update flags within group')
        updated+=current[0]['policy_updated']
        require(all(row['optimizer_step_count']==updated for row in current),'GRPO update counter mismatch')
    require(updated==env['optimizer_steps'],'Final update count disagrees with training records')


def validate_frozen_base(run_dir,cfg):
    """Bind immutable pre-SFT/original proof to a completed no-op resume.

    The caller separately verifies resume_verified.json's zero added steps and
    exact checkpoint counters. Sidecars attest that invocation's frozen bytes.
    """
    identity=dict(model_name_or_path=cfg.model_name_or_path,
                  model_revision=cfg.model_revision,training_mode=cfg.training_mode)
    initial_path=run_dir/'frozen_base_initial.json'
    original_path=run_dir/'frozen_base_verified.json'
    initial=read_json(initial_path); original=read_json(original_path)
    for label,evidence in (('initial',initial),('original',original)):
        require(evidence.get('config_hash')==cfg.config_hash() and evidence.get('model_identity')==identity,
                f'QLoRA frozen-base {label} model/config identity mismatch')
    before=initial.get('before')
    require(isinstance(before,dict) and re.fullmatch('[0-9a-f]{64}',before.get('sha256','')) is not None,
            'Original pre-SFT frozen-base digest missing')
    require(before.get('hash_version')=='frozen_stored_tensors_v1'
            and type(before.get('frozen_parameter_tensors')) is int and before['frozen_parameter_tensors']>0
            and before.get('trainable_parameters_excluded') is True,'Incomplete frozen-base digest scope')
    require(original.get('training_completed') is True and original.get('verified') is True,
            'Original QLoRA frozen-base training proof is incomplete or failed')
    require(original.get('before')==before and original.get('after')==before,
            'Frozen base changed from the original pre-SFT evidence')
    require(original.get('initial_evidence_file')==initial_path.name,'Original proof references different initial evidence')
    original_sha=sha256(original_path); valid=[]
    for path in sorted(run_dir.glob('frozen_base_verified.resume-*.json')):
        # Interrupted/failed sidecars remain in the raw export, but cannot be
        # mistaken for the completed matching resume required here.
        try:
            sidecar=read_json(path)
        except (ValueError,OSError):
            continue
        if not isinstance(sidecar,dict):
            continue
        if (sidecar.get('config_hash')==cfg.config_hash() and sidecar.get('model_identity')==identity
                and sidecar.get('resumed') is True and sidecar.get('training_completed') is True
                and sidecar.get('verified') is True and sidecar.get('original_proof_preserved') is True
                and sidecar.get('original_proof_file')==original_path.name
                and sidecar.get('original_proof_sha256')==original_sha
                and sidecar.get('initial_evidence_file')==initial_path.name
                and sidecar.get('before')==before and sidecar.get('after')==before):
            valid.append(path)
    require(valid,'No completed zero-step-resume frozen-base sidecar preserves the exact original training proof')
    return dict(frozen_base_sha256=before['sha256'],initial_evidence_sha256=sha256(initial_path),
                original_proof_sha256=original_sha,original_proof_preserved=True,
                resume_sidecars=[dict(file=path.name,sha256=sha256(path)) for path in valid])


def validate_run(repo,run_dir,study_commit,load_retry_run,ExperimentConfig,protocol):
    cfg=ExperimentConfig.from_json(run_dir/'config.json')
    require(cfg.prompt_version=='contract_v2' and cfg.training_reward_mode=='correctness_gated' and cfg.sft_epochs>0,'Not the frozen retry protocol')
    checkpoint_root=(repo/cfg.checkpoint_dir).resolve()
    require(checkpoint_root.is_relative_to((repo/'checkpoints').resolve()),'Checkpoint root outside repository checkpoints')
    env=completed(run_dir/'environment.json',cfg.config_hash(),study_commit)
    expected_id={'full_param_0.49b':'Qwen/Qwen2.5-Coder-0.5B-Instruct','qlora_7b_nf4':'Qwen/Qwen2.5-Coder-7B-Instruct'}[cfg.training_mode]
    require(env.get('model_id')==expected_id,'Unexpected upstream model identity')
    require(env.get('model_revision')==cfg.model_revision and env.get('training_mode')==cfg.training_mode,'Training model/revision mismatch')
    require(env.get('steps')==cfg.max_steps and env.get('sft_optimizer_steps')==cfg.sft_epochs*len(cfg.train_ops),'Frozen training schedule incomplete')
    main=completed(run_dir/'heldout_environment.json',cfg.config_hash(),study_commit)
    sft=completed(run_dir/'sft_only_environment.json',cfg.config_hash(),study_commit)
    proof=completed(run_dir/'resume_verified.json',cfg.config_hash(),study_commit)
    for field in ('restored_model_checksum_verified','restored_optimizer_and_rng'): require(proof.get(field) is True,'Checkpoint resume proof missing')
    require(proof.get('additional_training_steps')==0,'Resume proof added training')
    for stage,filename in ((main,'heldout.jsonl'),(sft,'sft_only.jsonl')): require(stage.get('output_sha256')==sha256(run_dir/filename),'Final evaluation output changed')
    final=checkpoint(checkpoint_root,cfg,main); supervised=checkpoint(checkpoint_root/'sft_only',cfg,sft,sft=True); state=final['state']
    for field,expected in dict(checkpoint_step=state['step'],checkpoint_weights_sha256=state['trainable_weights_sha256'],optimizer_steps=state['optimizer_step_count'],sft_optimizer_steps=state['sft_optimizer_steps']).items():
        require(proof.get(field)==expected,'Resume proof/exported checkpoint mismatch: '+field)
    require(env.get('trainable_weights_sha256')==state['trainable_weights_sha256'] and env.get('optimizer_steps')==state['optimizer_step_count'],'Training environment/checkpoint mismatch')
    validate_training(run_dir,cfg,env,supervised['state'])
    frozen_base=validate_frozen_base(run_dir,cfg) if cfg.training_mode=='qlora_7b_nf4' else None
    strict=load_retry_run(run_dir,protocol_path=protocol)
    quality_env=strict['environment']
    require(env.get('gpu')==quality_env.get('gpu'),'Training and strict quality GPU identities differ')
    require(env.get('dtype')==quality_env.get('dtype')==cfg.dtype,
            'Training/quality/configuration precision identities differ')
    require(quality_env.get('backend')==cfg.backend,'Strict quality backend differs from configuration')
    return dict(name=run_dir.name,path=run_dir,cfg=cfg,environment=env,final=final,sft=supervised,
                strict=strict,frozen_base_integrity=frozen_base)


def validate_verifier_regression(repo,study_commit):
    """Separate new-worker correctness regression, not the historical speed campaign."""
    from kernels.op_registry import OP_REGISTRY
    from sandbox_runner.gpu_verifier import validate_worker_payload
    directory=repo/'experiments/retry_v2/verifier_regression'
    env=read_json(directory/'environment.json')
    require(env.get('completed_utc') and not env.get('error'),'Strict-worker baseline regression is incomplete')
    require(env.get('source_commit')==study_commit,'Verifier regression source differs from frozen study code')
    require(env.get('warmup',env.get('warmup_iters'))==5 and env.get('iterations',env.get('bench_iters'))==30,'Verifier regression timing setup mismatch')
    rows=read_jsonl(directory/'baselines.jsonl')
    expected={(op,dtype) for op in OP_REGISTRY for dtype in ('fp32','bf16')}
    require(len(expected)==60 and len(rows)==60,'Expected all 60 strict-worker regression checks')
    require({(row.get('op_name'),row.get('dtype')) for row in rows}==expected,'Verifier regression coverage/uniqueness mismatch')
    for row in rows:
        require(row.get('status')=='PASS' and row.get('seed')==142 and row.get('independent_seed')==10142,'Verifier regression failure or seed mismatch; retain artifacts for root review')
        require(validate_worker_payload(row,expected_run_id=row.get('run_id'),expected_op_name=row['op_name'],expected_level=OP_REGISTRY[row['op_name']].level),'Invalid strict-worker regression payload')
    return dict(directory=directory,checks=60,passed=60,source_commit=study_commit,note='New-worker correctness regression; historical 180-check performance campaign remains unchanged')


def add_file(archive,path,name):
    require(path.is_file() and not path.is_symlink(),'Only ordinary files can be exported: '+str(path)); safe_name(name)
    if path.suffix.lower() in TEXT_SUFFIXES: scan_content(name,path.read_bytes())
    archive.write(path,name)


def zip_files(target,entries,extra=None):
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for path,name in entries: add_file(archive,path,name)
        for name,content in (extra or {}).items():
            safe_name(name); encoded=content.encode('utf-8'); scan_content(name,encoded); archive.writestr(name,encoded)
    verify_zip(target)


def attribution(run,phase,release_commit,study_commit):
    cfg=run['cfg']; state=run[phase]['state']; model=run['environment']['model_id']
    return ('# KernelBench-RL experimental checkpoint\n\n'
        f'Model: {model}\nPinned upstream revision: {cfg.model_revision}\n'
        f'Upstream model: https://huggingface.co/{model}/tree/{cfg.model_revision}\n'
        'The original upstream model LICENSE is included verbatim as UPSTREAM_MODEL_LICENSE.txt.\n'
        'Original model authors and copyright holders retain their attribution.\n\n'
        f'Release/source snapshot: {release_commit}\nActual study source: {study_commit}\n'
        f'Phase: {phase}; group step {state["step"]}; SFT updates {state["sft_optimizer_steps"]}; GRPO optimizer updates {state["optimizer_step_count"]}.\n'
        f'Trainable weight checksum: {state["trainable_weights_sha256"]}\n\n'
        'Exact extraction layout (do not edit recorded model/config paths):\n'
        'workspace/outputs/kernelbench-rl/                 source checkout\n'
        'workspace/outputs/kernelbench-rl/checkpoints/...  extract checkpoint ZIP under outputs\n'
        'workspace/work/models/Qwen2.5-Coder-0.5B-Instruct/ pinned public 0.49B snapshot\n'
        'workspace/work/models/Qwen2.5-Coder-7B-Instruct/   pinned public 7B snapshot\n'
        'Run commands from workspace/outputs/kernelbench-rl. Configs retain ../../work/models/... .\n'
        'Download the upstream snapshot to the matching model directory using the model/revision above, e.g.:\n'
        f'hf download {model} --revision {cfg.model_revision} --local-dir ../../work/models/{model.split("/")[-1]}\n'
        'The original sft_output_jsonl and model_name_or_path are hashed configuration fields.\n'
        'Editing them changes config_hash and defines a new experiment, not an exact resume.\n'
        'QLoRA bundles contain adapters only; the frozen public 7B base is not redistributed.\n'
        'Retrieve tokenizer/configuration and required public base weights at the pinned revision.\n'
        'Use matching recorded configuration and load_checkpoint to restore optimizer/RNG state.\n'
        'Artifacts are experimental; weight updates alone do not demonstrate kernel-quality gains.\n')


def split_package(path,staging,limit):
    """Keep complete oversized ZIP locally; publish bounded reassemblable byte chunks."""
    if path.stat().st_size<=limit: return [path],None
    parts=[]
    with path.open('rb') as source:
        index=1
        while True:
            first=source.read(min(8*1024*1024,limit))
            if not first: break
            part=staging/(path.name+f'.part{index:04d}')
            with part.open('xb') as stream:
                stream.write(first); remaining=limit-len(first)
                while remaining:
                    block=source.read(min(8*1024*1024,remaining))
                    if not block: break
                    stream.write(block); remaining-=len(block)
            parts.append(part); index+=1
    reassembled=hashlib.sha256()
    for part in parts:
        with part.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024),b''): reassembled.update(block)
    require(reassembled.hexdigest()==sha256(path),'Split reassembly checksum mismatch')
    descriptor=staging/(path.name+'.parts.json')
    info=dict(original_name=path.name,original_bytes=path.stat().st_size,original_sha256=sha256(path),parts=[dict(name=part.name,bytes=part.stat().st_size,sha256=sha256(part)) for part in parts],restore='Concatenate parts in listed order to original_name; verify original_sha256 before extracting the CRC-verified ZIP.')
    descriptor.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    return [*parts,descriptor],info


class PaperMetadata(HTMLParser):
    def __init__(self): super().__init__(); self.metadata={}
    def handle_starttag(self,tag,attrs):
        data=dict(attrs)
        if tag=='meta' and data.get('name'): self.metadata[data['name']]=data.get('content','')


def validate_publication(paper,report,site):
    require(paper.is_file() and paper.read_bytes().startswith(b'%PDF-'),'Final PDF missing (render/text QA remains owner responsibility)')
    require(report.is_file() and '<html' in report.read_text(encoding='utf-8').lower(),'Final HTML report missing')
    require((site/'index.html').is_file() and (site/'paper.html').is_file(),'Final website dist missing')
    config=read_json(site/'project-config.json'); require(config.get('draft') is False,'Website remains a draft')
    parser=PaperMetadata(); parser.feed((site/'paper.html').read_text(encoding='utf-8'))
    require(parser.metadata.get('citation_author')=='Eljo Mustafaj','Website author mismatch')
    require(config.get('public_base_url','').startswith('https://') and config.get('paper_pdf_path'),'Website origin/PDF not configured')
    site_pdf=relative_file(site,config['paper_pdf_path'])
    require(sha256(site_pdf)==sha256(paper),'Website PDF differs from final downloadable paper')
    require(parser.metadata.get('citation_pdf_url')==config['public_base_url'].rstrip('/')+'/'+config['paper_pdf_path'],'Website citation PDF URL mismatch')


def build(args):
    workspace=(args.workspace_root or ROOT).resolve()
    default_repo=workspace/'outputs/kernelbench-rl' if args.workspace_root else DEFAULT_REPO
    repo=(args.repo or default_repo).resolve(); out=(args.output_root or workspace/'outputs').resolve()
    work=(args.work_root or workspace/'work').resolve()
    args.model_05b_dir=args.model_05b_dir or workspace/'work/models/Qwen2.5-Coder-0.5B-Instruct'
    args.model_7b_dir=args.model_7b_dir or workspace/'work/models/Qwen2.5-Coder-7B-Instruct'
    args.paper=args.paper or out/'KernelBench-RL-Paper.pdf'
    args.report=args.report or out/'RETRY_FINAL_REPORT.html'
    args.site=args.site or out/'kernelbench-rl-site/dist'
    work.mkdir(parents=True,exist_ok=True)
    require(repo.is_dir() and out.is_dir(),'Repository/output directory missing')
    require(0<args.max_asset_bytes<=MAX_ASSET_BYTES,'Asset limit must be below 2 GiB')
    require(re.fullmatch(r'https://github\.com/Eljo-M/kernelbench-rl/actions/runs/[0-9]+',args.ci_url) is not None,'Expected owner-verified final GitHub Actions URL')
    check_source(repo,args.source_commit,args.study_source_commit)
    sys.path.insert(0,str(repo))
    from eval.compare_retry import load_retry_run,render_retry_markdown
    from rl_trainer.experiment_config import ExperimentConfig
    protocol=(args.protocol or repo/'experiments/retry_v2/protocol.json').resolve()
    run_dirs=args.run_dir or [repo/'experiments/retry_v2/laptop',repo/'experiments/retry_v2/laptop_7b']
    require(len(run_dirs)==2 and len({Path(path).resolve() for path in run_dirs})==2,'Both distinct laptop retry studies are required')
    runs=[validate_run(repo,Path(path).resolve(),args.study_source_commit,load_retry_run,ExperimentConfig,protocol) for path in run_dirs]
    require({run['cfg'].training_mode for run in runs}=={'full_param_0.49b','qlora_7b_nf4'},'Expected one full-parameter 0.49B and one 7B QLoRA study')
    regression=validate_verifier_regression(repo,args.study_source_commit)
    for run in runs:
        require(run['path'].is_relative_to(repo),'Run artifacts must be inside frozen source tree')
        model_dir=(args.model_05b_dir if run['cfg'].training_mode=='full_param_0.49b' else args.model_7b_dir).resolve()
        license_path=model_dir/'LICENSE'
        require(license_path.is_file() and license_path.stat().st_size>0 and not license_path.is_symlink(),'Missing original upstream model LICENSE: '+str(model_dir))
        run['upstream_license']=license_path
    paper=args.paper.resolve(); report=args.report.resolve(); site=args.site.resolve()
    validate_publication(paper,report,site)
    strict=dict(schema_version='kernelbench_retry_comparison_v1',pooled_across_runs=False,rl_only_causal_claim=False,historical_paired_same_protocol=False,runs=[run['strict'] for run in runs],historical_context=[])
    if args.validate_only:
        print(json.dumps(dict(validation='pass',runs=[run['name'] for run in runs],release_source=args.source_commit,study_source=args.study_source_commit))); return
    final_manifest=out/'RETRY_DELIVERY_MANIFEST.json'; upload_manifest=out/'retry-release-assets.json'
    previous=read_json(final_manifest) if final_manifest.exists() else None
    if previous or upload_manifest.exists(): require(args.rebuild and previous and previous.get('schema_version')==SCHEMA and previous.get('builder')==BUILDER,'Existing retry export requires explicit --rebuild and its valid builder manifest')
    owner_files=previous.get('generated_files',{}) if previous else {}
    prefix='kernelbench-rl-retry-v2-'+args.source_commit[:12]
    staging=Path(tempfile.mkdtemp(prefix='retry-export-',dir=work)); packages=[]; generated=[]
    try:
        source=staging/(prefix+'-source.zip')
        subprocess.run(['git','-C',str(repo),'archive','--format=zip','--prefix=kernelbench-rl/',f'--output={source}',args.source_commit],check=True)
        verify_zip(source,source=True); packages.append((source,'Frozen final source and committed result snapshot'))
        raw=staging/(prefix+'-raw-results.zip'); entries=[(protocol,'kernelbench-rl/'+protocol.relative_to(repo).as_posix())]
        for directory in [*[run['path'] for run in runs],regression['directory']]:
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    require(path.suffix.lower() not in {'.pt','.safetensors','.bin'},'Tensor in raw result directory')
                    entries.append((path,'kernelbench-rl/'+path.relative_to(repo).as_posix()))
        retry_root=repo/'experiments/retry_v2'
        evidence=list(retry_root.glob('*.json'))
        if (retry_root/'README.md').is_file():
            evidence.append(retry_root/'README.md')
        evidence += [path for path in (retry_root/'comparison').rglob('*') if path.is_file()]
        archived_names={name for _,name in entries}
        for path in sorted(evidence):
            require(path.suffix.lower() not in {'.pt','.safetensors','.bin'},'Tensor in raw comparison/provenance directory')
            name='kernelbench-rl/'+path.relative_to(repo).as_posix()
            if name not in archived_names:
                entries.append((path,name)); archived_names.add(name)
        entries.append((Path(__file__).resolve(),'RELEASE_EXPORT/export_retry.py'))
        extra={'retry_comparison.json':json.dumps(strict,indent=2,allow_nan=False)+'\n','retry_comparison.md':render_retry_markdown(strict),'RELEASE_PROVENANCE.txt':f'Release source: {args.source_commit}\nActual study source: {args.study_source_commit}\nFinal CI URL: {args.ci_url}\nSource and study commits are intentionally distinguished.\n'}
        zip_files(raw,entries,extra); packages.append((raw,'Raw completed retry studies, stage/error history, protocol and strict paired comparison'))
        # Legacy model ZIPs remain immutable. Supply their original upstream
        # licenses and pinned-model attribution as a separate small asset.
        license_entries=[]; license_notes=['# Upstream licenses for immutable historical checkpoint bundles','',
            'The historical model ZIPs are byte-for-byte unchanged. This companion contains upstream LICENSE copies and attribution; it contains no model tensors.',
            f'Release source snapshot: {args.source_commit}',f'Retry study source: {args.study_source_commit}','']
        for run in runs:
            full=run['cfg'].training_mode=='full_param_0.49b'
            historical_config=repo/('experiments/laptop_fullparam/config.json' if full else 'experiments/colab_t4/qlora7b_fresh/config.json')
            historic=read_json(historical_config)
            require(historic.get('model_revision')==run['cfg'].model_revision,
                    'Historical model revision differs from the available pinned LICENSE snapshot')
            model_id=run['environment']['model_id']; revision=run['cfg'].model_revision
            filename='kernelbench-05b-fullparam-checkpoint.zip' if full else 'kernelbench-7b-qlora-checkpoints.zip'
            license_entries.append((run['upstream_license'],'upstream/'+model_id.split('/')[-1]+'/LICENSE'))
            license_notes += [f'## {model_id}',f'Historical archive: {filename}',f'Pinned model revision: {revision}',
                f'Original model source: https://huggingface.co/{model_id}/tree/{revision}',
                f'Copied upstream LICENSE SHA-256: {sha256(run["upstream_license"])}',
                'Original upstream authors/copyright holders retain their attribution. The project checkpoint authors do not replace that attribution.','']
        licenses=staging/(prefix+'-upstream-model-licenses.zip')
        zip_files(licenses,license_entries,{'UPSTREAM_MODEL_ATTRIBUTION.md':'\n'.join(license_notes)})
        packages.append((licenses,'Original upstream model licenses and attribution accompanying unchanged historical checkpoint ZIPs'))
        for run in runs:
            model='05b-fullparam' if run['cfg'].training_mode=='full_param_0.49b' else '7b-qlora'
            for phase in ('final','sft'):
                target=staging/(prefix+f'-{model}-{phase}.zip')
                entries=[(path,'kernelbench-rl/'+path.relative_to(repo).as_posix()) for path in run[phase]['files']]
                entries += [(run['path']/name,'kernelbench-rl/'+(run['path']/name).relative_to(repo).as_posix()) for name in ('config.json','environment.json','resume_verified.json','sft_summary.json','sft.jsonl','train.jsonl','heldout_environment.json','sft_only_environment.json')]
                if run['frozen_base_integrity'] is not None:
                    proof_names=['frozen_base_initial.json','frozen_base_verified.json',
                                 *[item['file'] for item in run['frozen_base_integrity']['resume_sidecars']]]
                    entries += [(run['path']/name,'kernelbench-rl/'+(run['path']/name).relative_to(repo).as_posix()) for name in proof_names]
                entries.append((run['upstream_license'],'UPSTREAM_MODEL_LICENSE.txt'))
                if (repo/'LICENSE').is_file(): entries.append((repo/'LICENSE','PROJECT_LICENSE.txt'))
                zip_files(target,entries,{'UPSTREAM_MODEL_ATTRIBUTION.md':attribution(run,phase,args.source_commit,args.study_source_commit)})
                packages.append((target,f'{run["name"]}: '+('final hybrid checkpoint' if phase=='final' else 'SFT-only checkpoint')+' with optimizer, RNG, provenance and upstream license'))
        website=staging/(prefix+'-website.zip')
        site_entries=[(path,'kernelbench-rl-site/dist/'+path.relative_to(site).as_posix()) for path in sorted(site.rglob('*')) if path.is_file()]
        zip_files(website,site_entries); packages.append((website,'Complete static website and matching searchable paper'))
        package_details=[]; upload_assets=[]; staged_records={}
        for path,purpose in packages:
            bounded,split=split_package(path,staging,args.max_asset_bytes)
            for member in [path,*[part for part in bounded if part!=path]]:
                record=metadata(member,purpose); record['path']=str(out/member.name); staged_records[member.name]=record
                if member not in generated: generated.append(member)
            package_details.append(dict(file=path.name,purpose=purpose,bytes=path.stat().st_size,sha256=sha256(path),release_asset_names=[part.name for part in bounded],split=split))
            upload_assets += [dict(path=str(out/part.name),bytes=part.stat().st_size,sha256=sha256(part)) for part in bounded]
        external=[metadata(paper,'Searchable professional technical report'),metadata(report,'Final explanatory HTML report')]
        publication_verification=None
        if args.verification_file:
            verification_path=args.verification_file.resolve(); verification=read_json(verification_path)
            require(verification.get('schema_version')=='kernelbench_publication_byte_verification_v1'
                    and verification.get('verified') is True and verification.get('working_tree_clean') is True
                    and verification.get('source_commit')==args.source_commit,
                    'Publication byte-verification evidence does not match this final source commit')
            record=metadata(verification_path,'Post-commit saved-byte and optional clone checkout verification')
            external.append(record)
            publication_verification=dict(**record,source_commit=args.source_commit,verified=True,
                clone_check_performed=verification.get('clone_check_performed') is True)
        require(all(item['bytes']<=args.max_asset_bytes for item in external),'Report exceeds per-asset limit')
        upload_assets += [{key:item[key] for key in ('path','bytes','sha256')} for item in external]
        historical=[]; historical_manifest=out/'DELIVERY_MANIFEST.json'
        if args.include_historical:
            saved=read_json(historical_manifest)
            for name,expected in saved['files'].items():
                path=out/name; require(path.is_file(),'Historical asset missing: '+name)
                require(path.stat().st_size==expected['bytes'] and sha256(path)==expected['sha256'],'Historical asset differs from immutable manifest: '+name)
                if name.endswith('.zip'): verify_zip(path)
                else: scan_content(name,path.read_bytes())
                require(path.stat().st_size<=args.max_asset_bytes,'Historical asset exceeds upload limit; builder will not rewrite or split it')
                historical.append(metadata(path,'Unmodified historical export; different protocol'))
            historical.append(metadata(historical_manifest,'Unmodified historical delivery manifest'))
            upload_assets += [{key:item[key] for key in ('path','bytes','sha256')} for item in historical]
        require(len({Path(item['path']).name for item in upload_assets})==len(upload_assets),'Duplicate release asset basenames')
        require(all(item['bytes']<=args.max_asset_bytes for item in upload_assets),'A release asset exceeds the configured limit')
        manifest=dict(schema_version=SCHEMA,builder=BUILDER,created_utc=datetime.now(timezone.utc).isoformat(),repository=REPOSITORY,
            release_source_commit=args.source_commit,release_source_url=REPOSITORY+'/commit/'+args.source_commit,
            study_source_commit=args.study_source_commit,study_source_url=REPOSITORY+'/commit/'+args.study_source_commit,
            ci=dict(url=args.ci_url,verification='Final source CI URL supplied by owner; exporter performs no network calls'),
            strict_evaluation_validation='eval.compare_retry.load_retry_run passed independently for both completed studies',
            publication_byte_verification=publication_verification,
            verifier_regression={key:value for key,value in regression.items() if key!='directory'},
            runs=[dict(name=run['name'],source_commit=run['environment']['source_commit'],model_id=run['environment']['model_id'],model_revision=run['cfg'].model_revision,
                config_hash=run['cfg'].config_hash(),sft_optimizer_steps=run['final']['state']['sft_optimizer_steps'],grpo_optimizer_steps=run['final']['state']['optimizer_step_count'],
                final_checkpoint_sha256=run['final']['state']['trainable_weights_sha256'],sft_checkpoint_sha256=run['sft']['state']['trainable_weights_sha256'],
                upstream_license_sha256=sha256(run['upstream_license']),frozen_base_integrity=run['frozen_base_integrity']) for run in runs],
            protocol=dict(path=protocol.relative_to(repo).as_posix(),sha256=sha256(protocol)),packages=package_details,generated_files=staged_records,documents=external,
            website=dict(path=str(site),url=args.site_url,publication_verification='Hosting success is verified by owner, not by offline exporter',files=[metadata(path,'Static website asset') for path in sorted(site.rglob('*')) if path.is_file()]),
            historical_assets=historical,historical_manifest_sha256=sha256(historical_manifest) if historical_manifest.exists() else None,
            scope='Completed bounded studies. Fresh and inspected-development cohorts remain separate. SFT+GRPO does not establish an RL-only causal gain.')
        manifest_staged=staging/final_manifest.name
        manifest_staged.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        upload_assets.append(dict(path=str(final_manifest),bytes=manifest_staged.stat().st_size,sha256=sha256(manifest_staged)))
        upload_staged=staging/upload_manifest.name
        upload_staged.write_text(json.dumps(dict(builder=BUILDER,source_commit=args.source_commit,assets=upload_assets),indent=2)+'\n',encoding='utf-8')
        for path in generated:
            target=out/path.name; require(target.name not in PROTECTED,'Historical export overwrite refused')
            if target.exists():
                if sha256(target)==sha256(path): continue
                owned=owner_files.get(target.name)
                require(args.rebuild and owned and target.stat().st_size==owned['bytes'] and sha256(target)==owned['sha256'],'Refusing to replace unowned/changed export: '+str(target))
        if previous:
            history=work/'retry-export-history'/uuid.uuid4().hex; history.mkdir(parents=True)
            shutil.copy2(final_manifest,history/final_manifest.name)
            if upload_manifest.exists(): shutil.copy2(upload_manifest,history/upload_manifest.name)
        for path in generated:
            target=out/path.name
            if target.exists() and sha256(target)==sha256(path): continue
            path.replace(target)
        # Metadata commits last. Historical manifest and external PDF/report inputs
        # are never written; failed staging is retained rather than deleted.
        manifest_staged.replace(final_manifest); upload_staged.replace(upload_manifest)
        print(json.dumps(dict(manifest=str(final_manifest),upload_manifest=str(upload_manifest),release_assets=len(upload_assets),staging=str(staging)),indent=2),flush=True)
    except Exception:
        print('Export failed; retained staging evidence at '+str(staging),file=sys.stderr,flush=True)
        raise


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--workspace-root',type=Path,help='Exact-resume workspace containing outputs/kernelbench-rl and work/models')
    result.add_argument('--work-root',type=Path,help='Scratch/staging directory; defaults to workspace/work and is created if absent')
    result.add_argument('--repo',type=Path,help='Defaults to the enclosing repository, or workspace/outputs/kernelbench-rl')
    result.add_argument('--output-root',type=Path,help='Defaults to workspace/outputs')
    result.add_argument('--source-commit',required=True)
    result.add_argument('--study-source-commit',default=STUDY_COMMIT)
    result.add_argument('--ci-url',required=True)
    result.add_argument('--run-dir',action='append',type=Path)
    result.add_argument('--protocol',type=Path)
    result.add_argument('--model-05b-dir',type=Path,help='Defaults to workspace/work/models/Qwen2.5-Coder-0.5B-Instruct')
    result.add_argument('--model-7b-dir',type=Path,help='Defaults to workspace/work/models/Qwen2.5-Coder-7B-Instruct')
    result.add_argument('--paper',type=Path,help='Defaults to output-root/KernelBench-RL-Paper.pdf')
    result.add_argument('--report',type=Path,help='Defaults to output-root/RETRY_FINAL_REPORT.html')
    result.add_argument('--site',type=Path,help='Defaults to output-root/kernelbench-rl-site/dist')
    result.add_argument('--site-url',default=None)
    result.add_argument('--verification-file',type=Path,help='Completed post-commit artifact/checkout proof with matching final source SHA')
    result.add_argument('--include-historical',action='store_true',help='Verify/list historical ZIPs without modifying')
    result.add_argument('--max-asset-bytes',type=int,default=MAX_ASSET_BYTES)
    result.add_argument('--rebuild',action='store_true',help='Replace only byte-verified prior retry-builder outputs')
    result.add_argument('--validate-only',action='store_true')
    return result


if __name__=='__main__': build(parser().parse_args())
