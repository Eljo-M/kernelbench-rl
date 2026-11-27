"""Verify materialized public model snapshots without Torch, downloads or cache writes.

Metadata/small-file checks are a separate mode. Full safetensors hashing requires
the explicit --verify-local flag and should run after active model/GPU studies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import quote
import uuid


MODELS=(
    ('Qwen/Qwen2.5-Coder-0.5B-Instruct','ea3f2471cf1b1f0db85067f1ef93848e38e88c25'),
    ('Qwen/Qwen2.5-Coder-7B-Instruct','c03e6d358207e414f1eca0bb1891e29f1db0e242'),
)
RUNTIME_FILES={'config.json','generation_config.json','tokenizer_config.json','tokenizer.json',
               'vocab.json','vocab.txt','merges.txt','special_tokens_map.json','added_tokens.json',
               'tokenizer.model','model.safetensors.index.json'}


def layout():
    for parent in Path(__file__).resolve().parents:
        if (parent/'rl_trainer/experiment_config.py').is_file() and (parent/'kernels/op_registry.py').is_file():
            workspace=parent.parent.parent if parent.parent.name=='outputs' else parent
            return workspace,parent
    workspace=Path(__file__).resolve().parent.parent
    return workspace,workspace/'outputs/kernelbench-rl'


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_exclusive(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),'Refusing to overwrite prior evidence: '+str(path))
    temporary=path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False)
        stream.write('\n')
    # Windows rename is atomic and refuses an existing destination. POSIX link
    # provides the same no-clobber behavior without rename's overwrite semantics.
    import os
    if os.name=='nt':
        os.rename(temporary,path)
    else:
        os.link(temporary,path)
        temporary.unlink()


def safe_relative(name):
    require(isinstance(name,str) and bool(name),'Missing upstream filename')
    path=PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name,
            'Unsafe upstream filename: '+name)
    return path


def get_field(value,key):
    return value.get(key) if isinstance(value,dict) else getattr(value,key,None)


def fetch_expected():
    # Installed huggingface_hub 0.36.2 hf_api.py:2591-2663 documents this
    # exact method. token=False disables use of stored/implicit credentials.
    from huggingface_hub import HfApi
    api=HfApi(endpoint='https://huggingface.co',token=False)
    models=[]
    for model_id,revision in MODELS:
        fetched=now()
        info=api.model_info(model_id,revision=revision,files_metadata=True,token=False,timeout=60)
        require(info.sha==revision,'Official API resolved an unexpected revision for '+model_id)
        require(info.id==model_id,'Official API resolved an unexpected model identity')
        files=[]
        for sibling in info.siblings or []:
            name=sibling.rfilename; safe_relative(name)
            lfs=sibling.lfs
            files.append(dict(path=name,bytes=sibling.size,git_blob_sha1=sibling.blob_id,
                lfs_sha256=get_field(lfs,'sha256') if lfs is not None else None,
                lfs_bytes=get_field(lfs,'size') if lfs is not None else None,
                resolve_url=f'https://huggingface.co/{model_id}/resolve/{revision}/{quote(name,safe="/")}'))
        require(files and len({item['path'] for item in files})==len(files),'Missing/duplicate official file metadata')
        models.append(dict(model_id=model_id,requested_revision=revision,resolved_revision=info.sha,
            fetched_utc=fetched,response_recorded_utc=now(),
            api_url=f'https://huggingface.co/api/models/{model_id}/revision/{revision}?blobs=true',
            model_url=f'https://huggingface.co/{model_id}/tree/{revision}',files=sorted(files,key=lambda item:item['path'])))
        print('Fetched public metadata: '+model_id+' @ '+revision,flush=True)
    return dict(schema_version='kernelbench_upstream_model_expected_v1',metadata_only=True,
        actual_local_weight_hashes_verified=False,fetched_utc=now(),authentication='token=False (public unauthenticated API)',
        client='huggingface_hub',client_version=importlib.metadata.version('huggingface_hub'),
        request=dict(files_metadata=True,token=False,endpoint='https://huggingface.co'),models=models)


def validate_expected(expected):
    require(expected.get('schema_version')=='kernelbench_upstream_model_expected_v1','Unknown metadata schema')
    models=expected.get('models',[])
    require(len(models)==2 and {(item.get('model_id'),item.get('resolved_revision')) for item in models}==set(MODELS),
            'Metadata does not match both exact preregistered model revisions')
    for model in models:
        require(model.get('requested_revision')==model['resolved_revision'],'Metadata requested/resolved revisions differ')
        require(model.get('api_url')==f'https://huggingface.co/api/models/{model["model_id"]}/revision/{model["resolved_revision"]}?blobs=true',
                'Metadata API origin differs from the official pinned endpoint')
        for item in model['files']:
            safe_relative(item['path'])


def hash_local(path,expected,*,chunk_bytes):
    start=now(); before=path.stat(); began=time.perf_counter()
    size=expected.get('lfs_bytes') if expected.get('lfs_sha256') else expected.get('bytes')
    require(type(size) is int and size>=0,'Official size metadata is missing for '+expected['path'])
    require(before.st_size==size,'Local/official size mismatch: '+str(path))
    sha=hashlib.sha256(); blob=None if expected.get('lfs_sha256') else hashlib.sha1()
    if blob is not None:
        blob.update(b'blob '+str(before.st_size).encode('ascii')+b'\0')
    read_bytes=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(chunk_bytes),b''):
            sha.update(block)
            if blob is not None:
                blob.update(block)
            read_bytes+=len(block)
    after=path.stat()
    require(read_bytes==size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
            'Local file changed during verification: '+str(path))
    actual_sha=sha.hexdigest(); actual_blob=blob.hexdigest() if blob is not None else None
    if expected.get('lfs_sha256'):
        require(re.fullmatch('[0-9a-f]{64}',expected['lfs_sha256']) is not None,'Invalid official LFS SHA256')
        require(actual_sha==expected['lfs_sha256'],'Official LFS SHA256 mismatch: '+str(path))
        comparison='official_lfs_sha256_and_size'
    else:
        require(re.fullmatch('[0-9a-f]{40}',expected.get('git_blob_sha1','')) is not None,
                'Official Git blob SHA1 unavailable for small file '+expected['path'])
        require(actual_blob==expected['git_blob_sha1'],'Official Git blob SHA1 mismatch: '+str(path))
        comparison='official_git_blob_sha1_and_size'
    return dict(path=expected['path'],local_path=str(path.resolve()),bytes=read_bytes,sha256=actual_sha,
        actual_git_blob_sha1=actual_blob,expected_git_blob_sha1=expected.get('git_blob_sha1'),
        expected_lfs_sha256=expected.get('lfs_sha256'),comparison=comparison,verified=True,
        resolve_url=expected['resolve_url'],verification_started_utc=start,verification_completed_utc=now(),
        seconds=time.perf_counter()-began,local_mtime_utc=datetime.fromtimestamp(before.st_mtime,timezone.utc).isoformat(),
        local_mtime_ns=before.st_mtime_ns)


def verify_model(model,local,*,full,chunk_bytes):
    require(local.is_dir(),'Local model directory missing: '+str(local))
    files={item['path']:item for item in model['files']}
    expected_weights={name for name in files if name.endswith('.safetensors')}
    require(expected_weights,'Official snapshot has no safetensors weights')
    actual_weights={path.relative_to(local).as_posix() for path in local.rglob('*.safetensors') if path.is_file()}
    require(actual_weights==expected_weights,
            f'Weight shard inventory mismatch for {model["model_id"]}; missing={sorted(expected_weights-actual_weights)}, extra={sorted(actual_weights-expected_weights)}')
    required={name for name in files if name in RUNTIME_FILES}
    require('config.json' in required and 'tokenizer_config.json' in required,'Official model/tokenizer configuration metadata missing')
    for name in required:
        require((local/name).is_file(),'Required local model/tokenizer file missing: '+name)
    actual_runtime={path.name for path in local.iterdir() if path.is_file() and path.name in RUNTIME_FILES}
    require(not (actual_runtime-set(files)),'Extra local runtime model/tokenizer files absent upstream: '+str(sorted(actual_runtime-set(files))))
    selected=sorted(name for name in files if not name.endswith('.safetensors') and (local/name).is_file())
    if full:
        selected+=sorted(expected_weights)
    checked=[]
    for name in selected:
        path=local/name
        require(path.resolve().is_relative_to(local.resolve()),'Local model link escapes materialized snapshot: '+str(path))
        # Metadata-only mode never opens a weight shard, even on a malformed URL.
        if not full:
            require(path.stat().st_size<=64*1024*1024,'Unexpectedly large nonweight model file; small-only mode refused')
        item=hash_local(path,files[name],chunk_bytes=chunk_bytes); checked.append(item)
        print('Verified '+model['model_id']+' / '+name+f' ({item["bytes"]} bytes)',flush=True)
    index=local/'model.safetensors.index.json'
    if index.is_file():
        declared=set(read_json(index).get('weight_map',{}).values())
        require(declared==expected_weights,'Safetensors index/shard inventory mismatch')
    return dict(model_id=model['model_id'],requested_revision=model['requested_revision'],resolved_revision=model['resolved_revision'],
        model_url=model['model_url'],api_url=model['api_url'],metadata_fetched_utc=model['fetched_utc'],
        local_directory=str(local.resolve()),weight_shard_inventory_verified=True,expected_weight_shards=sorted(expected_weights),
        weight_shards_hash_verified=full,weight_shards_verified=len(expected_weights) if full else 0,
        weight_bytes_verified=sum(item['bytes'] for item in checked if item['path'].endswith('.safetensors')),
        checked_files=checked,verified=full,small_files_verified=True,completed_utc=now())


def main(argv=None):
    default_workspace,default_repo=layout()
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--metadata-only',action='store_true',help='Public official metadata; no weight bytes opened')
    modes.add_argument('--verify-local',action='store_true',help='Explicit full sequential weight hash pass; run after active studies')
    parser.add_argument('--check-small-files',action='store_true',help='In metadata-only mode verify small local configuration/tokenizer/license files too')
    parser.add_argument('--workspace-root',type=Path)
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--work-root',type=Path)
    parser.add_argument('--model-05b-dir',type=Path)
    parser.add_argument('--model-7b-dir',type=Path)
    parser.add_argument('--metadata-input',type=Path,help='Reuse captured official expected metadata; no network call')
    parser.add_argument('--metadata-output',type=Path,help='New expected metadata file, restricted to work-root')
    parser.add_argument('--output',type=Path,help='Full verification default: repo/experiments/retry_v2/upstream_model_integrity.json')
    parser.add_argument('--chunk-bytes',type=int,default=8*1024*1024)
    args=parser.parse_args(argv)
    require(0<args.chunk_bytes<=64*1024*1024,'Hash chunk size must be 1..64 MiB')
    workspace=(args.workspace_root or default_workspace).resolve()
    repo=(args.repo or (workspace/'outputs/kernelbench-rl' if args.workspace_root else default_repo)).resolve()
    work=(args.work_root or workspace/'work').resolve(); work.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    if args.metadata_input:
        expected_path=args.metadata_input.resolve(); expected=read_json(expected_path)
    else:
        expected=fetch_expected()
        expected_path=(args.metadata_output or work/('upstream-model-expected-'+stamp+'.json')).resolve()
        require(expected_path.is_relative_to(work),'Expected metadata must be saved under work-root only')
        write_exclusive(expected_path,expected)
    validate_expected(expected)
    print('EXPECTED_METADATA '+str(expected_path),flush=True)
    if args.metadata_only and not args.check_small_files:
        return
    locals=[args.model_05b_dir or workspace/'work/models/Qwen2.5-Coder-0.5B-Instruct',
            args.model_7b_dir or workspace/'work/models/Qwen2.5-Coder-7B-Instruct']
    by_id={item['model_id']:item for item in expected['models']}
    started=now()
    models=[verify_model(by_id[model_id],local.resolve(),full=args.verify_local,chunk_bytes=args.chunk_bytes)
            for (model_id,_),local in zip(MODELS,locals)]
    payload=dict(schema_version='kernelbench_upstream_model_integrity_v1',verified=bool(args.verify_local),
        metadata_only=not args.verify_local,all_local_weight_hashes_verified=bool(args.verify_local),
        small_model_files_verified=True,started_utc=started,completed_utc=now(),
        expected_metadata_path=str(expected_path),expected_metadata_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest(),
        authentication='token=False (public unauthenticated official metadata)',
        scope='Post-run materialized local files matched against declared pinned official revisions; no model load/replay or earlier-process state is inferred.',
        models=models)
    output=(args.output or (repo/'experiments/retry_v2/upstream_model_integrity.json' if args.verify_local
                           else work/('upstream-model-small-files-'+stamp+'.json'))).resolve()
    if not args.verify_local:
        require(output.is_relative_to(work),'Metadata/small-only evidence must remain under work-root')
    write_exclusive(output,payload)
    print('INTEGRITY_EVIDENCE '+str(output),flush=True)


if __name__=='__main__':
    main()
