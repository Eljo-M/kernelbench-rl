"""CPU-only checks of saved publication bytes; optional new local clone round trip.

No model tensors, downloads, network requests or GPU calls. Clone mode writes
only a new directory under work; it never removes or changes an existing clone.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import uuid

STUDY_COMMIT='cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e'
MODELS={'Qwen/Qwen2.5-Coder-0.5B-Instruct':'ea3f2471cf1b1f0db85067f1ef93848e38e88c25',
        'Qwen/Qwen2.5-Coder-7B-Instruct':'c03e6d358207e414f1eca0bb1891e29f1db0e242'}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(value,message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def check_digest(path,expected,label):
    require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected) is not None,
            label+': missing SHA-256')
    require(path.is_file(),label+': file missing '+str(path))
    actual=sha(path)
    require(actual==expected,label+': saved bytes differ '+str(path))
    return dict(path=str(path),bytes=path.stat().st_size,sha256=actual)


def within(root,name):
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name,
            'Nonportable relative artifact path')
    pure=PurePosixPath(name)
    require(not pure.is_absolute() and '..' not in pure.parts,'Artifact traversal refused')
    path=(root/name).resolve()
    require(path.is_relative_to(root.resolve()),'Artifact path escapes its root')
    return path


def complete(path,cfg_hash):
    data=read(path)
    require(data.get('completed_utc') and not data.get('error'),'Incomplete stage '+str(path))
    require(data.get('config_hash')==cfg_hash and data.get('source_commit')==STUDY_COMMIT,
            'Stage source/config mismatch '+str(path))
    require(data.get('source_code_dirty') is False,'Stage lacks clean-source attestation '+str(path))
    return data


def frozen_proofs(directory,cfg,config_hash):
    initial_path=directory/'frozen_base_initial.json'
    original_path=directory/'frozen_base_verified.json'
    initial=read(initial_path); original=read(original_path)
    identity={key:cfg[key] for key in ('model_name_or_path','model_revision','training_mode')}
    for data in (initial,original):
        require(data.get('config_hash')==config_hash and data.get('model_identity')==identity,
                'Frozen-base identity mismatch')
    before=initial.get('before')
    require(isinstance(before,dict) and re.fullmatch('[0-9a-f]{64}',before.get('sha256','')) is not None,
            'Initial frozen-base descriptor missing')
    require(original.get('training_completed') is True and original.get('verified') is True
            and original.get('before')==before and original.get('after')==before,
            'Original completed frozen-base proof failed')
    original_sha=sha(original_path); sidecars=[]
    for path in sorted(directory.glob('frozen_base_verified.resume-*.json')):
        try:
            data=read(path)
        except (OSError,ValueError):
            continue  # Interrupted evidence stays preserved, never counted.
        if not isinstance(data,dict) or data.get('training_completed') is not True:
            continue
        require(data.get('resumed') is True and data.get('verified') is True
                and data.get('original_proof_preserved') is True,'Completed frozen-base resume proof failed')
        require(data.get('config_hash')==config_hash and data.get('model_identity')==identity,
                'Frozen-base resume model/config mismatch')
        require(data.get('original_proof_file')==original_path.name
                and data.get('initial_evidence_file')==initial_path.name,
                'Frozen-base sidecar references changed')
        require(data.get('original_proof_sha256')==original_sha,'Original frozen proof bytes differ from sidecar digest')
        require(data.get('before')==before and data.get('after')==before,'Frozen base changed across resume')
        sidecars.append(dict(file=path.name,sha256=sha(path),original_proof_sha256=original_sha))
    require(sidecars,'No completed original-preserving frozen-base resume sidecar')
    return dict(initial_sha256=sha(initial_path),original_sha256=original_sha,sidecars=sidecars)


def verify_upstream(root):
    directory=root/'experiments/retry_v2'
    integrity_path=directory/'upstream_model_integrity.json'
    expected_path=directory/'upstream_expected_metadata.json'
    integrity=read(integrity_path); expected=read(expected_path)
    require(integrity.get('schema_version')=='kernelbench_upstream_model_integrity_v1',
            'Unexpected upstream integrity schema')
    require(integrity.get('verified') is True and integrity.get('metadata_only') is False
            and integrity.get('all_local_weight_hashes_verified') is True
            and integrity.get('small_model_files_verified') is True
            and integrity.get('completed_utc'),'Full upstream snapshot verification is incomplete')
    check_digest(expected_path,integrity.get('expected_metadata_sha256'),'Copied official expected metadata')
    require(expected.get('schema_version')=='kernelbench_upstream_model_expected_v1'
            and expected.get('request',{}).get('token') is False
            and expected.get('request',{}).get('files_metadata') is True,
            'Expected metadata is not the captured unauthenticated official file inventory')
    metadata={item['model_id']:item for item in expected['models']}
    actual={item['model_id']:item for item in integrity['models']}
    require(set(metadata)==set(actual)==set(MODELS),'Upstream model inventory differs')
    checked=[]
    for model_id,revision in MODELS.items():
        proof=actual[model_id]; official=metadata[model_id]
        require(proof.get('requested_revision')==proof.get('resolved_revision')==revision
                and official.get('requested_revision')==official.get('resolved_revision')==revision,
                'Pinned upstream revision differs')
        require(proof.get('verified') is True and proof.get('weight_shards_hash_verified') is True
                and proof.get('weight_shard_inventory_verified') is True,'Model weight proof incomplete')
        expected_files={item['path']:item for item in official['files']}
        weights={name for name in expected_files if name.endswith('.safetensors')}
        rows=proof.get('checked_files',[]); files={item['path']:item for item in rows}
        require(len(files)==len(rows),'Duplicate upstream file verification identity')
        require({name for name in files if name.endswith('.safetensors')}==weights
                and set(proof.get('expected_weight_shards',[]))==weights
                and proof.get('weight_shards_verified')==len(weights),'Missing/extra verified weight shard')
        total=0
        for name,item in files.items():
            require(name in expected_files and item.get('verified') is True,'Unverified/unregistered upstream file')
            entry=expected_files[name]
            if entry.get('lfs_sha256'):
                require(item.get('sha256')==entry['lfs_sha256'] and item.get('bytes')==entry['lfs_bytes'],
                        'Recorded LFS digest/size differs from official metadata')
            else:
                require(item.get('actual_git_blob_sha1')==entry['git_blob_sha1'] and item.get('bytes')==entry['bytes'],
                        'Recorded small-file Git blob digest/size differs from official metadata')
            if name in weights:
                total+=item['bytes']
        require(proof.get('weight_bytes_verified')==total,'Verified model-byte total differs')
        require({'config.json','tokenizer_config.json','LICENSE'}<=set(files),'Core configuration/license proof missing')
        checked.append(dict(model_id=model_id,resolved_revision=revision,weight_shards=len(weights),
                            recorded_verified_weight_bytes=total))
    return dict(integrity_file_sha256=sha(integrity_path),expected_metadata_sha256=sha(expected_path),models=checked,
                scope='Consistency of recorded completed snapshot hash evidence; model files are not rehashed by this checker')


def choose_file(root,explicit,preferred,pattern,label):
    if explicit:
        path=explicit.resolve()
    elif (root/preferred).is_file():
        path=root/preferred
    else:
        candidates=sorted((root/'paper').rglob(pattern))
        require(len(candidates)==1,'Specify '+label+' explicitly; expected one committed paper artifact')
        path=candidates[0]
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()),label+' must be within the publication source tree')
    return path.resolve()


def verify_saved_bytes(root,paper_pdf=None,plot_metadata=None):
    root=root.resolve(); run_checks=[]
    for name in ('laptop','laptop_7b'):
        directory=root/'experiments/retry_v2'/name; cfg=read(directory/'config.json')
        env=read(directory/'environment.json'); config_hash=env.get('config_hash')
        require(isinstance(config_hash,str) and config_hash,'Training configuration hash missing')
        env=complete(directory/'environment.json',config_hash)
        main=complete(directory/'heldout_environment.json',config_hash)
        sft=complete(directory/'sft_only_environment.json',config_hash)
        resume=complete(directory/'resume_verified.json',config_hash)
        require(env.get('steps')==cfg['max_steps'] and env.get('sft_optimizer_steps')==cfg['sft_epochs']*len(cfg['train_ops']),
                'Training schedule incomplete')
        require(main.get('checkpoint_weights_sha256')==env.get('trainable_weights_sha256')
                and main.get('checkpoint_step')==cfg['max_steps'],'Training/evaluation checkpoint identity differs')
        require(resume.get('additional_training_steps')==0 and resume.get('restored_model_checksum_verified') is True
                and resume.get('restored_optimizer_and_rng') is True,'Completed zero-step reload evidence missing')
        outputs=[check_digest(directory/'heldout.jsonl',main.get('output_sha256'),name+' heldout output'),
                 check_digest(directory/'sft_only.jsonl',sft.get('output_sha256'),name+' SFT-only output')]
        frozen=frozen_proofs(directory,cfg,config_hash) if cfg['training_mode']=='qlora_7b_nf4' else None
        run_checks.append(dict(name=name,study_source_commit=env['source_commit'],config_hash=config_hash,
                               outputs=outputs,frozen_base=frozen))
    plot_path=choose_file(root,plot_metadata,'paper/paper-assets/sft_training.json','sft_training.json','--plot-metadata')
    plot=read(plot_path); png=plot_path.with_suffix('.png')
    figure=check_digest(png,plot.get('figure_sha256'),'SFT figure PNG')
    require({track['name'] for track in plot['tracks']}=={'laptop','laptop_7b'} and len(plot['tracks'])==2,'SFT figure track inventory differs')
    for track in plot['tracks']:
        directory=root/'experiments/retry_v2'/track['name']
        check_digest(directory/'sft.jsonl',track.get('sft_log_sha256'),'SFT figure source log')
        check_digest(directory/'config.json',track.get('config_file_sha256'),'SFT figure source configuration')
    paper=choose_file(root,paper_pdf,'paper/KernelBench-RL-Paper.pdf','*.pdf','--paper-pdf')
    docs=root/'docs'; config=read(docs/'project-config.json')
    require(config.get('draft') is False,'Publication website is still a draft')
    docs_pdf=within(docs,config.get('paper_pdf_path'))
    paper_sha=sha(paper); check_digest(docs_pdf,paper_sha,'Published website/Paper PDF identity')
    upstream=verify_upstream(root)
    return dict(verified=True,runs=run_checks,plot=dict(metadata_path=str(plot_path),metadata_sha256=sha(plot_path),figure=figure),
        paper=dict(path=str(paper),docs_pdf=str(docs_pdf),bytes=paper.stat().st_size,sha256=paper_sha),upstream_model_evidence=upstream)


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args])


def source_state(root,expected):
    head=git(root,'rev-parse','HEAD').decode().strip()
    require(re.fullmatch('[0-9a-f]{40}',head) is not None,'Cannot resolve source HEAD')
    if expected:
        require(head==expected,'Source HEAD differs from --expected-commit')
    require(not git(root,'status','--porcelain=v1','--untracked-files=normal').strip(),'Publication source must be clean and fully committed')
    return head


def clone_check(source,work,commit,source_check):
    work.mkdir(parents=True,exist_ok=True)
    clone=work/('publication-clone-autocrlf-'+uuid.uuid4().hex)
    require(not clone.exists() and clone.resolve().is_relative_to(work.resolve()),'Clone target must be a new work directory')
    subprocess.run(['git','clone','--no-hardlinks','--config','core.autocrlf=true',str(source),str(clone)],check=True)
    require(source_state(clone,commit)==commit,'Clone checked out a different commit')
    require(git(clone,'config','--get','core.autocrlf').decode().strip()=='true','Clone does not use explicit core.autocrlf=true')
    names=[name.decode('utf-8') for name in git(source,'ls-files','-z','--','experiments','docs','paper').split(b'\0') if name]
    require(names,'No tracked publication/data files to compare')
    files=[]
    for name in names:
        original=within(source,name); copy=within(clone,name)
        require(original.is_file() and copy.is_file(),'Tracked file missing after clone: '+name)
        expected=sha(original); actual=sha(copy)
        # Compare committed blob bytes as well: a Windows CRLF checkout can
        # otherwise conceal an LF blob that breaks a Linux checkout's hashes.
        blob=git(source,'cat-file','blob',commit+':'+name)
        blob_sha=hashlib.sha256(blob).hexdigest()
        require(expected==actual==blob_sha,'Working/blob/autocrlf-clone bytes differ: '+name)
        files.append(dict(path=name,bytes=original.stat().st_size,sha256=expected,committed_blob_sha256=blob_sha,clone_sha256=actual))
    clone_saved=verify_saved_bytes(clone,
        paper_pdf=clone/Path(source_check['paper']['path']).relative_to(source),
        plot_metadata=clone/Path(source_check['plot']['metadata_path']).relative_to(source))
    require(source_state(source,commit)==commit,'Source changed during clone verification')
    return dict(verified=True,path=str(clone),core_autocrlf=True,tracked_files_compared=len(files),files=files,
                saved_byte_checks=clone_saved,no_deletions_performed=True)


def discover():
    for parent in Path(__file__).resolve().parents:
        if (parent/'rl_trainer/experiment_config.py').is_file():
            # Standalone clones use sibling scratch/output directories so the
            # proof remains outside the source tree even with default options.
            workspace=parent.parent.parent if parent.parent.name=='outputs' else parent.parent
            return workspace,parent
    workspace=Path(__file__).resolve().parent.parent
    return workspace,workspace/'outputs/kernelbench-rl'


def main(argv=None):
    workspace,default_repo=discover()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=default_repo)
    parser.add_argument('--work-root',type=Path,default=workspace/'work')
    parser.add_argument('--paper-pdf',type=Path)
    parser.add_argument('--plot-metadata',type=Path)
    parser.add_argument('--expected-commit')
    parser.add_argument('--clone-check',action='store_true')
    parser.add_argument('--output',type=Path,default=workspace/'outputs/SOURCE_CHECKOUT_VERIFICATION.json')
    args=parser.parse_args(argv)
    if args.clone_check and not args.expected_commit:
        parser.error('--clone-check requires a frozen --expected-commit')
    if args.expected_commit and re.fullmatch('[0-9a-f]{40}',args.expected_commit) is None:
        parser.error('--expected-commit must be a full lowercase SHA')
    root=args.repo.resolve(); output=args.output.resolve()
    require(not output.is_relative_to(root),'Verification output must be outside the repository')
    require(not output.exists(),'Preserve existing verification evidence; choose a new output path')
    result=dict(schema_version='kernelbench_publication_byte_verification_v1',verified=False,started_utc=now(),
                source_repository=str(root),clone_check_performed=False)
    try:
        commit=source_state(root,args.expected_commit)
        result.update(source_commit=commit,working_tree_clean=True)
        saved=verify_saved_bytes(root,args.paper_pdf,args.plot_metadata)
        result['saved_byte_checks']=saved
        if args.clone_check:
            result['clone_check']=clone_check(root,args.work_root.resolve(),commit,saved)
            result['clone_check_performed']=True
        require(source_state(root,commit)==commit,'Source changed during saved-byte verification')
        result.update(verified=True,completed_utc=now(),scope='Saved artifact hash consistency and, when requested, exact committed-blob/autocrlf checkout byte preservation; no model/GPU computations')
    except Exception as exc:
        result.update(error=type(exc).__name__+': '+str(exc),completed_utc=now())
        raise
    finally:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
        print(json.dumps(dict(verified=result['verified'],output=str(output),clone_check_performed=result['clone_check_performed'])),flush=True)


if __name__=='__main__':
    main()
