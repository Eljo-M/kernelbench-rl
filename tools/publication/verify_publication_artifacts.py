"""CPU-only checks of saved publication bytes; optional new local clone round trip.

No model tensors, downloads, network requests or GPU calls. Clone mode writes
only a new directory under work; it never removes or changes an existing clone.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path, PurePosixPath
import re
import random
import subprocess
import sys
import uuid

STUDY_COMMIT='cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e'
V3_STUDY_COMMIT='2c174abe268495159c10d538c92d071e7733828d'
V3_CONFIG_HASH='0dfb0a14abad3067'
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


def registered_v3_study(repo,*,expected_source=V3_STUDY_COMMIT,expected_config=V3_CONFIG_HASH):
    """Resolve the optional independent study; an unfinished v3 cannot disappear.

    Registration is written after the preregistration commit, avoiding a
    self-referential commit hash. Completed stage checks remain validate_run's
    responsibility. Production uses the frozen defaults; explicit expectations
    are provided only for clearly synthetic test fixtures. This function does
    not import models or inspect tensors.
    """
    repo=Path(repo).resolve(); root=repo/'experiments/retry_v3'
    if not root.exists():
        return None
    require(root.is_dir() and not root.is_symlink(),'Invalid retry_v3 evidence directory')
    run_dir=root/'laptop_7b'; provenance_path=root/'source_provenance.json'
    require(run_dir.is_dir() and (run_dir/'config.json').is_file(),
            'V3 is present but pending: completed independent laptop_7b study required')
    require(provenance_path.is_file(),'V3 source registration is missing; publication must wait')
    provenance=read(provenance_path)
    require(provenance.get('schema_version')=='kernelbench_study_source_provenance_v1',
            'Unexpected v3 source-provenance schema')
    source_commit=provenance.get('study_source_commit','')
    require(isinstance(source_commit,str) and re.fullmatch('[0-9a-f]{40}',source_commit) is not None,
            'V3 requires its exact full preregistration source commit')
    require(source_commit==expected_source,'V3 source differs from the frozen study identity')
    protocol=repo/'experiments/retry_v3/protocol.json'
    require(provenance.get('protocol_path')==protocol.relative_to(repo).as_posix(),
            'V3 provenance references a different protocol')
    require(protocol.is_file() and provenance.get('protocol_sha256')==sha(protocol),
            'V3 protocol bytes differ from the published registration')
    registered=read(protocol); hashes=provenance.get('config_hashes')
    require(isinstance(hashes,dict) and set(hashes)=={'laptop_7b'}
            and hashes==registered.get('config_hashes')
            and re.fullmatch('[0-9a-f]{16}',hashes.get('laptop_7b','')) is not None,
            'V3 provenance/protocol configuration identities differ')
    require(hashes['laptop_7b']==expected_config,'V3 configuration differs from the frozen study identity')
    ci_url=provenance.get('source_ci_url')
    require(provenance.get('published_before_training') is True and isinstance(ci_url,str)
            and re.fullmatch(r'https://github\.com/Eljo-M/kernelbench-rl/actions/runs/[0-9]+',
                             ci_url) is not None,
            'V3 requires recorded pre-training publication and source CI evidence')
    if 'source_ci_receipt_path' in provenance or 'source_ci_receipt_sha256' in provenance:
        receipt_path=root/'source_ci_verified.json'
        require(provenance.get('source_ci_receipt_path')==receipt_path.relative_to(repo).as_posix()
                and receipt_path.is_file() and provenance.get('source_ci_receipt_sha256')==sha(receipt_path),
                'V3 source CI receipt bytes/path differ from registration')
        receipt=read(receipt_path)
        require(receipt.get('authenticated_login')=='Eljo-M'
                and receipt.get('requested_source_commit')==source_commit,
                'V3 source CI receipt account/commit differs')
        matches=[run for run in receipt.get('runs',[]) if run.get('html_url')==ci_url]
        require(len(matches)==1 and matches[0].get('head_sha')==source_commit
                and matches[0].get('status')=='completed' and matches[0].get('conclusion')=='success'
                and provenance.get('source_ci_conclusion')=='success',
                'V3 source CI is not a completed successful run for its exact source')
    cfg=read(run_dir/'config.json')
    schedule={key:cfg[key] for key in ('max_steps','sft_epochs','group_size')}
    require(registered.get('run_schedules',{}).get('laptop_7b')==schedule,
            'V3 configuration schedule differs from preregistration')
    require(cfg.get('training_mode')=='qlora_7b_nf4' and not cfg.get('resume_from'),
            'Independent v3 must initialize from its pinned base, not resume v2')
    require(cfg.get('heldout_ops')==registered.get('fresh_heldout_ops')
            and cfg.get('development_eval_ops')==registered.get('development_regressions'),
            'V3 fresh/development cohorts differ from preregistration')
    require(cfg.get('train_ops')==registered.get('train_ops'),
            'V3 training operators/order differ from preregistration')
    adapter=registered.get('adapter',{})
    require(cfg.get('lora_r')==adapter.get('rank') and cfg.get('lora_alpha')==adapter.get('alpha')
            and cfg.get('lora_target_modules')==adapter.get('target_modules'),
            'V3 adapter architecture differs from preregistration')
    return dict(run_dir=run_dir,protocol=protocol,source_commit=source_commit,
                config_hash=hashes['laptop_7b'],study_key='v3/laptop_7b',
                provenance_path=provenance_path,provenance=provenance)


def complete(path,cfg_hash,source_commit=STUDY_COMMIT):
    data=read(path)
    require(data.get('completed_utc') and not data.get('error'),'Incomplete stage '+str(path))
    require(data.get('config_hash')==cfg_hash and data.get('source_commit')==source_commit,
            'Stage source/config mismatch '+str(path))
    require(data.get('source_code_dirty') is False,'Stage lacks clean-source attestation '+str(path))
    return data


def saved_config_hash(cfg):
    """Hash the fully serialized ExperimentConfig without importing model code.

    These conditional exclusions are the frozen config_hash contract, including
    its historical zero-SFT compatibility. Saved run configs contain defaults.
    """
    fields=dict(cfg)
    for key in ('max_steps','checkpoint_every','checkpoint_dir','resume_from','output_jsonl'):
        fields.pop(key)
    if cfg['prompt_version']=='contract_v1':
        fields.pop('prompt_version')
    if cfg['training_reward_mode']=='legacy_shaped':
        fields.pop('training_reward_mode')
    if cfg['sft_epochs']==0:
        for key in ('sft_epochs','sft_learning_rate','sft_max_tokens','sft_output_jsonl'):
            fields.pop(key)
    if cfg['onpolicy_logp_tolerance']==0.001:
        fields.pop('onpolicy_logp_tolerance')
    if not cfg['development_eval_ops']:
        fields.pop('development_eval_ops')
    if not cfg['qlora_cpu_offload']:
        fields.pop('qlora_cpu_offload')
    return hashlib.sha256(json.dumps(fields,sort_keys=True,allow_nan=False).encode('utf-8')).hexdigest()[:16]


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


def timestamp(value,label):
    require(isinstance(value,str) and value,label+': UTC timestamp missing')
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.tzinfo is not None,label+': timezone missing')
    return result


def verify_upstream(root,integrity_relative='experiments/retry_v2/upstream_model_integrity.json',not_before=None):
    directory=root/'experiments/retry_v2'
    integrity_path=within(root,integrity_relative)
    expected_path=directory/'upstream_expected_metadata.json'
    integrity=read(integrity_path); expected=read(expected_path)
    require(integrity.get('schema_version')=='kernelbench_upstream_model_integrity_v1',
            'Unexpected upstream integrity schema')
    require(integrity.get('verified') is True and integrity.get('metadata_only') is False
            and integrity.get('all_local_weight_hashes_verified') is True
            and integrity.get('small_model_files_verified') is True
            and integrity.get('completed_utc'),'Full upstream snapshot verification is incomplete')
    started=timestamp(integrity.get('started_utc'),'Upstream integrity start')
    finished=timestamp(integrity.get('completed_utc'),'Upstream integrity completion')
    require(finished>=started,'Upstream integrity timestamps are reversed')
    if not_before is not None:
        require(started>=not_before,'V3 requires a new full upstream snapshot hash pass after its completed stages')
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
    return dict(integrity_path=integrity_relative,integrity_file_sha256=sha(integrity_path),
                expected_metadata_sha256=sha(expected_path),models=checked,
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


def scientific_identity(value):
    """Ignore only the three declared location fields; retain JSON value types."""
    value=copy.deepcopy(value)
    for path in (('protocol_path',),('environment','run_dir'),
                 ('strict_main_comparison','environment','run_dir')):
        parent=value
        for key in path[:-1]:
            parent=parent.get(key,{}) if isinstance(parent,dict) else {}
        if isinstance(parent,dict) and path[-1] in parent:
            require(isinstance(parent[path[-1]],str),'Location-only metadata must remain a string')
            parent[path[-1]]='<location-only>'
    return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))


def plain(value):
    return ' '.join(str(value).split())


class PublishedHTML(HTMLParser):
    """Read structured fields/tables, without interpreting surrounding prose."""
    def __init__(self,text):
        super().__init__(convert_charrefs=True)
        self.meta={};self.ids={};self.h1=[];self.abstract=[];self.tables=[];self.sections=[]
        self.capture=None;self.abstract_stack=[];self.table=None;self.row=None;self.row_header=False
        self.feed(text)

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='meta' and attrs.get('name'):
            require(attrs['name'] not in self.meta,'Duplicate publication metadata '+attrs['name'])
            self.meta[attrs['name']]=attrs.get('content','')
        if tag=='section':
            self.abstract_stack.append('abstract' in attrs.get('class','').split() or any(self.abstract_stack))
        kind=None
        if tag=='h1':kind='h1'
        elif tag=='h3':kind='h3'
        elif tag=='code' and self.sections and self.capture is None:kind='code'
        elif tag=='p' and attrs.get('id') in ('latest-hybrid-score','latest-hybrid-label'):kind=attrs['id']
        elif tag=='p' and any(self.abstract_stack):kind='abstract'
        elif tag=='table':
            require(self.table is None,'Nested report table refused')
            self.table=dict(caption='',headers=[],rows=[])
        elif tag=='caption':kind='caption'
        elif tag=='tr':self.row=[];self.row_header=False
        elif tag in ('th','td'):
            kind='cell';self.row_header=self.row_header or tag=='th'
        if kind:
            require(self.capture is None,'Overlapping structured publication field')
            self.capture=(kind,tag,[])

    def handle_data(self,data):
        if self.capture:self.capture[2].append(data)

    def handle_endtag(self,tag):
        if self.capture and self.capture[1]==tag:
            kind,_,parts=self.capture;value=plain(''.join(parts));self.capture=None
            if kind=='h1':self.h1.append(value)
            elif kind=='h3':self.sections.append(dict(label=value,codes=[]))
            elif kind=='code':self.sections[-1]['codes'].append(value)
            elif kind=='abstract':self.abstract.append(value)
            elif kind=='caption':self.table['caption']=value
            elif kind=='cell':self.row.append(value)
            else:
                require(kind not in self.ids,'Duplicate latest-study headline field')
                self.ids[kind]=value
        if tag=='section' and self.abstract_stack:self.abstract_stack.pop()
        elif tag=='tr' and self.row is not None:
            if self.row_header:self.table['headers'].append(self.row)
            else:self.table['rows'].append(self.row)
            self.row=None
        elif tag=='table' and self.table is not None:
            self.tables.append(self.table);self.table=None


def verify_published_metrics(root,studies,strict_reports=None):
    """Bind public derivatives to fresh strict metadata, without model execution.

    This is called only after completed-stage checks. Tests may supply clearly
    synthetic strict reports; production always recomputes them from raw runs.
    """
    if strict_reports is None:
        sys.path.insert(0,str(root))
        from eval.compare_retry import load_retry_run
        strict_reports={study['study_key']:load_retry_run(study['run_dir'],protocol_path=study['run_dir'].parent/'protocol.json')
                        for study in studies}
    manuscript_path=root/'paper/manuscript.json';manuscript=read(manuscript_path)
    require(str(manuscript.get('status','')).startswith('FINAL')
            and manuscript.get('pending_results',{}).get('required') is False,'Publication manuscript remains pending')
    metrics=manuscript.get('verified_retry_metrics',[])
    keys=[metric.get('study_key') for metric in metrics];expected=[study['study_key'] for study in studies]
    require(len(keys)==len(set(keys)) and set(keys)==set(expected)==set(strict_reports),
            'Manuscript/strict publication study inventory differs')
    by_key={metric['study_key']:metric for metric in metrics}
    labels={'v2/laptop':'v2 0.49B FP32','v2/laptop_7b':'v2 7B rank 8','v3/laptop_7b':'v3 7B rank 16'}
    for study in studies:
        key=study['study_key'];metric=by_key[key];directory=study['run_dir'];version,name=key.split('/')
        require(scientific_identity(metric['quality'])==scientific_identity(strict_reports[key]),
                'Manuscript quality differs from fresh strict records: '+key)
        require(metric.get('environment')==read(directory/'environment.json')
                and metric.get('config_snapshot')==read(directory/'config.json')
                and metric.get('sft_summary')==read(directory/'sft_summary.json'),
                'Manuscript source/config/training identity changed: '+key)
        require(metric.get('protocol_version')==version and metric.get('run_name')==name
                and metric.get('short_label')==labels[key]
                and metric['environment'].get('source_commit')==study['source_commit'],
                'Manuscript canonical study identity differs: '+key)
        counts=Counter(json.loads(line)['status'] for line in (directory/'train.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip())
        require(metric.get('training_status_counts')==dict(counts),'Manuscript training counts differ: '+key)
        for field in ('operational_pause','training_update_analysis'):
            path=directory.parent/(field+'.json')
            if path.exists():
                record=metric.get(field,{})
                require(record.get('path')==path.relative_to(root).as_posix() and record.get('sha256')==sha(path)
                        and record.get('record')==read(path),'Manuscript evidence binding differs: '+field)
    site=read(root/'docs/results.json').get('retry',{})
    require(site.get('status')=='completed' and site.get('published_results') is True,'Website retry evidence remains pending')
    runs=site.get('runs',[]);site_keys=[run.get('study_key') for run in runs]
    require(len(site_keys)==len(set(site_keys)) and set(site_keys)==set(keys),'Website/manuscript study inventory differs')
    methods={'base_pass1':'Base pass@1','base_best_of_8':'Base best-of-8',
             'sft_only_checkpoint':'SFT-only pass@1','trained_checkpoint':'Hybrid pass@1'}
    tables={spec['id']:spec for spec in manuscript['tables']}
    require(len(tables)==len(manuscript['tables']),'Duplicate manuscript table identity')
    table_ids={'fresh_heldout','inspected_development','paired_fresh_status'}
    if 'v3/laptop_7b' in expected:table_ids|={'v3_fresh_heldout','v3_inspected_development'}
    require(set(tables)==table_ids,'Publication table inventory differs from registered studies')
    for run in runs:
        key=run['study_key'];metric=by_key[key];quality=strict_reports[key];env=metric['environment']
        fields=dict(status='completed',protocol_version=key.split('/')[0],source_commit=env['source_commit'],
                    config_hash=env['config_hash'],model_revision=env['model_revision'],hardware=env['gpu'],
                    training_mode=env['training_mode'],sft_updates=quality['sft_optimizer_steps'],
                    optimizer_updates=quality['grpo_optimizer_steps'])
        require(all(run.get(field)==value for field,value in fields.items()),'Website study/source/counters differ: '+key)
        for short,cohort in (('fresh','fresh_heldout'),('development','inspected_development')):
            actual=quality[cohort]
            rows=[dict(method=methods[method],passed=stats['pass_count'],total=stats['total_ops'],budget=stats['budget_k'])
                  for method,stats in actual['methods'].items() if stats.get('available') is not False]
            require(run.get('cohorts',{}).get(short)==rows and run.get('cohort_ops',{}).get(short)==actual['ops'],
                    'Website cohort/count/budget differs: '+key+' '+short)
        require(run.get('fresh_ops')==quality['fresh_heldout']['ops']
                and run.get('development_ops')==quality['inspected_development']['ops'],'Website cohort aliases differ: '+key)
    for version in ('v2','v3'):
        version_keys=[key for key in expected if key.startswith(version+'/')]
        if not version_keys:continue
        for cohort in ('fresh_heldout','inspected_development'):
            expected_rows=[]
            for key in version_keys:
                for method,stats in strict_reports[key][cohort]['methods'].items():
                    if stats.get('available') is False:continue
                    row=dict(method=methods[method],budget=stats['budget_k'],passes=f"{stats['pass_count']}/{stats['total_ops']}",
                             fast=f"{stats['fast_vs_eager_count']}/{stats['total_ops']}")
                    if version=='v2':row['track']=labels[key]
                    else:row['failures']=', '.join(f'{status}={count}' for status,count in stats['failure_counts_by_status'].items()) or 'None'
                    expected_rows.append(row)
            identity=cohort if version=='v2' else 'v3_'+cohort
            require(tables.get(identity,{}).get('rows')==expected_rows,'Manuscript table differs from strict metrics: '+identity)
    statuses={'PASS':'PASS','CHEAT':'POLICY','COMPILE_ERROR':'COMPILE','NUMERICAL_MISMATCH':'NUMERIC',
              'RUNTIME_ERROR':'RUNTIME','TIMEOUT':'TIMEOUT','OOM':'OOM'}
    paired=[]
    for key in (key for key in expected if key.startswith('v2/')):
        quality=strict_reports[key];short=labels[key]
        for op in quality['fresh_heldout']['ops']:
            row=dict(track=short,operator=op)
            for column,method in (('base','base_pass1'),('best','base_best_of_8'),('sft','sft_only_checkpoint'),('hybrid','trained_checkpoint')):
                found=[record for record in quality['fresh_heldout']['methods'][method]['per_operator'] if record['op_name']==op]
                require(len(found)==1,'Strict paired operator inventory differs')
                row[column]=statuses[found[0]['status']]
            paired.append(row)
    require(tables.get('paired_fresh_status',{}).get('rows')==paired,'Paired manuscript status table differs')
    paper=PublishedHTML((root/'docs/paper.html').read_text(encoding='utf-8'))
    require(paper.meta.get('citation_title')==manuscript['title'] and paper.h1==[plain(manuscript['title'])]
            and paper.meta.get('citation_author')==manuscript.get('author',{}).get('display_name')=='Eljo Mustafaj'
            and paper.meta.get('citation_publication_date')==manuscript['publication_date'].replace('-','/'),
            'Paper landing title/author/date differs from manuscript')
    require(plain(' '.join(paper.abstract))==plain(manuscript['abstract']),'Paper landing abstract differs from manuscript')
    index=PublishedHTML((root/'docs/index.html').read_text(encoding='utf-8'))
    latest='v3/laptop_7b' if 'v3/laptop_7b' in by_key else 'v2/laptop_7b'
    score=strict_reports[latest]['fresh_heldout']['methods']['trained_checkpoint']
    require(index.ids.get('latest-hybrid-score')==f"{score['pass_count']} / {score['total_ops']}"
            and index.ids.get('latest-hybrid-label')==latest.split('/')[0]+' 7B hybrid fresh held-out passes · one attempt',
            'Website latest-study headline differs from actual latest strict metrics')
    report=PublishedHTML((root/'docs/report.html').read_text(encoding='utf-8'))
    expected_tables=[dict(caption=plain(spec['caption']),headers=[[plain(column['label']) for column in spec['columns']]],
                         rows=[[plain(row[column['key']]) for column in spec['columns']] for row in spec['rows']])
                     for spec in manuscript['tables']]
    require(report.tables==expected_tables,'HTML report tables differ from validated manuscript tables')
    expected_sections=[dict(label=plain(by_key[key]['label']),codes=[by_key[key]['environment'][field] for field in
                         ('source_commit','config_hash','trainable_weights_sha256')]) for key in expected]
    require(report.sections==expected_sections,'HTML report study/source/checkpoint identities differ')
    return dict(verified=True,studies=expected,manuscript_sha256=sha(manuscript_path),
                website_results_sha256=sha(root/'docs/results.json'),report_sha256=sha(root/'docs/report.html'),
                scope='Fresh strict metrics matched manuscript/site structured data, report tables/identities and landing metadata/headline; no free-prose equivalence claim')


def verify_pdf_manuscript_binding(pdf,manuscript):
    # ReportLab writes this ASCII /Info subject uncompressed. Match the unique
    # marker, not arbitrary hex strings elsewhere in the PDF. This is a stale
    # render guard for the trusted builder, not a signature/hostile-PDF proof.
    matches=re.findall(rb'/Subject\s*\(KernelBench-RL manuscript SHA-256: ([0-9a-f]{64})\)',pdf.read_bytes())
    require(len(matches)==1 and matches[0].decode('ascii')==sha(manuscript),
            'PDF input manuscript SHA-256 missing, duplicated or stale; rerender the exact committed manuscript')
    return dict(verified=True,input_manuscript_sha256=matches[0].decode('ascii'),
                scope='Unique trusted-renderer input digest equals exact saved manuscript bytes; no cryptographic signature claim')


def verify_followup_plot(root,v3,required=False):
    """Bind the published v3 figure using committed metadata, never tensors.

    Checkpoint JSON/tensors are distributed separately. The SFT environment and
    summary bind its trainable identity; the original metadata-file digest is
    retained as provenance without claiming that CI reopened that excluded file.
    """
    paper=root/'paper'
    sidecars=sorted(paper.rglob('sft_followup.json'))
    figures=sorted(paper.rglob('sft_followup.png'))
    if not sidecars and not figures and not required:
        return None
    require(v3 is not None,'A follow-up figure requires the registered v3 study')
    require(len(sidecars)==len(figures)==1 and sidecars[0].with_suffix('.png')==figures[0],
            'V3 publication requires exactly one paired sft_followup JSON/PNG')
    path=sidecars[0]; plot=read(path); directory=v3['run_dir']
    cfg=read(directory/'config.json'); summary=read(directory/'sft_summary.json')
    sft=read(directory/'sft_only_environment.json')
    expected=dict(schema_version='kernelbench_sft_followup_plot_v1',study_key=v3['study_key'],
                  config_hash=v3['config_hash'],source_commit=v3['source_commit'],
                  seed=cfg['seed'],epochs=cfg['sft_epochs'],examples_per_epoch=len(cfg['train_ops']),
                  completed_steps=cfg['sft_epochs']*len(cfg['train_ops']),
                  adapter_rank=cfg['lora_r'],adapter_alpha=cfg['lora_alpha'],
                  adapter_target_modules=cfg['lora_target_modules'],train_ops=cfg['train_ops'],
                  model_revision=cfg['model_revision'],sft_learning_rate=cfg['sft_learning_rate'],
                  inference_quality_claim=False,
                  sft_checkpoint_identity=sft['checkpoint_identity'],
                  sft_checkpoint_weights_sha256=sft['checkpoint_weights_sha256'])
    require(all(plot.get(key)==value for key,value in expected.items()),
            'Follow-up figure study/config/source/schedule/checkpoint identity differs')
    require(summary.get('ran') is True and summary.get('optimizer_step_count')==expected['completed_steps']
            and summary.get('weight_sha256_after')==sft['checkpoint_weights_sha256'],
            'Follow-up figure SFT summary/checkpoint identity differs')
    files=dict(figure_sha256=figures[0],config_file_sha256=directory/'config.json',
               sft_log_sha256=directory/'sft.jsonl',sft_summary_sha256=directory/'sft_summary.json',
               protocol_sha256=v3['protocol'],source_provenance_sha256=v3['provenance_path'],
               token_feasibility_sha256=directory.parent/'token_feasibility.json')
    bindings={key:check_digest(file,plot.get(key),'Follow-up figure '+key) for key,file in files.items()}
    require(plot.get('bytes')==figures[0].stat().st_size,'Follow-up figure recorded byte count differs')
    records=[json.loads(line) for line in (directory/'sft.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(len(records)==expected['completed_steps'] and summary.get('records')==records,
            'Follow-up figure requires the completed durable SFT log and summary')
    order=[]
    for epoch in range(1,cfg['sft_epochs']+1):
        indices=list(range(len(cfg['train_ops'])))
        random.Random(cfg['seed']+1000*epoch).shuffle(indices)
        order.extend((epoch,cfg['train_ops'][index]) for index in indices)
    require([row.get('step') for row in records]==list(range(1,expected['completed_steps']+1))
            and [(row.get('epoch'),row.get('op_name')) for row in records]==order,
            'Follow-up figure SFT rows differ from the configured seeded order')
    inventory=read(directory.parent/'token_feasibility.json')
    examples=inventory.get('examples',[])
    require(inventory.get('config_hash')==v3['config_hash'] and len(examples)==len(cfg['train_ops'])
            and Counter(item.get('op_name') for item in examples)==Counter(cfg['train_ops']),
            'Follow-up figure preflight target inventory differs')
    by_op={item['op_name']:item for item in examples}
    for row in records:
        require(all(row.get(key)==by_op[row['op_name']].get(key) for key in
                    ('prompt_tokens','completion_tokens','prompt_sha256','source_sha256')),
                'Follow-up figure SFT prompt/source/token identity differs from preflight')
    require(plot.get('observed_operator_order')==[row['op_name'] for row in records],
            'Follow-up figure operator order differs from the SFT log')
    means=[]
    for epoch in range(1,cfg['sft_epochs']+1):
        losses=[row['loss'] for row in records if row['epoch']==epoch]
        require(len(losses)==len(cfg['train_ops'])
                and all(type(value) in (int,float) and math.isfinite(value) and value>=0 for value in losses),
                'Follow-up figure has incomplete/nonfinite epoch evidence')
        means.append(math.fsum(losses)/len(losses))
    supplied=plot.get('epoch_means_nats_per_token')
    require(isinstance(supplied,list) and len(supplied)==len(means)
            and all(type(value) in (int,float) and math.isfinite(value)
                    and math.isclose(value,actual,rel_tol=1e-12,abs_tol=1e-12)
                    for value,actual in zip(supplied,means)),
            'Follow-up figure epoch means differ from its saved SFT log')
    checkpoint_digest=plot.get('sft_checkpoint_metadata_sha256')
    require(isinstance(checkpoint_digest,str) and re.fullmatch('[0-9a-f]{64}',checkpoint_digest) is not None,
            'Follow-up figure original checkpoint-metadata digest missing')
    return dict(verified=True,metadata_path=str(path),metadata_sha256=sha(path),bindings=bindings,
                sft_checkpoint_identity=sft['checkpoint_identity'],
                original_checkpoint_metadata_sha256=checkpoint_digest,
                scope='Committed figure/source bytes and recorded SFT checkpoint identity; excluded checkpoint metadata/tensors are not reopened')


def verify_saved_bytes(root,paper_pdf=None,plot_metadata=None,require_v3=False):
    root=root.resolve(); run_checks=[]; v3=registered_v3_study(root)
    require(not require_v3 or v3 is not None,'This publication requires the independent v3 study')
    studies=[dict(study_key='v2/'+name,run_dir=root/'experiments/retry_v2'/name,source_commit=STUDY_COMMIT)
             for name in ('laptop','laptop_7b')]
    if v3 is not None:
        studies.append(v3)
    v3_complete=None
    for study in studies:
        key=study['study_key']; directory=study['run_dir']; name=directory.name
        cfg=read(directory/'config.json')
        env=read(directory/'environment.json'); config_hash=env.get('config_hash')
        require(isinstance(config_hash,str) and config_hash,'Training configuration hash missing')
        require(saved_config_hash(cfg)==config_hash,'Saved configuration contents/hash differ: '+key)
        if key.startswith('v3/'):
            require(config_hash==study['config_hash'],'V3 training/provenance configuration mismatch')
        env=complete(directory/'environment.json',config_hash,study['source_commit'])
        main=complete(directory/'heldout_environment.json',config_hash,study['source_commit'])
        sft=complete(directory/'sft_only_environment.json',config_hash,study['source_commit'])
        resume=complete(directory/'resume_verified.json',config_hash,study['source_commit'])
        require(env.get('steps')==cfg['max_steps'] and env.get('sft_optimizer_steps')==cfg['sft_epochs']*len(cfg['train_ops']),
                'Training schedule incomplete')
        require(main.get('checkpoint_weights_sha256')==env.get('trainable_weights_sha256')
                and main.get('checkpoint_step')==cfg['max_steps'],'Training/evaluation checkpoint identity differs')
        require(resume.get('additional_training_steps')==0 and resume.get('restored_model_checksum_verified') is True
                and resume.get('restored_optimizer_and_rng') is True,'Completed zero-step reload evidence missing')
        for field in ('checkpoint_step','checkpoint_weights_sha256','optimizer_steps','sft_optimizer_steps'):
            require(resume.get(field)==main.get(field),'Reload/final-evaluation identity mismatch: '+key+' '+field)
        require(sft.get('checkpoint_step')==0 and sft.get('optimizer_steps')==0
                and sft.get('sft_optimizer_steps')==env.get('sft_optimizer_steps'),
                'SFT-only stage phase counters differ: '+key)
        outputs=[check_digest(directory/'heldout.jsonl',main.get('output_sha256'),key+' heldout output'),
                 check_digest(directory/'sft_only.jsonl',sft.get('output_sha256'),key+' SFT-only output')]
        frozen=frozen_proofs(directory,cfg,config_hash) if cfg['training_mode']=='qlora_7b_nf4' else None
        if key.startswith('v3/'):
            require(read(directory/'frozen_base_verified.json').get('resumed') is False,
                    'V3 original frozen-base proof does not attest fresh initialization')
            v3_complete=max(timestamp(stage['completed_utc'],key+' completed stage') for stage in (env,main,sft,resume))
        run_checks.append(dict(name=name,study_key=key,study_source_commit=env['source_commit'],config_hash=config_hash,
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
    v3_upstream=verify_upstream(root,'experiments/retry_v3/upstream_model_integrity.json',v3_complete) if v3 is not None else None
    followup=verify_followup_plot(root,v3,required=require_v3)
    publication=verify_published_metrics(root,studies)
    pdf_input=verify_pdf_manuscript_binding(paper,root/'paper/manuscript.json') if require_v3 else None
    return dict(verified=True,runs=run_checks,plot=dict(metadata_path=str(plot_path),metadata_sha256=sha(plot_path),figure=figure),
        paper=dict(path=str(paper),docs_pdf=str(docs_pdf),bytes=paper.stat().st_size,sha256=paper_sha),upstream_model_evidence=upstream,
        v3_source_provenance=dict(path=str(v3['provenance_path']),sha256=sha(v3['provenance_path']),
                                  source_commit=v3['source_commit']) if v3 is not None else None,
        v3_upstream_model_evidence=v3_upstream,v3_required=require_v3,followup_plot=followup,
        publication_metrics=publication,pdf_manuscript_binding=pdf_input)


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args])


def source_state(root,expected):
    head=git(root,'rev-parse','HEAD').decode().strip()
    require(re.fullmatch('[0-9a-f]{40}',head) is not None,'Cannot resolve source HEAD')
    if expected:
        require(head==expected,'Source HEAD differs from --expected-commit')
    require(not git(root,'status','--porcelain=v1','--untracked-files=normal').strip(),'Publication source must be clean and fully committed')
    v3=registered_v3_study(root)
    if v3 is not None:
        subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',v3['source_commit'],head],check=True)
        protocol_name=v3['protocol'].relative_to(root).as_posix()
        blob=git(root,'cat-file','blob',v3['source_commit']+':'+protocol_name)
        require(hashlib.sha256(blob).hexdigest()==v3['provenance']['protocol_sha256'],
                'V3 registered protocol differs from its preregistration committed bytes')
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
        plot_metadata=clone/Path(source_check['plot']['metadata_path']).relative_to(source),
        require_v3=source_check.get('v3_required',False))
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
    parser.add_argument('--require-v3',action='store_true',help='Refuse a v2-only snapshot; final v3 publication must include its completed independent study')
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
        if args.require_v3:
            require(registered_v3_study(root) is not None,'This publication requires the independent v3 study')
        commit=source_state(root,args.expected_commit)
        result.update(source_commit=commit,working_tree_clean=True)
        saved=verify_saved_bytes(root,args.paper_pdf,args.plot_metadata,require_v3=args.require_v3)
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
