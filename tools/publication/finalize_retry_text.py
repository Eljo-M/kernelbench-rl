"""Derive publication text from complete, strictly validated retry artifacts."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def latest_hybrid_headline(index, version, passed, total):
    """Always display the latest completed study, including a worse outcome."""
    score=f'<p class="metric" id="latest-hybrid-score">{passed} / {total}</p>'
    label=f'<p id="latest-hybrid-label">{version} 7B hybrid fresh held-out passes · one attempt</p>'
    if 'id="latest-hybrid-score"' in index or 'id="latest-hybrid-label"' in index:
        index,n_score=re.subn(r'<p\b[^>]*\bid="latest-hybrid-score"[^>]*>.*?</p>',lambda match:score,index,flags=re.S)
        index,n_label=re.subn(r'<p\b[^>]*\bid="latest-hybrid-label"[^>]*>.*?</p>',lambda match:label,index,flags=re.S)
        if n_score!=1 or n_label!=1: raise ValueError('Latest-study headline IDs are missing or duplicated')
        return index
    pattern=r'<p class="metric">[^<]*</p>\s*<p>(?:historical trained-model held-out passes, per model|(?:corrected|v[23]) 7B hybrid fresh held-out passes · one attempt)</p>'
    index,count=re.subn(pattern,lambda match:score+label,index)
    if count!=1: raise ValueError('Cannot uniquely locate the historical/latest 7B headline')
    return index


def comparison_science(run):
    """Normalize only three declared location-only fields for clone relocation.

    Preserve field presence/types and every hash, source/config/model/checkpoint
    identity, cohort, seed, hardware field and measured value. Stored report
    bytes are never rewritten by this normalization.
    """
    normalized=copy.deepcopy(run)
    for path in (('protocol_path',),('environment','run_dir'),
                 ('strict_main_comparison','environment','run_dir')):
        parent=normalized
        for key in path[:-1]:
            parent=parent.get(key,{}) if isinstance(parent,dict) else {}
        if isinstance(parent,dict) and path[-1] in parent:
            if not isinstance(parent[path[-1]],str):
                raise ValueError('Location-only comparison metadata must remain a string')
            # Fresh strict objects can alias the two environment dictionaries;
            # JSON-decoded saved reports do not. One marker preserves equality
            # without making normalization depend on object aliasing.
            parent[path[-1]]='<location-only>'
    return normalized


def aware_time(value,context):
    stamp=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    if stamp.tzinfo is None: raise ValueError(context+' requires a timezone')
    return stamp


def pause_evidence(repo,metric):
    """Validate recorded OS evidence without inferring timer sleep inclusion."""
    path=repo/'experiments/retry_v3/operational_pause.json'
    if not path.exists(): return None,None
    evidence=read(path); env=metric['environment']
    if evidence.get('source_commit')!=env['source_commit'] or evidence.get('source_provenance_modified') is not False:
        raise ValueError('Operational pause evidence has a different or modified study source')
    if evidence.get('original_quality_records_preserved') is not True or evidence.get('timer_sleep_inclusion_not_assumed') is not True:
        raise ValueError('Operational pause must preserve original records and timer uncertainty')
    sleep=aware_time(evidence['reported_sleep_time_utc'],'Reported sleep')
    wake=aware_time(evidence['reported_wake_time_utc'],'Reported wake')
    duration=evidence.get('reported_interval_seconds')
    if type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0 or not math.isclose((wake-sleep).total_seconds(),duration,abs_tol=.001,rel_tol=0):
        raise ValueError('Operational pause interval differs from its recorded timestamps')
    started=aware_time(env['started_utc'],'Training start')
    finished=aware_time(env['completed_utc'],'Training completion')
    if not started<=sleep<wake<=finished:
        raise ValueError('Recorded operational pause does not fall within completed training')
    utc_elapsed=(finished-started).total_seconds(); timer=env.get('seconds')
    if type(timer) not in (int,float) or not math.isfinite(timer) or timer<0:
        raise ValueError('Operational pause comparison requires a finite recorded elapsed timer')
    difference=abs(timer-utc_elapsed); tolerance=1.0
    # Require the pause to exceed the agreement uncertainty substantially, so
    # agreement cannot also fit a timer that excluded a very short pause.
    agrees=difference<=tolerance and duration>2*tolerance
    elapsed_observation=dict(recorded_timer_seconds=timer,utc_training_interval_seconds=utc_elapsed,
        absolute_difference_seconds=difference,agreement_tolerance_seconds=tolerance,
        timer_agrees_with_utc_interval_including_documented_pause=agrees,
        raw_duration_adjusted=False,gpu_compute_time_derived=False)
    events={(event.get('provider'),event.get('event_id')) for event in evidence.get('events',[])}
    if not {('Microsoft-Windows-Power-Troubleshooter',1),('Microsoft-Windows-Kernel-Power',42)}.issubset(events):
        raise ValueError('Operational pause lacks its recorded Windows sleep/wake event provenance')
    anomaly=any(event.get('provider')=='Microsoft-Windows-Kernel-Power' and event.get('event_id')==107
                and event.get('event_time_utc') and aware_time(event['event_time_utc'],'Resume provider event')<wake
                for event in evidence.get('events',[]))
    provider_note=(' The saved provider timestamps, including the separately noted resume-event time anomaly, remain unchanged.'
                   if anomaly else ' The saved provider timestamps remain unchanged.')
    display=lambda stamp:(stamp+dt.timedelta(microseconds=500)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    timer_note=(f" The recorded elapsed timer ({timer:.3f} seconds) agrees with the UTC start-to-completion interval ({utc_elapsed:.3f} seconds) to {difference:.3f} seconds; this observed agreement indicates that it includes the documented sleep interval."
                if agrees else ' Whether the recorded timer includes the OS sleep interval is not inferred from these timestamps.')
    note=(f"Recorded Windows power-event evidence reports a sleep/wake interval from {display(sleep)} UTC to {display(wake)} UTC ({duration:.3f} seconds), with the sleep reason reported as {evidence.get('reason','unspecified')}."
          +provider_note+timer_note+' Training-stage duration remains the original elapsed-timer measurement, not GPU compute time. No pause subtraction or derived GPU-only training duration is reported.')
    return dict(path=path.relative_to(repo).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                record=evidence,elapsed_observation=elapsed_observation),note


def update_analysis_evidence(repo,metric,training_rows):
    """Cross-check post-run flags against completed, immutable training rows."""
    path=repo/'experiments/retry_v3/training_update_analysis.json'
    if not path.exists(): return None,None
    evidence=read(path); env=metric['environment']; cfg=metric['config_snapshot']
    training_path=repo/metric['run_dir']/'train.jsonl'
    expected={'source_commit':env['source_commit'],'config_hash':env['config_hash'],
              'analysis_only':True,'new_generation_performed':False,
              'training_output_sha256':hashlib.sha256(training_path.read_bytes()).hexdigest()}
    if any(evidence.get(key)!=value for key,value in expected.items()):
        raise ValueError('Post-run update analysis identity/source-log binding differs')
    groups=evidence.get('groups',[])
    if len(groups)!=cfg['max_steps'] or [group.get('step') for group in groups]!=list(range(1,cfg['max_steps']+1)):
        raise ValueError('Post-run analysis must cover every frozen training group exactly once')
    repeated=[]; repeated_updates=[]; same_tokens=[]
    timing_keys=('candidate_ms','eager_ms','compile_ms','speedup_vs_eager','speedup_vs_compile')
    for group in groups:
        step=group['step']; actual=sorted((row for row in training_rows if row['step']==step),key=lambda row:row['group_index'])
        if len(actual)!=cfg['group_size']: raise ValueError('Update analysis group inventory differs')
        hashes=[row['candidate_sha256'] for row in actual]
        tokens=[row['completion_token_ids'] for row in actual]
        rewards=[row['reward'] for row in actual]
        identical_hash=len(set(hashes))==1
        identical_tokens=all(value==tokens[0] for value in tokens[1:])
        repeated_flag=all(row['status']=='PASS' for row in actual) and identical_hash and len(set(rewards))>1
        timing_flag=len({tuple(row.get(key) for key in timing_keys) for row in actual})>1
        expected_group={'op_name':actual[0]['op_name'],'statuses':[row['status'] for row in actual],
            'rewards':rewards,'candidate_sha256':hashes,
            'all_candidate_sha256_identical':identical_hash,
            'all_completion_token_ids_identical':identical_tokens,
            'grpo_advantages':[row['grpo_advantage'] for row in actual],
            'has_nonzero_advantage':any(abs(row['grpo_advantage'])>1e-8 for row in actual),
            'policy_updated':actual[0]['policy_updated'],
            'optimizer_step_count':actual[0]['optimizer_step_count'],
            'identical_code_pass_with_different_rewards':repeated_flag,
            'timing_variation_observed':timing_flag}
        if any(group.get(key)!=value for key,value in expected_group.items()):
            raise ValueError('Post-run analysis flags/data differ from raw group '+str(step))
        for key in ('weight_l2_delta','old_new_max_abs_logp_diff','skipped_non_finite_grad','skipped_non_finite_loss'):
            if key in actual[0] and group.get(key)!=actual[0][key]:
                raise ValueError('Post-run analysis update diagnostic differs: '+key)
        if repeated_flag:
            repeated.append(step)
            if identical_tokens: same_tokens.append(step)
            if actual[0]['policy_updated']: repeated_updates.append(step)
    note=None
    if repeated:
        steps=', '.join(str(step) for step in repeated)
        note=(f"The completed-run analysis identifies {len(repeated)} of {len(groups)} GRPO groups ({steps}) in which the same verified candidate-source SHA-256 received different PASS rewards across repeated measurements; {len(same_tokens)} of these groups also had identical completion-token sequences. Of these same-source groups, {len(repeated_updates)} performed optimizer updates. These comparisons distinguish measured rewards for identical source rather than distinct candidate programs. Timing/reference variation, KL regularization, optimizer state and finite precision can contribute to parameter behavior; this analysis does not causally decompose them. Optimizer-step counts and tensor changes therefore document activity, not automatically useful kernel learning.")
    summary=dict(identical_source_different_reward_groups=repeated,
                 identical_source_and_tokens_different_reward_groups=same_tokens,
                 updated_identical_source_different_reward_groups=repeated_updates)
    return dict(path=path.relative_to(repo).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                record=evidence,recomputed_summary=summary),note


def publication_abstract(metrics):
    """Keep the three-study abstract concise; exact counters stay in results."""
    by_key={metric['study_key']:metric for metric in metrics}
    methods=('base_pass1','base_best_of_8','sft_only_checkpoint','trained_checkpoint')
    def fractions(key):
        values=[by_key[key]['quality']['fresh_heldout']['methods'][method] for method in methods]
        return ', '.join(f"{value['pass_count']}/{value['total_ops']}" for value in values[:-1])+' and '+f"{values[-1]['pass_count']}/{values[-1]['total_ops']}"
    paragraphs=[
        'KernelBench-RL is an execution-graded GPU kernel synthesis system for a memory-constrained workstation. It combines a custom thirty-operator registry, curated Triton implementations, numerical checks, bounded execution, supervised initialization and group-relative optimization. The suite is distinct from the complete official KernelBench benchmark. Fixed baselines passed 180/180 operator/precision/input-seed checks, with geometric mean eager speedups of 2.494x in FP32 and 5.417x in BF16. Verified compiled comparisons cover twenty-nine operators. These are fixed-implementation measurements, not learned-policy gains.',
        'Corrected v2 experiments use training-only supervision, correctness-gated speed rewards, separate fresh and inspected development cohorts, and an SFT-only control. For each fresh N=3 cohort, methods are original-base pass@1, base best-of-eight, SFT-only pass@1 and hybrid pass@1. V2 0.49B FP32 recorded respective passes '+fractions('v2/laptop')+', while v2 7B rank 8 recorded '+fractions('v2/laptop_7b')+'.'
    ]
    if 'v3/laptop_7b' in by_key:
        paragraphs.append('A separate prospective v3 study starts from the original pinned 7B base with fresh rank-16 adapters, trains on the former v2 test operators, and uses a different three-operator fresh cohort. Its respective passes were '+fractions('v3/laptop_7b')+'. Results remain separated by model, precision, configuration and protocol. The hybrid includes SFT; update counts alone do not establish useful learning, and no RL-only causal gain is inferred.')
    paragraphs.append('The study preserves failures, checkpoint identities, timing boundaries and unavailable comparisons. One training seed and three fresh operators per study cannot establish statistical significance, a general learning advantage, production acceleration or formal security. The artifact provides a traceable basis for examining both successful generated programs and limitations, rather than a broad model or hardware ranking.')
    return ' '.join(paragraphs)


def residual_case_study(repo):
    """Optional, identity-bound inspection of two already exposed v2 failures."""
    folder=repo/'experiments/retry_v2/laptop_7b'
    cases={}
    for filename,method in (('sft_only.jsonl','sft_only_checkpoint'),('heldout.jsonl','trained_checkpoint')):
        matches=[row for row in rows(folder/filename) if row.get('op_name')=='matmul_residual_add_512'
                 and row.get('method')==method and row.get('eval_phase')=='heldout_test']
        if len(matches)!=1: return None
        cases[method]=matches[0]
    expected={'sft_only_checkpoint':'304785e2b89e76e531d2a1dcd2f28bbb38861f7a6156bc13bd31454dce811789',
              'trained_checkpoint':'b006f8a6b723a5e1177c62ff4455beb701ab2baead3dc3248fc63aaf237edcc6'}
    if any(cases[method].get('candidate_sha256')!=digest for method,digest in expected.items()): return None
    for row in cases.values():
        source=next((row[key] for key in ('candidate_source','source','source_code') if row.get(key)),None)
        if not isinstance(source,str) or hashlib.sha256(source.encode('utf-8')).hexdigest()!=row['candidate_sha256']:
            raise ValueError('V2 case-study source bytes differ from the preserved candidate identity')
    sft,hybrid=cases['sft_only_checkpoint'],cases['trained_checkpoint']
    if not (sft['status']=='NUMERICAL_MISMATCH' and sft['compiled'] and not sft['correct']
            and 'non-finite' in sft.get('error_message','') and hybrid['status']=='RUNTIME_ERROR'
            and 'TypeError' in hybrid.get('error_message','')):
        raise ValueError('V2 case-study statuses differ from the preserved failures')
    return ('In v2, the 7B SFT-only matmul-residual candidate (SHA-256 prefix 304785e2b89e, twelve characters only) compiled but failed canonical FP16 checks with nonfinite output. Source inspection found doubled reduction/column pointer offsets, a 128-column block and an empty_like output, consistent with indexing/coverage concerns. The hybrid candidate (prefix b006f8a6b723) instead raised a tuple TypeError. Its nested launch grid is visible in the source; separately, the grid counts sixty-four rows per program while the body selects one row. These are source-inspection explanations, not controlled causal attribution; the latent coverage issue was not independently executed. Failed samples remain failures in every relevant pass-rate denominator.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--paper-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--publication-date', required=True, help='Actual author-local publication date, YYYY-MM-DD')
    parser.add_argument('--training-figure', type=Path)
    parser.add_argument('--followup-training-figure', type=Path,
                        help='Optional actual completed v3 SFT PNG; default auto-detects out/paper-assets/sft_followup.png')
    parser.add_argument('--manuscript-template', type=Path)
    args=parser.parse_args()
    repo=args.repo.resolve(); out=args.out.resolve(); site=args.site.resolve()
    sys.path.insert(0,str(repo))
    from eval.compare_retry import compare_retry_runs, load_retry_run
    from rl_trainer.experiment_config import ExperimentConfig
    from export_retry import validate_run, validate_verifier_regression, registered_v3_study, STUDY_COMMIT
    run_dirs=[repo/'experiments/retry_v2'/n for n in ('laptop','laptop_7b')]
    validated_runs={'v2/'+folder.name:validate_run(repo,folder,STUDY_COMMIT,load_retry_run,ExperimentConfig,repo/'experiments/retry_v2/protocol.json') for folder in run_dirs}
    v3_registration=registered_v3_study(repo)
    if v3_registration is not None:
        registration=v3_registration
        validated_runs[registration['study_key']]=validate_run(repo,registration['run_dir'],registration['source_commit'],load_retry_run,ExperimentConfig,registration['protocol'])
    verifier_regression=validate_verifier_regression(repo,STUDY_COMMIT)
    # Reuse the original v2 comparison bytes when present. A v3 publication
    # must never regenerate or append to that historical comparison artifact.
    comparison_paths={'v2':repo/'experiments/retry_v2/comparison/retry_comparison.json'}
    if comparison_paths['v2'].exists():
        report=read(comparison_paths['v2'])
        if [comparison_science(run) for run in report.get('runs',[])]!=[comparison_science(validated_runs['v2/'+folder.name]['strict']) for folder in run_dirs]:
            raise ValueError('Preserved v2 comparison differs from its strictly validated original records')
    else:
        report=compare_retry_runs(run_dirs,
            comparison_paths['v2'].parent, historical_run_dirs=[repo/'experiments/laptop_fullparam',repo/'experiments/colab_t4/qlora7b_fresh'])
    # Metric identities come from the current strict load; only the explicitly
    # normalized location metadata may differ from the preserved report.
    records=[('v2',validated_runs['v2/'+folder.name]['strict']) for folder in run_dirs]
    if v3_registration is not None:
        comparison_paths['v3']=repo/'experiments/retry_v3/comparison/retry_comparison.json'
        if comparison_paths['v3'].exists():
            v3_report=read(comparison_paths['v3'])
            if [comparison_science(run) for run in v3_report.get('runs',[])]!=[comparison_science(validated_runs['v3/laptop_7b']['strict'])]:
                raise ValueError('Preserved v3 comparison differs from its strictly validated records')
        else:
            v3_report=compare_retry_runs([v3_registration['run_dir']],comparison_paths['v3'].parent,
                                        protocol_path=v3_registration['protocol'])
        records.append(('v3',validated_runs['v3/laptop_7b']['strict']))
    labels={'v2/laptop':'v2 0.49B full-parameter FP32',
            'v2/laptop_7b':'v2 7B QLoRA rank 8 / NF4 / FP16 decoder / CPU vocabulary',
            'v3/laptop_7b':'v3 7B QLoRA rank 16 / NF4 / FP16 decoder / CPU vocabulary'}
    short_labels={'v2/laptop':'v2 0.49B FP32','v2/laptop_7b':'v2 7B rank 8','v3/laptop_7b':'v3 7B rank 16'}
    method_labels={'base_pass1':'Base pass@1','base_best_of_8':'Base best-of-8','sft_only_checkpoint':'SFT-only pass@1','trained_checkpoint':'Hybrid pass@1'}
    metrics=[]
    for version,run in records:
        run_name=run['environment']['run_name']; name=version+'/'+run_name
        folder=repo/'experiments'/('retry_'+version)/run_name
        validated=validated_runs[name]
        env=validated['environment']; train=rows(folder/'train.jsonl'); sft=read(folder/'sft_summary.json')
        if validated['strict']!=run: raise ValueError('Strict report changed during publication validation')
        if run['grpo_optimizer_steps']==0 and validated['final']['state']['trainable_weights_sha256']!=validated['sft']['state']['trainable_weights_sha256']:
            raise ValueError('Zero-GRPO study has different final and SFT-only tensor checksums')
        status_counts=dict(Counter(r['status'] for r in train))
        metric=dict(name=name,study_key=name,run_name=run_name,protocol_version=version,
            run_dir=folder.relative_to(repo).as_posix(),protocol_path='experiments/retry_'+version+'/protocol.json',
            comparison_path=comparison_paths[version].relative_to(repo).as_posix(),
            label=labels[name],short_label=short_labels[name],environment=env,sft_summary=sft,
            config_snapshot=read(folder/'config.json'),
            training_status_counts=status_counts,training_failures={key:value for key,value in status_counts.items() if key!='PASS'},quality=run)
        if version=='v3':
            metric['source_provenance']=v3_registration['provenance']
            metric['operational_notes']=[]
            pause,pause_note=pause_evidence(repo,metric)
            analysis,analysis_note=update_analysis_evidence(repo,metric,train)
            if pause is not None: metric['operational_pause']=pause
            if analysis is not None: metric['training_update_analysis']=analysis
            metric['operational_notes'].extend(note for note in (pause_note,analysis_note) if note)
        metrics.append(metric)
    v2_metrics=[m for m in metrics if m['protocol_version']=='v2']
    v3_metric=next((m for m in metrics if m['protocol_version']=='v3'),None)
    publication_date=dt.date.fromisoformat(args.publication_date).isoformat()
    training_figure=(args.training_figure or out/'paper-assets/sft_training.png').resolve()
    plot=read(training_figure.with_suffix('.json'))
    if plot['figure_sha256']!=hashlib.sha256(training_figure.read_bytes()).hexdigest(): raise ValueError('Training figure changed')
    if {track['name'] for track in plot['tracks']}!={'laptop','laptop_7b'}: raise ValueError('Training plot track mismatch')
    for track in plot['tracks']:
        folder=repo/'experiments/retry_v2'/track['name']
        if track['sft_log_sha256']!=hashlib.sha256((folder/'sft.jsonl').read_bytes()).hexdigest(): raise ValueError('Training plot log mismatch')
        if track['config_file_sha256']!=hashlib.sha256((folder/'config.json').read_bytes()).hexdigest(): raise ValueError('Training plot config mismatch')
    followup_figure=(args.followup_training_figure or out/'paper-assets/sft_followup.png').resolve()
    followup_plot=None
    if args.followup_training_figure is not None or followup_figure.exists() or followup_figure.with_suffix('.json').exists():
        if v3_metric is None: raise ValueError('A v3 figure cannot be included without a completed registered v3 study')
        followup_plot=read(followup_figure.with_suffix('.json'))
        folder=repo/v3_metric['run_dir']
        required={'schema_version':'kernelbench_sft_followup_plot_v1','study_key':'v3/laptop_7b',
                  'config_hash':v3_metric['environment']['config_hash'],
                  'source_commit':v3_metric['environment']['source_commit'],
                  'completed_steps':v3_metric['quality']['sft_optimizer_steps'],
                  'epochs':8,'examples_per_epoch':11,'adapter_rank':16,'adapter_alpha':32,
                  'inference_quality_claim':False,
                  'sft_checkpoint_identity':v3_metric['quality']['sft_only_checkpoint_identity']}
        if any(followup_plot.get(key)!=value for key,value in required.items()):
            raise ValueError('Follow-up plot study/config/source/schedule/checkpoint identity differs')
        bound_files={'figure_sha256':followup_figure,'sft_log_sha256':folder/'sft.jsonl',
                     'config_file_sha256':folder/'config.json','sft_summary_sha256':folder/'sft_summary.json',
                     'protocol_sha256':v3_registration['protocol'],
                     'source_provenance_sha256':v3_registration['provenance_path'],
                     'token_feasibility_sha256':folder.parent/'token_feasibility.json',
                     'sft_checkpoint_metadata_sha256':validated_runs['v3/laptop_7b']['sft']['step_root']/'checkpoint_state.json'}
        for key,path in bound_files.items():
            if followup_plot.get(key)!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Stale follow-up plot binding: '+key)
        logged=rows(folder/'sft.jsonl')
        means=[math.fsum(row['loss'] for row in logged if row['epoch']==epoch)/11 for epoch in range(1,9)]
        supplied=followup_plot.get('epoch_means_nats_per_token',[])
        if len(supplied)!=8 or any(not math.isclose(actual,recorded,rel_tol=1e-12,abs_tol=1e-12) for actual,recorded in zip(means,supplied)):
            raise ValueError('Follow-up plot epoch means differ from the actual training log')
        if followup_plot.get('observed_operator_order')!=[row['op_name'] for row in logged]:
            raise ValueError('Follow-up plot operator order differs')
        if followup_plot.get('train_ops')!=v3_metric['config_snapshot']['train_ops'] or followup_plot.get('adapter_target_modules')!=v3_metric['config_snapshot']['lora_target_modules']:
            raise ValueError('Follow-up plot training/adapter targets differ')
    template=(args.manuscript_template or args.paper_source/'manuscript_template.json')
    if not template.exists() and args.manuscript_template is None: template=args.paper_source/'manuscript_draft.json'
    manuscript=copy.deepcopy(read(template))
    manuscript.update(status='FINAL_VERIFIED_COMPLETED_RESOURCE_CONSTRAINED_STUDIES',publication_date=publication_date)
    manuscript['publication_type']='Independent technical report (preprint; not peer reviewed)'
    manuscript['pending_results']={'required':False}
    result_sentences=[]
    for m in metrics:
        q=m['quality']; methods=q['fresh_heldout']['methods']
        counts=', '.join(f"{method_labels[key]} {value['pass_count']}/{value['total_ops']}" for key,value in methods.items())
        update_word='update' if q['grpo_optimizer_steps']==1 else 'updates'
        result_sentences.append(f"The corrected {m['label']} study completed {q['sft_optimizer_steps']} supervised updates and {q['grpo_optimizer_steps']} GRPO optimizer {update_word}; fresh held-out counts were {counts}.")
    outcome=' '.join(result_sentences)
    manuscript['abstract']=publication_abstract(metrics)
    sections={s['id']:s for s in manuscript['sections']}
    for section in manuscript['sections']:
        section['paragraphs']=[re.sub(r'\s*\[DRAFT:.*?\]','',p) for p in section['paragraphs']]
    case_study=residual_case_study(repo)
    if case_study and 'failure_diagnosis' in sections: sections['failure_diagnosis']['paragraphs'].append(case_study)
    sections['introduction']['paragraphs'][-1]+=' '+outcome
    hardware=read(repo/'experiments/retry_v2/hardware_environment.json')
    gpu_name,driver,device_memory=[part.strip() for part in hardware['gpu_csv'].split(',')]
    sections['system_methods']['paragraphs']=[paragraph.replace('The 7B track trains 5,046,272','The corrected v2 7B track trains 5,046,272') for paragraph in sections['system_methods']['paragraphs']]
    sections['system_methods']['paragraphs'].append(f"Both corrected v2 studies used an {hardware['cpu']['Name'].strip()} CPU ({hardware['cpu']['NumberOfCores']} physical cores), {hardware['reported_physical_memory_bytes']/2**30:.2f} GiB reported physical RAM, and an {gpu_name} with {device_memory} reported device memory. The system ran {hardware['os']['Caption']} build {hardware['os']['BuildNumber']} with NVIDIA driver {driver}. PyTorch used {hardware['torch_cpu_threads']} CPU threads. The hardware descriptor was recorded during the v2 studies.")
    sections['original_evidence']['paragraphs'].append('After the retry hardened output dtype/device and post-timing checks, a separate regression passed all sixty operator/precision records across the thirty fixed baselines in fp32 and bf16, with seed 142 and independent seed 10142. This verifies the updated worker on its canonical and registered stress inputs. It is a distinct correctness campaign and does not replace or re-estimate the historical 180-record speedup aggregates.')
    protocol=sections['corrected_protocol']['paragraphs']
    protocol[0]=protocol[0].replace('The retry freezes','The frozen retry used').replace('The intended supervised phase contains thirty-two example updates; actual completed counters must be read from durable logs and checkpoints.','Both tracks completed thirty-two supervised example updates, verified in durable logs and checkpoints.')
    protocol[1]=protocol[1].replace('not evidence that supervision or the gate has already improved quality','not an assumption that supervision or the gate improves quality')
    protocol[-1]=protocol[-1].replace('the retry is planned sequentially on the laptop','the retry completed sequentially on the laptop')
    protocol.append('Both tracks retain the original pre-SFT base as the reference-policy anchor: a frozen CPU copy for the full-parameter model and disabled adapters over the frozen quantized base for QLoRA. The reference penalty therefore regularizes toward the original model rather than the post-SFT checkpoint. The three fresh targets are unseen operator programs within normalization, activation and matrix-product families represented in training; this is not an unseen-family out-of-distribution test. Here, fresh means excluded from model training and from previous generated-program test inspection, not that their fixed reference implementations were absent from the public project.')
    for m in v2_metrics:
        e=m['environment']; q=m['quality']; fail=', '.join(f'{k}={v}' for k,v in m['training_status_counts'].items())
        protocol.append(f"{m['label']}: training completed in {e['seconds']:.2f} seconds, with {e['peak_allocated_bytes']/2**30:.3f} GiB parent-process peak allocated CUDA memory during training. This allocator measurement excludes child-process and other applications' device allocations. Generated training statuses were {fail}. The full study includes a model-checksum-verified reload restoring optimizer and RNG state with zero extra training steps, main evaluation and the separate SFT control. The reload is not an independent next-step trajectory replication or a separate optimizer/RNG digest comparison. Its final trainable tensor SHA-256 prefix is {e['trainable_weights_sha256'][:12]} (12 characters only; the full 64-character digest is preserved in artifact metadata and raw checkpoint manifests). Tensor digests are distinct from serialized-file and ZIP digests.")
        if q['grpo_optimizer_steps']==0:
            protocol.append(f"For {m['label']}, the correctness-gated schedule made no GRPO optimizer update. The final checkpoint consequently retains the SFT-only weights; any difference in sampled outputs or timing is not an RL parameter gain. Zero-valued update diagnostics on skipped groups are sentinels, not measured old/new-likelihood agreement for a performed update.")
    q7_run=next(m['quality'] for m in v2_metrics if m['name']=='v2/laptop_7b')
    q7_methods=q7_run['fresh_heldout']['methods']
    q7_passes={method:{row['op_name'] for row in q7_methods[method]['per_operator']
                      if row['status']=='PASS' and row['compiled'] and row['correct']}
               for method in ('sft_only_checkpoint','trained_checkpoint')}
    control_comparison=''
    if q7_passes['sft_only_checkpoint']==q7_passes['trained_checkpoint']:
        shared=sorted(q7_passes['sft_only_checkpoint'])
        details=' The identical passing operators were '+', '.join(shared)+'.' if shared else ''
        control_comparison=(f"The v2 7B SFT-only and hybrid checkpoints each passed {len(shared)}/{q7_methods['sft_only_checkpoint']['total_ops']} fresh held-out operators."
                            +details+' No incremental GRPO held-out correctness gain over the SFT-only control was observed in this single configured attempt.')
        protocol.append(control_comparison)
    protocol.append(' '.join(result_sentences[:len(v2_metrics)])+' All failures remain in pass-rate and fast-versus-eager denominators; correct-only speedup and compiled-reference availability are labeled separately. These single-attempt observations do not establish statistical significance or a general learning advantage.')
    tables=[]
    for cohort,key in [('Fresh held-out','fresh_heldout'),('Previously inspected development','inspected_development')]:
        table_rows=[]
        for m in v2_metrics:
            for method,stats in m['quality'][key]['methods'].items():
                if stats.get('available') is False: continue
                table_rows.append({'track':m['short_label'],'method':method_labels[method],'budget':stats['budget_k'],'passes':f"{stats['pass_count']}/{stats['total_ops']}",'fast':f"{stats['fast_vs_eager_count']}/{stats['total_ops']}"})
        tables.append(dict(id=key,section_id='corrected_protocol',caption='V2 '+cohort.lower()+' results, separated by model track.',columns=[{'key':'track','label':'Model / training','weight':2.2},{'key':'method','label':'Method','weight':1.6},{'key':'budget','label':'Budget','weight':.65,'align':'RIGHT'},{'key':'passes','label':'Pass / N','weight':.8,'align':'RIGHT'},{'key':'fast','label':'Fast / N (eager)','weight':1,'align':'RIGHT'}],rows=table_rows,notes=['N=3 operators per method and model. Selection rows are excluded. The hybrid includes SFT; no RL-only effect is established.']))
    paired=[]
    short_status={'PASS':'PASS','CHEAT':'POLICY','COMPILE_ERROR':'COMPILE','NUMERICAL_MISMATCH':'NUMERIC','RUNTIME_ERROR':'RUNTIME','TIMEOUT':'TIMEOUT','OOM':'OOM'}
    for m in v2_metrics:
        folder=repo/m['run_dir']
        finals={(r['method'],r['op_name']):r['status'] for filename in ('heldout.jsonl','sft_only.jsonl') for r in rows(folder/filename) if r['eval_phase']=='heldout_test'}
        for op in m['quality']['fresh_heldout']['ops']:
            paired.append(dict(track=m['short_label'],operator=op,
                base=short_status[finals[('base_pass1',op)]],best=short_status[finals[('base_best_of_8',op)]],
                sft=short_status[finals[('sft_only_checkpoint',op)]],hybrid=short_status[finals[('trained_checkpoint',op)]]))
    tables.append(dict(id='paired_fresh_status',section_id='corrected_protocol',caption='Paired fresh-operator outcomes, retaining every failure.',columns=[{'key':'track','label':'Model','weight':1.2},{'key':'operator','label':'Operator','weight':2.2},{'key':'base','label':'Base @1','weight':1},{'key':'best','label':'Best @8','weight':1},{'key':'sft','label':'SFT @1','weight':1},{'key':'hybrid','label':'Hybrid @1','weight':1}],rows=paired,notes=['POLICY: source-policy rejection; COMPILE: compile error; NUMERIC: numerical mismatch; RUNTIME: candidate runtime error. Full diagnostics and source are in the raw records.']))
    v3_discussion=''
    if v3_metric is not None:
        m=v3_metric; q=m['quality']; e=m['environment']; cfg=m['config_snapshot']
        fresh=q['fresh_heldout']['methods']; dev=q['inspected_development']['methods']
        former=set(q7_run['fresh_heldout']['ops'])
        if not former.issubset(cfg['train_ops']) or former.intersection(cfg['heldout_ops']+cfg['development_eval_ops']):
            raise ValueError('V3 must label all former v2 fresh targets training-seen, never fresh/development')
        if cfg.get('resume_from') or cfg['lora_r']!=16 or cfg['lora_alpha']!=32:
            raise ValueError('V3 reporting expects registered fresh initialization and rank-16/alpha-32 adapters')
        target_modules=', '.join(cfg['lora_target_modules'])
        trainable_parameters=read(v3_registration['protocol'])['adapter']['expected_trainable_parameters']
        if type(trainable_parameters) is not int or trainable_parameters!=40370176:
            raise ValueError('V3 registered adapter parameter count differs from its verified rank-16 architecture')
        m['trainable_adapter_parameters']=trainable_parameters
        fresh_counts=', '.join(f"{method_labels[key]} {value['pass_count']}/{value['total_ops']}" for key,value in fresh.items())
        dev_counts=', '.join(f"{method_labels[key]} {value['pass_count']}/{value['total_ops']}" for key,value in dev.items() if value.get('available') is not False)
        pass_sets={method:{row['op_name'] for row in fresh[method]['per_operator'] if row['status']=='PASS'}
                   for method in ('sft_only_checkpoint','trained_checkpoint')}
        v3_discussion='V3 fresh held-out counts were '+fresh_counts+'. Development counts were '+dev_counts+'.'
        if pass_sets['sft_only_checkpoint']==pass_sets['trained_checkpoint']:
            v3_discussion+=' The SFT-only and hybrid arms have the same passing-operator set; no incremental GRPO held-out correctness gain over SFT was observed in this attempt.'
        v3_paragraphs=[
            f"The prospective v3 follow-up is a separate 7B study initialized from the pinned original base with fresh LoRA adapters. It does not resume or transfer the v2 adapters. Rank {cfg['lora_r']}, alpha {cfg['lora_alpha']} adapters target {target_modules}, totaling {trainable_parameters:,} trainable adapter parameters; the decoder remains NF4 with FP16 compute and fixed FP32 CPU embedding/output computations. The configuration uses seed {cfg['seed']}, {cfg['sft_epochs']} supervised epochs over {len(cfg['train_ops'])} training operators, SFT learning rate {cfg['sft_learning_rate']:g} and GRPO learning rate {cfg['learning_rate']:g}.",
            'All three former v2 fresh operators (l2_normalize_2048, fused_bias_silu_4096, matmul_residual_add_512) are now training targets. They are training-seen in v3 and excluded from both fresh and development denominators. The three original inspected development operators remain development-only. The prospective fresh operators are '+', '.join(q['fresh_heldout']['ops'])+'. The exposure inventory records no earlier repository model-generated or supervised-target records for this trio; public fixed references/baselines and related training families were already available. This is a repository-local prospective-program claim, not an unseen-family or pretraining-contamination claim.',
            f"The completed v3 schedule records {q['sft_optimizer_steps']} actual SFT updates, {e['steps']} GRPO groups, {sum(m['training_status_counts'].values())} generated training candidates and {q['grpo_optimizer_steps']} GRPO optimizer updates. Training statuses were "+', '.join(f'{status}={count}' for status,count in m['training_status_counts'].items())+'. Correctness gating preserves measured PASS speed rewards and gives every non-PASS candidate zero training reward; it is not a binary PASS=1 objective. Equal group rewards can yield zero relative advantage even for correct candidates. The original pre-SFT base remains the KL anchor.',
            v3_discussion,
            f"The recorded v3 training-stage elapsed timer is {e['seconds']:.2f} seconds, not a GPU-compute-time measurement. Training reached {e['peak_allocated_bytes']/2**30:.3f} GiB parent-process peak allocated CUDA memory, excluding child-worker and other applications' allocations. Its final trainable tensor SHA-256 prefix is {e['trainable_weights_sha256'][:12]} (12 characters only; full identities remain in artifact metadata). The source commit prefix is {e['source_commit'][:12]}; the registered protocol and full source provenance are published separately from v2.",
            'The v3 evaluation retains its own comparison artifact, cohorts, failed rows and checkpoint identities. V2 tables and strict comparison remain separate. Changes in tasks, training exposure, schedule, adapter rank/targets, learning rates and seed prevent a causal v2-versus-v3 comparison or a model-capacity claim. One read of three fresh operators does not establish broad generalization or statistical significance, irrespective of whether all three pass.'
        ]
        v3_paragraphs[3:3]=m.get('operational_notes',[])
        cosine_rows=[row for row in rows(repo/m['run_dir']/'heldout.jsonl')
                     if row.get('method')=='trained_checkpoint' and row.get('eval_phase')=='heldout_test'
                     and row.get('op_name')=='cosine_similarity_1024']
        if len(cosine_rows)==1:
            cosine=cosine_rows[0]
            inspected_sha='9d62a1bcbb30d2cc9cdd46a61916cf009fefb49ee76827a004cd59f0937ef2c7'
            if (cosine.get('candidate_sha256')==inspected_sha and cosine.get('status')=='COMPILE_ERROR'
                    and hashlib.sha256(cosine.get('candidate_source','').encode('utf-8')).hexdigest()==inspected_sha
                    and 'Mask argument cannot be block type if pointer argument is not a block' in cosine.get('error_message','')):
                v3_paragraphs.append('Source inspection explains the hybrid cosine-similarity compilation failure. The generated program reduces each input row to scalar dot-product and norm terms, then passes its vector column mask to a store through the scalar output pointer C_ptr + row. Triton rejects that pointer/mask type combination. The preserved candidate SHA-256 prefix is 9d62a1bcbb30; its complete identity and compiler diagnostic remain in the raw test record. The candidate was retained unchanged. Scalar/vector store compatibility is a concrete target for future training material, whose effect would require another prospectively specified evaluation.')
        if q['grpo_optimizer_steps']==0:
            zero_update_note=('No v3 GRPO optimizer update was performed. The full final and SFT-only trainable-tensor SHA-256 checksums are verified identical, so the GRPO phase made no trainable-parameter change. Here, hybrid denotes the checkpoint after the completed SFT-plus-GRPO schedule; it does not denote additional RL parameter learning. Any sampled-output or measured-timing difference between these two arms is not evidence of RL learning.')
            v3_paragraphs.append(zero_update_note)
            v3_discussion+=' '+zero_update_note
        manuscript['sections']=[section for section in manuscript['sections'] if section['id']!='prospective_v3']
        insertion=next(i for i,section in enumerate(manuscript['sections']) if section['id']=='applications')
        manuscript['sections'].insert(insertion,dict(id='prospective_v3',heading='7. Prospective Expanded-Adapter Follow-Up (v3)',paragraphs=v3_paragraphs))
        for number,section in enumerate(manuscript['sections'],1):
            section['heading']=re.sub(r'^\d+\.',str(number)+'.',section['heading'])
        for cohort,key in [('Prospective fresh held-out','fresh_heldout'),('Previously inspected development','inspected_development')]:
            table_rows=[dict(method=method_labels[method],budget=stats['budget_k'],passes=f"{stats['pass_count']}/{stats['total_ops']}",fast=f"{stats['fast_vs_eager_count']}/{stats['total_ops']}",failures=', '.join(f'{status}={count}' for status,count in stats['failure_counts_by_status'].items()) or 'None')
                        for method,stats in q[key]['methods'].items() if stats.get('available') is not False]
            tables.append(dict(id='v3_'+key,section_id='prospective_v3',caption='V3 7B rank 16: '+cohort.lower()+'.',columns=[{'key':'method','label':'Method','weight':1.8},{'key':'budget','label':'Budget','weight':.6},{'key':'passes','label':'Pass / N','weight':.7},{'key':'fast','label':'Fast / N (eager)','weight':1},{'key':'failures','label':'Failures','weight':2}],rows=table_rows,notes=['V3 only; N=3 per method. Fresh base initialization, rank/schedule and fresh operators differ from v2. No pooling or causal attribution. Development best-of-eight unavailable.']))
    manuscript['tables']=tables
    manuscript['figures']=[dict(id='observed_sft_training',section_id='corrected_protocol',after_paragraph=6,path=training_figure.relative_to(out).as_posix(),caption='V2 studies only. '+plot['caption'],max_height_inches=4.2)]
    if followup_plot is not None:
        manuscript['figures'].append(dict(id='observed_v3_sft_training',section_id='prospective_v3',after_paragraph=3,
            path=followup_figure.relative_to(out).as_posix(),caption=followup_plot['caption'],max_height_inches=4.2))
    repro=sections['reproducibility_ethics']['paragraphs']
    repro[0]=repro[0].replace('The public artifact should link','The public artifact links').replace('The final paper and explanatory website should share','The final paper and explanatory website share')
    repro[0]+=f' Both v2 corrected quality studies executed from public source commit prefix {STUDY_COMMIT[:12]} (12 characters only; full identity in artifact metadata), frozen and pushed before their fresh test generation. The later publication snapshot also contains result files and publication assets; its distinct commit is recorded in the versioned delivery manifest.'
    repro[-1]+=' Implementation, analysis tooling and manuscript preparation used Codex and Gemini as coding and drafting tools. They are not listed as authors. Reported measurements are derived from executed runs and preserved artifacts, rather than generated examples.'
    repro[-1]+=' Fixed baselines are curated implementations prepared with coding-assistant support, distinct from programs sampled by the evaluated Qwen policies.'
    manuscript['verified_retry_metrics']=metrics
    rendered=' '.join([manuscript['abstract'],*[p for s in manuscript['sections'] for p in s['paragraphs']]])
    if re.search(r'\[DRAFT\b|\bTBD\b|\[PENDING\b|results are pending',rendered,re.I): raise ValueError('Unresolved manuscript placeholder')
    write(out/'KernelBench-RL-Manuscript.json',manuscript)
    write(out/'RETRY_VERIFIED_METRICS.json',metrics)
    data=read(site/'results.json'); normalized=[]
    for m in metrics:
        q=m['quality']; e=m['environment']; cohorts={}
        for c,key in [('fresh','fresh_heldout'),('development','inspected_development')]:
            cohorts[c]=[dict(method=method_labels[method],passed=s['pass_count'],total=s['total_ops'],budget=s['budget_k']) for method,s in q[key]['methods'].items() if s.get('available') is not False]
        cohort_label='v3 prospective fresh trio' if m['protocol_version']=='v3' else 'v2 original fresh trio'
        normalized.append(dict(study_key=m['study_key'],protocol_version=m['protocol_version'],
            label=m['label'],short_label=m['short_label'],cohort_label=cohort_label,
            cohort_ops=dict(fresh=q['fresh_heldout']['ops'],development=q['inspected_development']['ops']),
            fresh_ops=q['fresh_heldout']['ops'],development_ops=q['inspected_development']['ops'],
            comparison_path=m['comparison_path'],hardware=e['gpu'],training_mode=e['training_mode'],
            config_hash=e['config_hash'],model_revision=e['model_revision'],notes=m.get('operational_notes',[]),
            status='completed',source_commit=e['source_commit'],sft_updates=q['sft_optimizer_steps'],
            optimizer_updates=q['grpo_optimizer_steps'],cohorts=cohorts))
    data['retry']=dict(status='completed',published_results=True,runs=normalized,
        note=outcome+' '+control_comparison+' '+v3_discussion+' Fresh and inspected development cohorts are reported separately; the hybrid includes supervised warmup. V2 and v3 use different fresh operators and configurations and are never pooled.')
    data['verifier_regression']={key:value for key,value in verifier_regression.items() if key!='directory'}
    provenance_paths=list(comparison_paths.values())
    if v3_registration is not None:
        provenance_paths.extend([v3_registration['protocol'],v3_registration['provenance_path']])
        for key in ('operational_pause','training_update_analysis'):
            if key in v3_metric: provenance_paths.append(repo/v3_metric[key]['path'])
    for provenance_path in provenance_paths:
        provenance_name=provenance_path.relative_to(repo).as_posix()
        data['provenance']=[item for item in data['provenance'] if item['path']!=provenance_name]
        data['provenance'].append(dict(path=provenance_name,sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest()))
    write(site/'results.json',data)
    index=(site/'index.html').read_text(encoding='utf-8').replace('Open source ·','Public code ·').replace('Results pending.','Corrected studies completed.')
    highlighted=v3_metric or next(m for m in v2_metrics if m['name']=='v2/laptop_7b')
    q7=highlighted['quality']['fresh_heldout']['methods']['trained_checkpoint']
    index=latest_hybrid_headline(index,highlighted['protocol_version'],q7['pass_count'],q7['total_ops'])
    proof_note='<p class="chart-note" id="updated-verifier-proof">Updated verifier regression: 60/60 fixed-kernel records passed across all 30 operators in FP32/BF16 using independent input checks. This separate correctness campaign preserves the historical performance aggregates. <a href="https://github.com/Eljo-M/kernelbench-rl/tree/main/experiments/retry_v2/verifier_regression">Inspect the regression traces</a>.</p>'
    if 'id="updated-verifier-proof"' not in index: index=index.replace('<div class="learning-result">',proof_note+'<div class="learning-result">',1)
    index=re.sub(r'(<p id="retry-copy">).*?(</p>)',lambda match:match[1]+html.escape(data['retry']['note'])+match[2],index,flags=re.S)
    (site/'index.html').write_text(index,encoding='utf-8')
    paper=(site/'paper.html').read_text(encoding='utf-8')
    paper=re.sub(r'(<meta name="citation_title" content=")[^"]*(">)',lambda m:m[1]+html.escape(manuscript['title'],quote=True)+m[2],paper)
    paper=re.sub(r'(<meta name="citation_publication_date" content=")[^"]*(">)',lambda m:m[1]+publication_date.replace('-','/')+m[2],paper)
    paper=re.sub(r'<h1>.*?</h1>','<h1>'+html.escape(manuscript['title'])+'</h1>',paper,count=1,flags=re.S)
    paper=re.sub(r'(<section class="abstract"[^>]*><h2[^>]*>Abstract</h2>).*?(</section>)',lambda m:m[1]+'<p>'+html.escape(manuscript['abstract'])+'</p>'+m[2],paper,flags=re.S)
    paper=re.sub(r'title = \{.*?\},',lambda m:'title = {'+manuscript['title']+'},',paper,count=1,flags=re.S)
    paper=paper.replace('Historical studies complete<br>Revised study pending','Historical and corrected studies complete')
    human_date=dt.date.fromisoformat(publication_date).strftime('%d %B %Y').lstrip('0')
    paper=re.sub(r'(<p class="eyebrow">Independent technical report · )[^<]*(</p>)',lambda match:match[1]+human_date+match[2],paper)
    (site/'paper.html').write_text(paper,encoding='utf-8')
    # Standalone report preserves historical exports and links to the complete raw comparison.
    esc=html.escape
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KernelBench-RL corrected study report</title><style>body{font:17px/1.65 system-ui;max-width:1100px;margin:3rem auto;padding:0 1.5rem;color:#102235}p{max-width:90ch}table{border-collapse:collapse;width:100%;min-width:650px;font-size:15px;margin:0}th,td{padding:.7rem;border-bottom:1px solid #bbc9d5;text-align:left;vertical-align:top}th{font-weight:650}code{overflow-wrap:anywhere;word-break:break-word}a{color:#006a7b}.note{background:#eaf2f6;padding:1rem}h1{line-height:1.15}caption{text-align:left;font-weight:700;padding:0 0 1rem}.report-table-scroll{overflow-x:auto;max-width:100%;margin:2rem 0;-webkit-overflow-scrolling:touch}.report-table-scroll:focus-visible{outline:3px solid #087ba3;outline-offset:4px}@media(max-width:600px){body{margin:2rem auto;padding:0 1rem}table{font-size:14px}th,td{padding:.65rem}}@media print{body{max-width:none;margin:0;padding:0;font:10.5pt/1.5 Georgia,serif;color:black}.report-table-scroll{overflow:visible;margin:1.2rem 0}table{min-width:0;width:100%;table-layout:fixed;font-size:9pt}thead{display:table-header-group}tfoot{display:table-footer-group}tr{break-inside:avoid}th,td{padding:.35rem;overflow-wrap:anywhere;word-break:break-word}h1,h2,h3,caption{break-after:avoid}a{color:black}.note{border:1px solid #bbb;background:white}code{font-size:8.5pt}}</style><h1>KernelBench-RL: corrected study report</h1><p>Eljo Mustafaj · '+publication_date+'</p><p class="note">'+esc(outcome)+'</p><p>These are bounded, single-attempt empirical results. The hybrid includes supervised training. Curated fixed-baseline performance is separate: those implementations were prepared with coding-assistant support and are distinct from programs sampled by the evaluated policies. No formal security or functional proof, statistical significance, production speedup, or full-parameter 7B result is claimed.</p>']
    if control_comparison: parts.append('<p>'+esc(control_comparison)+'</p>')
    if v3_discussion:
        parts.append('<h2>Separate prospective v3 follow-up</h2><p>'+esc(v3_discussion)+'</p><p>V3 uses fresh base/adapters, rank 16, a longer schedule and a new fresh trio. Former v2 fresh operators are v3 training targets. V2 and v3 are unpaired; no pooling, model-capacity or causal SFT/GRPO claim is supported.</p>')
    for table in tables:
        parts.append('<div class="report-table-scroll" role="region" tabindex="0" aria-label="'+esc(table['caption'],quote=True)+'"><table><caption>'+esc(table['caption'])+'</caption><thead><tr>'+''.join('<th scope="col">'+esc(c['label'])+'</th>' for c in table['columns'])+'</tr></thead><tbody>')
        for row in table['rows']: parts.append('<tr>'+''.join('<td>'+esc(str(row[c['key']]))+'</td>' for c in table['columns'])+'</tr>')
        parts.append('</tbody></table></div>')
    parts.append('<h2>Traceable evidence</h2><p><a href="https://github.com/Eljo-M/kernelbench-rl/tree/main/experiments/retry_v2">Raw traces, frozen protocol and strict comparison</a> · <a href="https://github.com/Eljo-M/kernelbench-rl/blob/main/RESULTS.md">Historical baseline and model results</a></p>')
    if v3_metric is not None:
        parts.append('<p><a href="https://github.com/Eljo-M/kernelbench-rl/tree/main/experiments/retry_v3">Separate v3 traces, prospective protocol, source provenance and strict comparison</a></p>')
    for m in metrics:
        e=m['environment']; parts.append('<h3>'+esc(m['label'])+'</h3><p>Study source <code>'+esc(e['source_commit'])+'</code>; config <code>'+esc(e['config_hash'])+'</code>. Recorded training-stage elapsed timer '+f"{e['seconds']:.2f}"+' s; parent-process peak allocated CUDA memory '+f"{e['peak_allocated_bytes']/2**30:.3f}"+' GiB. Elapsed time is not GPU compute time. This PyTorch allocator peak excludes child-worker and other applications\' device allocations.</p><p>Final trainable tensor SHA-256: <code>'+esc(e['trainable_weights_sha256'])+'</code>.</p>')
        for note in m.get('operational_notes',[]): parts.append('<p>'+esc(note)+'</p>')
    parts.append('<p>All numerical failures are retained. The three fresh operators share families with training operators. The original pre-SFT base anchors reference regularization. Source checks and process timeouts are development guards. Checkpoint reload evidence does not imply statistical reproducibility across new training runs.</p></html>')
    (out/'RETRY_FINAL_REPORT.html').write_text(''.join(parts),encoding='utf-8')
    cff=('cff-version: 1.2.0\nmessage: "If you use this artifact, please cite the technical report."\ntitle: "KernelBench-RL"\nversion: "0.2.0"\n'
         f'date-released: {publication_date}\n'
         'authors:\n  - family-names: "Mustafaj"\n    given-names: "Eljo"\n'
         'repository-code: "https://github.com/Eljo-M/kernelbench-rl"\nurl: "https://eljo-m.github.io/kernelbench-rl/"\n'
         'preferred-citation:\n  type: report\n  authors:\n    - family-names: "Mustafaj"\n      given-names: "Eljo"\n'
         f'  title: "{manuscript["title"]}"\n  year: {publication_date[:4]}\n'
         '  url: "https://eljo-m.github.io/kernelbench-rl/paper.html"\n')
    (out/'CITATION.cff').write_text(cff,encoding='utf-8')
    print(json.dumps({'manuscript':str(out/'KernelBench-RL-Manuscript.json'),'tracks':len(metrics),'publication_date':publication_date,'quality':outcome},ensure_ascii=False))


if __name__=='__main__': main()
