"""Derive publication text from complete, strictly validated retry artifacts."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import datetime as dt
import hashlib
import html
import json
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--paper-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--publication-date', required=True, help='Actual author-local publication date, YYYY-MM-DD')
    parser.add_argument('--training-figure', type=Path)
    parser.add_argument('--manuscript-template', type=Path)
    args=parser.parse_args()
    repo=args.repo.resolve(); out=args.out.resolve(); site=args.site.resolve()
    sys.path.insert(0,str(repo))
    from eval.compare_retry import compare_retry_runs, load_retry_run
    from rl_trainer.experiment_config import ExperimentConfig
    from export_retry import validate_run, validate_verifier_regression, STUDY_COMMIT
    run_dirs=[repo/'experiments/retry_v2'/n for n in ('laptop','laptop_7b')]
    validated_runs={folder.name:validate_run(repo,folder,STUDY_COMMIT,load_retry_run,ExperimentConfig,repo/'experiments/retry_v2/protocol.json') for folder in run_dirs}
    verifier_regression=validate_verifier_regression(repo,STUDY_COMMIT)
    report=compare_retry_runs(run_dirs,
        repo/'experiments/retry_v2/comparison', historical_run_dirs=[repo/'experiments/laptop_fullparam',repo/'experiments/colab_t4/qlora7b_fresh'])
    labels={'laptop':'0.49B full-parameter FP32','laptop_7b':'7B QLoRA NF4 / FP16 decoder / CPU vocabulary'}
    method_labels={'base_pass1':'Base pass@1','base_best_of_8':'Base best-of-8','sft_only_checkpoint':'SFT-only pass@1','trained_checkpoint':'Hybrid pass@1'}
    metrics=[]
    for run in report['runs']:
        name=run['environment']['run_name']; folder=repo/'experiments/retry_v2'/name
        validated=validated_runs[name]
        env=validated['environment']; train=rows(folder/'train.jsonl'); sft=read(folder/'sft_summary.json')
        if validated['strict']!=run: raise ValueError('Strict report changed during publication validation')
        if run['grpo_optimizer_steps']==0 and validated['final']['state']['trainable_weights_sha256']!=validated['sft']['state']['trainable_weights_sha256']:
            raise ValueError('Zero-GRPO study has different final and SFT-only tensor checksums')
        status_counts=dict(Counter(r['status'] for r in train))
        metric=dict(name=name,label=labels[name],environment=env,sft_summary=sft,
            training_status_counts=status_counts,training_failures={key:value for key,value in status_counts.items() if key!='PASS'},quality=run)
        metrics.append(metric)
    publication_date=dt.date.fromisoformat(args.publication_date).isoformat()
    training_figure=(args.training_figure or out/'paper-assets/sft_training.png').resolve()
    plot=read(training_figure.with_suffix('.json'))
    if plot['figure_sha256']!=hashlib.sha256(training_figure.read_bytes()).hexdigest(): raise ValueError('Training figure changed')
    if {track['name'] for track in plot['tracks']}!={'laptop','laptop_7b'}: raise ValueError('Training plot track mismatch')
    for track in plot['tracks']:
        folder=repo/'experiments/retry_v2'/track['name']
        if track['sft_log_sha256']!=hashlib.sha256((folder/'sft.jsonl').read_bytes()).hexdigest(): raise ValueError('Training plot log mismatch')
        if track['config_file_sha256']!=hashlib.sha256((folder/'config.json').read_bytes()).hexdigest(): raise ValueError('Training plot config mismatch')
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
        result_sentences.append(f"The corrected {m['label']} study completed {q['sft_optimizer_steps']} supervised updates and {q['grpo_optimizer_steps']} GRPO optimizer updates; fresh held-out counts were {counts}.")
    outcome=' '.join(result_sentences)
    manuscript['abstract']=re.sub(r'\[DRAFT:.*?\]',outcome,manuscript['abstract'])
    sections={s['id']:s for s in manuscript['sections']}
    for section in manuscript['sections']:
        section['paragraphs']=[re.sub(r'\s*\[DRAFT:.*?\]','',p) for p in section['paragraphs']]
    sections['introduction']['paragraphs'][-1]+=' '+outcome
    hardware=read(repo/'experiments/retry_v2/hardware_environment.json')
    gpu_name,driver,device_memory=[part.strip() for part in hardware['gpu_csv'].split(',')]
    sections['system_methods']['paragraphs'].append(f"Both corrected studies used an {hardware['cpu']['Name'].strip()} CPU ({hardware['cpu']['NumberOfCores']} physical cores), {hardware['reported_physical_memory_bytes']/2**30:.2f} GiB reported physical RAM, and a {gpu_name} with {device_memory} reported device memory. The system ran {hardware['os']['Caption']} build {hardware['os']['BuildNumber']} with NVIDIA driver {driver}. PyTorch used {hardware['torch_cpu_threads']} CPU threads. The hardware descriptor was recorded during the studies.")
    sections['original_evidence']['paragraphs'].append('After the retry hardened output dtype/device and post-timing checks, a separate regression passed all sixty operator/precision records across the thirty fixed baselines in fp32 and bf16, with seed 142 and independent seed 10142. This verifies the updated worker on its canonical and registered stress inputs. It is a distinct correctness campaign and does not replace or re-estimate the historical 180-record speedup aggregates.')
    protocol=sections['corrected_protocol']['paragraphs']
    protocol[0]=protocol[0].replace('The retry freezes','The frozen retry used').replace('The intended supervised phase contains thirty-two example updates; actual completed counters must be read from durable logs and checkpoints.','Both tracks completed thirty-two supervised example updates, verified in durable logs and checkpoints.')
    protocol[1]=protocol[1].replace('not evidence that supervision or the gate has already improved quality','not an assumption that supervision or the gate improves quality')
    protocol[-1]=protocol[-1].replace('the retry is planned sequentially on the laptop','the retry completed sequentially on the laptop')
    protocol.append('Both tracks retain the original pre-SFT base as the reference-policy anchor: a frozen CPU copy for the full-parameter model and disabled adapters over the frozen quantized base for QLoRA. The reference penalty therefore regularizes toward the original model rather than the post-SFT checkpoint. The three fresh targets are unseen operator programs within normalization, activation and matrix-product families represented in training; this is not an unseen-family out-of-distribution test. Here, fresh means excluded from model training and from previous generated-program test inspection, not that their fixed reference implementations were absent from the public project.')
    for m in metrics:
        e=m['environment']; q=m['quality']; fail=', '.join(f'{k}={v}' for k,v in m['training_status_counts'].items())
        protocol.append(f"{m['label']}: training completed in {e['seconds']:.2f} seconds, with {e['peak_allocated_bytes']/2**30:.3f} GiB parent-process peak allocated CUDA memory during training. This allocator measurement excludes child-process and other applications' device allocations. Generated training statuses were {fail}. The full study includes a model-checksum-verified reload restoring optimizer and RNG state with zero extra training steps, main evaluation and the separate SFT control. The reload is not an independent next-step trajectory replication or a separate optimizer/RNG digest comparison. Its final trainable tensor checksum is {e['trainable_weights_sha256']}. These tensor digests are distinct from serialized-file and ZIP digests.")
        if q['grpo_optimizer_steps']==0:
            protocol.append(f"For {m['label']}, the correctness-gated schedule made no GRPO optimizer update. The final checkpoint consequently retains the SFT-only weights; any difference in sampled outputs or timing is not an RL parameter gain. Zero-valued update diagnostics on skipped groups are sentinels, not measured old/new-likelihood agreement for a performed update.")
    q7_run=next(m['quality'] for m in metrics if m['name']=='laptop_7b')
    q7_methods=q7_run['fresh_heldout']['methods']
    q7_passes={method:{row['op_name'] for row in q7_methods[method]['per_operator']
                      if row['status']=='PASS' and row['compiled'] and row['correct']}
               for method in ('sft_only_checkpoint','trained_checkpoint')}
    control_comparison=''
    if q7_passes['sft_only_checkpoint']==q7_passes['trained_checkpoint']:
        shared=sorted(q7_passes['sft_only_checkpoint'])
        details=' The identical passing operators were '+', '.join(shared)+'.' if shared else ''
        control_comparison=(f"The 7B SFT-only and hybrid checkpoints each passed {len(shared)}/{q7_methods['sft_only_checkpoint']['total_ops']} fresh held-out operators."
                            +details+' No incremental GRPO held-out correctness gain over the SFT-only control was observed in this single configured attempt.')
        protocol.append(control_comparison)
    protocol.append(outcome+' All failures remain in pass-rate and fast-versus-eager denominators; correct-only speedup and compiled-reference availability are labeled separately. These single-attempt observations do not establish statistical significance or a general learning advantage.')
    tables=[]
    for cohort,key in [('Fresh held-out','fresh_heldout'),('Previously inspected development','inspected_development')]:
        table_rows=[]
        for m in metrics:
            for method,stats in m['quality'][key]['methods'].items():
                if stats.get('available') is False: continue
                table_rows.append({'track':m['label'],'method':method_labels[method],'budget':stats['budget_k'],'passes':f"{stats['pass_count']}/{stats['total_ops']}",'fast':f"{stats['fast_vs_eager_count']}/{stats['total_ops']}"})
        tables.append(dict(id=key,section_id='corrected_protocol',caption=cohort+' results, separated by model track.',columns=[{'key':'track','label':'Model / training','weight':2.2},{'key':'method','label':'Method','weight':1.6},{'key':'budget','label':'Budget','weight':.65,'align':'RIGHT'},{'key':'passes','label':'Pass / N','weight':.8,'align':'RIGHT'},{'key':'fast','label':'Fast / N (eager)','weight':1,'align':'RIGHT'}],rows=table_rows,notes=['N=3 operators per method and model. Selection rows are excluded. The hybrid includes SFT; no RL-only effect is established.']))
    paired=[]
    short_status={'PASS':'PASS','CHEAT':'POLICY','COMPILE_ERROR':'COMPILE','NUMERICAL_MISMATCH':'NUMERIC','RUNTIME_ERROR':'RUNTIME','TIMEOUT':'TIMEOUT','OOM':'OOM'}
    for m in metrics:
        folder=repo/'experiments/retry_v2'/m['name']
        finals={(r['method'],r['op_name']):r['status'] for filename in ('heldout.jsonl','sft_only.jsonl') for r in rows(folder/filename) if r['eval_phase']=='heldout_test'}
        for op in m['quality']['fresh_heldout']['ops']:
            paired.append(dict(track='0.49B FP32' if m['name']=='laptop' else '7B QLoRA',operator=op,
                base=short_status[finals[('base_pass1',op)]],best=short_status[finals[('base_best_of_8',op)]],
                sft=short_status[finals[('sft_only_checkpoint',op)]],hybrid=short_status[finals[('trained_checkpoint',op)]]))
    tables.append(dict(id='paired_fresh_status',section_id='corrected_protocol',caption='Paired fresh-operator outcomes, retaining every failure.',columns=[{'key':'track','label':'Model','weight':1.2},{'key':'operator','label':'Operator','weight':2.2},{'key':'base','label':'Base @1','weight':1},{'key':'best','label':'Best @8','weight':1},{'key':'sft','label':'SFT @1','weight':1},{'key':'hybrid','label':'Hybrid @1','weight':1}],rows=paired,notes=['POLICY: source-policy rejection; COMPILE: compile error; NUMERIC: numerical mismatch; RUNTIME: candidate runtime error. Full diagnostics and source are in the raw records.']))
    manuscript['tables']=tables
    manuscript['figures']=[dict(id='observed_sft_training',section_id='corrected_protocol',after_paragraph=6,path=training_figure.relative_to(out).as_posix(),caption=plot['caption'],max_height_inches=4.2)]
    repro=sections['reproducibility_ethics']['paragraphs']
    repro[0]=repro[0].replace('The public artifact should link','The public artifact links').replace('The final paper and explanatory website should share','The final paper and explanatory website share')
    repro[0]+=f' Both corrected quality studies executed from public source commit {STUDY_COMMIT}, frozen and pushed before their fresh test generation. The later publication snapshot also contains result files and publication assets; its distinct commit is recorded in the versioned delivery manifest.'
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
        normalized.append(dict(label=m['label'],hardware=e['gpu'],training_mode=e['training_mode'],status='completed',source_commit=e['source_commit'],sft_updates=q['sft_optimizer_steps'],optimizer_updates=q['grpo_optimizer_steps'],cohorts=cohorts))
    data['retry']=dict(status='completed',published_results=True,runs=normalized,note=outcome+' '+control_comparison+' Fresh and inspected development cohorts are reported separately; the hybrid includes supervised warmup.')
    data['verifier_regression']={key:value for key,value in verifier_regression.items() if key!='directory'}
    comparison_path=repo/'experiments/retry_v2/comparison/retry_comparison.json'
    comparison_name=str(comparison_path.relative_to(repo)).replace('\\','/')
    data['provenance']=[item for item in data['provenance'] if item['path']!=comparison_name]
    data['provenance'].append(dict(path=comparison_name,sha256=hashlib.sha256(comparison_path.read_bytes()).hexdigest()))
    write(site/'results.json',data)
    index=(site/'index.html').read_text(encoding='utf-8').replace('Open source ·','Public code ·').replace('Results pending.','Corrected studies completed.')
    q7=next(m['quality'] for m in metrics if m['name']=='laptop_7b')['fresh_heldout']['methods']['trained_checkpoint']
    index=index.replace('<p class="metric">0 / 3</p><p>historical trained-model held-out passes, per model</p>',f'<p class="metric">{q7["pass_count"]} / {q7["total_ops"]}</p><p>corrected 7B hybrid fresh held-out passes · one attempt</p>')
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
    paper=paper.replace('26 November 2026',human_date)
    (site/'paper.html').write_text(paper,encoding='utf-8')
    # Standalone report preserves historical exports and links to the complete raw comparison.
    esc=html.escape
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KernelBench-RL corrected study report</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:3rem auto;padding:0 1.5rem;color:#102235}table{border-collapse:collapse;width:100%;font-size:15px;margin:2rem 0}th,td{padding:.7rem;border-bottom:1px solid #bbc9d5;text-align:left;vertical-align:top}code{overflow-wrap:anywhere}a{color:#006a7b}.note{background:#eaf2f6;padding:1rem}h1{line-height:1.15}caption{text-align:left;font-weight:700}</style><h1>KernelBench-RL: corrected study report</h1><p>Eljo Mustafaj · '+publication_date+'</p><p class="note">'+esc(outcome)+'</p><p>These are bounded, single-attempt empirical results. The hybrid includes supervised training. Curated fixed-baseline performance is separate: those implementations were prepared with coding-assistant support and are distinct from programs sampled by the evaluated policies. No formal security or functional proof, statistical significance, production speedup, or full-parameter 7B result is claimed.</p>']
    if control_comparison: parts.append('<p>'+esc(control_comparison)+'</p>')
    for table in tables:
        parts.append('<div style="overflow-x:auto"><table style="min-width:650px"><caption>'+esc(table['caption'])+'</caption><thead><tr>'+''.join('<th>'+esc(c['label'])+'</th>' for c in table['columns'])+'</tr></thead><tbody>')
        for row in table['rows']: parts.append('<tr>'+''.join('<td>'+esc(str(row[c['key']]))+'</td>' for c in table['columns'])+'</tr>')
        parts.append('</tbody></table></div>')
    parts.append('<h2>Traceable evidence</h2><p><a href="https://github.com/Eljo-M/kernelbench-rl/tree/main/experiments/retry_v2">Raw traces, frozen protocol and strict comparison</a> · <a href="https://github.com/Eljo-M/kernelbench-rl/blob/main/RESULTS.md">Historical baseline and model results</a></p>')
    for m in metrics:
        e=m['environment']; parts.append('<h3>'+esc(m['label'])+'</h3><p>Study source <code>'+esc(e['source_commit'])+'</code>; config <code>'+esc(e['config_hash'])+'</code>. Training '+f"{e['seconds']:.2f}"+' s; parent-process peak allocated CUDA memory '+f"{e['peak_allocated_bytes']/2**30:.3f}"+' GiB. This PyTorch allocator peak excludes child-worker and other applications\' device allocations.</p><p>Final trainable tensor SHA-256: <code>'+esc(e['trainable_weights_sha256'])+'</code>.</p>')
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
