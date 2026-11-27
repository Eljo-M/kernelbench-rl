"""TEST ONLY: portable three-study publication fixture, never experiment evidence.

The finalizer's scientific/model validators are stubbed; only postprocessing is
executed. Data are synthetic and isolated in containment-checked TMP/work.
"""
from contextlib import contextmanager, redirect_stdout
import copy
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory,gettempdir
from types import ModuleType
from unittest.mock import patch
from tools.publication import finalize_retry_text as finalizer

REPO_ROOT=Path(__file__).resolve().parents[1]
TEST_WORK_ROOT=Path(os.environ.get('KERNELBENCH_PUBLICATION_TEST_WORK',str(Path(gettempdir())/'kernelbench-publication-tests'))).resolve()
TEST_WORK_ROOT.mkdir(parents=True,exist_ok=True)
METHODS=('base_pass1','base_best_of_8','sft_only_checkpoint','trained_checkpoint')
V2_FRESH=['l2_normalize_2048','fused_bias_silu_4096','matmul_residual_add_512']
V3_FRESH=['cosine_similarity_1024','fused_bias_gelu_4096','matmul_bias_silu_512']
DEV=['rmsnorm_4096','int8_weight_only_gemm_512','scaled_dot_product_attn_128']

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def save_rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(value)+'\n' for value in values),encoding='utf-8')


@contextmanager
def fixture(v3_updates=5,v3_equal_weights=False):
    with TemporaryDirectory(prefix='TEST_ONLY_three_study_rendering_',dir=TEST_WORK_ROOT) as tmp:
        base=Path(tmp).resolve()
        if base.parent!=TEST_WORK_ROOT: raise ValueError('TEST ONLY scratch path escaped work')
        repo=base/'TEST_ONLY_repository';site=base/'TEST_ONLY_site';paper=base/'TEST_ONLY_template';out=base/'TEST_ONLY_generated'
        repo.mkdir();site.mkdir();paper.mkdir();out.mkdir()
        checked={};reports={};sources={'v2':'1'*40,'v3':'2'*40}
        for key in ('v2/laptop','v2/laptop_7b','v3/laptop_7b'):
            version,name=key.split('/');root=repo/'experiments'/('retry_'+version);run=root/name
            run.mkdir(parents=True)
            v3=version=='v3';small=name=='laptop';epochs,groups=(8,16) if v3 else (4,8)
            training=[f'TEST_ONLY_training_{i}' for i in range(8)]+(V2_FRESH if v3 else [])
            fresh=V3_FRESH if v3 else V2_FRESH
            config=dict(train_ops=training,heldout_ops=fresh,development_eval_ops=DEV,seed=242 if v3 else 142,
                sft_epochs=epochs,max_steps=groups,group_size=2,sft_learning_rate=.0001 if v3 else .0002,
                learning_rate=.00002 if v3 else .00005,lora_r=16 if v3 else 8,lora_alpha=32 if v3 else 16,
                lora_target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if v3 else ['q_proj','k_proj','v_proj','o_proj'],
                resume_from='',training_mode='full_param_0.49b' if small else 'qlora_7b_nf4')
            save(run/'config.json',config)
            updates=v3_updates if v3 else (0 if small else 1)
            sft_sha=('a' if small else 'b')*64
            final_sha=sft_sha if small or (v3 and v3_equal_weights) else ('c' if v3 else 'd')*64
            env=dict(source_commit=sources[version],config_hash=('e' if v3 else 'f')*16,
                trainable_weights_sha256=final_sha,model_revision='9'*40,training_mode=config['training_mode'],
                gpu='TEST ONLY GPU; not a measurement',seconds=30.,peak_allocated_bytes=2**30,
                steps=groups,sft_optimizer_steps=epochs*len(training),optimizer_steps=updates)
            train=[dict(step=step,group_index=index,status='COMPILE_ERROR') for step in range(1,groups+1) for index in range(2)]
            save_rows(run/'train.jsonl',train)
            save_rows(run/'sft.jsonl',[dict(step=i,loss=1.) for i in range(1,epochs*len(training)+1)])
            save(run/'sft_summary.json',dict(optimizer_step_count=epochs*len(training),weight_sha256_after=sft_sha))
            per_cohort={};main=[];sft=[]
            for cohort,ops in [('fresh_heldout',fresh),('inspected_development',DEV)]:
                methods={}
                for method in METHODS:
                    if cohort=='inspected_development' and method=='base_best_of_8':
                        methods[method]=dict(available=False);continue
                    passes=0 if v3 or small or method in ('base_pass1','base_best_of_8') else (2 if cohort=='fresh_heldout' else 1)
                    records=[dict(op_name=op,status='PASS' if index<passes else 'COMPILE_ERROR',compiled=index<passes,correct=index<passes,
                        candidate_sha256='7'*64,method=method,eval_phase='heldout_test' if cohort=='fresh_heldout' else 'development_test') for index,op in enumerate(ops)]
                    methods[method]=dict(pass_count=passes,total_ops=3,budget_k=8 if method=='base_best_of_8' else 1,
                        fast_vs_eager_count=passes,failure_counts_by_status={'COMPILE_ERROR':3-passes} if passes<3 else {},per_operator=records)
                    (sft if method=='sft_only_checkpoint' else main).extend(records)
                per_cohort[cohort]=dict(ops=ops,methods=methods)
            main.extend(dict(op_name=op,method='base_best_of_8',eval_phase='selection_val',status='COMPILE_ERROR') for op in fresh for _ in range(8))
            save_rows(run/'heldout.jsonl',main);save_rows(run/'sft_only.jsonl',sft)
            quality_env=dict(run_name=name,run_dir=str(run),source_commit=sources[version],config_hash=env['config_hash'])
            q=dict(environment=quality_env,protocol_path=str(root/'protocol.json'),protocol_sha256='6'*64,
                sft_optimizer_steps=env['sft_optimizer_steps'],grpo_optimizer_steps=updates,
                sft_only_checkpoint_identity='checkpoint:0:'+sft_sha,trained_checkpoint_identity=f'checkpoint:{groups}:'+final_sha,
                strict_main_comparison={'environment':quality_env},**per_cohort)
            checked[key]=dict(environment=env,strict=q,final={'state':{'trainable_weights_sha256':final_sha}},sft={'state':{'trainable_weights_sha256':sft_sha}})
            reports.setdefault(version,[]).append(q)
            save(root/'protocol.json',dict(fixture_notice='TEST ONLY',adapter={'expected_trainable_parameters':40370176}))
        for version,runs in reports.items():save(repo/'experiments'/('retry_'+version)/'comparison/retry_comparison.json',dict(fixture_notice='TEST ONLY',runs=runs))
        v3root=repo/'experiments/retry_v3';save(v3root/'source_provenance.json',{'fixture_notice':'TEST ONLY'})
        registration=dict(study_key='v3/laptop_7b',run_dir=v3root/'laptop_7b',source_commit=sources['v3'],protocol=v3root/'protocol.json',provenance_path=v3root/'source_provenance.json',provenance={'fixture_notice':'TEST ONLY'})
        save(repo/'experiments/retry_v2/hardware_environment.json',dict(cpu={'Name':'TEST ONLY CPU','NumberOfCores':4},reported_physical_memory_bytes=16*2**30,
            gpu_csv='TEST ONLY GPU, TEST DRIVER, TEST MEMORY',os={'Caption':'TEST ONLY OS','BuildNumber':'TEST'},torch_cpu_threads=4))
        figure=out/'paper-assets/sft_training.png';figure.parent.mkdir();figure.write_bytes(b'TEST ONLY opaque hash fixture; not an image')
        save(figure.with_suffix('.json'),dict(figure_sha256=hashlib.sha256(figure.read_bytes()).hexdigest(),caption='TEST ONLY layout fixture',tracks=[dict(name=name,
            sft_log_sha256=hashlib.sha256((repo/'experiments/retry_v2'/name/'sft.jsonl').read_bytes()).hexdigest(),
            config_file_sha256=hashlib.sha256((repo/'experiments/retry_v2'/name/'config.json').read_bytes()).hexdigest()) for name in ('laptop','laptop_7b')]))
        section_ids=['introduction','related_work','system_methods','original_evidence','failure_diagnosis','corrected_protocol','applications','limitations','reproducibility_ethics']
        sections=[dict(id=id,heading=f'{i}. TEST ONLY {id}',paragraphs=['TEST ONLY fixture paragraph.']*(6 if id=='corrected_protocol' else 2)) for i,id in enumerate(section_ids,1)]
        sections[2]['paragraphs'][0]='The 7B track trains 5,046,272 adapter parameters.'
        save(paper/'manuscript_draft.json',dict(title='TEST ONLY publication rendering fixture',abstract='TEST ONLY',sections=sections))
        save(site/'results.json',dict(fixture_notice='TEST ONLY',retry={'runs':[]},provenance=[]))
        for filename in ('index.html','paper.html'):
            archive=REPO_ROOT/'website/archive-v0.2'
            original_site=archive if archive.is_dir() else REPO_ROOT/'docs'
            (site/filename).write_text((original_site/filename).read_text(encoding='utf-8'),encoding='utf-8')
        compare_module=ModuleType('eval.compare_retry');compare_module.load_retry_run=lambda *args,**kwargs:None
        def no_regenerate(*args,**kwargs):raise AssertionError('Preserved fixture comparison was regenerated')
        compare_module.compare_retry_runs=no_regenerate
        export_module=ModuleType('export_retry');export_module.STUDY_COMMIT=sources['v2']
        export_module.registered_v3_study=lambda repo:registration
        export_module.validate_run=lambda repo,folder,*args:checked[folder.parent.name.replace('retry_','')+'/'+folder.name]
        export_module.validate_verifier_regression=lambda *args:dict(directory=repo/'TEST_ONLY_regression',checks=60,passed=60,source_commit=sources['v2'])
        cfg_module=ModuleType('rl_trainer.experiment_config');cfg_module.ExperimentConfig=object
        eval_package=ModuleType('eval');eval_package.__path__=[]
        trainer_package=ModuleType('rl_trainer');trainer_package.__path__=[]
        modules={'eval':eval_package,'eval.compare_retry':compare_module,'rl_trainer':trainer_package,'rl_trainer.experiment_config':cfg_module,'export_retry':export_module}
        argv=['TEST_ONLY_finalizer','--repo',str(repo),'--site',str(site),'--paper-source',str(paper),'--out',str(out),'--publication-date','2026-11-27']
        original_path=list(sys.path)
        try:
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',argv),redirect_stdout(StringIO()):
                yield dict(repo=repo,site=site,out=out,run=finalizer.main)
        finally:sys.path[:]=original_path
