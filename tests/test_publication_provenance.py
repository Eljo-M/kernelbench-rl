"""Portable publication regressions using labeled synthetic metadata only.

No model/GPU/network/Git operations. Scratch stays under the selected work/TMP
root; the generated fixture directory is containment-checked before cleanup.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextlib import redirect_stdout
import copy
import os
import random
import tempfile
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from tools.publication import export_retry as export
from tools.publication import verify_publication_artifacts as check

TEST_WORK_ROOT=Path(os.environ.get('KERNELBENCH_PUBLICATION_TEST_WORK',
    str(Path(tempfile.gettempdir())/'kernelbench-publication-tests'))).resolve()
TEST_WORK_ROOT.mkdir(parents=True,exist_ok=True)
SOURCE='a'*40
CONFIG='b'*16


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


@contextmanager
def fixture():
    temporary=TemporaryDirectory(prefix='synthetic-publication-v3-',dir=TEST_WORK_ROOT)
    root=Path(temporary.name).resolve()
    if root.parent!=TEST_WORK_ROOT:
        raise ValueError('Synthetic fixture escaped the intended work directory')
    try:
        yield root
    finally:
        temporary.cleanup()


def registration(root):
    directory=root/'experiments/retry_v3'; run=directory/'laptop_7b'
    cfg=dict(max_steps=16,sft_epochs=8,group_size=2,training_mode='qlora_7b_nf4',
             resume_from='',train_ops=['synthetic_train_'+str(i) for i in range(11)],
             heldout_ops=['synthetic_fresh'],development_eval_ops=['synthetic_dev'],
             lora_r=16,lora_alpha=32,lora_target_modules=['synthetic_projection'])
    protocol=dict(config_hashes={'laptop_7b':CONFIG},
                  run_schedules={'laptop_7b':{k:cfg[k] for k in ('max_steps','sft_epochs','group_size')}},
                  fresh_heldout_ops=cfg['heldout_ops'],development_regressions=cfg['development_eval_ops'],
                  train_ops=cfg['train_ops'],
                  adapter=dict(rank=16,alpha=32,target_modules=cfg['lora_target_modules']))
    save(run/'config.json',cfg); save(directory/'protocol.json',protocol)
    provenance=dict(schema_version='kernelbench_study_source_provenance_v1',study_source_commit=SOURCE,
                    protocol_path='experiments/retry_v3/protocol.json',
                    protocol_sha256=hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest(),
                    config_hashes={'laptop_7b':CONFIG},published_before_training=True,
                    source_ci_url='https://github.com/Eljo-M/kernelbench-rl/actions/runs/123')
    save(directory/'source_provenance.json',provenance)
    return run,directory/'protocol.json',directory/'source_provenance.json'


class RegistrationTests(unittest.TestCase):
    def test_absent_and_pending_are_distinct(self):
        with fixture() as root:
            for module in (export,check):
                self.assertIsNone(module.registered_v3_study(root,expected_source=SOURCE,expected_config=CONFIG))
            (root/'experiments/retry_v3').mkdir(parents=True)
            for module in (export,check):
                with self.assertRaisesRegex(ValueError,'pending'):
                    module.registered_v3_study(root,expected_source=SOURCE,expected_config=CONFIG)
            absent_repo=root/'synthetic_v2_only_repo'; absent_repo.mkdir()
            output=root/'synthetic_required_v3_refusal.json'
            with redirect_stdout(StringIO()),self.assertRaisesRegex(ValueError,'requires the independent v3'):
                check.main(['--repo',str(absent_repo),'--require-v3','--output',str(output)])
            self.assertIs(check.read(output)['verified'],False)

    def test_registration_keeps_distinct_source_and_key(self):
        with fixture() as root:
            registration(root)
            results=[module.registered_v3_study(root,expected_source=SOURCE,expected_config=CONFIG) for module in (export,check)]
            self.assertEqual(results[0],results[1])
            self.assertEqual(results[0]['source_commit'],SOURCE)
            self.assertEqual(results[0]['study_key'],'v3/laptop_7b')
            self.assertNotEqual(results[0]['source_commit'],export.STUDY_COMMIT)
            for module in (export,check):
                with self.assertRaisesRegex(ValueError,'frozen study identity'):
                    module.registered_v3_study(root)

    def test_changed_protocol_bytes_are_refused(self):
        with fixture() as root:
            _,protocol,_=registration(root)
            protocol.write_bytes(protocol.read_bytes()+b'\r\n')
            for module in (export,check):
                with self.assertRaisesRegex(ValueError,'protocol bytes'):
                    module.registered_v3_study(root,expected_source=SOURCE,expected_config=CONFIG)

    def test_cross_protocol_hash_and_v2_resume_are_refused(self):
        with fixture() as root:
            run,_,provenance=registration(root)
            data=check.read(provenance); data['config_hashes']['laptop_7b']='c'*16; save(provenance,data)
            for module in (export,check):
                with self.assertRaisesRegex(ValueError,'configuration identities'):
                    module.registered_v3_study(root,expected_source=SOURCE,expected_config=CONFIG)
            registration(root)
            cfg=check.read(run/'config.json'); cfg['resume_from']='checkpoints/retry_v2_laptop_7b'; save(run/'config.json',cfg)
            for module in (export,check):
                with self.assertRaisesRegex(ValueError,'not resume v2'):
                    module.registered_v3_study(root,expected_source=SOURCE,expected_config=CONFIG)
            _,_,provenance=registration(root)
            receipt=provenance.with_name('source_ci_verified.json')
            save(receipt,dict(authenticated_login='Eljo-M',requested_source_commit=SOURCE,
                 runs=[dict(html_url='https://github.com/Eljo-M/kernelbench-rl/actions/runs/123',
                            head_sha='d'*40,status='completed',conclusion='success')]))
            data=check.read(provenance)
            data.update(source_ci_receipt_path='experiments/retry_v3/source_ci_verified.json',
                        source_ci_receipt_sha256=check.sha(receipt),source_ci_conclusion='success')
            save(provenance,data)
            for module in (export,check):
                with self.assertRaisesRegex(ValueError,'exact source'):
                    module.registered_v3_study(root,expected_source=SOURCE,expected_config=CONFIG)

    def test_completed_stage_requires_its_own_source(self):
        with fixture() as root:
            path=root/'synthetic_stage.json'
            save(path,dict(completed_utc='2026-01-01T00:00:00+00:00',config_hash=CONFIG,
                           source_commit=SOURCE,source_code_dirty=False))
            self.assertEqual(check.complete(path,CONFIG,SOURCE)['source_commit'],SOURCE)
            self.assertEqual(export.completed(path,CONFIG,SOURCE)['source_commit'],SOURCE)
            with self.assertRaisesRegex(ValueError,'source/config'):
                check.complete(path,CONFIG,export.STUDY_COMMIT)

    def test_v3_requires_new_post_run_upstream_proof(self):
        with fixture() as root:
            original=root/'experiments/retry_v2/upstream_expected_metadata.json'
            save(original,{'synthetic_fixture':True})
            relative='experiments/retry_v3/upstream_model_integrity.json'
            save(root/relative,dict(schema_version='kernelbench_upstream_model_integrity_v1',
                verified=True,metadata_only=False,all_local_weight_hashes_verified=True,
                small_model_files_verified=True,started_utc='2026-01-01T00:00:00+00:00',
                completed_utc='2026-01-01T00:01:00+00:00'))
            after=check.timestamp('2026-01-01T01:00:00+00:00','Synthetic completed stage')
            with self.assertRaisesRegex(ValueError,'new full upstream'):
                check.verify_upstream(root,relative,not_before=after)


class ScheduleTests(unittest.TestCase):
    def synthetic_schedule(self,root,op_count,epochs,groups):
        cfg=SimpleNamespace(train_ops=['synthetic_op_'+str(i) for i in range(op_count)],
                            sft_epochs=epochs,max_steps=groups,group_size=2,backend='triton',dtype='fp16',
                            config_hash=lambda:CONFIG)
        updates=epochs*op_count; state={'trainable_weights_sha256':'d'*64}
        save(root/'sft_summary.json',dict(ran=True,optimizer_step_count=updates,weight_sha256_after='d'*64))
        sft=[dict(step=(epoch-1)*op_count+i+1,epoch=epoch,op_name=op,loss=1.0,grad_norm=0.1)
             for epoch in range(1,epochs+1) for i,op in enumerate(cfg.train_ops)]
        rows=[dict(step=step,group_index=index,op_name=cfg.train_ops[(step-1)%op_count],config_hash=CONFIG,
                   backend='triton',dtype='fp16',status='COMPILE_ERROR',gpu_model='unavailable',compiled=False,
                   correct=False,sft_optimizer_steps=updates,training_reward_mode='correctness_gated',
                   policy_updated=False,optimizer_step_count=0,reward=0.0)
              for step in range(1,groups+1) for index in range(2)]
        for filename,records in (('sft.jsonl',sft),('train.jsonl',rows)):
            (root/filename).write_text(''.join(json.dumps(row)+'\n' for row in records),encoding='utf-8')
        env=dict(gpu='Synthetic test GPU',dtype='fp16',optimizer_steps=0)
        return cfg,env,state

    def test_v2_and_v3_counts_use_the_configured_schedule(self):
        for op_count,epochs,groups in ((8,4,8),(11,8,16)):
            with self.subTest(ops=op_count),fixture() as root:
                cfg,env,state=self.synthetic_schedule(root,op_count,epochs,groups)
                export.validate_training(root,cfg,env,state)
                rows=(root/'train.jsonl').read_text(encoding='utf-8').splitlines()
                self.assertEqual(len(rows),groups*2)
                (root/'train.jsonl').write_text('\n'.join(rows[:-1])+'\n',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'rollout inventory'):
                    export.validate_training(root,cfg,env,state)


def followup_fixture(root):
    run=root/'experiments/retry_v3/laptop_7b'; paper=root/'paper/paper-assets'
    cfg=dict(seed=242,sft_epochs=8,train_ops=['synthetic_op_'+str(i) for i in range(11)],
             lora_r=16,lora_alpha=32,lora_target_modules=['synthetic_projection'],
             model_revision='e'*40,sft_learning_rate=0.0001)
    save(run/'config.json',cfg)
    protocol=run.parent/'protocol.json'; provenance=run.parent/'source_provenance.json'
    save(protocol,dict(synthetic_fixture=True));save(provenance,dict(synthetic_fixture=True))
    records=[]; examples=[dict(op_name=op,prompt_tokens=1,completion_tokens=1,
                              prompt_sha256='a'*64,source_sha256='b'*64) for op in cfg['train_ops']]
    for epoch in range(1,9):
        order=list(range(11));random.Random(242+1000*epoch).shuffle(order)
        for i in order:
            records.append(dict(examples[i],step=len(records)+1,epoch=epoch,loss=1.0))
    (run/'sft.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records),encoding='utf-8')
    save(run/'sft_summary.json',dict(ran=True,optimizer_step_count=88,weight_sha256_after='d'*64,records=records))
    identity='checkpoint:0:'+'d'*64
    save(run/'sft_only_environment.json',dict(checkpoint_identity=identity,checkpoint_weights_sha256='d'*64))
    save(run.parent/'token_feasibility.json',dict(config_hash='c'*16,examples=examples))
    paper.mkdir(parents=True,exist_ok=True)
    png=paper/'sft_followup.png';png.write_bytes(b'\x89PNG\r\n\x1a\nsynthetic-fixture')
    v3=dict(run_dir=run,study_key='v3/laptop_7b',config_hash='c'*16,source_commit='f'*40,
            protocol=protocol,provenance_path=provenance)
    metadata=dict(schema_version='kernelbench_sft_followup_plot_v1',study_key=v3['study_key'],
        config_hash=v3['config_hash'],source_commit=v3['source_commit'],seed=242,epochs=8,
        examples_per_epoch=11,completed_steps=88,adapter_rank=16,adapter_alpha=32,
        adapter_target_modules=cfg['lora_target_modules'],train_ops=cfg['train_ops'],
        model_revision=cfg['model_revision'],sft_learning_rate=cfg['sft_learning_rate'],
        inference_quality_claim=False,sft_checkpoint_identity=identity,sft_checkpoint_weights_sha256='d'*64,
        sft_checkpoint_metadata_sha256='e'*64,epoch_means_nats_per_token=[1.0]*8,
        observed_operator_order=[row['op_name'] for row in records],bytes=png.stat().st_size)
    for key,path in dict(figure_sha256=png,config_file_sha256=run/'config.json',
        sft_log_sha256=run/'sft.jsonl',sft_summary_sha256=run/'sft_summary.json',
        protocol_sha256=protocol,source_provenance_sha256=provenance,
        token_feasibility_sha256=run.parent/'token_feasibility.json').items():
        metadata[key]=check.sha(path)
    save(png.with_suffix('.json'),metadata)
    return v3,png.with_suffix('.json')


class FollowupTests(unittest.TestCase):
    def test_missing_required_plot_is_refused(self):
        with fixture() as root:
            self.assertIsNone(check.verify_followup_plot(root,None))
            with self.assertRaisesRegex(ValueError,'registered v3'):
                check.verify_followup_plot(root,None,required=True)

    def test_complete_source_bindings_need_no_excluded_checkpoint_files(self):
        with fixture() as root:
            v3,_=followup_fixture(root)
            result=check.verify_followup_plot(root,v3,required=True)
            self.assertTrue(result['verified'])
            self.assertFalse((root/'checkpoints').exists())

    def test_stale_figure_and_incorrect_mean_are_refused(self):
        with fixture() as root:
            v3,path=followup_fixture(root)
            metadata=check.read(path);metadata['epoch_means_nats_per_token'][0]=2.0;save(path,metadata)
            with self.assertRaisesRegex(ValueError,'epoch means'):
                check.verify_followup_plot(root,v3,required=True)
            v3,path=followup_fixture(root)
            png=path.with_suffix('.png');png.write_bytes(png.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'saved bytes differ'):
                check.verify_followup_plot(root,v3,required=True)


class AdapterTests(unittest.TestCase):
    def test_rank_target_and_structural_overrides_are_refused(self):
        cfg=SimpleNamespace(lora_r=16,lora_alpha=32,lora_dropout=0.0,
            lora_target_modules=['q_proj','v_proj'],model_revision='a'*40,
            model_name_or_path='../../work/models/synthetic-test-model')
        valid=dict(r=16,lora_alpha=32,lora_dropout=0.0,bias='none',peft_type='LORA',
            task_type='CAUSAL_LM',base_model_name_or_path=cfg.model_name_or_path,
            revision=None,target_modules=['v_proj','q_proj'],rank_pattern={},alpha_pattern={})
        with fixture() as root:
            path=root/'adapter_config.json';save(path,valid)
            export.validate_adapter_config(path,cfg)
            for key,value in (('r',8),('target_modules',['q_proj']),
                              ('rank_pattern',{'q_proj':8}),('base_model_name_or_path','wrong')):
                wrong=copy.deepcopy(valid);wrong[key]=value;save(path,wrong)
                with self.subTest(field=key),self.assertRaises(ValueError):
                    export.validate_adapter_config(path,cfg)


URL='https://github.com/Eljo-M/kernelbench-rl/actions/runs/123'


class FinalGateTests(unittest.TestCase):
    def test_exact_successful_main_ci_and_owner_required(self):
        with fixture() as root:
            path=root/'synthetic_ci.json'
            run=dict(name='KernelBench-RL CI',head_sha=SOURCE,head_branch='main',status='completed',
                     conclusion='success',html_url=URL)
            data=dict(authenticated_login='Eljo-M',requested_source_commit=SOURCE,
                      observed_utc='2026-01-01T00:00:00+00:00',runs=[run])
            save(path,data);self.assertTrue(export.validate_ci_verification(path,SOURCE,URL)['verified'])
            for field,value in (('head_branch','codex/test'),('head_sha','b'*40),
                                ('status','in_progress'),('conclusion','failure')):
                original=run[field];run[field]=value;save(path,data)
                with self.subTest(field=field),self.assertRaises(ValueError):
                    export.validate_ci_verification(path,SOURCE,URL)
                run[field]=original

    def test_v3_requires_clone_complete_figure_and_exact_study_inventory(self):
        with fixture() as root:
            path=root/'synthetic_checkout.json'
            saved=dict(verified=True,v3_required=True,followup_plot=dict(verified=True),
                       publication_metrics=dict(verified=True),pdf_manuscript_binding=dict(verified=True),
                       v3_upstream_model_evidence=dict(synthetic_fixture=True),runs=[
                           dict(study_key='v2/laptop',study_source_commit=export.STUDY_COMMIT),
                           dict(study_key='v2/laptop_7b',study_source_commit=export.STUDY_COMMIT),
                           dict(study_key='v3/laptop_7b',study_source_commit=export.V3_STUDY_COMMIT)])
            data=dict(schema_version='kernelbench_publication_byte_verification_v1',verified=True,
                      working_tree_clean=True,source_commit=SOURCE,clone_check_performed=True,
                      clone_check=dict(verified=True),saved_byte_checks=saved)
            save(path,data);self.assertTrue(export.validate_checkout_verification(path,SOURCE,True)['verified'])
            data['clone_check_performed']=False;save(path,data)
            with self.assertRaises(ValueError):export.validate_checkout_verification(path,SOURCE,True)
            data['clone_check_performed']=True;saved['runs'].pop();save(path,data)
            with self.assertRaisesRegex(ValueError,'inventory'):
                export.validate_checkout_verification(path,SOURCE,True)

    def test_external_publication_copies_match_committed_tree(self):
        with fixture() as root:
            repo=root/'synthetic_repo';site=root/'synthetic_dist'
            (repo/'paper').mkdir(parents=True);(repo/'docs').mkdir();site.mkdir()
            (repo/'paper/KernelBench-RL-Paper.pdf').write_bytes(b'%PDF-synthetic-fixture')
            report=root/'synthetic_report.html';report.write_bytes(b'<html>synthetic fixture</html>')
            (repo/'docs/report.html').write_bytes(report.read_bytes())
            (repo/'docs/index.html').write_bytes(b'<html>synthetic index</html>')
            (repo/'docs/.nojekyll').write_bytes(b'')
            for name in ('index.html','report.html'):(site/name).write_bytes((repo/'docs'/name).read_bytes())
            export.validate_publication_source(repo,repo/'paper/KernelBench-RL-Paper.pdf',report,site)
            (site/'index.html').write_bytes(b'changed synthetic index')
            with self.assertRaisesRegex(ValueError,'differs'):
                export.validate_publication_source(repo,repo/'paper/KernelBench-RL-Paper.pdf',report,site)


if __name__=='__main__':
    unittest.main()
