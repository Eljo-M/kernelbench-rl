"""TEST ONLY small fixtures; no experiments, models, network or GPU work."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest

from website import validate_archive as archive
from website import validate_showcase as showcase
from tools.publication import verify_publication_artifacts as legacy
from tests.publication_rendering_fixtures import fixture as legacy_fixture
from tests.test_publication_semantics import prepare


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')


@contextmanager
def selected_fixture():
    with tempfile.TemporaryDirectory(prefix='TEST_ONLY_selected_cases_') as temporary:
        root=Path(temporary);site=root/'site';site.mkdir()
        env=dict(source_commit=showcase.STUDY_SOURCE,config_hash=showcase.CONFIG_HASH,
                 completed_utc='2026-11-27T00:00:00Z',source_code_dirty=False,
                 gpu='TEST ONLY GPU',dtype='fp16',torch='TEST ONLY',triton='TEST ONLY',cuda='TEST ONLY',
                 model_id='TEST ONLY MODEL',model_revision='1'*40,checkpoint_identity='TEST ONLY checkpoint')
        records=[]
        for name in sorted(showcase.OPS):
            source='TEST ONLY inert text; never executed: '+name+'\n'
            records.append(dict(fixture_notice='TEST ONLY, not experimental evidence',op_name=name,
                method='trained_checkpoint',eval_phase='heldout_test',status='PASS',compiled=True,correct=True,
                cheat_detected=False,config_hash=showcase.CONFIG_HASH,model_revision=env['model_revision'],
                dtype='fp16',checkpoint_identity=env['checkpoint_identity'],gpu_model=env['gpu'],
                timing_mode='cuda_graph',candidate_source=source,candidate_sha256=showcase.digest(source.encode()),
                max_abs_err=0.,seed=1,independent_seed=2,budget_k=1,
                **{field:1. for field in showcase.TIMINGS}))
        raw=('\n'.join(json.dumps(record) for record in records)+'\n').encode()
        raw_path=root/showcase.RAW_PATH;raw_path.parent.mkdir(parents=True);raw_path.write_bytes(raw)
        env['output_sha256']=showcase.digest(raw)
        save(raw_path.with_name('heldout_environment.json'),env)
        data=dict(schema_version='kernelbench_selected_case_studies_v1',
                  author={'name':'Eljo Mustafaj','github':'Eljo-M'},
                  selection=dict(selected_count=2,verified_count=2,scope=showcase.SCOPE),
                  environment={**env,'method':'trained_checkpoint','timing_mode':'cuda_graph'},cases=[])
        for line,record in enumerate(records,1):
            name=record['op_name'];provenance=dict(candidate_sha256=record['candidate_sha256'],
                code_path='kernels/'+name+'.py',receipt_path='evidence/'+name+'.json',
                original_record_path=showcase.RAW_PATH,jsonl_line=line,raw_file_sha256=env['output_sha256'],
                record_canonical_sha256=showcase.digest(showcase.canonical(record)),
                canonical_record_encoding='json_sorted_keys_utf8_compact_v1')
            code=site/provenance['code_path'];code.parent.mkdir(exist_ok=True);code.write_bytes(record['candidate_source'].encode())
            save(site/provenance['receipt_path'],dict(schema_version='kernelbench_selected_case_receipt_v1',
                selection_scope='One selected successful case; not an aggregate model-success estimate.',
                case_id=name,study_source_commit=showcase.STUDY_SOURCE,record=record,provenance=provenance))
            data['cases'].append(dict(id=name,op_name=name,canonical_shapes=[[1]],dtype='fp16',max_abs_err=0.,
                evaluation={field:record[field] for field in ('status','compiled','correct','cheat_detected','seed','independent_seed','method','budget_k')},
                timings={field:record[field] for field in showcase.TIMINGS},provenance=provenance))
        save(root/'website/archive-v0.2/results.json',dict(baseline={'operators':[dict(name=name,input_shapes=[[1]]) for name in sorted(showcase.OPS)]}))
        yield root,site,data


class ArchiveTests(unittest.TestCase):
    def test_archive_exact_bytes_and_changed_bytes(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_archive_') as temporary:
            root=Path(temporary);(root/'index.html').write_bytes(b'TEST ONLY\r\n')
            self.assertTrue(archive.verify_files(root,{'index.html':b'TEST ONLY\r\n'})['verified'])
            with self.assertRaisesRegex(ValueError,'bytes differ'):
                archive.verify_files(root,{'index.html':b'TEST ONLY\n'})

    def test_archive_missing_extra_and_routing(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_archive_') as temporary:
            root=Path(temporary);self.assertEqual(legacy.publication_site(root),root/'docs')
            mirror=root/'website/archive-v0.2';mirror.mkdir(parents=True)
            self.assertEqual(legacy.publication_site(root),mirror)
            with self.assertRaisesRegex(ValueError,'inventory'):
                archive.verify_files(mirror,{'missing':b'TEST ONLY'})
            (mirror/'extra').write_bytes(b'TEST ONLY')
            with self.assertRaisesRegex(ValueError,'inventory'):
                archive.verify_files(mirror,{})

    def test_legacy_semantics_use_preserved_mirror_and_still_reject_mutation(self):
        with legacy_fixture() as item:
            root,studies,strict=prepare(item)
            (root/'website').mkdir();mirror=root/'website/archive-v0.2'
            (root/'docs').rename(mirror);(root/'docs').mkdir()
            (root/'docs/index.html').write_text('TEST ONLY separate selected presentation')
            self.assertTrue(legacy.verify_published_metrics(root,studies,strict)['verified'])
            path=mirror/'results.json';data=legacy.read(path);data['retry']['runs'].pop();save(path,data)
            with self.assertRaisesRegex(ValueError,'study inventory'):
                legacy.verify_published_metrics(root,studies,strict)


class SelectedCaseTests(unittest.TestCase):
    def test_two_selected_receipts_and_source_programs_match(self):
        with selected_fixture() as (root,site,data):
            self.assertEqual(len(showcase.validate_cases(data,root,site)),2)

    def test_selected_scope_timing_dtype_and_candidate_mutations_refused(self):
        with selected_fixture() as (root,site,original):
            for kind in ('scope','timing','dtype','candidate'):
                with self.subTest(kind=kind):
                    data=copy.deepcopy(original)
                    if kind=='scope':data['selection']['scope']='TEST ONLY false complete accuracy claim'
                    if kind=='timing':data['cases'][0]['timings']['candidate_ms']=2.
                    if kind=='dtype':data['cases'][0]['dtype']='fp32'
                    if kind=='candidate':data['cases'][0]['provenance']['candidate_sha256']='0'*64
                    with self.assertRaises(ValueError):showcase.validate_cases(data,root,site)

    def test_selected_receipt_cannot_replace_original_pass_record(self):
        with selected_fixture() as (root,site,data):
            path=site/data['cases'][0]['provenance']['receipt_path'];receipt=showcase.load(path)
            receipt['record']['status']='COMPILE_ERROR';save(path,receipt)
            with self.assertRaisesRegex(ValueError,'receipt differs'):
                showcase.validate_cases(data,root,site)


if __name__=='__main__':unittest.main()
