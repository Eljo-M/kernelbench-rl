"""TEST ONLY: stale-publication regressions; no real quality/model/GPU calls."""
import copy
import json
from pathlib import Path
import shutil
import unittest

from tools.publication import verify_publication_artifacts as check
from tests.publication_rendering_fixtures import fixture,save


def prepare(item):
    item['run']()
    root=item['repo'];out=item['out'];site=item['site']
    manuscript=check.read(out/'KernelBench-RL-Manuscript.json')
    manuscript['author']={'display_name':'Eljo Mustafaj'}
    save(root/'paper/manuscript.json',manuscript)
    docs=root/'docs';docs.mkdir()
    for name in ('index.html','paper.html','results.json'):shutil.copy2(site/name,docs/name)
    shutil.copy2(out/'RETRY_FINAL_REPORT.html',docs/'report.html')
    studies=[];strict={}
    for metric in manuscript['verified_retry_metrics']:
        run=root/metric['run_dir'];save(run/'environment.json',metric['environment'])
        studies.append(dict(study_key=metric['study_key'],run_dir=run,
                            source_commit=metric['environment']['source_commit']))
        strict[metric['study_key']]=copy.deepcopy(metric['quality'])
    return root,studies,strict


class PublicationSemanticTests(unittest.TestCase):
    def test_complete_three_study_output_and_stale_two_study_site(self):
        with fixture() as item:
            root,studies,strict=prepare(item)
            self.assertTrue(check.verify_published_metrics(root,studies,strict)['verified'])
            path=root/'docs/results.json';data=check.read(path)
            data['retry']['runs'].pop();save(path,data)
            with self.assertRaisesRegex(ValueError,'study inventory'):
                check.verify_published_metrics(root,studies,strict)

    def test_scientific_identity_and_table_or_report_mutations_refused(self):
        with fixture() as item:
            root,studies,strict=prepare(item);path=root/'paper/manuscript.json'
            original=check.read(path);changed=copy.deepcopy(original)
            changed['verified_retry_metrics'][-1]['quality']['grpo_optimizer_steps']=0
            save(path,changed)
            with self.assertRaisesRegex(ValueError,'fresh strict'):
                check.verify_published_metrics(root,studies,strict)
            changed=copy.deepcopy(original);changed['tables'][-1]['rows'][0]['budget']=8;save(path,changed)
            with self.assertRaisesRegex(ValueError,'table differs'):
                check.verify_published_metrics(root,studies,strict)
            changed=copy.deepcopy(original)
            extra=copy.deepcopy(changed['tables'][0]);extra['id']='TEST_ONLY_stale_extra_table'
            changed['tables'].append(extra);save(path,changed)
            with self.assertRaisesRegex(ValueError,'table inventory'):
                check.verify_published_metrics(root,studies,strict)
            save(path,original)
            report=root/'docs/report.html'
            report.write_text(report.read_text(encoding='utf-8').replace('<td>0/3</td>','<td>3/3</td>',1),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'HTML report tables'):
                check.verify_published_metrics(root,studies,strict)

    def test_latest_worse_headline_and_paper_abstract_are_bound(self):
        with fixture() as item:
            root,studies,strict=prepare(item);index=root/'docs/index.html'
            original=index.read_text(encoding='utf-8')
            index.write_text(original.replace('id="latest-hybrid-score">0 / 3','id="latest-hybrid-score">2 / 3'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'headline'):
                check.verify_published_metrics(root,studies,strict)
            index.write_text(original,encoding='utf-8')
            paper=root/'docs/paper.html'
            paper.write_text(paper.read_text(encoding='utf-8').replace('KernelBench-RL is an execution-graded','STALE TEST ONLY abstract: KernelBench-RL is an execution-graded'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'abstract'):
                check.verify_published_metrics(root,studies,strict)

    def test_stale_two_study_pdf_input_missing_and_duplicate_marker(self):
        with fixture() as item:
            root=item['repo'];path=root/'TEST_ONLY_manuscript.json';pdf=root/'TEST_ONLY_marker_fixture.pdf'
            save(path,dict(fixture_notice='TEST ONLY',studies=['v2/laptop','v2/laptop_7b']))
            old=check.sha(path)
            def marker(digest):return b'/Subject (KernelBench-RL manuscript SHA-256: '+digest.encode('ascii')+b')'
            pdf.write_bytes(b'%PDF-TEST-ONLY\n'+marker(old))
            self.assertTrue(check.verify_pdf_manuscript_binding(pdf,path)['verified'])
            save(path,dict(fixture_notice='TEST ONLY',studies=['v2/laptop','v2/laptop_7b','v3/laptop_7b']))
            with self.assertRaisesRegex(ValueError,'stale'):
                check.verify_pdf_manuscript_binding(pdf,path)
            pdf.write_bytes(b'%PDF-TEST-ONLY old render without input binding')
            with self.assertRaisesRegex(ValueError,'missing'):
                check.verify_pdf_manuscript_binding(pdf,path)
            pdf.write_bytes(marker(check.sha(path))+b'\n'+marker(check.sha(path)))
            with self.assertRaisesRegex(ValueError,'duplicated'):
                check.verify_pdf_manuscript_binding(pdf,path)


if __name__=='__main__':unittest.main()
