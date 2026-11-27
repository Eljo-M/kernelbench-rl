"""Check selected-case presentation against existing records, without execution."""
from __future__ import annotations
import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

OPS={'fused_bias_gelu_4096','matmul_bias_silu_512'}
STUDY_SOURCE='2c174abe268495159c10d538c92d071e7733828d'
CONFIG_HASH='0dfb0a14abad3067'
RAW_PATH='experiments/retry_v3/laptop_7b/heldout.jsonl'
SCOPE='Selected successful case studies; not a complete held-out or model success rate.'
TIMINGS=('candidate_ms','eager_ms','compile_ms','speedup_vs_eager','speedup_vs_compile')


def require(condition,message):
    if not condition:raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))


def digest(data):return hashlib.sha256(data).hexdigest()


def canonical(record):
    return json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def local(root,name):
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name,
            'Invalid selected artifact path')
    path=(root/name).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(),'Missing/escaping selected artifact: '+name)
    return path


def validate_cases(data,repo,site):
    require(data.get('schema_version')=='kernelbench_selected_case_studies_v1','Unknown selected-case schema')
    require(data.get('author')=={'name':'Eljo Mustafaj','github':'Eljo-M'},'Selected-case author differs')
    require(data.get('selection')==dict(selected_count=2,verified_count=2,scope=SCOPE),
            'Selected scope must not become an aggregate accuracy claim')
    cases=data.get('cases',[])
    require(len(cases)==2 and {item.get('op_name') for item in cases}==OPS
            and {item.get('id') for item in cases}==OPS,'Selected case inventory differs')
    env=load(repo/'experiments/retry_v3/laptop_7b/heldout_environment.json')
    require(env.get('source_commit')==STUDY_SOURCE and env.get('config_hash')==CONFIG_HASH
            and env.get('completed_utc') and env.get('source_code_dirty') is False,
            'Selected study source/config/completion differs')
    raw=local(repo,RAW_PATH).read_bytes();raw_sha=digest(raw)
    require(env.get('output_sha256')==raw_sha,'Original selected-record file digest differs')
    rows=[json.loads(line) for line in raw.decode('utf-8-sig').splitlines()]
    actual_env=data.get('environment',{})
    for field in ('gpu','dtype','torch','triton','cuda','model_id','model_revision','source_commit','config_hash','checkpoint_identity'):
        require(actual_env.get(field)==env.get(field),'Selected environment differs: '+field)
    require(actual_env.get('method')=='trained_checkpoint' and actual_env.get('timing_mode')=='cuda_graph',
            'Selected method/timing scope differs')
    archived=load(repo/'website/archive-v0.2/results.json')
    shapes={op['name']:op['input_shapes'] for op in archived['baseline']['operators'] if op['name'] in OPS}
    checked=[]
    for case in cases:
        name=case['op_name'];provenance=case.get('provenance',{})
        require(case.get('id')==name and case.get('canonical_shapes')==shapes.get(name),
                'Selected workload identity/shape differs: '+name)
        require(provenance.get('original_record_path')==RAW_PATH
                and provenance.get('raw_file_sha256')==raw_sha
                and provenance.get('canonical_record_encoding')=='json_sorted_keys_utf8_compact_v1',
                'Selected raw-record provenance differs: '+name)
        line=provenance.get('jsonl_line')
        require(type(line) is int and 1<=line<=len(rows),'Invalid selected source line')
        record=rows[line-1]
        require(record.get('op_name')==name and record.get('method')=='trained_checkpoint'
                and record.get('eval_phase')=='heldout_test' and record.get('fixture') is not True,
                'Selected record phase/method differs: '+name)
        require(record.get('status')=='PASS' and record.get('compiled') is True
                and record.get('correct') is True and record.get('cheat_detected') is False,
                'Selected record is not a verified PASS: '+name)
        for field in ('config_hash','model_revision','dtype','checkpoint_identity'):
            require(record.get(field)==actual_env.get(field),'Selected record identity differs: '+field)
        require(record.get('gpu_model')==actual_env['gpu'] and record.get('timing_mode')==actual_env['timing_mode'],
                'Selected record hardware/timing differs')
        record_sha=digest(canonical(record))
        require(provenance.get('record_canonical_sha256')==record_sha,'Selected canonical-record digest differs')
        receipt=load(local(site,provenance.get('receipt_path')))
        require(receipt.get('schema_version')=='kernelbench_selected_case_receipt_v1'
                and receipt.get('case_id')==name and receipt.get('study_source_commit')==STUDY_SOURCE
                and receipt.get('selection_scope')=='One selected successful case; not an aggregate model-success estimate.'
                and receipt.get('record')==record and receipt.get('provenance')==provenance,
                'Selected receipt differs from original record: '+name)
        source=local(site,provenance.get('code_path')).read_bytes()
        require(source==record.get('candidate_source','').encode('utf-8')
                and digest(source)==record.get('candidate_sha256')==provenance.get('candidate_sha256'),
                'Selected candidate source/digest differs: '+name)
        require(case.get('dtype')==record['dtype'] and case.get('max_abs_err')==record.get('max_abs_err'),
                'Selected precision/error differs: '+name)
        for field in ('status','compiled','correct','cheat_detected','seed','independent_seed','method','budget_k'):
            actual=case.get('evaluation',{}).get(field)
            require(type(actual) is type(record.get(field)) and actual==record.get(field),'Selected evaluation differs: '+field)
        for field in TIMINGS:
            value=case.get('timings',{}).get(field)
            require(type(value) in (int,float) and math.isfinite(value) and value>0 and value==record.get(field),
                    'Selected timing differs: '+field)
        checked.append(dict(op_name=name,candidate_sha256=digest(source),record_canonical_sha256=record_sha))
    return checked


class Page(HTMLParser):
    def __init__(self,path):
        super().__init__();self.path=path;self.links=[];self.ids={};self.stack=[];self.meta={};self.cases=set();self.h1=[];self.capture=[]
        self.feed(path.read_text(encoding='utf-8'))
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='html':require(attrs.get('lang')=='en','Missing document language')
        if tag=='img':require('alt' in attrs,'Selected-case image lacks alt text')
        if attrs.get('data-case'):self.cases.add(attrs['data-case'])
        if tag=='meta' and attrs.get('name'):self.meta[attrs['name']]=attrs.get('content','')
        for attr in ('href','src'):
            if attr in attrs:self.links.append(attrs[attr])
        marker=attrs.get('id')
        if marker:
            require(marker not in self.ids,'Duplicate selected-page ID')
            self.ids[marker]=''
        if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:
            self.stack.append((tag,marker))
        if tag=='h1':self.capture.append('')
    def handle_data(self,data):
        for _,marker in self.stack:
            if marker:self.ids[marker]+=data
        if self.capture:self.capture[-1]+=data
    def handle_endtag(self,tag):
        if tag=='h1' and self.capture:self.h1.append(self.capture.pop().strip())
        if self.stack and self.stack[-1][0]==tag:self.stack.pop()


def validate_pages(data,site):
    require(not (site/'results.json').exists() and not (site/'assets/kernelbench-rl-paper.pdf').exists()
            and not any(path.name=='archive-v0.2' for path in site.rglob('*')),
            'Legacy full-study site must remain outside the served selected presentation')
    pages={name:Page(site/name) for name in ('index.html','paper.html','report.html')}
    for name,page in pages.items():
        require(len(page.h1)==1,'Selected page must have one h1: '+name)
        require(page.cases==OPS,'Selected page case inventory differs: '+name)
        for link in page.links:
            parsed=urlparse(link)
            if parsed.scheme or parsed.netloc:continue
            target=local(site,unquote(parsed.path)) if parsed.path else page.path
            if parsed.fragment and target.suffix=='.html':
                target_page=pages.get(target.name) or Page(target)
                require(parsed.fragment in target_page.ids,'Broken selected-page anchor: '+link)
    index=pages['index.html']
    require(index.ids.get('featured-verified-count','').strip()=='2/2'
            and index.ids.get('featured-scope-label','').strip()=='featured kernels verified',
            'Featured count must be explicitly scoped to displayed kernels')
    paper=pages['paper.html'];title=data['title']
    require(data['paper']['title']==title and paper.h1==[title]
            and paper.meta.get('citation_title')==title and paper.meta.get('citation_author')=='Eljo Mustafaj'
            and paper.meta.get('citation_publication_date')==data['publication_date'].replace('-','/'),
            'Selected paper landing metadata differs')
    config=load(site/'project-config.json')
    pdf=data['paper']['pdf_path']
    require(config.get('draft') is False and config.get('paper_pdf_path')==pdf
            and config.get('public_base_url')=='https://eljo-m.github.io/kernelbench-rl',
            'Selected publication configuration differs')
    require(paper.meta.get('citation_pdf_url')==config['public_base_url']+'/'+pdf,'Selected citation PDF URL differs')
    local(site,pdf)
    return pages


def validate(repo,site):
    data=load(site/'case-studies.json');cases=validate_cases(data,repo,site);validate_pages(data,site)
    manuscript=repo/'paper/case-studies/manuscript.json';paper=load(manuscript)
    require(paper.get('title')==data['title'] and paper.get('publication_date')==data['publication_date']
            and paper.get('author',{}).get('display_name')=='Eljo Mustafaj','Selected manuscript metadata differs')
    source_pdf=repo/'paper/case-studies/KernelBench-RL-Case-Studies.pdf'
    pdf=local(site,data['paper']['pdf_path'])
    require(pdf.read_bytes()==source_pdf.read_bytes(),'Selected source/public PDF bytes differ')
    sys.path.insert(0,str(repo))
    from tools.publication.verify_publication_artifacts import PublishedHTML,plain,verify_pdf_manuscript_binding
    landing=PublishedHTML((site/'paper.html').read_text(encoding='utf-8'))
    require(plain(' '.join(landing.abstract))==plain(paper.get('abstract','')) and bool(landing.abstract),
            'Selected paper landing abstract differs from manuscript')
    verify_pdf_manuscript_binding(pdf,manuscript)
    return dict(verified=True,scope=SCOPE,featured_cases=cases,paper_sha256=digest(pdf.read_bytes()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--site',type=Path,default=Path('docs'))
    args=parser.parse_args();repo=args.repo.resolve();site=args.site if args.site.is_absolute() else repo/args.site
    print(json.dumps(validate(repo,site.resolve()),indent=2))
