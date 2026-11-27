"""Check buildless local assets, anchors, key semantics, data and publication metadata."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(); self.path=path; self.ids=set(); self.links=[]; self.errors=[]; self.h1=0; self.meta={}
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if a.get('id'):
            if a['id'] in self.ids: self.errors.append('duplicate id '+a['id'])
            self.ids.add(a['id'])
        if tag=='html' and a.get('lang')!='en': self.errors.append('missing document language')
        if tag=='h1': self.h1+=1
        if tag=='img' and 'alt' not in a: self.errors.append('image lacks alt')
        if tag=='meta' and a.get('name'): self.meta[a['name']]=a.get('content','')
        for attr in ('href','src'):
            if attr in a: self.links.append(a[attr])


def retry_errors(retry):
    """Validate optional cohort identity metadata without changing results."""
    errors=[]; keys=set()
    for run in retry.get('runs',[]):
        if 'short_label' in run and (not isinstance(run['short_label'],str) or not run['short_label'].strip()): errors.append('invalid retry short label')
        for field,length in (('config_hash',16),('model_revision',40)):
            if field in run and (not isinstance(run[field],str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}',run[field])): errors.append('invalid retry '+field)
        if 'notes' in run and (not isinstance(run['notes'],list) or not all(isinstance(note,str) for note in run['notes'])): errors.append('invalid retry prose notes')
        key=run.get('study_key')
        if key is not None:
            if key not in {'v2/laptop','v2/laptop_7b','v3/laptop_7b'} or key in keys:
                errors.append('duplicate/unknown retry study_key')
            keys.add(key)
            if run.get('protocol_version')!=key.split('/')[0]: errors.append('retry protocol/study_key mismatch')
            if not isinstance(run.get('cohort_label'),str) or not run['cohort_label'].strip(): errors.append('missing retry cohort label')
            if 'cohort_ops' not in run: errors.append('named retry study lacks cohort workload identities')
        operators=run.get('cohort_ops')
        if operators is not None:
            if not isinstance(operators,dict) or set(operators)!={'fresh','development'}:
                errors.append('invalid retry cohort_ops inventory'); continue
            valid={}
            for cohort,names in operators.items():
                if not isinstance(names,list) or not names or not all(isinstance(name,str) and name for name in names) or len(set(names))!=len(names):
                    errors.append('duplicate/invalid retry cohort workload'); continue
                valid[cohort]=set(names)
                methods=run.get('cohorts',{}).get(cohort,[])
                if not methods or any(method.get('total')!=len(names) for method in methods):
                    errors.append('cohort workloads do not match method denominators')
            if set(valid)=={'fresh','development'} and valid['fresh']&valid['development']:
                errors.append('fresh/development retry workloads overlap')
        for methods in run.get('cohorts',{}).values():
            for method in methods:
                passed,total,budget=(method.get(key) for key in ('passed','total','budget'))
                if type(passed) is not int or type(total) is not int or type(budget) is not int or not 0<=passed<=total or total<=0 or budget<=0:
                    errors.append('invalid retry method count/budget')
    return errors


def validate(root):
    pages={path.name: Page(path) for path in root.glob('*.html')}; errors=[]
    for page in pages.values():
        page.feed(page.path.read_text(encoding='utf-8'))
    for page in pages.values():
        errors += [f'{page.path.name}: {error}' for error in page.errors]
        if page.h1!=1: errors.append(f'{page.path.name}: expected one h1')
        for link in page.links:
            parsed=urlparse(link)
            if parsed.scheme or parsed.netloc: continue
            path=(page.path.parent/unquote(parsed.path)).resolve() if parsed.path else page.path
            if not path.is_relative_to(root.resolve()) or not path.is_file(): errors.append(f'{page.path.name}: missing local link {link}'); continue
            if parsed.fragment and path.suffix=='.html' and parsed.fragment not in pages[path.name].ids:
                errors.append(f'{page.path.name}: missing anchor {link}')
    data=json.loads((root/'results.json').read_text(encoding='utf-8'))
    config=json.loads((root/'project-config.json').read_text(encoding='utf-8'))
    baseline=data['baseline']
    if baseline['passed']!=180 or baseline['total']!=180 or len(baseline['operators'])!=30: errors.append('baseline coverage mismatch')
    for op in baseline['operators']:
        if not op['source'] or not re.fullmatch('[0-9a-f]{64}',op['source_sha256']): errors.append('missing source evidence '+op['name'])
        for precision in ('fp32','bf16'):
            if op['precision'][precision]['total']!=3: errors.append('unexpected seed count '+op['name'])
    if data['retry']['status']=='pending' and (data['retry']['published_results'] or data['retry']['runs']): errors.append('pending retry contains results')
    errors+=retry_errors(data['retry'])
    if pages['paper.html'].meta.get('citation_author')!='Eljo Mustafaj': errors.append('incorrect paper author')
    if not config['draft']:
        expected=config['public_base_url']+'/'+config['paper_pdf_path']
        if pages['paper.html'].meta.get('citation_pdf_url')!=expected: errors.append('paper PDF metadata differs from verified config')
        if not (root/config['paper_pdf_path']).is_file(): errors.append('publication PDF missing')
    for script in root.glob('assets/*.js'):
        source=script.read_text(encoding='utf-8')
        for id in re.findall(r"\$\('([^']+)'\)",source):
            if not any(id in page.ids for page in pages.values()): errors.append('unknown JS element '+id)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(json.dumps(dict(result='pass', html_pages=len(pages), operators=len(baseline['operators']), records=baseline['total'], draft=config['draft'], browser_qa='not performed; owner will verify responsive layouts and interactions')))


if __name__=='__main__': validate(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).parent/'dist')
