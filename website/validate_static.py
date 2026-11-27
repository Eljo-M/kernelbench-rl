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
