"""Finalize verified origin, local PDF, and externally verified download links."""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse


def configure(dist, origin, pdf, downloads, ready):
    origin = origin.rstrip('/')
    parsed = urlparse(origin)
    if (parsed.scheme != 'https' or not parsed.netloc or parsed.query or parsed.fragment
            or any(part in ('.', '..') for part in parsed.path.split('/'))):
        raise ValueError('PUBLIC_BASE_URL must be the verified HTTPS deployment base, including any project path')
    pdf_path = (dist/pdf).resolve()
    if not pdf_path.is_relative_to(dist.resolve()) or not pdf_path.is_file() or pdf_path.suffix.lower() != '.pdf':
        raise ValueError('Copy the rendered searchable PDF into dist before configuring publication')
    if not pdf_path.read_bytes().startswith(b'%PDF-'):
        raise ValueError('The configured local paper is not a PDF')
    for item in downloads:
        if not isinstance(item.get('label'), str) or not item['label'].strip():
            raise ValueError('Each verified download requires a label')
        url = urlparse(item.get('url', ''))
        if url.scheme != 'https' or not url.netloc:
            raise ValueError('Each release download must have a separately verified HTTPS URL')
    config = dict(draft=not ready, public_base_url=origin, paper_pdf_path=pdf, downloads=downloads)
    (dist/'project-config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    paper_path = dist/'paper.html'
    paper = paper_path.read_text(encoding='utf-8')
    metadata = f'<!-- PUBLICATION_METADATA_START -->\n<meta name="citation_pdf_url" content="{html.escape(origin+"/"+pdf, quote=True)}">\n<meta name="citation_abstract_html_url" content="{html.escape(origin+"/paper.html", quote=True)}">\n<link rel="canonical" href="{html.escape(origin+"/paper.html", quote=True)}">\n<!-- PUBLICATION_METADATA_END -->'
    paper = re.sub(r'<!-- PUBLICATION_METADATA_START -->.*?<!-- PUBLICATION_METADATA_END -->', metadata, paper, flags=re.S)
    # Static href and citation text remain visible to crawlers and without JS.
    paper = re.sub(r'<a class="button primary" id="paper-download"[^>]*>', f'<a class="button primary" id="paper-download" href="{html.escape(pdf, quote=True)}">', paper)
    paper = re.sub(r'url = \{https://[^}]+\}', f'url = {{{origin}/paper.html}}', paper)
    paper = re.sub(r'name="robots" content="[^"]+"', f'name="robots" content="{"index,follow" if ready else "noindex,nofollow"}"', paper)
    paper = re.sub(r'data-draft-notice(?: hidden)?', 'data-draft-notice'+(' hidden' if ready else ''), paper)
    paper_path.write_text(paper, encoding='utf-8')
    index_path=dist/'index.html'
    index=index_path.read_text(encoding='utf-8')
    index=re.sub(r'name="robots" content="[^"]+"', f'name="robots" content="{"index,follow" if ready else "noindex,nofollow"}"', index)
    index=re.sub(r'data-draft-notice(?: hidden)?', 'data-draft-notice'+(' hidden' if ready else ''), index)
    index_path.write_text(index, encoding='utf-8')
    (dist/'robots.txt').write_text('User-agent: *\n'+('Allow: /\n' if ready else 'Disallow: /\n'), encoding='utf-8')
    if ready:
        (dist/'sitemap.xml').write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{html.escape(origin)}/</loc></url><url><loc>{html.escape(origin)}/paper.html</loc></url></urlset>\n', encoding='utf-8')
    print(json.dumps(dict(draft=not ready, origin=origin, paper_pdf_url=origin+'/'+pdf, downloads=len(downloads))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, default=Path(__file__).parent/'dist')
    parser.add_argument('--public-base-url', default=os.environ.get('PUBLIC_BASE_URL', ''))
    parser.add_argument('--pdf', default='assets/kernelbench-rl-paper.pdf')
    parser.add_argument('--downloads-json', type=Path, help='JSON list of {label,url} already verified by the site owner')
    parser.add_argument('--ready', action='store_true', help='Remove draft/noindex only after content, PDF, links and browser QA are complete')
    args=parser.parse_args()
    links=json.loads(args.downloads_json.read_text(encoding='utf-8')) if args.downloads_json else []
    configure(args.dist.resolve(), args.public_base_url, args.pdf, links, args.ready)
