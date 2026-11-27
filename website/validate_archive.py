"""Require the preserved full-study website to match its original Git blobs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PUBLICATION_SOURCE = '9e1c103bf4d3afe7cd5078868e9a00b4f1fdba65'


def verify_files(archive, expected):
    """Compare every byte, missing file and extra file; fixtures use tiny data."""
    if not archive.is_dir() or archive.is_symlink():
        raise ValueError('Missing or linked full-study archive')
    entries=list(archive.rglob('*'))
    if any(path.is_symlink() for path in entries):
        raise ValueError('Archive symlinks are not exact saved files')
    actual={path.relative_to(archive).as_posix():path for path in entries if path.is_file()}
    if set(actual)!=set(expected):
        raise ValueError('Full-study archive file inventory differs')
    records=[]
    for name,path in sorted(actual.items()):
        data=path.read_bytes()
        if data!=expected[name]:
            raise ValueError('Full-study archive bytes differ: '+name)
        records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    return dict(verified=True,files=records)


def verify_archive(repo):
    def git(*args):
        return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True).stdout
    source=git('rev-parse',PUBLICATION_SOURCE+'^{commit}').decode().strip()
    if source!=PUBLICATION_SOURCE:
        raise ValueError('Original publication source is unavailable')
    expected={}
    for row in git('ls-tree','-rz',PUBLICATION_SOURCE,'--','docs').split(b'\0'):
        if not row:continue
        metadata,name=row.split(b'\t',1)
        mode,kind,oid=metadata.split()
        if kind!=b'blob' or mode not in (b'100644',b'100755'):
            raise ValueError('Unexpected original docs object')
        relative=name.decode('utf-8').removeprefix('docs/')
        expected[relative]=git('cat-file','blob',oid.decode('ascii'))
    if not expected:
        raise ValueError('Original publication website is empty')
    result=verify_files(repo/'website/archive-v0.2',expected)
    result.update(source_commit=source,archive_path='website/archive-v0.2')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify_archive(parser.parse_args().repo.resolve()),indent=2))
