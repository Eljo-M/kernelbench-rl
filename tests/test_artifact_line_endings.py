"""Synthetic Git line-ending regression fixtures; NOT experiment evidence.

Only brand-new TemporaryDirectory repositories under work are changed. No
network, model code, active repository state, or global Git config is used.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def work_root():
    explicit=os.environ.get('KERNELBENCH_TEST_WORK_ROOT')
    if explicit:
        return Path(explicit).resolve()
    script=Path(__file__).resolve()
    if script.parent.name=='work':
        return script.parent
    for parent in script.parents:
        if (parent/'rl_trainer/experiment_config.py').is_file():
            workspace=parent.parent.parent if parent.parent.name=='outputs' else parent
            return workspace/'work'
    raise RuntimeError('Set KERNELBENCH_TEST_WORK_ROOT to an explicit scratch work directory')


GIT=shutil.which('git')


@unittest.skipUnless(GIT,'Git is required for this local synthetic regression')
class ArtifactLineEndingTests(unittest.TestCase):
    def setUp(self):
        self.work=work_root().resolve()
        self.work.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='line-ending-fixture-NOT-EXPERIMENT-',dir=self.work)
        # Disable unverified destructor cleanup. The explicit cleanup below
        # checks resolved containment immediately before recursive removal.
        self.temporary._finalizer.detach()
        self.fixture=Path(self.temporary.name).resolve()
        self.assertEqual(self.fixture.parent,self.work)
        self.assertTrue(self.fixture.is_relative_to(self.work))
        self.addCleanup(self.cleanup_fixture)
        self.env={key:value for key,value in os.environ.items() if not key.startswith('GIT_')}
        self.env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,
            GIT_ATTR_NOSYSTEM='1',GIT_CEILING_DIRECTORIES=str(self.work),
            GIT_AUTHOR_NAME='Eljo-M',GIT_AUTHOR_EMAIL='mustafajeljo@gmail.com',
            GIT_COMMITTER_NAME='Eljo-M',GIT_COMMITTER_EMAIL='mustafajeljo@gmail.com',
            GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
        self.source=self.fixture/'source'
        self.source.mkdir()
        self.relative='experiments/fixture-NOT-EXPERIMENT.jsonl'
        self.saved=b'{"synthetic_fixture":true,"experiment":false}\r\n'
        self.saved_sha=hashlib.sha256(self.saved).hexdigest()

    def cleanup_fixture(self):
        target=Path(self.temporary.name).resolve()
        if target.parent!=self.work or not target.is_relative_to(self.work):
            raise RuntimeError('Refusing recursive fixture cleanup outside checked work root')
        self.temporary.cleanup()

    def git(self,*args,cwd=None):
        result=subprocess.run([GIT,'-C',str(cwd or self.source),*args],env=self.env,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        if result.returncode:
            self.fail('Synthetic fixture Git command failed: '+repr(args)+'\n'+result.stderr.decode(errors='replace'))
        return result.stdout

    def commit_fixture(self,preserve):
        self.git('init','--initial-branch=main')
        self.git('config','core.autocrlf','true')
        self.git('config','core.safecrlf','false')
        self.git('config','commit.gpgsign','false')
        self.git('config','core.hooksPath',str(self.fixture/'no-hooks'))
        self.git('config','user.name','Eljo-M')
        self.git('config','user.email','mustafajeljo@gmail.com')
        attributes=b'* text=auto\n'+(b'experiments/** -text\n' if preserve else b'')
        (self.source/'.gitattributes').write_bytes(attributes)
        artifact=self.source/self.relative
        artifact.parent.mkdir()
        artifact.write_bytes(self.saved)
        self.git('add','--','.gitattributes',self.relative)
        self.git('commit','-m','test: synthetic line-ending preservation fixture')
        identity=self.git('show','-s','--format=%an <%ae>|%cn <%ce>').decode().strip()
        self.assertEqual(identity,'Eljo-M <mustafajeljo@gmail.com>|Eljo-M <mustafajeljo@gmail.com>')
        self.assertEqual(artifact.read_bytes(),self.saved)
        return self.git('cat-file','blob','HEAD:'+self.relative)

    def clone_bytes(self,autocrlf):
        clone=self.fixture/('clone-'+('crlf' if autocrlf else 'lf'))
        self.assertFalse(clone.exists())
        self.assertEqual(clone.resolve().parent,self.fixture)
        # text=auto can still use core.eol=native on Windows when autocrlf is
        # false. Explicit lf models the other platform's checkout faithfully.
        eol=[] if autocrlf else ['--config','core.eol=lf']
        self.git('clone','--no-hardlinks','--config','core.autocrlf='+str(autocrlf).lower(),
                 *eol,str(self.source),str(clone),cwd=self.fixture)
        setting=self.git('config','--get','core.autocrlf',cwd=clone).decode().strip()
        self.assertEqual(setting,str(autocrlf).lower())
        return (clone/self.relative).read_bytes()

    def test_text_auto_crlf_roundtrip_can_hide_lf_blob_hash_break(self):
        blob=self.commit_fixture(preserve=False)
        self.assertEqual(blob,self.saved.replace(b'\r\n',b'\n'))
        self.assertNotEqual(hashlib.sha256(blob).hexdigest(),self.saved_sha)
        # A same-setting Windows clone appears byte-identical, concealing the
        # normalized blob that breaks the recorded saved-byte hash elsewhere.
        windows_checkout=self.clone_bytes(autocrlf=True)
        self.assertEqual(windows_checkout,self.saved)
        self.assertEqual(hashlib.sha256(windows_checkout).hexdigest(),self.saved_sha)
        lf_checkout=self.clone_bytes(autocrlf=False)
        self.assertEqual(lf_checkout,blob)
        self.assertNotEqual(hashlib.sha256(lf_checkout).hexdigest(),self.saved_sha)

    def test_minus_text_preserves_saved_crlf_blob_and_both_checkouts(self):
        blob=self.commit_fixture(preserve=True)
        self.assertEqual(blob,self.saved)
        self.assertEqual(hashlib.sha256(blob).hexdigest(),self.saved_sha)
        for setting in (True,False):
            checkout=self.clone_bytes(autocrlf=setting)
            self.assertEqual(checkout,self.saved)
            self.assertEqual(hashlib.sha256(checkout).hexdigest(),self.saved_sha)


if __name__=='__main__':
    unittest.main(verbosity=2)
