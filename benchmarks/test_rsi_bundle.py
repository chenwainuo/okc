import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from rsi_bundle import materialize


class BundleTest(unittest.TestCase):
    def fixture(self, root, name='package/a.py'):
        archive=root/'bundle.tar.gz';data=b'VALUE = 1\n'
        with tarfile.open(archive,'w:gz') as t:
            member=tarfile.TarInfo(name);member.size=len(data);t.addfile(member,io.BytesIO(data))
        manifest={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'files':{name:hashlib.sha256(data).hexdigest()},'symlinks':{}}
        path=root/'manifest.json';path.write_text(json.dumps(manifest));return archive,path

    def test_verified_reuse_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a,m=self.fixture(root);dest=root/'tree'
            materialize(a,m,dest);materialize(a,m,dest)
            file=dest/'package/a.py';file.chmod(0o644);file.write_text('CHANGED')
            with self.assertRaises(ValueError):materialize(a,m,dest)
            # Restore writable dirs so non-root cleanup works.
            dest.chmod(0o755);(dest/'package').chmod(0o755)

    def test_traversal_is_rejected_before_member_write(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a,m=self.fixture(root,'../escaped')
            with self.assertRaises(ValueError):materialize(a,m,root/'tree')
            self.assertFalse((root/'escaped').exists())

    def test_archive_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a,m=self.fixture(root);a.write_bytes(a.read_bytes()+b'extra')
            with self.assertRaises(ValueError):materialize(a,m,root/'tree')
