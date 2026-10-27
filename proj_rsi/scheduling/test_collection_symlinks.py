"""Collect link metadata without following links from extracted source bundles."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .dispatch import _INVENTORY


class CollectionSymlinks(unittest.TestCase):
    def test_collection_records_links_but_source_verification_rejects_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / 'artifacts'
            artifacts.mkdir()
            (artifacts / 'regular').write_text('safe')
            (root / 'outside').mkdir()
            (root / 'outside' / 'file').write_text('must not be collected')
            (artifacts / 'directory').symlink_to('../outside', target_is_directory=True)
            (artifacts / 'file').symlink_to('../outside/file')
            (artifacts / 'broken').symlink_to('../missing')
            command = [sys.executable, '-c', _INVENTORY, str(root), json.dumps(['artifacts'])]
            result = json.loads(subprocess.check_output(command + ['allow_missing']))
            self.assertEqual(set(result['files']), {'artifacts/regular'})
            self.assertEqual(result['symlinks'], {
                'artifacts/directory': '../outside',
                'artifacts/file': '../outside/file',
                'artifacts/broken': '../missing',
            })
            self.assertEqual(result['missing'], [])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
