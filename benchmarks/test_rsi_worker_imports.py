"""Regression checks for editable-install fallback escaping frozen workers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class WorkerImportIsolationTests(unittest.TestCase):
    def test_missing_frozen_submodule_cannot_fall_back_to_editable_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'bundle'
            package=root/'serving/sglang/python/sglang'
            package.mkdir(parents=True)
            (root/'muse_moe_w4a16').mkdir();(root/'benchmarks').mkdir()
            (package/'__init__.py').write_text('from .version import __version__\n')
            (package/'version.py').write_text(
                'try:\n    from sglang._version import __version__\n'
                'except ImportError:\n    __version__ = "frozen-fallback"\n')
            (package/'local_module.py').write_text('VALUE = "frozen-local"\n')
            outside=Path(directory)/'installed'
            outside.mkdir()
            marker=outside/'leaked.txt'
            (outside/'_version.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("leaked")\n__version__="outside"\n')
            (outside/'thirdparty.py').write_text('VALUE="runtime-third-party"\n')
            script=f'''
import importlib, importlib.abc, importlib.util, json, sys
from pathlib import Path
sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})
from rsi_worker import setup_paths
class EditableFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname=='sglang._version':
            return importlib.util.spec_from_file_location(fullname,{str(outside/'_version.py')!r})
        if fullname=='rsi_thirdparty_probe':
            return importlib.util.spec_from_file_location(fullname,{str(outside/'thirdparty.py')!r})
        return None
sys.meta_path.insert(0,EditableFinder())
setup_paths(Path({str(root)!r}))
import sglang
import sglang.local_module
import rsi_thirdparty_probe
try:
    importlib.import_module('sglang._version')
except ModuleNotFoundError:
    blocked=True
else:
    blocked=False
print(json.dumps(dict(version=sglang.__version__,local=sglang.local_module.VALUE,
                     thirdparty=rsi_thirdparty_probe.VALUE,blocked=blocked,
                     leaked='sglang._version' in sys.modules)))
'''
            run=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,check=True)
            result=json.loads(run.stdout)
            self.assertEqual(result,dict(version='frozen-fallback',local='frozen-local',
                                         thirdparty='runtime-third-party',blocked=True,leaked=False))
            self.assertFalse(marker.exists())

    def test_symlinked_sglang_submodule_outside_bundle_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'bundle';package=root/'serving/sglang/python/sglang'
            package.mkdir(parents=True);(root/'muse_moe_w4a16').mkdir();(root/'benchmarks').mkdir()
            (package/'__init__.py').write_text('')
            outside=Path(directory)/'outside.py';outside.write_text('VALUE="escaped"\n')
            (package/'escape.py').symlink_to(outside)
            script=f'''
import importlib,sys
from pathlib import Path
sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})
from rsi_worker import setup_paths
setup_paths(Path({str(root)!r}))
try:
    importlib.import_module('sglang.escape')
except ModuleNotFoundError:
    print('blocked')
else:
    raise RuntimeError('symlink escape succeeded')
'''
            run=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,check=True)
            self.assertEqual(run.stdout.strip(),'blocked')


if __name__=='__main__':unittest.main()
