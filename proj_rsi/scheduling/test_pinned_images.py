"""Focused CPU-only tests; run in disposable package with proposed model/dispatch."""
import copy,hashlib,shlex,unittest
from pathlib import Path
from proj_rsi.scheduling.model import launch_tools,DEFAULT_GPU_IMAGE,SGLANG_GPU_IMAGE,validate_job
from proj_rsi.scheduling.test_dispatch import DispatcherTests,FakeTransport

def digest(b):return hashlib.sha256(b).hexdigest()
class PinnedImageTests(DispatcherTests):
 def add_sglang(self):
  self.job['launch']['image']=SGLANG_GPU_IMAGE
  for name in ('h100_sglang_trial.sh','h100_sglang_container.sh'):
   rel='proj_rsi/tools/'+name;data=('# scoped '+name+'\n').encode();p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);self.job['files'][rel]=digest(data)
  original=self.transport.runner
  def runner(host,path='/root/okc_run.sh'):
   if path=='/root/okc_run.sh':return original(host)
   assert path=='/root/okc_run_sglang.sh'
   return {'sha256':'0'*64 if self.transport.runner_bad else digest(self.transport.remote['proj_rsi/tools/h100_sglang_container.sh']),'mode':self.transport.runner_mode}
  self.transport.runner=runner
 def test_default_and_explicit_default_unchanged_command(self):
  launch=copy.deepcopy(self.job['launch']);handles={'host_log':'/root/okc/logs/scheduler/same.log'}
  before=self.dispatch._command(launch,handles);launch['image']=DEFAULT_GPU_IMAGE
  self.assertEqual(before,self.dispatch._command(launch,handles));self.assertEqual(launch_tools({})[2],'/root/okc_run.sh')
 def test_sglang_launch_pins_selected_installed_helper_and_guard(self):
  self.add_sglang();job=self.launch();self.assertEqual(job['state'],'RUNNING')
  script=shlex.split(self.transport.commands[0])[2];self.assertIn('/proj_rsi/tools/h100_sglang_trial.sh',script);self.assertNotIn('/proj_rsi/tools/h100_trial.sh',script)
 def test_sglang_wrong_installed_runner_hash_holds(self):
  self.add_sglang();self.transport.runner_bad=True;self.assertEqual(self.launch()['state'],'HELD');self.assertFalse(self.transport.commands)
 def test_sglang_wrong_installed_mode_holds(self):
  self.add_sglang();self.transport.runner_mode=0o644;self.assertEqual(self.launch()['state'],'HELD');self.assertFalse(self.transport.commands)
 def test_sglang_missing_selected_guard_pin_rejected(self):
  self.add_sglang();del self.job['files']['proj_rsi/tools/h100_sglang_trial.sh']
  with self.assertRaises(ValueError):self.dispatch._validate(self.job)
 def test_sglang_cannot_substitute_default_runner_pin(self):
  self.add_sglang();del self.job['files']['proj_rsi/tools/h100_sglang_container.sh']
  with self.assertRaises(ValueError):self.dispatch._validate(self.job)
 def test_unknown_images_rejected_by_model_and_dispatch(self):
  for value in ('ubuntu:latest','lmsysorg/sglang:latest',SGLANG_GPU_IMAGE+'x',None,[]):
   self.job['launch']['image']=value
   self.assertTrue(any('image' in e for e in validate_job(self.job)))
   with self.assertRaises(ValueError):self.dispatch._validate(self.job)
if __name__=='__main__':unittest.main()
