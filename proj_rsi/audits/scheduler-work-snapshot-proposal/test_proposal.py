"""CPU-only coherent-ranking/read-count/concurrency proposal tests."""
import copy,importlib.util,threading,unittest
from pathlib import Path
from unittest import mock
from proj_rsi.scheduling import test_dispatch as T
from proj_rsi.scheduling.dispatch import GPUDispatcher as Original
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('proj_rsi.scheduling._isolated_work_snapshot',HERE/'dispatch.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)

class ReadCountStore:
 def __init__(self,n=1000):
  self.jobs=[dict(id=str(i),resource='gpu',state='READY',launch={'outer_seconds':60})for i in range(n)]
  self.reads={'snapshot':0,'rank':0,'get':0};self.claims=0
 def snapshot(self):self.reads['snapshot']+=1;return {'jobs':{j['id']:copy.deepcopy(j)for j in self.jobs}}
 def rank(self,resource):self.reads['rank']+=1;return copy.deepcopy(self.jobs)
 def get(self,id):self.reads['get']+=1;return copy.deepcopy(self.jobs[int(id)])
 def claim(self,*args):self.claims+=1;return None

class ProposalChecks(unittest.TestCase):
 def test_rank_rows_avoid_N_full_WAL_reads(self):
  for cls,expected in ((Original,1001),(P.GPUDispatcher,0)):
   s=ReadCountStore();d=cls(s,HERE,transport=object())
   self.assertEqual(d.work(max_wall_seconds=1)['status'],'INSUFFICIENT_RESERVATION_BUDGET')
   self.assertEqual(s.reads,dict(snapshot=1,rank=1,get=expected));self.assertEqual(s.claims,0)
 def test_current_rank_state_not_older_snapshot_state(self):
  s=ReadCountStore(1);original=s.snapshot
  def old_snapshot():
   snapshot=original();snapshot['jobs']['0']['state']='HELD';return snapshot
  s.snapshot=old_snapshot
  self.assertEqual(P.GPUDispatcher(s,HERE,transport=object()).work(max_wall_seconds=1)['status'],'INSUFFICIENT_RESERVATION_BUDGET')
  self.assertEqual(s.reads['get'],0)
 def test_current_rank_HELD_does_not_launch_old_READY(self):
  s=ReadCountStore(1);original=s.rank
  def current_rank(resource):
   rows=original(resource);rows[0]['state']='HELD';return rows
  s.rank=current_rank
  self.assertEqual(P.GPUDispatcher(s,HERE,transport=object()).work(max_wall_seconds=100)['status'],'NO_READY_JOB');self.assertEqual(s.claims,0)
 def test_claim_rechecks_after_ranking(self):
  s=ReadCountStore(1);d=P.GPUDispatcher(s,HERE,transport=object())
  with mock.patch.object(d,'launch_next',side_effect=lambda *a,**kw:s.claim('gpu','root'))as launch:
   self.assertEqual(d.work(max_wall_seconds=100)['status'],'NO_CLAIM')
  self.assertEqual(s.claims,1);launch.assert_called_once()

class ProposalRegression(T.DispatcherTests):
 def setUp(self):
  super().setUp();self.dispatch=P.GPUDispatcher(self.store,self.root,transport=self.transport)
 def test_other_controller_claim_between_rank_and_our_claim(self):
  self.register();rank=self.store.rank;observed=[]
  def racing_rank(resource):
   rows=rank(resource)
   def competitor():observed.append(self.store.claim('gpu','other-controller'))
   thread=threading.Thread(target=competitor);thread.start();thread.join(timeout=5)
   self.assertFalse(thread.is_alive());return rows
  with mock.patch.object(self.store,'rank',side_effect=racing_rank):
   self.assertEqual(self.dispatch.work(max_wall_seconds=120)['status'],'NO_CLAIM')
  self.assertEqual(observed[0]['owner'],'other-controller');self.assertEqual(len(self.transport.commands),0)
  self.assertEqual(self.store.get(self.job['id'])['state'],'RUNNING')
 def test_new_higher_ROI_large_job_after_rank_still_checks_claim_budget(self):
  self.register();rank=self.store.rank
  def racing_rank(resource):
   rows=rank(resource);high=copy.deepcopy(self.job);high['id']='higher-later';high['launch']['container']='okc-proj-later';high['launch']['outer_seconds']=300;high['estimate']['direct']['scenario_score_us']=100
   self.store.register(high);return rows
  with mock.patch.object(self.store,'rank',side_effect=racing_rank):
   result=self.dispatch.work(max_wall_seconds=120)
  self.assertEqual(result['status'],'HELD');self.assertEqual(result['job_id'],'higher-later')
  self.assertFalse(self.transport.commands)
  self.assertEqual(self.store.get('higher-later')['state'],'HELD')
  self.assertEqual(self.store.get(self.job['id'])['state'],'READY')

if __name__=='__main__':unittest.main()
