"""CPU-only policy contract tests. Run with unittest discovery."""
import copy
import math
import tempfile
import json
from pathlib import Path
import unittest
from .model import load_policy, score, validate_job

def fixture(resource='gpu'):
    return {'id':'family.v1.attempt01','resource':resource,'kind':'optimization','state':'BACKLOG','author':'author','created_at':1000,'depends_on':[],'write_paths':['proj_rsi/candidates/new'],
        'estimate':{'direct':{'scenario_score_us':30,'p_gain':.2,'p_valid':.8},'information':{'scenario_score_us':0,'p_resolve':0,'p_followup_gain':0,'unblocks':[],'decision':''},'basis':'subjective conditional prior; not measured performance','confidence':'low','evidence':['proj_rsi/audits/sol16-score-opportunity.md']},
        'cost':{'gpu_seconds':120,'followup_gpu_seconds':0,'cpu_minutes':10},
        'files':{'proj_rsi/tools/lab.py':'a'*64,'proj_rsi/tools/h100_trial.sh':'c'*64,'proj_rsi/tools/h100_container.sh':'d'*64},'review':{'actor':'critic','path':'proj_rsi/audits/review.md','sha256':'b'*64,'decision':'GO'},
        'launch':{'host':'root@107.170.6.60','remote_repo':'/root/okc','container':'okc-proj-example-attempt01','inner_seconds':240,'outer_seconds':300,'profile':False,'env':{'MAX_JOBS':'1'},'argv':['python3','/okc/proj_rsi/tools/lab.py'],'artifacts':['/root/okc/logs/attempt01.json']}}

class PolicyTests(unittest.TestCase):
    def test_quantified_expected_roi(self):
        got=score(fixture(),now=1000)
        self.assertAlmostEqual(got['direct_expected_us'],4.8)
        self.assertAlmostEqual(got['cost_usd'],1/6)
        self.assertAlmostEqual(got['priority'],28.8)
        self.assertTrue(got['rankable'])
    def test_lower_cost_beats_larger_scenario(self):
        small=fixture(); big=fixture()
        big['estimate']['direct']['scenario_score_us']=100
        big['cost']['gpu_seconds']=1200
        self.assertGreater(score(small,now=1000)['priority'],score(big,now=1000)['priority'])
    def test_information_max_not_sum_and_chain_cost(self):
        job=fixture(); job['estimate']['information']={'scenario_score_us':100,'p_resolve':.5,'p_followup_gain':.2,'unblocks':['followup.v1'],'decision':'choose X2 or X4 using measured readiness evidence'}
        self.assertIsNone(score(job,now=1000)['priority'])
        job['cost']['followup_gpu_seconds']=240
        got=score(job,now=1000)
        self.assertEqual(got['information_expected_us'],10)
        self.assertEqual(got['effective_value_us'],10)
        self.assertAlmostEqual(got['cost_usd'],.5)
        self.assertEqual(got['priority'],20)
        for key in ('decision','unblocks'):
            broken=copy.deepcopy(job); broken['estimate']['information'].pop(key)
            self.assertIsNone(score(broken)['priority'])
    def test_zero_is_rankable_unknown_is_held(self):
        job=fixture(); job['estimate']['direct']['scenario_score_us']=0
        self.assertEqual(score(job,now=1000)['priority'],0)
        job['estimate']['direct']['scenario_score_us']=None
        self.assertIsNone(score(job)['priority'])
        self.assertIsNone(score(job)['effective_value_us'])
    def test_nonfinite_and_probability_bounds(self):
        for value in (math.nan,math.inf,-math.inf,-1,True,None,3201,1e308):
            job=fixture(); job['estimate']['direct']['scenario_score_us']=value
            self.assertTrue(validate_job(job)); self.assertIsNone(score(job)['priority'])
        for value in (1.01,-.01,math.nan,True,None):
            job=fixture(); job['estimate']['direct']['p_gain']=value
            self.assertTrue(validate_job(job))
    def test_age_capped_and_no_future_bonus(self):
        job=fixture(); fresh=score(job,now=1000)
        self.assertEqual(score(job,now=0)['age_multiplier'],1)
        old=score(job,now=1000+864000000)
        self.assertEqual(old['age_multiplier'],1.1)
        self.assertAlmostEqual(old['priority'],fresh['priority']*1.1)
    def test_cpu_per_agent_minute(self):
        job=fixture('cpu'); job.pop('launch');job.pop('review');job.pop('files')
        self.assertEqual(validate_job(job),[])
        self.assertAlmostEqual(score(job,now=1000)['priority'],.48)
    def test_hypothetical_not_promoted_to_observation(self):
        got=score(fixture(),now=1000)
        self.assertTrue(any('subjective' in x for x in got['assumptions']))
        self.assertTrue(any('H100' in x for x in got['assumptions']))
        self.assertNotIn('measured_speedup',got)
        self.assertNotIn('disposition',got)
    def test_pinned_hash_path_independence_and_launch(self):
        self.assertEqual(validate_job(fixture(),for_admission=True),[])
        for change in ('traversal','bad_hash','author_review','hold_review','deadline','host','shell','unknown_launch'):
            job=fixture()
            if change=='traversal': job['files']={'../escape':'a'*64}
            if change=='bad_hash': job['review']['sha256']='wrong'
            if change=='author_review':job['review']['actor']='author'
            if change=='hold_review':job['review']['decision']='HOLD'
            if change=='deadline':job['launch']['outer_seconds']=260
            if change=='host':job['launch']['host']='host; echo unsafe'
            if change=='shell':job['launch']['argv']=['bash','-c','echo unsafe']
            if change=='unknown_launch':job['launch']['privileged']=True
            self.assertTrue(validate_job(job,for_admission=True),change)
    def test_guard_and_runner_pins_required_for_gpu_only(self):
        for path in ('proj_rsi/tools/h100_trial.sh','proj_rsi/tools/h100_container.sh'):
            job=fixture();job['files'].pop(path)
            self.assertTrue(any('pinned tool' in error for error in validate_job(job,for_admission=True)))
        cpu=fixture('cpu')
        for key in ('files','review','launch'):cpu.pop(key)
        self.assertEqual(validate_job(cpu,for_admission=True),[])
    def test_incumbent_bound_is_configuration_not_forecast(self):
        policy=load_policy();self.assertEqual(policy['incumbent_score_us'],3200.6150084373335)
        job=fixture();job['estimate']['direct']['scenario_score_us']=policy['incumbent_score_us']
        self.assertTrue(score(job,now=1000)['rankable'])
        configured=copy.deepcopy(policy);configured['incumbent_score_us']=20
        self.assertIsNone(score(fixture(),now=1000,policy=configured)['priority'])
        job['estimate']['direct']['scenario_score_us']=policy['incumbent_score_us']+1e-6
        self.assertIsNone(score(job,now=1000)['priority'])
    def test_profile_flag_constraints(self):
        job=fixture(); job['launch']['env']['PROJ_GPU_PROFILE']='1'
        self.assertTrue(validate_job(job,for_admission=True))
        job['launch']['profile']=True
        job['kind']='profile';job['estimate']['direct']['scenario_score_us']=0
        self.assertEqual(validate_job(job,for_admission=True),[])
        job['launch']['env']['PROJ_GPU_PROFILE']='0'
        self.assertTrue(validate_job(job,for_admission=True))
    def test_explicit_ncu_profile(self):
        for executable in ('ncu','/usr/local/cuda/bin/ncu'):
            for kind in ('profile','diagnostic'):
                job=fixture();job['kind']=kind;job['launch']['profile']=True;job['estimate']['direct']['scenario_score_us']=0
                job['launch']['argv']=[executable,'--replay-mode','kernel','--cache-control','all','--clock-control','none','python3','/okc/proj_rsi/tools/lab.py','--mode','profile']
                self.assertEqual(validate_job(job,for_admission=True),[])
    def test_ncu_rejects_wrong_profile_kind_shell_and_missing_driver(self):
        job=fixture();job['kind']='profile';job['launch']['profile']=True;job['estimate']['direct']['scenario_score_us']=0
        job['launch']['argv']=['ncu','--clock-control','none','--replay-mode','kernel','python3','/okc/proj_rsi/tools/lab.py']
        for mutation in ('profile_false','wrong_kind','shell','absent_driver','outside_driver','unreviewed_driver','python_code'):
            broken=copy.deepcopy(job)
            if mutation=='profile_false':broken['launch']['profile']=False
            if mutation=='wrong_kind':broken['kind']='optimization'
            if mutation=='shell':broken['launch']['argv']=['bash','-c','ncu python3 driver.py']
            if mutation=='absent_driver':broken['launch']['argv']=['ncu','--version']
            if mutation=='outside_driver':broken['launch']['argv']=['ncu','python3','/okc/outside.py']
            if mutation=='unreviewed_driver':broken['files'].pop('proj_rsi/tools/lab.py')
            if mutation=='python_code':broken['launch']['argv']=['ncu','python3','-c','print(1)']
            self.assertTrue(validate_job(broken,for_admission=True),mutation)
    def test_ncu_actual_target_and_no_clock_changes(self):
        job=fixture();job['kind']='diagnostic';job['launch']['profile']=True
        job['estimate']['direct']['scenario_score_us']=0
        for argv in [
            ['ncu','--clock-control','none','bash','-c','code','python3','/okc/proj_rsi/tools/lab.py'],
            ['ncu','python3','/okc/proj_rsi/tools/lab.py'],
            ['ncu','--clock-control','base','python3','/okc/proj_rsi/tools/lab.py'],
            ['ncu','--unknown','value','--clock-control','none','python3','/okc/proj_rsi/tools/lab.py'],
            ['ncu','--clock-control','none','--clock-control','base','python3','/okc/proj_rsi/tools/lab.py']]:
            job['launch']['argv']=argv
            self.assertTrue(validate_job(job,for_admission=True),argv)
    def test_diagnostic_direct_and_profile_capability_kind(self):
        for kind in ('profile','diagnostic'):
            job=fixture();job['kind']=kind
            self.assertTrue(validate_job(job))
        job=fixture();job['launch']['profile']=True
        self.assertTrue(validate_job(job))
    def test_huge_integer_and_policy_override_rejected(self):
        job=fixture();job['estimate']['direct']['scenario_score_us']=10**1000
        self.assertTrue(validate_job(job))
        self.assertIsNone(score(job)['priority'])
        for field, key, value in [('active_capacity','cpu',11),('active_capacity','gpu',2),('ready_capacity','gpu',11),('aging','max_bonus_fraction',.11)]:
            policy=load_policy();policy[field][key]=value
            with self.assertRaises(ValueError):score(fixture(),policy=policy)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'policy.json';path.write_text(json.dumps(policy))
                with self.assertRaises(ValueError):load_policy(path)
    def test_deadline_upper_bound_and_fraction(self):
        for inner,outer in [(3601,3720),(10,3721),(10.5,100),(0,100)]:
            job=fixture();job['launch']['inner_seconds']=inner;job['launch']['outer_seconds']=outer
            self.assertTrue(validate_job(job,for_admission=True))
    def test_capacity_and_policy(self):
        p=load_policy();self.assertEqual(p['ready_capacity'],{'gpu':10,'cpu':20});self.assertEqual(p['active_capacity'],{'gpu':1,'cpu':10})
    def test_terminal_historical_import(self):
        job=fixture();job['kind']='import';job['state']='DONE'
        for key in ('estimate','launch','review','files','cost'): job.pop(key)
        self.assertEqual(validate_job(job),[])
        self.assertIsNone(score(job)['priority'])
        self.assertTrue(validate_job(job,for_admission=True))

if __name__=='__main__':unittest.main()
