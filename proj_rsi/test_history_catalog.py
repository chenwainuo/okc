"""CPU tests for preserving the complete evidence partition and provenance."""
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from .history_catalog import INPUTS, ROOT, build, render,resolve_sources,merged_live_jobs,archive_jobs

class HistoryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.catalog=build()
 def test_live_authority_partition_keeps_all_ten_runs_and_forty_checks(self):
  _,config=resolve_sources(ROOT);live,_=merged_live_jobs(ROOT,config)
  cat=self.catalog
  self.assertEqual({r['job_id'] for r in live},{r['job_id'] for r in cat['all_official_jobs']})
  self.assertEqual(len(cat['all_official_jobs']),len(live))
  self.assertEqual(len(cat['submissions']),sum(r['mode']=='run' for r in live))
  self.assertEqual(len(cat['checks']),sum(r['mode']=='check' for r in live))
  self.assertEqual(cat['submissions'][0]['job_id'],'23cecdef1bb54c9e')
  self.assertEqual(cat['submissions'][0]['candidate'],'r6-c12')
  self.assertEqual(cat['submissions'][0]['score_us'],3427.959139938749)
  self.assertTrue(all(r['tests_passed']==r['tests']==9 for r in cat['submissions'] if r['status']=='done' and r['passed'] is True))
 def test_pw43_local_and_profile_failures_never_create_official_jobs(self):
  cat=self.catalog;rows=[r for r in cat['local_paired_observations'] if r['version']=='pw43']
  self.assertEqual(len(rows),30);self.assertEqual(sum(r['gain_us']>0 for r in rows),14)
  self.assertEqual(sum(r['gain_us']<0 for r in rows),12)
  self.assertTrue(all(r['shipping_gain'] is False for r in rows))
  self.assertFalse(any(r['candidate']=='pw43' for r in cat['all_official_jobs']))
  finding=next(r for r in cat['improvements'] if r['immutable_id']=='improvement.pw43.stack.fullscope.01')
  self.assertEqual(finding['score_tested_jobs'],[])
  profile=next(r for r in cat['improvements'] if r['immutable_id']=='improvement.pw38.selected.profile.context.01')
  self.assertIn('INVALID_PROFILE',profile['classification']);self.assertEqual(profile['score_tested_jobs'],[])
 def test_pw45_mixed_points_and_pw46_losses_preserve_tiny_fixed_gains(self):
  findings={r['immutable_id']:r for r in self.catalog['improvements']}
  self.assertIn('SCORED_COMPLETED',findings['improvement.pw45.stack.fullscope.01']['classification'])
  self.assertIn('REGRESSION',findings['improvement.pw46.stack.fullscope.01']['classification'])
  rows=[r for r in self.catalog['local_paired_observations'] if r['version'] in ('pw45','pw46')]
  self.assertEqual(len(rows),60);self.assertTrue(all(r['shipping_gain'] is False for r in rows))
  fixed=[r for r in self.catalog['local_paired_observations'] if r['version']=='pw45_fixed_o32']
  self.assertEqual(len(fixed),3)
  tiny=next(r for r in fixed if r['shape'].endswith('old_st5_early_zi_minus_half_st5_fullk'))
  self.assertGreater(tiny['gain_us'],0);self.assertLess(tiny['gain_us'],.3)
  self.assertTrue(any(r['gain_us']<0 for r in fixed))
  self.assertFalse(any(r['candidate']=='pw46' for r in self.catalog['all_official_jobs']))
 def test_pw45_completed_check_and_scored_outcome_remain_distinct(self):
  rows={r['job_id']:r for r in self.catalog['all_official_jobs']}
  check=rows['87db1d6a076c4da4'];run=rows['d5a125fcdcdc408d']
  self.assertEqual(check['mode'],'check');self.assertEqual(check['score_us'],3198.0269280312727)
  self.assertEqual(check['archive_status'],'ARCHIVED_30_SHAPES');self.assertEqual(len(check['actual_timed_profiles']),30)
  self.assertEqual(next(p for p in check['actual_picks'] if p['layer']=='o' and p['m']==32)['selected'],'r6')
  self.assertEqual(run['mode'],'run');self.assertEqual(run['status'],'done');self.assertEqual(run['score_us'],3179.285623687654)
  self.assertEqual(run['archive_status'],'ARCHIVED_30_SHAPES');self.assertEqual(len(run['actual_timed_profiles']),30)
  self.assertEqual(run['source']['sha256'],check['source']['sha256'])
  self.assertEqual(self.catalog['authority']['best_job'],'9916d34357b745cf')
  self.assertEqual(self.catalog['counts']['pending_scored_submissions'],0)
 def test_failed_unarchived_checks_are_not_deleted_or_assigned_sources(self):
  rows=[r for r in self.catalog['all_official_jobs'] if r['passed'] is False]
  self.assertEqual(len(rows),5)
  self.assertTrue(all(r['mode']=='check' and not r['passed'] for r in rows))
  self.assertTrue(all(r['source'] is None and r['candidate'] is None and r['score_us'] is None for r in rows))
  self.assertEqual({r['job_id'] for r in rows},{'2d9ba3f9ed534f34','140fb59bbf2c490b','8f4c78fd88f5469c','0a2f1bbff69148c5','5c994bf40c144571'})
  partial=next(r for r in rows if r['job_id']=='5c994bf40c144571')
  self.assertEqual(partial['archive_status'],'ARCHIVED_PARTIAL')
 def test_no_small_positive_threshold_and_no_duplicate_feature_inflation(self):
  cat=self.catalog;observations=cat['positive_observations']
  self.assertEqual(len(cat['improvements']),57)
  original=[r for r in observations if r['catalog_cohort']=='initial_45_job_audit']
  self.assertEqual(len(original),224)
  self.assertGreater(len(observations),224)
  self.assertEqual(len({r['observation_id'] for r in observations}),len(observations))
  self.assertEqual(sum(r['positive_bake_difference_us']<.3 for r in original),76)
  self.assertEqual(sum(r['positive_bake_difference_us']<.5 for r in original),107)
  self.assertTrue(all(r['feature_level'] is False for r in observations))
  self.assertTrue(any(r['positive_bake_difference_us']<.04 for r in observations))
 def test_actual_selected_routes_are_separate_from_available_code(self):
  rows={r['job_id']:r for r in self.catalog['submissions']}
  skipped=rows['5a39372456514f20'];best=rows['fb548b5dc6f04696']
  self.assertIn('RQ128_registerA_splitK',skipped['compiled_or_available_features'])
  self.assertFalse(any(p['selected'].startswith('rq:') for p in skipped['actual_picks']))
  self.assertTrue(any(p['layer']=='o' and p['m']==32 and p['selected'].startswith('rq:') for p in best['actual_picks']))
  self.assertNotIn('RTK_TMA_router128_256',best['compiled_or_available_features'])
  self.assertIn('RTK_TMA_router128_256',rows['e22a4b9d693d4902']['compiled_or_available_features'])
  self.assertEqual(len(self.catalog['current_exact_source_components']),22)
  self.assertEqual(len(self.catalog['current_exact_selected_routes']),30)
 def test_cursor_latest_valid_evidence_keeps_regressions_and_old_invalid_attempt(self):
  latest=self.catalog['cursor_v3_latest']
  self.assertEqual(latest['validation']['raw_calls'],770)
  row=next(r for r in self.catalog['improvements'] if r['id'].startswith('smallm cursor'))
  self.assertIn('INVALID_ATTEMPT01_THEN_VALID_V3',row['classification'])
  self.assertIn('−0.144',row['finding'])
  self.assertIn('+0.032',row['finding'])
  self.assertEqual(row['score_tested_jobs'],[])
 def test_prospective_stacks_not_in_submission_table(self):
  candidates={r['candidate']:r for r in self.catalog['candidates']}
  self.assertFalse(candidates['pw37']['leaderboard_submitted'])
  self.assertEqual(candidates['pw37']['official_jobs'],[])
  self.assertNotIn('pw37',{r['candidate'] for r in self.catalog['submissions']})
  self.assertTrue(candidates['pw38']['leaderboard_submitted'])
  self.assertIn('a3ed4ce5886a4415',candidates['pw38']['official_jobs'])
  self.assertIn('a3f0ebf25a754c01',candidates['pw38']['official_jobs'])
  self.assertIn('optional_EH_bit16_no_cursor',candidates['pw38']['compiled_or_available_features'])
  self.assertNotIn('smallm_cursor',candidates['pw38']['compiled_or_available_features'])
 def test_small_new_parent_gains_and_actual_losses_preserved_not_shipped(self):
  rows={r['id']:r for r in self.catalog['improvements']}
  for name,parent,actual in [('Native-X STMatrix attempt01',.1759994775056839,-7.455999031662941),('Native-X4 attempt01',.35200081765651703,-7.4720000848174095)]:
   row=rows[name]
   self.assertEqual(row['parent_paired_gain_us'],parent)
   self.assertEqual(row['actual_incumbent_paired_gain_us'],actual)
   self.assertIn('NOT_SHIPPED',row['classification'])
   self.assertEqual(row['score_tested_jobs'],[])
   self.assertIsNone(row['official_score_us'])
   self.assertTrue(all('sha256' in ref for ref in row['evidence']))
  self.assertEqual(len(self.catalog['local_paired_observations']),319)
 def test_normalized_release_profile_discovered_without_candidate_directory(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);path=root/'results/proj-rsi/releases/frozen-v1/profiles/submission.pending.json'
   path.parent.mkdir(parents=True);path.write_text(json.dumps({'job_id':'pending.id','benchmarks':[],'summary':{'passed':True}}))
   records=archive_jobs(root)
   self.assertIn('pending.id',records)
   self.assertIsNone(records['pending.id']['candidate'])
   self.assertTrue(records['pending.id']['ref']['sha256'])
 def test_pending_check_retains_release_association_without_becoming_scored(self):
  row=next(r for r in self.catalog['all_official_jobs'] if r['job_id']=='a3ed4ce5886a4415')
  self.assertEqual(row['mode'],'check');self.assertEqual(row['candidate'],'pw38')
  self.assertFalse(row['leaderboard_submission'])
  self.assertEqual(row['source']['sha256'],'29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e')
  self.assertTrue(row['release_attestation']['sha256'])
  self.assertEqual(row['status'],'done')
  self.assertEqual(row['score_us'],3193.637501792908)
  self.assertEqual(row['tests_passed'],9)
  self.assertEqual(len(row['actual_timed_profiles']),30)
  self.assertTrue(any('EH' in pick['selected'] for pick in row['actual_picks']))
  self.assertTrue(row['additional_release_attestations'])
  self.assertEqual(self.catalog['authority']['best_job'],'9916d34357b745cf')
  self.assertIn('23cecdef1bb54c9e',{r['job_id'] for r in self.catalog['submissions']})
  self.assertGreaterEqual(len(self.catalog['all_official_jobs']),51)
 def test_completed_scored_row_uses_own_result_and_verified_board_authority(self):
  row=next(r for r in self.catalog['submissions'] if r['job_id']=='a3f0ebf25a754c01')
  self.assertEqual(row['candidate'],'pw38');self.assertEqual(row['mode'],'run')
  self.assertEqual(row['score_us'],3198.6175079817267)
  self.assertNotEqual(row['score_us'],3193.637501792908)
  self.assertEqual(row['tests'],9);self.assertEqual(len(row['actual_timed_profiles']),30)
  self.assertEqual(self.catalog['authority']['source_sha256'],'501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9')
  self.assertEqual(self.catalog['authority']['board_row']['rank'],59)
  self.assertEqual(self.catalog['authority']['board_row']['attempts'],18)
  self.assertEqual(len(self.catalog['submissions']),18);self.assertEqual(len(self.catalog['checks']),48)
 def test_current_source_view_uses_scored_EH_flags_and_preserves_old_rs32_audit(self):
  view=json.loads((ROOT/'proj_rsi/audits/selected-hotpaths-pw38/current-stack-snapshot.json').read_text())
  self.assertEqual(view['official_job_id'],'a3f0ebf25a754c01')
  self.assertEqual(view['historical_rs32_job'],'fb548b5dc6f04696')
  self.assertEqual(len(view['components']),20);self.assertEqual(len(view['routes']),30)
  pooler=next(r for r in view['routes'] if r['layer']=='pooler' and r['m']==128)
  self.assertTrue(pooler['EH_selected']);self.assertEqual(pooler['RQ_flags'],65538)
  self.assertEqual(pooler['actual_prepare_pick']['selected'],'rq:s4s5zEH')
  self.assertTrue(all(binding['source_sha256']==view['source']['sha256'] for r in view['routes'] for binding in r['bindings']))
  self.assertEqual(self.catalog['historical_rs32_source_map']['path'],'proj_rsi/audits/current-leaderboard-stack-source.json')
 def test_historical_pw45_source_and_actual_scored_routes_do_not_infer_new_HALF_selection(self):
  view=json.loads((ROOT/'results/proj-rsi/releases/pw45-v1-835acb90b47b/historical-current-stack-before-gv.json').read_text())
  self.assertEqual(view['official_job_id'],'d5a125fcdcdc408d');self.assertEqual(len(view['routes']),30)
  self.assertEqual(len(view['components']),21);self.assertEqual(len(view['route_differences_from_pw38']),11)
  self.assertEqual(view['new_HALF52_selected_routes'],[])
  self.assertFalse(any(r['EH_selected'] for r in view['routes']))
  o=next(r for r in view['routes'] if r['layer']=='o' and r['m']==32)
  self.assertEqual(o['RQ_flags'],34);self.assertEqual(o['actual_prepare_pick']['selected'],'rq:t32s5ze-k2')
  self.assertEqual(self.catalog['authority']['previous_best_job'],'d5a125fcdcdc408d')
  self.assertAlmostEqual(self.catalog['authority']['observed_package_point_change_us'],22.247220034975726)
  self.assertFalse(self.catalog['authority']['point_change_is_isolated_feature_gain'])
 def test_current_gv_best_uses_actual_scored_predxor_routes_and_no_synthetic_check(self):
  view=self.catalog['current_stack_view'];self.assertEqual(view['official_job_id'],'9916d34357b745cf')
  self.assertIsNone(view['check_job']);self.assertEqual(len(view['routes']),30)
  actual={(r['layer'],r['m']) for r in view['routes'] if r['GV_predxor_selected']}
  self.assertEqual(actual,{('pooler',1),('pooler',8),('pooler',16),('unpooler',16)})
  self.assertTrue(all(len(r['GV_template'])==10 for r in view['routes'] if r['GV_predxor_selected']))
  self.assertTrue(all(b['source_sha256']==view['source']['sha256'] for r in view['routes'] for b in r['bindings']))
  rows={r['job_id']:r for r in self.catalog['all_official_jobs']}
  self.assertEqual(rows['87044866f4494726']['mode'],'check');self.assertEqual(rows['7b2da0f017d54e3d']['mode'],'run')
  self.assertNotEqual(rows['87044866f4494726']['score_us'],rows['7b2da0f017d54e3d']['score_us'])
 def test_pw39_and_pw40_official_regressions_do_not_replace_best(self):
  checks={r['job_id']:r for r in self.catalog['checks']}
  previous=checks['49d4063f97c14250'];completed=checks['6c4eff6d63784f90']
  self.assertEqual(previous['score_us'],3212.6810592521706);self.assertEqual(previous['tests_passed'],9)
  self.assertEqual(previous['candidate'],'pw39');self.assertEqual(completed['candidate'],'pw40')
  self.assertEqual(completed['score_us'],3252.054799599241);self.assertEqual(completed['archive_status'],'ARCHIVED_30_SHAPES')
  self.assertEqual(completed['tests_passed'],9)
  finding=next(r for r in self.catalog['improvements'] if r['id']=='PW40 full-policy local validation + official check')
  self.assertEqual(finding['official_prepare']['total_seconds'],161)
  self.assertFalse(finding['official_prepare']['RQ_budget_skip_present'])
  self.assertEqual(finding['official_prepare']['QKV128_screen_us']['rq:wring_x6w4hz'],17.12)
  self.assertEqual(finding['official_prepare']['actual_QKV128_pick']['selected'],'rq:s4s5hz')
  self.assertEqual(finding['score_tested_jobs'],[])
  self.assertEqual(self.catalog['authority']['best_job'],'9916d34357b745cf')
  self.assertEqual(len(self.catalog['all_official_jobs']),66)
  self.assertEqual(len(self.catalog['submissions']),18)
  observations=[r for r in self.catalog['local_paired_observations'] if r['version'] in ('pw39','pw40')]
  self.assertEqual(len(observations),120)
  self.assertTrue(any(r['gain_us']>0 and r['F1_gain_us']<0 for r in observations))
  self.assertTrue(any(r['gain_us']<0 for r in observations))
  self.assertEqual(self.catalog['pw40_latest_local']['Wring_selected'],[])
 def test_complete_stack_own_scored_result_and_local_controls_remain_separate(self):
  cat=self.catalog;rows={r['job_id']:r for r in cat['submissions']}
  complete=rows['161bdb8de7354344']
  self.assertEqual(complete['candidate'],'complete_gv_rq_eh_o32_v1')
  self.assertEqual(complete['score_us'],3187.14239967648)
  self.assertEqual(complete['tests_passed'],9);self.assertEqual(complete['tests'],9)
  self.assertEqual(complete['source']['sha256'],'d1ca973956a3a968796964679e8685741dba0d390e8f4c09f4b0794be108872b')
  self.assertEqual(len(complete['actual_timed_profiles']),30)
  self.assertEqual(cat['authority']['best_job'],'9916d34357b745cf')
  self.assertTrue(all(rows[j]['score_us']>cat['authority']['score_us'] for j in ['161bdb8de7354344','c2b3b2f924e043a8','22d677fa4501409f','ed28c762f3274df1']))
  full=[r for r in cat['local_paired_observations'] if r['version']=='complete_gv_rq_eh_o32_v1']
  inc=[r for r in cat['local_paired_observations'] if r['version']=='gv_private_eh_o32_v1']
  self.assertEqual(len(full),30);self.assertEqual(len(inc),30)
  self.assertEqual({r['control'] for r in full},{'gv_predxor_pw45_v2'})
  self.assertEqual({r['control'] for r in inc},{'gv_private_eh_screen_v1'})
  self.assertTrue(all(r['shipping_gain'] is False for r in full+inc))
  self.assertTrue(any(r['gain_us']<0 for r in full));self.assertTrue(any(r['gain_us']<0 for r in inc))
  self.assertNotIn('gv_private_eh_o32_v1',{r['candidate'] for r in cat['submissions']})
  self.assertEqual(cat['complete_stack_latest_local']['actual_policy_geometric_load_speedup']['paired'],1.0016700108951944)
 def test_new_completed_checks_keep_receipt_provenance_and_local_signs(self):
  rows={r['job_id']:r for r in self.catalog['checks']}
  for jid,score in [('8e9cba04c9d644c8',3221.2172600872086),('323514d58af84e72',3173.2818089965185),('abbaf85681f74e53',3178.0129977697934)]:
   row=rows[jid];self.assertEqual(row['score_us'],score)
   self.assertEqual(row['tests_passed'],9);self.assertFalse(row['leaderboard_submission'])
   self.assertEqual(row['status_evidence_kind'],'accepted_receipt_and_own_completed_official_result_not_API_capture')
   self.assertEqual(len(row['actual_timed_profiles']),30)
  self.assertEqual(self.catalog['authority']['best_job'],'9916d34357b745cf')
  m1=[r for r in self.catalog['local_paired_observations'] if r['version']=='gv53_splitcta_m1_selection_v1']
  self.assertEqual(len(m1),30);self.assertTrue(any(r['gain_us']<0 for r in m1));self.assertTrue(any(r['gain_us']>0 for r in m1))
  self.assertTrue(all(r['shipping_gain'] is False for r in m1))
 def test_lut_whole_parent_progress_keeps_anchor_losses_and_unmatched_F1(self):
  rows=[r for r in self.catalog['local_paired_observations'] if r['attempt_id']=='research.moe.ss.lut.whole.01']
  self.assertEqual(len(rows),5);self.assertTrue(all(r['F1_gain_us'] is None and not r['shipping_gain'] for r in rows))
  anchor=[r for r in rows if r['control']=='anchor'];self.assertEqual(len(anchor),3)
  self.assertTrue(all(r['gain_us']<0 and r['wins']==0 for r in anchor))
  parent=next(r for r in rows if r['control']=='SS' and r['version']=='LUT')
  self.assertGreater(parent['gain_us'],280);self.assertEqual(parent['wins'],8)
 def test_server_content_hash_not_claimed_equal_to_local_sha(self):
  row=next(r for r in self.catalog['submissions'] if r['candidate']=='rs32')
  self.assertNotEqual(row['source']['sha256'],row['api_content_hash'])
  self.assertIn('UNKNOWN',row['api_content_hash_algorithm'])
  self.assertTrue(all(r['causal_delta_vs_other_jobs_us'] is None for r in self.catalog['all_official_jobs']))
 def test_csv_roundtrip_preserves_all_rows_and_structured_evidence(self):
  outputs=render(self.catalog)
  expected={'submissions':len(self.catalog['submissions']),'checks':len(self.catalog['checks']),'all_official_jobs':len(self.catalog['all_official_jobs']),'improvements':57,'positive_observations':len(self.catalog['positive_observations']),'local_observations':319,'candidates':self.catalog['counts']['candidates']}
  for table,count in expected.items():
   rows=list(csv.DictReader(io.StringIO(outputs[f'proj_rsi/tables/{table}.csv'])))
   self.assertEqual(len(rows),count,table)
  rows=list(csv.DictReader(io.StringIO(outputs['proj_rsi/tables/submissions.csv'])))
  self.assertEqual(json.loads(rows[0]['source'])['sha256'],self.catalog['submissions'][0]['source']['sha256'])
 def test_render_is_deterministic_and_checked_in_outputs_current(self):
  outputs=render(self.catalog)
  self.assertEqual(outputs,render(build()))
  for name,text in outputs.items():self.assertEqual((ROOT/name).read_text(),text,name)
 def test_immutable_audited_source_change_fails_in_isolated_fixture(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   _,config=resolve_sources(ROOT)
   dependencies=set(INPUTS.values())|{'proj_rsi/history_sources.json'}|{r['path'] for r in config['jobs_snapshots']}
   dependencies.update(r['path'] for r in config.get('board_snapshots',[]))
   for association in config['job_source_associations'].values():dependencies.update(association[k]['path'] for k in ('source','attestation'));dependencies.update(r['path'] for r in association.get('additional_attestations',[]))
   for entry in config.get('completed_receipt_jobs',[]):dependencies.update(entry[k]['path'] for k in ('receipt','result'))
   for name in dependencies:
    dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
   source=root/'proj_rsi/candidates/rs32/submission.py'
   source.parent.mkdir(parents=True,exist_ok=True);source.write_text('# altered fixture, never GPU executed\n')
   with self.assertRaisesRegex(ValueError,'Immutable audited source changed: rs32'):build(root)

if __name__=='__main__':unittest.main()
