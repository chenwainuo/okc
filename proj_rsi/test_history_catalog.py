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
  self.assertTrue(all(r['tests_passed']==r['tests']==9 for r in cat['submissions']))
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
  self.assertEqual(len(cat['improvements']),37)
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
  self.assertEqual(len(self.catalog['current_exact_source_components']),18)
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
  for name in ('pw37','pw38'):
   self.assertFalse(candidates[name]['leaderboard_submitted'])
   if name=='pw37':self.assertEqual(candidates[name]['official_jobs'],[])
   else:self.assertIn('a3ed4ce5886a4415',candidates[name]['official_jobs'])
   self.assertNotIn(name,{r['candidate'] for r in self.catalog['submissions']})
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
  self.assertEqual(len(self.catalog['local_paired_observations']),10)
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
  self.assertEqual(self.catalog['authority']['best_job'],'fb548b5dc6f04696')
  self.assertIn('23cecdef1bb54c9e',{r['job_id'] for r in self.catalog['submissions']})
  self.assertGreaterEqual(len(self.catalog['all_official_jobs']),51)
 def test_server_content_hash_not_claimed_equal_to_local_sha(self):
  row=next(r for r in self.catalog['submissions'] if r['candidate']=='rs32')
  self.assertNotEqual(row['source']['sha256'],row['api_content_hash'])
  self.assertIn('UNKNOWN',row['api_content_hash_algorithm'])
  self.assertTrue(all(r['causal_delta_vs_other_jobs_us'] is None for r in self.catalog['all_official_jobs']))
 def test_csv_roundtrip_preserves_all_rows_and_structured_evidence(self):
  outputs=render(self.catalog)
  expected={'submissions':len(self.catalog['submissions']),'checks':len(self.catalog['checks']),'all_official_jobs':len(self.catalog['all_official_jobs']),'improvements':37,'positive_observations':len(self.catalog['positive_observations']),'local_observations':10,'candidates':self.catalog['counts']['candidates']}
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
   for association in config['job_source_associations'].values():dependencies.update(association[k]['path'] for k in ('source','attestation'));dependencies.update(r['path'] for r in association.get('additional_attestations',[]))
   for name in dependencies:
    dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
   source=root/'proj_rsi/candidates/rs32/submission.py'
   source.parent.mkdir(parents=True,exist_ok=True);source.write_text('# altered fixture, never GPU executed\n')
   with self.assertRaisesRegex(ValueError,'Immutable audited source changed: rs32'):build(root)

if __name__=='__main__':unittest.main()
