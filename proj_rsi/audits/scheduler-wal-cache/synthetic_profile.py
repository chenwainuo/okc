"""1000-event temporary-store correctness+descriptive perf; never opens actual scheduler."""
from copy import deepcopy
import hashlib,json,statistics,tempfile,time,subprocess,sys,importlib
from pathlib import Path
from unittest import mock
LegacyModule=importlib.import_module('proj_rsi.audits.scheduler-wal-cache.baseline_loader').load()
Legacy=LegacyModule.QueueStore
from proj_rsi.scheduling import _store_cached_proposal as C
HERE=Path(__file__).resolve().parent

def run():
    with tempfile.TemporaryDirectory(prefix='queue-cache-1000-') as tmp:
        directory=Path(tmp);new=C.QueueStore(directory)
        state={'version':1,'sequence':0,'jobs':{'synthetic':{'metadata':'s'*65536,'last':-1,'state':'RUNNING','owner':'synthetic-owner','lease':{'id':'must-not-expire','heartbeat_at':0,'expiry_is_not_death':True}}}}
        started=time.perf_counter()
        with new._locked():
            for i in range(1000):
                state['jobs']['synthetic']['last']=i;new._commit(state,'synthetic',data={'i':i})
        build=time.perf_counter()-started;expected=deepcopy(state);wal_sha=hashlib.sha256(new.wal_path.read_bytes()).hexdigest()
        def read(store):
            start=time.perf_counter()
            with store._locked():got=store._read()
            elapsed=time.perf_counter()-start;assert got==expected;return elapsed
        with mock.patch.object(C.QueueStore,'_record',wraps=C.QueueStore._record) as check:
            cold=read(new);assert check.call_count==1000
        times={};process_wall={}
        expected_hash=hashlib.sha256(C._json(expected)).hexdigest()
        code="""import hashlib,json,sys,time
import importlib
Legacy=importlib.import_module('proj_rsi.audits.scheduler-wal-cache.baseline_loader').load().QueueStore
from proj_rsi.scheduling._store_cached_proposal import QueueStore as Cached
store=(Legacy if sys.argv[2]=='original_strict' else Cached)(sys.argv[1])
start=time.perf_counter()
with store._locked():state=store._read()
elapsed=time.perf_counter()-start
raw=json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
print(json.dumps({'read_seconds':elapsed,'state_sha256':hashlib.sha256(raw).hexdigest()}))
"""
        for label in ('original_strict','cached_new_process'):
            times[label]=[];process_wall[label]=[]
            for _ in range(5):
                start=time.perf_counter()
                child=subprocess.run([sys.executable,'-c',code,str(directory),label],capture_output=True,text=True,check=True)
                process_wall[label].append(time.perf_counter()-start)
                record=json.loads(child.stdout);assert record['state_sha256']==expected_hash
                times[label].append(record['read_seconds'])
        counts=[]
        for _ in range(5):
            with mock.patch.object(C.QueueStore,'_record',wraps=C.QueueStore._record) as check:
                read(C.QueueStore(directory));assert check.call_count==1;counts.append(check.call_count)
        assert hashlib.sha256(new.wal_path.read_bytes()).hexdigest()==wal_sha,'reads changed authoritative WAL'
        legacy=Legacy(directory)
        with legacy._locked():
            state=legacy._read()
            for i in range(3):state['jobs']['synthetic']['last']=1000+i;legacy._commit(state,'legacy-tail')
        expected=deepcopy(state)
        with mock.patch.object(C.QueueStore,'_record',wraps=C.QueueStore._record) as check:
            tail=read(C.QueueStore(directory));assert check.call_count==4
        med={label:statistics.median(values) for label,values in times.items()}
        return {'candidate_source_sha256':hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),'baseline_source_sha256':hashlib.sha256(LegacyModule._review_source_path.read_bytes()).hexdigest(),'status':'SYNTHETIC_1000_EVENT_CORRECTNESS_PASS','temporary_only':True,'events_before_tail':1000,'WAL_bytes':new.wal_path.stat().st_size,'build_seconds':build,'cold_candidate_seconds':cold,'cold_record_validations':1000,'hot_record_validations':counts,'three_legacy_tail_events_record_validations':4,'legacy_tail_seconds':tail,'each_sample_fresh_interpreter':True,'whole_process_wall_samples_seconds':process_wall,'samples_seconds':times,'medians_seconds':med,'observed_median_ratio_original_over_cached':med['original_strict']/med['cached_new_process'],'speed_threshold_asserted':False,'WAL_byte_identity_across_reads':'PASS','full_snapshot_and_lease_equal':'PASS','event_format_change':False,'GPU_calls':0}
if __name__=='__main__':
    result=run();(HERE/'synthetic-1000-profile.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
