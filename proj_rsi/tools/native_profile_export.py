"""Offline NCU report import only. No CUDA API, kernel launch or extra capability.
Run inside the normal serialized root guard; never concurrently with timing.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import traceback

ROOT=Path(__file__).resolve().parents[2]
INPUT_REL='proj_rsi/audits/native-profile-01/input/native-profile-01.ncu-rep'
INPUT_SHA='4bbde4f4d2a157dde8504dee2764c1a80706dd4ae1ff735725703681b39c6ddf'
NCU=Path('/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu')
CLIENT_VERSION='2025.1.1'
EXPECTED=('rx_pack','rq_px_dp','rq_tma_x')
BOUNDS={'inner_seconds':120,'outer_seconds':180,'version_timeout_seconds':5,'per_import_timeout_seconds':45}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()


def require_input(path):
 path=Path(path)
 assert path.resolve()==path and path.is_file() and not path.is_symlink(),'unsafe/missing report'
 assert path.stat().st_size==1080736,'unexpected report size'
 assert sha(path)==INPUT_SHA,'captured input hash mismatch'
 return path


def require_version(text):
 assert re.search(r'\b2025\.1\.1(?:\.0)?(?=\s|$)',text),'unsupported NCU client version'
 return CLIENT_VERSION


def commands(report):
 # Fixed argument grammar. Never invoke a target program, replay, capture or metric collection.
 report=str(Path(report))
 return [('raw.csv',[str(NCU),'--import',report,'--page','raw','--csv']),
         ('source.csv',[str(NCU),'--import',report,'--page','source','--print-source','sass','--csv'])]


def raw_rows(text):
 rows=list(csv.reader(io.StringIO(text)))
 header=next((row for row in rows if 'Kernel Name' in row and 'ID' in row),None)
 assert header is not None,'raw CSV header unavailable'
 result=[];started=False
 for row in rows:
  if row==header:started=True;continue
  if not started or not row:continue
  if len(row)!=len(header):continue
  item=dict(zip(header,row))
  if re.fullmatch(r'\d+',item.get('ID','')):result.append(item)
 return result


def source_tables(text):
 result=[];table=None;header=None
 for row in csv.reader(io.StringIO(text)):
  if not row:continue
  if row[0]=='Kernel Name':
   table={'kernel':row[1],'header':[],'rows':[]};result.append(table);header=None
  elif row[0]=='Address' and table is not None:
   header=row;table['header']=row
  elif table is not None and header and row[0].startswith('0x'):
   assert len(row)==len(header),'source CSV column mismatch'
   table['rows'].append(dict(zip(header,row)))
 return result


def numeric(value):
 try:return int(value.replace(',',''))
 except (ValueError,AttributeError):return None


def summarize_exports(raw,source):
 assert len(raw)==len(source)==3,'expected exactly three imported nodes'
 result={}
 for token in EXPECTED:
  matches=[r for r in raw if token in r['Kernel Name']]
  tables=[t for t in source if token in t['kernel']]
  assert len(matches)==len(tables)==1,('unexpected/duplicate imported kernel',token)
  record=matches[0];table=tables[0];assert table['rows'],'source PCs unavailable'
  samples=[numeric(row.get('# Samples')) for row in table['rows']]
  source_total=sum(samples) if all(v is not None for v in samples) else None
  rawkey='smsp__pcsamp_sample_count'
  rawtotal=numeric(record.get(rawkey))
  dropped={key:value for key,value in record.items() if 'pcsamp' in key and ('drop' in key or 'pass' in key or 'interval' in key)}
  result[token]={'kernel_name':record['Kernel Name'],'source_PC_rows':len(table['rows']),
   'source_samples':source_total,'raw_samples':rawtotal,'sample_count_metric_available':rawtotal is not None,
   'source_sample_column_available':'# Samples' in table['header'],
   'source_raw_sample_reconciliation':'PASS' if source_total is not None and source_total==rawtotal else 'UNKNOWN_OR_MISMATCH_ANALYST_REVIEW_REQUIRED',
   'sampling_metadata_raw':dropped,'role_sample_denominators':'PENDING_CPU_SOURCE_SASS_BINDING_AND_ANALYSIS',
   'interpretation':'Counts/samples are not microseconds; no inferred speed result.'}
 return result


def cpu_check():
 pairs=commands(ROOT/INPUT_REL)
 assert BOUNDS=={'inner_seconds':120,'outer_seconds':180,'version_timeout_seconds':5,'per_import_timeout_seconds':45}
 for filename,argv in pairs:
  assert argv[0]==str(NCU) and argv[1:3]==['--import',str(ROOT/INPUT_REL)]
  assert argv[-1]=='--csv' and set(argv[3:])<= {'--page','raw','source','--print-source','sass','--csv'}
 assert require_version('Version 2025.1.1.0 (build 35528883)')==CLIENT_VERSION
 for wrong in ('Version 2025.1.2.0','Version 2025.1.1.9','Version 2024.1.1.0','Version 2025.1.1beta','Version 2025.1.1.0garbage','Version 2025.1.1-rc1','Version 2025.1.1+build','missing'):
  try:require_version(wrong)
  except AssertionError:pass
  else:raise AssertionError(('unsupported version accepted',wrong))
 require_input(ROOT/INPUT_REL)
 with tempfile.TemporaryDirectory() as directory:
  altered=Path(directory)/'wrong.ncu-rep';altered.write_bytes(b'wrong')
  try:require_input(altered)
  except AssertionError:pass
  else:raise AssertionError('changed report accepted')
 raw='"ID","Kernel Name","smsp__pcsamp_sample_count"\n"","","sample"\n'+''.join('"%s","%s","%s"\n'%(i,name,i+1) for i,name in enumerate(EXPECTED))
 source=''.join('"Kernel Name","%s"\n"Address","Source","# Samples"\n"0x1000","NOP","%s"\n'%(name,i+1) for i,name in enumerate(EXPECTED))
 parsed=summarize_exports(raw_rows(raw),source_tables(source));assert len(parsed)==3 and all(v['source_raw_sample_reconciliation']=='PASS' for v in parsed.values())
 missing=raw.replace('"smsp__pcsamp_sample_count"','"metric_not_available"')
 assert all(v['raw_samples'] is None for v in summarize_exports(raw_rows(missing),source_tables(source)).values())
 assert numeric('N/A') is None and numeric('-') is None and numeric('1,024')==1024
 return {'status':'CPU_PASS_EXPORT_UNRUN','GPU_calls':0,'NCU_invocations':0,'root_only_serialized_normal_guard':True,'bounds':BOUNDS,
  'input_sha256':INPUT_SHA,'input_bytes':1080736,'client_path':str(NCU),'required_client_version':CLIENT_VERSION,
  'capture_binary_sha256':'UNKNOWN_NOT_INVENTED; runtime records export-client SHA/version/image separately',
  'fixed_import_only_commands':pairs,'negative_changed_report':'PASS','negative_unsupported_versions':8,
  'three_node_CSV_PC_parser':'PASS','missing_metric_stays_unknown_not_zero':'PASS','result_scope':'Offline export only; actual source/sample/role analysis remains separately required.'}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cpu-check',action='store_true');ap.add_argument('--output',type=Path)
 args=ap.parse_args()
 if args.cpu_check:print(json.dumps(cpu_check(),indent=2));return
 assert args.output is not None
 out=args.output
 assert out.is_absolute() and out.resolve()==out and (ROOT/'logs/scheduler') in out.parents,'unsafe output path'
 assert not out.exists(),'export output child must be nonexistent'
 out.mkdir(parents=True,exist_ok=False)
 result={'status':'EXPORT_STARTED_NO_COUNTER_INTERPRETATION','input_path':INPUT_REL,'input_sha256_expected':INPUT_SHA,'source_sha256':sha(__file__),
  'bounds':BOUNDS,'capture_binary_sha256':'UNKNOWN_NOT_RECORDED_BY_CAPTURE; do not equate export binary hash with capture provenance',
  'image_runtime_identity':{'hostname':socket.gethostname(),'os_release':Path('/etc/os-release').read_text() if Path('/etc/os-release').is_file() else None},'commands':[]}
 def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 save()
 try:
  report=require_input(ROOT/INPUT_REL)
  assert NCU.is_file() and os.access(NCU,os.X_OK),'pinned client path unavailable'
  binary=NCU.resolve();assert binary==NCU,'unexpected client symlink/path'
  client_sha=sha(binary)
  version=subprocess.run([str(NCU),'--version'],capture_output=True,text=True,timeout=BOUNDS['version_timeout_seconds'],check=True,shell=False)
  versiontext=version.stdout+version.stderr;require_version(versiontext)
  (out/'ncu-version.txt').write_text(versiontext)
  result['export_client']={'path':str(NCU),'sha256':client_sha,'version':versiontext,'version_pin':CLIENT_VERSION};save()
  for filename,argv in commands(report):
   record={'argv':argv,'stdout':filename,'stderr':filename+'.stderr.txt'};result['commands'].append(record);save()
   with (out/filename).open('x') as stdout,(out/(filename+'.stderr.txt')).open('x') as stderr:
    proc=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=BOUNDS['per_import_timeout_seconds'],shell=False)
   record['returncode']=proc.returncode;save()
   assert proc.returncode==0,('offline import failed',filename,proc.returncode)
  raw=raw_rows((out/'raw.csv').read_text());source=source_tables((out/'source.csv').read_text())
  summary=summarize_exports(raw,source)
  (out/'PC-source-rows.json').write_text(json.dumps(source,indent=2)+'\n')
  (out/'raw-kernel-rows.json').write_text(json.dumps(raw,indent=2)+'\n')
  result['three_nodes']=summary
  require_input(report);assert sha(binary)==client_sha,'export client changed during imports'
  result['input_immutable']='PASS';result['status']='OFFLINE_EXPORT_COMPLETE_SOURCE_COUNTER_ANALYSIS_PENDING'
  result['no_GPU_context_or_kernel_requested_by_script']=True
  result['artifact_hashes']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='result.json'}
  save();print(json.dumps({'status':result['status'],'three_nodes':summary},indent=2))
 except BaseException:
  result['status']='INVALID_OR_INCOMPLETE_OFFLINE_EXPORT_NO_COUNTER_INTERPRETATION';result['failure']=traceback.format_exc();save();raise

if __name__=='__main__':main()
