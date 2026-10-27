"""Exact stock-walker requested word loads, distinct from allocation or HBM traffic."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loads(lengths):
 # Initial 32-lane load, then another full window after every32 consumed
 # entries, including a sentinel-only window when length is a multiple32.
 return [32*(n//32+1) for n in lengths]
def run():
 source=ROOT/'proj_rsi/candidates/rs32/submission.py';text=source.read_text()
 assert 'uint32_t ec = warp < NW ? __ldg(el + lane) : 0u;' in text
 assert 'if ((ei & 31) == 0) ec = __ldg(el + ei + lane);' in text
 src=HERE/'actual-cpu-reference.json';d=json.loads(src.read_text());out={'input_sha256':sha(src),'walker_source_sha256':sha(source),'accounting_source_sha256':sha(Path(__file__)),'meaning':'Source-requested bytes from the unchanged stock32-entry lane-prefetch walker, not observed HBM/L2 traffic or measured timing. Conditional on using that walker for RQ11. Allocation includes global max-CAP slack and guards; requested loads stop after each list sentinel window.','layers':{}}
 for layer,r in d['layers'].items():
  rows=[]
  for old,new in zip(r['formats']['12']['geometry'],r['formats']['11']['geometry']):
   a=sum(loads(old['group_lengths']))*4;b=sum(loads(new['group_lengths']))*4
   assert a<=old['escape_allocation_bytes']-128 and b<=new['escape_allocation_bytes']-128
   ta=old['payload_bytes']+a;tb=new['payload_bytes']+b
   rows.append({'KS':old['KS'],'RQ12_requested_escape_bytes':a,'RQ11_requested_escape_bytes':b,'RQ12_payload_plus_requested_escape_bytes':ta,'RQ11_payload_plus_requested_escape_bytes':tb,'requested_bytes_saved':ta-tb,'requested_bytes_fraction_saved':(ta-tb)/ta,'RQ12_requested_window_padding_bytes':a-old['actual_escape_bytes'],'RQ11_requested_window_padding_bytes':b-new['actual_escape_bytes']})
  out['layers'][layer]=rows
 return out
if __name__=='__main__':
 result=run();(HERE/'window-costs.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
