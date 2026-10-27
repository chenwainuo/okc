"""Read-only exact kernel-body extraction; no author import/build/compile."""
import ast,re,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
def cu(text):
 return next(ast.literal_eval(n.value)for n in ast.parse(text).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
def kernel(text,name):
 m=re.search(r'\b'+re.escape(name)+r'\(G p\)\s*\{',text);assert m,name
 start=text.rfind('template <int FMT',0,m.start());assert start>=0
 # Ignore comments/strings while balancing C++ body braces; preserve exact bytes.
 masked=list(text)
 for skip in re.finditer(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',text,re.S):
  masked[skip.start():skip.end()]=' '*(skip.end()-skip.start())
 depth=0;body=text.index('{',m.start())
 for i in range(body,len(masked)):
  depth+=(masked[i]=='{')-(masked[i]=='}')
  if depth==0:return text[start:i+1]
 raise AssertionError('unclosed kernel')
def check():
 source=ROOT/'proj_rsi/candidates/gv53_selected_earlyload_v1/submission.py';pilot=source.with_name('pilot.cu');a=cu(source.read_text());b=pilot.read_text();out={}
 for name,path in [('gv_predxor','proj_rsi/candidates/gv53_keyed_half_v1/submission.py'),('gv_keyed','proj_rsi/candidates/gv53_keyed_half_v1/submission.py'),('gv_earlyload','proj_rsi/candidates/gv53_earlyload_v1/submission.py')]:
  authored=kernel(a,name);assert authored==kernel(b,name)==kernel(cu((ROOT/path).read_text()),name)
  out[name]={'exact_production_pilot_and_original_body':True,'body_sha256':hashlib.sha256(authored.encode()).hexdigest(),'original_source':path}
 return {'state':'EXACT_NATIVE_BODY_LINEAGE_CPU_ONLY','production_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pilot_sha256':hashlib.sha256(pilot.read_bytes()).hexdigest(),'bodies':out,'combined_keyed_earlyload_body':False,'selection':'Fixedrolesonly; actual production selection not inferred.'}
if __name__=='__main__':print(json.dumps(check(),indent=2))
