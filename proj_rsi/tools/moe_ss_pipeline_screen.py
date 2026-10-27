"""Thin stdlib-only queue entrypoint for the isolated MoE SS screen."""
import json,runpy,sys,time,traceback
from pathlib import Path
if __name__=='__main__':
 started=time.time();output=None
 if '--output' in sys.argv and sys.argv.index('--output')+1<len(sys.argv):output=Path(sys.argv[sys.argv.index('--output')+1])
 existed_at_entry=output is not None and output.exists()
 try:runpy.run_path(str(Path(__file__).resolve().parents[2]/'tools/moe_ss_pipeline_screen.py'),run_name='__main__')
 except BaseException:
  failure=traceback.format_exc()
  # Never write into an existing attempt, including a directory another process
  # creates concurrently. A run-created result already records its own failure.
  if output is not None and not existed_at_entry and not output.exists():
   try:output.mkdir(parents=True,exist_ok=False)
   except FileExistsError:pass
   else:
    with (output/'launcher-error.json').open('x') as f:f.write(json.dumps({'state':'INVALID_NO_ADMISSION','launcher_started_at':started,'failed_at':time.time(),'error':failure},indent=2)+'\n')
  raise
