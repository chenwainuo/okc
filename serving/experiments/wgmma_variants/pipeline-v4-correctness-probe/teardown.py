"""Host-side bounded cleanup proof; never declares unknown container state terminal."""
import json,subprocess,sys
from pathlib import Path
name,output,original_code=sys.argv[1:]
records=[]
def call(args,timeout):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
        d={'argv':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except subprocess.TimeoutExpired as e:
        d={'argv':args,'timeout':True,'returncode':None,'stdout':'','stderr':''}
    records.append(d);return d
call(['docker','rm','-f',name],15)
r=call(['docker','inspect',name],10)
absent=r['returncode'] not in (None,0) and any(x in (r['stderr']+r['stdout']).lower() for x in ('no such container:','no such object:'))
report={'status':'terminal' if absent else 'termination_unknown','container':name,'container_absent':absent,'client_exit_code':int(original_code),'observations':records,'root_instruction':'Close lease only if terminal proof passes; termination_unknown leaves canonical lease active and blocks further GPU work pending root reconciliation.'}
Path(output,'host-terminal.json').write_text(json.dumps(report,indent=2)+'\n')
sys.exit(0 if absent else 42)
