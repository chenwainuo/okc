"""Construct strict host-only PW39 selection patch. Never imports production."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'pw38'
PARENT_MANIFEST='48b697b288e71617bdf5e688325c17e90d6541d140153acfb73a464e01dbc89b'
PARENT_SOURCE='29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e'
def sha(data):return hashlib.sha256(data).hexdigest()
def apply(source,delta,reverse=False):
    for edit in reversed(delta['edits']) if reverse else delta['edits']:
        before,after=(edit['after'],edit['before']) if reverse else (edit['before'],edit['after'])
        if source.count(before)!=1:raise ValueError('nonunique/missing strict anchor '+edit['id'])
        source=source.replace(before,after,1)
    return source

def build():
    assert sha((PARENT/'manifest.json').read_bytes())==PARENT_MANIFEST
    original=(PARENT/'submission.py').read_text();assert sha(original.encode())==PARENT_SOURCE
    helper=(HERE/'selection_helper.py.txt').read_text();edits=[dict(id='bounded-confirmation-helper',before='def _rqbake(st, thr):',after=helper+'def _rqbake(st, thr):')]
    before='''            for us, nm, p in sorted(cands, key=lambda c: c[0]):
                if us < best[2] * MARGIN:
                    best = ("rq:" + nm, p, us)
                break
            if best[1] is not None:
'''
    after=before.replace('            if best[1] is not None:\n','''            best = _rqconfirm(st, layer, m, cands, lambda xx, oo: _r6(xx, layer, n, st),
                              x, out, flush, sink, best, ("r6", None, min(r6a[m], r6b)))
            if best[1] is not None:
''')
    edits.append(dict(id='RQ128-confirm-after-existing-screen',before=before,after=after))
    before='''            for us, tag, p in sorted(cands, key=lambda c: c[0]):
                if us < best[2] * MARGIN:
                    best = ("rq:" + tag, p, us)
                break
            if best[1] is not None:
'''
    after=before.replace('            if best[1] is not None:\n','''            best = _rqconfirm(st, layer, m, cands, inc, x, out, flush, sink,
                              best, (iname, None, min(ia, ib)))
            if best[1] is not None:
''')
    edits.append(dict(id='RQ32-confirm-after-existing-screen',before=before,after=after))
    delta=dict(version='pw39_v1_bounded_selection_confirmation',parent_manifest_sha256=PARENT_MANIFEST,parent_source_sha256=PARENT_SOURCE,edits=edits,
      policy='Host-only top-two confirmation,7x5replay blocks (5blocks when onlyone),2s shared elapsed allowance,original210s cutoff; incomplete retains oldscreen decision with reason,errors retain incumbent; same0.99margin,allshape generic,all CUDA unchanged')
    candidate=apply(original,delta);assert apply(candidate,delta,True)==original
    (HERE/'submission.py').write_text(candidate);(HERE/'delta.json').write_text(json.dumps(delta,indent=2)+'\n')
if __name__=='__main__':build()
