"""Host-only PW43 patch. Running this never imports the submission or torch."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
PARENT=HERE.parents[2]/'results/proj-rsi/releases/pw38-v1-29e557ea53f3/submission.py'
# HERE parents[2] is proj_rsi parent (repository).
PARENT=HERE.parents[2]/'results/proj-rsi/releases/pw38-v1-29e557ea53f3/submission.py'
SHA='29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e'
def build():
    old=PARENT.read_text();assert hashlib.sha256(old.encode()).hexdigest()==SHA
    new=old.replace('def _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0):','def _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0, heldout=None):',1)
    call='        st["lt"][(layer, m)] = _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0)'
    replacement='''        heldout = None
        if m == 256 and not f32:
            def heldout(fixture):
                # Private generator: confirmation cannot perturb subsequent prepare RNG.
                gen = torch.Generator(device=dev).manual_seed(4300 + fixture)
                xx = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
                yy = torch.empty((m, n), dtype=dt, device=dev)
                rr = torch.mm(xx.float(), W[:n].float().T)
                def valid(code):
                    return ext.lt_mm(xx, W, Wt, yy, n, code) == 0 and _close(yy, rr)
                def timed(code):
                    fn = (lambda: torch.mm(xx, W[:n].T)) if code < 0 else (lambda: ext.lt_mm(xx, W, Wt, yy, n, code))
                    return _tm(fn, flush, 3)
                return valid, timed
        st["lt"][(layer, m)] = _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0, heldout)'''
    assert new.count(call)==1;new=new.replace(call,replacement)
    marker='    njob = wsum / LTW[layer]\n'
    extra='''    target = heldout is not None and m == 256 and not f32
    observations = []
    # Translate existing absolute wall deadline once; never restart either budget.
    remaining = min(dl, _T0 + PREPARE_BUDGET_S) - time.time() if target else 0.0
    end = time.monotonic() + max(0.0, remaining)
    explore_end = time.monotonic() + max(0.0, remaining) * 0.55
    def explore_live():
        return not target or time.monotonic() < explore_end
'''
    assert new.count(marker)==1;new=new.replace(marker,marker+extra)
    new=new.replace('        if code < 0 or code in tried:\n','        if code < 0 or code in tried or not explore_live():\n',1)
    new=new.replace('        if ok(code):\n            res.append((_tm(lambda: run(code), flush, 5), code))', '''        valid = ok(code)
        observation = {"code": code, "valid": bool(valid), "time_ms": None}
        if target:
            observations.append(observation)
        if valid and explore_live():
            value = _tm(lambda: run(code), flush, 5)
            finite = __import__("math").isfinite(value) and value > 0
            observation["time_ms"] = value if finite else None
            observation["within_exploration_deadline"] = explore_live()
            if (not target or finite) and explore_live():
                res.append((value, code))''',1)
    new=new.replace('    def addc(lay, cfg):\n        c = ext.lt_add','    def addc(lay, cfg):\n        if not explore_live():\n            return\n        c = ext.lt_add',1)
    new=new.replace('    t_torch = _tm(torch_fn, flush, 9)','    t_torch = _tm(torch_fn, flush, 9) if explore_live() else float("inf")',1)
    marker='    res.sort()\n    best = (t_torch, -1)\n'
    extra='''    res.sort()
    if target:
        codes = [code for _, code in res[:2]]
        chosen, record = _lt43confirm(codes, heldout, end, time.monotonic)
        record.update(layer=layer, m=m, tried=len(tried), exploration_ms=res, exploration_observations=observations, selected=chosen,
                      configs={str(c): [c // 100000, list(ext.lt_cfg(m, n, K, f32, c))] for c in codes})
        print("PW43_LT " + __import__("json").dumps(record, allow_nan=False), flush=True)
        _LTDONE.append((layer, m))
        return chosen
    best = (t_torch, -1)
'''
    assert new.count(marker)==1;new=new.replace(marker,extra)
    helper=(HERE/'helper.py.txt').read_text()
    new=new.replace('\ndef _ltsearch(', '\n'+helper+'\n\ndef _ltsearch(',1)
    return old,new
if __name__=='__main__':
    old,new=build();(HERE/'submission.py').write_text(new)
