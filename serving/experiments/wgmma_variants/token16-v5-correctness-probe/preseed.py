"""Optional ROOT CPU preparation BEFORE GPU lease. Never mutate source cache.
Only baseline SGLang subtree from reviewed standalone coalesced-v3 provenance.
"""
import hashlib,json,shutil
from pathlib import Path
root=Path('/root/okc');pins=json.loads(Path(__file__).with_name('source-pins.json').read_text())
report=root/'reports/2026-10-25-wgmma-coalesced-correctness.json'
assert hashlib.sha256(report.read_bytes()).hexdigest()==pins['preseed_prior_report_sha256']
manifest=root/'results/rsi/bundles/baseline-v2.tar.gz.manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==pins['baseline_manifest_sha256']
files=json.loads(manifest.read_text())['files']
prior=next(r for r in json.loads(report.read_text())['routed_and_full_results'] if 'sglang_imports' in r)
for info in prior['sglang_imports'].values():
    rel=str(Path(info['path']).relative_to('/baseline'))
    assert files[rel]==info['sha256'],'prior cache source provenance differs'
assert all(str(p).startswith('/cache/') for p in prior['jit_libraries'])
src=root/'.gpu-cache/wgmma-coalesced-v3/sglang';dst=root/'.gpu-cache/wgmma-token16-v5/sglang'
assert src.is_dir() and not dst.exists(),'source absent or destination already populated'
# Drop standalone candidate-specific modules; all reused leaves still undergo
# normal SGLang transitive dependency/runtime verification. No forced acceptance.
shutil.copytree(src,dst,symlinks=True,ignore=lambda path,names:[n for n in names if n.startswith('sgl_kernel_jit_rsi_wgmma')])
print(json.dumps({'status':'preseeded_baseline_only','source':str(src),'target':str(dst),'GPU_executed':False}))
