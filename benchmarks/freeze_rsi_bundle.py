#!/usr/bin/env python3
"""Freeze source, benchmark and fixture inputs for isolated future RSI runs.

Archives are execution inputs, not proof that a runner used them. Reviewer must
verify read-only extraction/import paths and pinned container before GPU lease.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = 'serving/sglang/python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pipeline-stages', type=int, choices=[2,3,4], default=4)
    args = p.parse_args()
    checkout = ROOT / 'serving/sglang'
    tracked = subprocess.check_output(['git','-C',str(checkout),'ls-files','-z','python/sglang'])
    paths = [Path('serving/sglang')/x.decode() for x in tracked.split(b'\0') if x]
    paths += [x.relative_to(ROOT) for x in (ROOT/'benchmarks').glob('*.py')]
    paths += [x.relative_to(ROOT) for x in (ROOT/'muse_moe_w4a16').glob('*.py') if not x.name.startswith('candidate')]
    paths += [Path('fixture-hashes.json'),Path('serving/benchmark-contract.json')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    symlinks = {}
    with args.output.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz, tarfile.open(fileobj=gz,mode='w|') as archive:
        for relative in sorted(set(paths)):
            path = ROOT/relative
            if path.is_symlink():
                target = os.readlink(path)
                symlinks[str(relative)] = target
                files[str(relative)] = hashlib.sha256(('symlink:' + target).encode()).hexdigest()
                info = tarfile.TarInfo(str(relative)); info.type=tarfile.SYMTYPE;info.linkname=target;info.mode=0o777
                archive.addfile(info)
                continue
            data = path.read_bytes()
            if str(relative) == PATCH_FILE and args.pipeline_stages != 4:
                anchor = b'namespace device::marlin_moe {'
                if data.count(anchor)!=1 or b'pipe_stages' not in data:
                    raise ValueError('Pinned MoE pipeline anchors changed')
                data = data.replace(b'pipe_stages', b'okc_moe_pipe_stages')
                data = data.replace(anchor, anchor + f'\nstatic constexpr int okc_moe_pipe_stages = {args.pipeline_stages};'.encode())
            files[str(relative)] = hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(str(relative)); info.size=len(data);info.mode=0o644
            archive.addfile(info,io.BytesIO(data))
    manifest=dict(schema_version=1,archive_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  pipeline_stages=args.pipeline_stages,files=files,symlinks=symlinks,
                  execution_requirements=['Extract into dedicated immutable tree','Run baseline and candidate in separate processes with imports rooted in their own tree','Use pinned container digest','Verify extracted files against hashes before and after run'])
    args.output.with_suffix(args.output.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','symlinks')}))


if __name__=='__main__':
    main()
