"""Finite exact full-CUDA commutativity/inverse proof, CPU only."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
PINS={'reverse_tma_x':'025f4a2de43f7c14a4ee4779dcb6a929974311b0380b433710ed4022beef50fc',
      'reverse_tma_x4':'d47ed1663eac87f5bcc05b5f17df8fa62e7c9e91780a13d02ce9b95ac0f05c8a',
      'reverse_tma_x_stmatrix':'2c132b82a854e476f433335462b2122f21bc3cb490dec6f2b691c01cd9775045'}
DEPTH=[('+2*16384+2*8;','+4*16384+4*8;'),('xfull=xring+32768;','xfull=xring+65536;'),
       ('slot=s&1','slot=s&3'),('(s/2)&1','(s/4)&1'),('s+2<KC','s+4<KC'),
       ('(s+2)*64','(s+4)*64'),('for(int s=0;s<2;++s)','for(int s=0;s<4;++s)')]
def sha(data):return hashlib.sha256(data).hexdigest()
def normalize(source):
    for name in ('rq_tma_x4_stmx','rq_tma_x_stmx','rq_tma_x4','rq_tma_x'):
        source=source.replace(name,'COMPOSITE_KERNEL')
    return source
def store_block(source):
    a=source.index('      for(int k=0;k<4;++k)');b=source.index('      asm volatile("fence.proxy.async.shared::cta;',a)
    return source[a:b]
def proof():
    import json,re
    source={}
    for folder,pin in PINS.items():
        root=HERE.parent/folder;assert sha((root/'manifest.json').read_bytes())==pin
        m=json.loads((root/'manifest.json').read_text())
        for name,digest in m['files'].items():assert sha((root/name).read_bytes())==digest,(folder,name)
        source[folder]=normalize((root/'extension_source.cu').read_text())
    base=source['reverse_tma_x'];x4=source['reverse_tma_x4'];st=source['reverse_tma_x_stmatrix']
    scalar=store_block(base);matrix=store_block(st)
    def depth(text,reverse=False):
        for a,b in DEPTH:
            if reverse:a,b=b,a
            assert text.count(a)==1,(a,text.count(a))
            text=text.replace(a,b,1)
        return text
    def store(text,reverse=False):
        a,b=(matrix,scalar) if reverse else (scalar,matrix)
        assert text.count(a)==1
        return text.replace(a,b,1)
    assert depth(base)==x4 and store(base)==st
    order_x4_store=store(depth(base));order_store_x4=depth(store(base))
    actual=normalize((HERE/'extension_source.cu').read_text())
    assert order_x4_store==order_store_x4==actual
    assert store(actual,True)==x4 and depth(actual,True)==st
    assert depth(store(actual,True),True)==base==store(depth(actual,True),True)
    lo=base.index(scalar);hi=lo+len(scalar);spans=[]
    for old,new in DEPTH:
        start=base.index(old);stop=start+len(old)
        assert stop<=lo or hi<=start
        spans.append(dict(old=old,new=new,source_span=[start,stop]))
    # Identity normalization must affect exactly the kernel declaration and host
    # launch pointer in extension source; generated source still uses one name.
    raw=(HERE/'extension_source.cu').read_text()
    assert raw.count('rq_tma_x4_stmx')==2 and 'COMPOSITE_KERNEL' not in raw
    return dict(status='PASS_EXACT_FULL_CUDA_COMMUTATIVITY_AND_BOTH_PARENT_INVERSES',
                normalized_full_source_sha256=sha(actual.encode()),normalized_bytes=len(actual),
                edit_orders=['depth_then_store','store_then_depth'],depth_edit_spans=spans,
                store_edit_span=[lo,hi],disjoint_source_edits=True,
                inverse_targets=list(PINS),frozen_manifests=PINS,
                unchanged='entire codec/FP32/mathA4/hostlaunch/output/proxy/bar3/bar4/bar5 except finite Xdepth edits and store block',
                limits='source composition equivalence; speed additivity and GPU correctness are not assumed')
