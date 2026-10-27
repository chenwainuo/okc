"""CPU source/index/resource/physical admission only; no CUDA context."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import prototype as P
import liveness
from schedule_proof import proof
HERE=P.HERE

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def sourceproof():
    body=(HERE/'kernel_body.cuh').read_text()
    restored=body.replace('// Fixed unpooler256, two independent128-token tiles.','// One fixed unpooler128 policy.')
    restored=restored.replace('// Every call writes all2MiB','// Every call writes all1MiB')
    restored=restored.replace('  x+=blockIdx.y*262144u;packed+=blockIdx.y*262144u;\n','')
    restored=restored.replace('rq_px_g8_m256(', 'rq_px_g8(')
    restored=restored.replace('(uint64_t)p.XP+uint32_t(blockIdx.y)*1048576u+uint32_t(warp)','(uint64_t)p.XP+uint32_t(warp)')
    restored=restored.replace('[%0,{%1,%3}],[%2]','[%0,{%1,0}],[%2]')
    restored=restored.replace('"r"(cb),"r"(bring),"r"(128*blockIdx.y):"memory");','"r"(cb),"r"(bring):"memory");')
    assert restored==(P.PARENT/'kernel_body.cuh').read_text()
    assert 'p.cnt' not in body and 'atomic' not in body
    cu=P.source().replace(body,restored)
    for a,b in [('rq_px_g8_m256<','rq_px_g8<'),('if(p.m!=256 ||','if(p.m!=128 ||'),('xp.numel()!=524288','xp.numel()!=262144'),('rx_pack<<<dim3(256,2),256,','rx_pack<<<256,256,'),('dim3(p.NB * ((FL & 16) ? 2 : 1), 2)','dim3(p.NB * ((FL & 16) ? 2 : 1), KS)')]:
        assert cu.count(a)==1;cu=cu.replace(a,b)
    assert hashlib.sha256(cu.encode()).hexdigest()==P.PARENT_SOURCE
    assert 'rq_map(&to, out.data_ptr(), p.N, p.m, NT)' in P.source()
    return {'exact_restored_parent_source':P.PARENT_SOURCE,'NT':128,'M':256,'N':8192,'K':4096,'grid':[128,2],'pack_grid':[256,2],'descriptor_outer':256,'descriptor_box_outer':128,'unchanged_codec_producers_math_and_smem':True}

def maps():
    seen=bytearray(524288);dstseen=bytearray(524288);mapping=[None]*524288
    for tile in range(2):
        for thread in range(65536):
            v=thread&3;lane=(thread>>2)&31;chunk=(thread>>7)&3;warp=(thread>>9)&7;s=thread>>12
            row=64*(warp//4)+16*(warp%4)+(lane>>2)+8*(v&1)
            k=64*s+16*chunk+8*((v>>1)&1)+2*(lane&3)
            for u in range(4):
                dst=tile*262144+thread+65536*u;src=tile*262144+row*2048+k//2+512*u
                assert 0<=src<524288 and not seen[src] and not dstseen[dst]
                seen[src]=dstseen[dst]=1;mapping[dst]=src
    assert all(seen) and all(dstseen)
    patterns=set()
    for src in mapping:patterns.update(((2*src)&65535,(2*src+1)&65535))
    assert len(patterns)==65536
    for tile in range(2):
        for s in range(64):
            for warp in range(8):
                for lane in range(32):
                    for chunk in range(4):
                        addr=tile*1048576+16384*s+2048*warp+16*lane+512*chunk
                        assert addr%16==0 and 0<=addr<=2097136
                        for v in range(4):
                            row=tile*128+64*(warp//4)+16*(warp%4)+(lane>>2)+8*(v&1)
                            k=64*s+16*chunk+8*((v>>1)&1)+2*(lane&3)
                            assert mapping[addr//4+v]==row*2048+k//2
    # Shared epilogue bijection inherited; enumerate all global TMA cells.
    cells=bytearray(256*8192)
    for tile in range(2):
        for cb in range(0,8192,64):
            for row in range(128):
                for col in range(64):
                    index=(128*tile+row)*8192+cb+col
                    assert not cells[index];cells[index]=1
    assert all(cells)
    # W source/CSR index does not depend on y; validate both copies unchanged.
    sequences=[]
    for tile in range(2):
        seq=[((bx//2*64+s)*12288+(bx%2)*6144) for bx in range(128) for s in range(64)]
        assert min(seq)==0 and max(seq)+6144==64*64*12288
        sequences.append(seq)
    assert sequences[0]==sequences[1]
    return {'status':'PASS_EXHAUSTIVE_2TILE_U32_AND_OUTPUT_BIJECTIONS','XP_bytes':2097152,'BF16_bits_preserved':1048576,'all65536_patterns':True,'output_cells':len(cells),'input_XP_max_byte':2097151,'output_max_byte':4194303,'per_tile_W_sequences_identical':True,'u32_offset_max_below_2pow32':True,'consumer_vectors':2*64*8*32*4,'counters':'unchanged no source access except inherited fields; no y-shared atomics','alias':'each y owns disjoint XP1MiB and output rows128; W immutable shared'}

def main():
    result={'state':'START_CPU_ONLY_GPU_UNRUN','source':sourceproof(),'maps':maps(),'schedule_per_independent_tile':proof()}
    try:
        old=load('m256_retained_maps',HERE.parent/'operand_reverse_packed_x/preflight.py')
        result['W_bit_map']=old.wproof();result['shared_epilogue_bijection']=old.outputproof()
        parent=load('m256_parent_preflight',P.PARENT/'preflight.py')
        source=P.source();(HERE/'extension_source.cu').write_text(source);(HERE/'device_source.cu').write_text(P.device_source())
        cmd=['/usr/local/cuda-12.8/bin/nvcc',*P.S.RQFLAGS,'-Xptxas=-v','-cubin',str(HERE/'device_source.cu'),'-o',str(HERE/'device.sm90a.cubin')]
        run=subprocess.run(cmd,text=True,capture_output=True,timeout=90);log=run.stdout+run.stderr;(HERE/'resource.log').write_text(log)
        assert run.returncode==0,log
        assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log),log
        assert not re.search(r'C75\d\d|wgmma.*serial',log,re.I),log
        regs=[int(n) for n in re.findall(r'Used (\d+) registers',log)]
        assert len(regs)==2 and 0<regs[0]<=128 and 0<regs[1]<=64,regs
        pool=((regs[0]+7)//8)*8*512;assert pool<=65536
        sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(HERE/'device.sm90a.cubin')],text=True,timeout=20)
        (HERE/'sass.txt').write_text(sass);assert not re.search(r'\b(?:LDL|STL|USETMAXREG)\b',sass)
        result['physical']=liveness.check(sass);result['negative_controls']=parent.physical_negative_controls(sass)
        storecheck=load('m256_cpu_storecheck',HERE.parent/'operand_reverse_dual_stmatrix/liveness.py')
        result['store_form']=storecheck.store_form(sass,'rq_px_g8_m256',False,dual=True)
        result.update(state='CPU_PASS_GPU_UNRUN_ROOT_CRITIC_GO_REQUIRED',registers=regs,SMEM_bytes=103600,threads=512,rounded_register_pool=pool,source_sha256=hashlib.sha256(source.encode()).hexdigest(),device_source_sha256=hashlib.sha256(P.device_source().encode()).hexdigest(),gpu_calls=0)
    except Exception as error:
        result.update(state='FAILED_CPU_RETAIN_DIAGNOSE',error=repr(error));raise
    finally:
        (HERE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k in ('state','registers','rounded_register_pool','source_sha256','error','gpu_calls')},indent=2))
if __name__=='__main__':main()
