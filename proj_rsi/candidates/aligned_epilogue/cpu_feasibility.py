"""CPU-only feasibility: exact BF16 power2 encoding, aligned-window metadata and FP32 domain limits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.parent / "pw36" / "submission.py"
PARENT_SHA = "2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf"


def fp32(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def rebase(bits, shift):
    bits = np.asarray(bits, dtype=np.uint16)
    e = ((bits >> 7) & 255).astype(np.int32)
    zero = (bits & 0x7fff) == 0
    ok = zero | ((e > 0) & (e < 255) & (e + shift > 0) & (e + shift < 255))
    changed = (bits & 0x807f) | (((e + shift).astype(np.uint16) & 255) << 7)
    return np.where(zero, bits, changed).astype(np.uint16), ok


def exhaustive():
    bits = np.arange(65536, dtype=np.uint16)
    cases, accepted, onehot, overflow = 0, 0, 0, 0
    shifts = {}
    for B in range(121):
        aligned = ((B + 4) // 8) * 8
        shift = 2 * (aligned - B)
        assert (aligned * 0x01010101) & 0x87878787 == 0  # Thus add==OR for every possible packed C.
        shifted, ok = rebase(bits, shift)
        back, backok = rebase(shifted, -shift)
        assert np.all(backok[ok]) and np.array_equal(back[ok], bits[ok])
        assert len(np.unique(shifted[ok])) == int(ok.sum())
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            numeric = np.ldexp(fp32(bits[ok]), shift)
            assert np.array_equal(numeric.view(np.uint32), fp32(shifted[ok]).view(np.uint32))
            restored = np.ldexp(fp32(shifted[ok]), -shift)
            assert np.array_equal(restored.view(np.uint32), fp32(bits[ok]).view(np.uint32))
            for power in (3, 4, 5, 6, 7):
                want = np.ldexp(fp32(bits[ok]), power)
                intermediate = np.ldexp(fp32(shifted[ok]), power)
                safe = np.isfinite(want) & np.isfinite(intermediate)
                got = np.ldexp(intermediate[safe], -shift)
                assert np.array_equal(got.view(np.uint32), want[safe].view(np.uint32))
                onehot += int(safe.sum())
                overflow += int((np.isfinite(want) & ~np.isfinite(intermediate)).sum())
        cases += len(bits)
        accepted += int(ok.sum())
        shifts[str(B)] = {"aligned_B": aligned, "exponent_shift": shift,
                          "accepted_bitpatterns": int(ok.sum()), "fallback_bitpatterns": int((~ok).sum())}
    # Weight bit bijection alone does NOT ensure finite scaled FP32 products.
    huge = np.array([246 << 7], dtype=np.uint16)
    scaled, ok = rebase(huge, 4)
    with np.errstate(over="ignore"):
        want = np.ldexp(fp32(huge), 7)
        scaled_product = np.ldexp(fp32(scaled), 7)
    assert bool(ok[0]) and bool(np.isfinite(want[0])) and bool(np.isinf(scaled_product[0]))
    return {"B_bitpattern_cases":cases,"accepted_reversible_cases":accepted,
            "safe_onehot_power_cases_exact":onehot,"unsafe_scaled_product_cases_detected":overflow,
            "fallback":"zero preserved; reject nonzero subnormal/nonfinite or exponent shifted outside1..254",
            "overflow_counterexample":{"original_BF16_bits":int(huge[0]),"shift":4,"input_power":7,
                                      "original_product_finite":True,"scaled_product_overflows":True},
            "B_policy":shifts}


def main():
    torch.set_num_threads(4)
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    result={"state":"CPU_FEASIBILITY_GPU_UNRUN","parent_sha256":PARENT_SHA,"exhaustive":exhaustive(),"actual_packs":{}}
    spec=importlib.util.spec_from_file_location("aligned_frozen_pw36",PARENT)
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    sys.path.insert(0,str(ROOT/"muse_bf16_proj"))
    import reference
    weights=reference.generate_weights(5151,device="cpu")
    for layer,geometry in (("pooler",(2,8,16,1,1,0)),("unpooler",(1,4,16,1,1,2))):
        w=weights[layer]
        original=parent._c12(parent._pack(w),1)
        assert original is not None
        data,tab,esc=original
        B=tab&255;aligned=((B+4)//8)*8;shift=2*(aligned-B)
        bits=w.view(torch.int16).numpy().view(np.uint16)
        changed,ok=rebase(bits,shift)
        assert bool(ok.all())
        shifted=torch.from_numpy(changed.view(np.int16).copy()).view(torch.bfloat16)
        shifted_pack=parent._c12(parent._pack(shifted),1)
        assert shifted_pack is not None
        sdata,stab,sesc=shifted_pack
        assert stab&255==aligned
        assert torch.equal(sdata,data) and torch.equal(sesc[:,:5],esc[:,:5])
        packed_esc=torch.cat((esc,torch.zeros((len(esc),1),dtype=esc.dtype)),1)
        shifted_esc=torch.cat((sesc,torch.zeros((len(sesc),1),dtype=sesc.dtype)),1)
        a=parent._gvesc(packed_esc,w.shape[0]//16,geometry)
        b=parent._gvesc(shifted_esc,w.shape[0]//16,geometry)
        assert a is not None and b is not None and a.shape==b.shape
        assert torch.equal(a.long()&~127,b.long()&~127)
        # Payload bytes invariant, XOR metadata recomputed; low8 bits invariant because shift is even.
        restored=torch.ldexp(shifted.float(),torch.tensor(-shift)).to(torch.bfloat16)
        assert torch.equal(restored.view(torch.int16),w.view(torch.int16))
        x=torch.randn((16,w.shape[1]),generator=torch.Generator().manual_seed(7171)).to(torch.bfloat16).float()
        y=x@w.float().T
        ys=(x@shifted.float().T)*(2.0**(-shift))
        assert torch.equal(y,ys), "CPU FP32 normal-domain homogeneity failed"
        for power in (3,4,5,6,7):
            assert torch.equal(w.float()*(2.0**power),shifted.float()*(2.0**(power-shift)))
        row={"shape":list(w.shape),"GV_geometry_WR_WK_KU_S_RT_XS":geometry,
             "original_B":B,"aligned_B":aligned,"shift":shift,"restore_scale":2.0**(-shift),
             "all_weight_bits_bijective":True,"c12_payload_identical":True,"escape_positions_identical":True,
             "escape_count":len(esc),"changed_XOR_records":int((esc[:,-1]!=sesc[:,-1]).sum()),
             "per_warp_escape_list_nonXOR_bits_identical":True,"CPU_FP32_matmul_homogeneous_exact":True,
             "allK_probe_powers8to128_exact":True,
             "scope":"CPU reference seed5151; GPU MMA rounding/subnormal semantics and judge coverage NOT established"}
        result["actual_packs"][layer]=row
        print(json.dumps(row),flush=True)
        (HERE/"cpu-feasibility.json").write_text(json.dumps(result,indent=2)+"\n")
    (HERE/"cpu-feasibility.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
