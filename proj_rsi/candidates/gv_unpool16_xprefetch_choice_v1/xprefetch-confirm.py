def _gvxprefetch_confirm(st, flush):
    # All original bakes finish before this optional, same-geometry comparison.
    target = (1,2,2,1,4,16,1,1,2,1)
    deadline = _T0 + PREPARE_BUDGET_S
    for m in (16,):
        layer, n, key = "unpooler", 8192, ("unpooler",_bk(m))
        record = {"m":m,"choice":"parent","status":"UNSELECTED_PARENT_RETAINED","pairs_us":[],"samples_us":[]}
        st.setdefault("gvxprefetch_choices", {})[(layer,m)] = record
        def receipt():
            print("GV_XPREFETCH_CHOICE " + repr(record),flush=True)
        if time.time() >= deadline:
            record["status"] = "BUDGET_SKIP_PARENT_RETAINED"; receipt(); continue
        rp = st.get("rq",{}).get((layer,m))
        if rp is not None and rp["n"] == n:
            record["status"] = "RQ_PRECEDENCE_PARENT_RETAINED"; receipt(); continue
        owner = st["sel"]; p = owner.get(key)
        if p is None or p[4] != n:
            owner = st["gv"]; p = owner.get(key)
        if p is None or p[4] != n or len(p) != 7 or not (0 <= p[0] < len(GVI)) or tuple(GVI[p[0]]) != target:
            receipt(); continue
        w = st[layer+"_raw"][:n]
        if tuple(w.shape) != (8192,4096):
            record["status"] = "SHAPE_SKIP_PARENT_RETAINED"; receipt(); continue
        record.update({"template":target,"id":p[0],"owner":"sel" if owner is st["sel"] else "gv"})
        plans = (p,tuple(p)+("gv_xprefetch_choice",))
        gen = torch.Generator(device=w.device).manual_seed(89101+m)
        x = torch.randn((m,w.shape[1]),generator=gen,device=w.device).to(torch.bfloat16)
        xb = x.clone()
        outputs = [torch.empty((m,n),dtype=torch.bfloat16,device=w.device) for _ in range(2)]
        ref = x.double() @ w.double().T
        def invoke(index):
            assert _gvcall(st,plans[index],x,outputs[index]) == 0
        def check(index):
            got = outputs[index].double()
            assert torch.equal(x.view(torch.uint8),xb.view(torch.uint8)) and bool(torch.isfinite(got).all())
            assert float((got-ref).norm()/ref.norm().clamp(min=1e-30)) <= 4e-3
            assert float(((got-ref).norm(dim=-1)/ref.norm(dim=-1).clamp(min=1e-30)).max()) <= 2e-2
        complete = True
        for block in range(4):
            sample = {}
            for index in ((0,1) if block % 2 == 0 else (1,0)):
                if time.time() >= deadline:
                    record["status"] = "BUDGET_SKIP_PARENT_RETAINED"; complete = False; break
                sample[index] = _tm(lambda: invoke(index),flush,9)*1e3
                record["samples_us"].append({"block":block,"arm":"parent" if index==0 else "xprefetch","us":sample[index]})
                check(index)  # immediate own-terminal/current-X check after this arm's actual graph timing
                if not (0 < sample[index] < float("inf")):
                    record["status"] = "INVALID_TIMING_PARENT_RETAINED"; complete = False; break
            if not complete:
                break
            record["pairs_us"].append({"parent":sample[0],"xprefetch":sample[1],"order":"AB" if block%2==0 else "BA"})
        if complete and time.time() < deadline:
            ds = sorted(r["parent"]-r["xprefetch"] for r in record["pairs_us"])
            effect = (ds[1]+ds[2])/2
            record["paired_median_parent_minus_xprefetch_us"] = effect
            if effect > 0:
                owner[key] = plans[1]; record["choice"] = "xprefetch"
            record["status"] = "COMPLETE_ZERO_THRESHOLD"
        elif complete:
            record["status"] = "BUDGET_SKIP_PARENT_RETAINED"
        receipt()
