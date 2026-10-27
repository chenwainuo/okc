"""Independent CPU-only per-warp async-X dependency expansion, no CUDA imports."""
from collections import defaultdict, deque

def graph():
    dep=defaultdict(set)
    def add(n,*ps): dep[n].update(ps)
    for s in range(64):
        add(('X',s), *([] if s<2 else [('I',s-2)]))
        for w in range(8):
            add(('C',s,w),('X',s))
            if s: add(('C',s,w),('C',s-1,w))
            if s>=4:add(('C',s,w),('M',s-3,w//4))
            add(('A',s,w),('C',s,w)) # fullwarp ordered release-arrival
        add(('E',s),*(('A',s,w) for w in range(8)))
        add(('I',s), *([] if s<2 else [('B',s-2)]))
        if s<62:add(('I',s),('E',s))
        add(('W',s), *([] if s<6 else [('B',s-6)]))
        add(('B',s),('I',s),('W',s))
        if s>=8:add(('B',s),('M',s-7,0),('M',s-7,1))
        for g in range(2):
            add(('M',s,g),('B',s),*(('C',s,w) for w in range(4*g,4*g+4)))
            if s:add(('M',s,g),('M',s-1,g))
            if s<2:add(('M',s,g),*(('C',3,w) for w in range(4*g,4*g+4)))
            elif s+2<64:add(('M',s,g),*(('C',s+2,w) for w in range(4*g,4*g+4)))
    add(('drain',),('M',63,0),('M',63,1));add(('out',),('drain',))
    return dep

def check(dep):
    nodes=set(dep).union(*(ps for ps in dep.values()));ind={n:len(dep[n]) for n in nodes};out=defaultdict(set)
    for n,ps in dep.items():
        for p in ps:out[p].add(n)
    todo=deque(n for n in nodes if ind[n]==0);seen=[]
    while todo:
        n=todo.popleft();seen.append(n)
        for q in out[n]:
            ind[q]-=1
            if not ind[q]:todo.append(q)
    assert len(seen)==len(nodes),'DEPENDENCY_CYCLE'
    for s in range(64):assert dep['E',s]=={('A',s,w) for w in range(8)},'MISSING_WARP_ARRIVAL'
    return len(nodes),sum(map(len,dep.values()))

if __name__=='__main__':
    d=graph();print('PER_WARP_DAG_PASS',check(d),'Xissues64,refills62,arrivals512')
    for bad in ('startup_cycle','missing_warp'):
        d=graph()
        if bad=='startup_cycle':d['C',0,0].add(('B',0))
        else:d['E',0].remove(('A',0,7))
        try:check(d)
        except AssertionError as e:print('NEGATIVE_REJECT',bad,str(e))
        else:raise AssertionError(bad)
