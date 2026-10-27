# Rolled-down stage candidate, retaining measured v1 LUT up

The bounded candidate is `ss_decode_lut_v2/down_only/`. Its only new device
function is T8DIV1. The decoder helper/table and matrix/protocol remain inherited;
the change fuses lookup into shared scatter and rolls the four producer steps.
Offline CUDA12.8 uses48registers with no spills/stack. Actual CUDA13 resources and
loop-overhead effects remain unmeasured.

`pilot.py` derives from the reviewed and successfully run v1 stage screen. It
reuses the exact original/v1LUT SOs and immutable original tree, prepares weights
once, and compiles/gates only the new down function. Actual serving captures use
v1LUT up and down. A separate private-globals class uses v1LUT up plus rolled down;
current/fresh/empty outputs and its own post-timing terminal must match v1 and the
existing task reference. Original classes/module globals remain untouched.

Down stage graphs include ordinary wrapper zeroing and the original allocation
grid. The three-block F1 bracket uses **one frozen seed7904 record**, copied into
each private graph buffer; x/routes/experts/post byteSHA256s and the complete route
histogram document exact workload identity. Each block has3samples and its own
terminal. Eight balanced seed7900..7907 pairs follow. Every raw event time is
saved before its output/current-input check, and every paired seed saves its
actual routing histogram. Signed losses and runtime/numerical failures remain
recorded. A loss rejects this measured configuration.

Optional profiling is disabled in the descriptor and only runs after direct
measurements, their terminals, mixed own output and source/bank checks are saved.
Trace/export failure is optional; numeric/runtime failure of attempted replay
propagates outside the optional handler. No counter privileges are requested.

CPU readiness has run: import/help without Torch, frozen file pins, exact v1 and
candidate module-factory sources using mocked builders, private-globals mechanism,
and inherited onefunction SASS positive/missingwait/missingbarrier checks. It does
not rerun unchanged exhaustive scalar or matrix proofs. CUDA is uninitialized.

Root owns review binding, transfer, admission and GPU execution. Proposed bounds
are130s inner/180s outer, with one new JIT and no new full archive/tree. The focused
rq review is requested before a root-ready descriptor is bound. No GPU/queue/API
work was performed by this author.
