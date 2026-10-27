# Verified-prefix scheduler replay proposal — CPU tested, NOT ACTIVATED

Live `scheduling/store.py` and `test_store.py` remain byte-exact commitfbc90c7.
Candidate is isolated as `scheduling/_store_cached_proposal.py`; all candidate
cases import that private sibling in `test_store_cached_proposal.py`. No actual
QueueStore was instantiated, initialized, recovered, snapshotted or transitioned
by this author. No actual WAL/cache/checkpoint/lock/queue mutation, GPU or API
operation occurred. Root alone may activate after independent critic review.

## Read-only problem measurement

`readonly_profile.py` used open rb and a fixed initial byte count, without taking
queue locks or using QueueStore. The actual WAL prefix was589,393,887B,
402completeevents/36jobs; latest event2,959,927B. SHA256 of that exact prefix:
7fe9b9a1ed8b1ccb84d92d99afa05953b4307813fca45763f2a6080b50e1c1ab.
Byte hashing took0.348s; independent original canonical-payload checksum,
sequence and full-snapshot envelope replay took4.140s. A second pure-read model
rehashes those same589MB and validates only its attested last event in0.380s.
It is a read-cost model, not a deployed QueueStore/CLI benchmark. Active later
appends are outside the fixed prefix; no tail was repaired or truncated here.

## Mechanism and authority

Every read still hashes EVERY byte of the previously verified prefix. The small
optional replay-cache.json contains version,byteoffset,prefixSHA,sequence,last
recordoffset andlastrecordSHA. It carries an HMAC validation attestation under
an optional local32byte0600key. The key/cache attest only that this code already
strictly validated those prefix bytes; they contain no authoritative snapshot.
The actual WAL last event is reread, its whole line hash and original payload
checksum/sequence/snapshot envelope verified, and state reconstructed from THAT
actual event. Then every newly appended event receives original full checks.
Checkpoint is never trusted, even if it is wellformed or has a higher sequence.
No mutable returned state or Python cache object is reused as authoritative state.

Authentication matters: changing an unkeyed cache digest to match a corrupted
prefix cannot bless history that never passed replay. A queue-directory owner
who can read the key and reseal arbitrary cache data is outside this integrity
threat model; that owner can already rewrite completely valid WAL events under
the existing unkeyed event checksums. This proposal adds no authorization/security
boundary against malicious root and does not claim signatures or tamper-proof
storage. SHA256 collision resistance and cooperative advisory locking are assumed.

Missing,bad,oversized,deeply malformed,wrong-sealed,stale or misbound cache/key
falls back to complete strict streaming replay. Key/cache loss or disk errors
cannot change operation results or WAL/checkpoint durability. Prefix changes,
including same-length middle corruption, force replay and expose old checksum/
sequence errors. Cache payload and key reads are bounded. Last-event mismatch
also falls back; it never repairs malformed authoritative data. Replayed valid
prefixes may be attested only after original checks succeed. A future change to
validation semantics must bump/invalidate cache version; this proposal keeps
exact original event validation rules and WAL format.

Stable .lock inode and LOCK_EX scope remain unchanged. Only original public
operations perform their original commits; WAL records are byte-format identical
complete-snapshot envelopes. Existing torn-final behavior is preserved: durable
content-hashed quarantine before trimming ONLY an incomplete final record,
before checking prior records. Complete corrupt records are never trimmed or
ignored. This is existing application recovery, not administrative compaction or
manual live-WAL editing. No new WAL truncation/rewriting strategy is introduced.

After own commits, an immediately verified preceding byte prefix may be extended
with the newly fsynced event hash and optionally attested. Crash after WALfsync
but before checkpoint/cache remains recoverable by tail replay. Checkpoint errors
retain their original failure behavior; cache-write errors are best effort.
Atomic key/cache writes fsync and rename; orphan temporary files are ignored.
A newly appended own event must also pass the original strict record checks
before extending an attestation. A malformed private commit followed by a valid
last event cannot hide an invalid middle snapshot; both loaders still refuse.

Old workers ignore new cache/key files and continue to append the unchanged event
format under the same lock. A new worker can use the older valid cache prefix
then validate old-worker tails. Rollback to original code just ignores optional
files. No worker is killed, lease expired, live source auto-reloaded or GPU lease
released by this patch. Old long-lived readers retain their original replay cost
until root chooses a normal restart; coexistence is correctness, not a promise
that old code gets faster.

## Tests and descriptive synthetic performance

All106 selected CPU tests pass:41 private proposal cases plus original store,
dispatch and policy suites. Tests use temporary stores only. They cover original
ready/active caps,source-review admission,dependencies,guards,cleanup,immutable
terminal history,stale-lease behavior and process races; plus middle/final record
tamper,checkpoint forgery,cache/key loss/corruption/oversize/deep JSON,wrong last
record/sequence bindings,complete bad tail,torn tail plus corrupt middle,
post-WAL-fsync crash,optional cache-write failure,orphan temporary cache,
shorter/replaced WAL,checksum-valid history edit semantics and mixed old/new
workers with exactly one GPUclaim. The word GPU in these synthetic descriptors
never launches hardware or transport.

A1000event temporary WAL (~66MB including three later legacy events) preserves
full snapshots and an old RUNNING lease. Cold candidate validates1000events;
each fresh-process warm read validates only1event while rehashing every prefix
byte; three old-worker events require1+3=4validations. Five original strict reads
have median0.2785s versus0.0421s for five cached reads, an observed6.61ratio here.
There is NO speed threshold assertion; deterministic parse counts,full-state/
lease equality and unchanged WALbytes are asserted. These are environment/cache
observations, not guaranteed latency, official scores or kernel speedups.

Reproduce from repository root:

```bash
python3 -m unittest proj_rsi.scheduling.test_store_cached_proposal proj_rsi.scheduling.test_store proj_rsi.scheduling.test_dispatch proj_rsi.scheduling.test_policy
python3 -m proj_rsi.audits.scheduler-wal-cache.synthetic_profile
```

The live-file read-only scripts are retained for provenance, not automatically
rerun by tests. They never instantiate the live store or use recovery code.
Synthetic profile binds candidate and baseline source hashes in its JSON.
Every timing sample uses a fresh Python interpreter; process wall time is also
retained. Exact legacy fbc90c7 source and a hash-checked loader are preserved in
this audit so legacy-worker and benchmark comparisons stay genuinely legacy
even after root later activates the candidate. No live module is imported as a
proxy for historical code in these compatibility tests. Prior draft8b source
and its preliminary measurement are retained under draft-history; the current
source-bound fresh-process profile is authoritative for this proposal.

## Activation boundary

Independent critic must review the exact proposal manifest and this threat/scope
model. Root may then explicitly replace live store.py with reviewed proposal
bytes and run normal fresh-process tests, preserving unrelated changes and any
new baseline edits. Root must not blindly overwrite a diverged baseline; rebase
and review if source hashes differ. No activation script or actual queue command
is provided/executed by the author. Descriptor/event/checkpoint formats and public
method signatures remain compatible; optional cache files are disposable.
