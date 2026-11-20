# Single-seed local implementation checks — 2026-11-18

These are development-runtime correctness checks, not trained-model performance or publication acceptance. See `summary.json` for exact component source fingerprints and runtime.

- 55 final unit tests passed, including partial-evaluation recovery, cache reuse, damaged-row rejection, single-seed statistics and teleport-removal reconciliation.
- Grid/Monaco × MA2C/DQL/PPO real-SUMO workflows passed: parent, offline WCE, five continuation methods and fresh-process online-WCE recovery.
- Both maps: two-episode uninterrupted versus interrupted/resumed offline WCE and fixed-WCE continuation produce matching tensors and episode queues.
- The initial full evaluation exposed two teleport removals counted as SUMO arrivals. The evaluation-only fix preserves raw arrived, separates normal completed and removed, reconciles TripInfo records, and uses only normal completions for completion rate/trip means.
- After that fix, both maps passed complete 6600-second controller validation and 3600-second Uniform/peak-1.5 evaluations, each with one realization plus audit rerun. Evaluation resumption preserves saved shard bytes.
- Current code also passed forced SIGKILL recovery, duplicate-ownership rejection and two concurrent independent SUMO runs.

The six-case training workflows predate the evaluation-only fix. Subsequent full-horizon evaluation and recovery checks record the current source separately. No single-source publication gate is issued from combined evidence. Formal execution requires the locked local/container runtime and a fresh complete `scripts/verify.py --extended` gate. The host uses a SUMO development build and mismatched dependencies.

Training pilots use 110-second episodes and reduced DQL replay/warm-up to exercise updates. These checks do not establish convergence, full-replay resource needs or performance superiority. No long tuning or formal training has started.

Only compact evidence is retained here. Checkpoints and audit NPZs remain in temporary run directories; audit certificates retain their original hashes. Canonical evaluation shards and index hashes are included. The initial failure log and exact follow-up scripts are retained for auditability; use the repository verifier for a fresh reproducible complete run.
