# Replace one lost generation worker after a durable save

The generation worker disappeared during weight publication after checkpoint97.
The local and S3 pointers both identify checkpoint97; all checkpoint and rollout
state file sizes match the archived objects. All4,096 responses and16 optimizer
updates in resume7 are retained, through100,352 responses,780,349,493 output
tokens and392 updates. No completed work from this interruption is discarded.
The node received SIGTERM; its underlying deletion cause is not established.

Resume8 restores the same model, Adam, scheduler, trainer RNG, data position
and counters. Keep sixteen Tier-2 B200s across three nodes: eight TP1/DP8
trainers on one node and eight TP1 generation engines across two four-GPU
nodes. One generation node is a prepared replacement. Explicit training-node
priority keeps checkpoint shards together. Actual placement and the first
resumed batch must be audited before accepting the continuation.

P128/G8/B256/S4, constant LR1e-6, response cap16384, dynamic packing8192,
validation every8192 responses and the fixed131072-response endpoint remain
unchanged. Recovery saves occur every1024 responses. Selection remains limited
to the initial model and32768-response checkpoints. There are30,720 responses
and120 updates left. SGLang restarts do not guarantee identical future samples.

A CPU/I/O probe of the same archived checkpoint took77.20 seconds with32
concurrent S3 requests versus85.64 seconds for its original transfer with10.
The copied object sizes and multipart ETags matched, and the probe copy was
removed. The attempted live tuning process exited before changing any setting
when resume7 failed. Apply32 requests before resume8 and measure the next three
native uploads; restore the previous configuration if no improvement appears.
This single probe is operational evidence, not a controlled speedup estimate.
The checkpoint pointer is still published last, after all state is durable.
