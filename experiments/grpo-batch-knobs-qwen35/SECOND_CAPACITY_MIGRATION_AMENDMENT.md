# Restore the established twelve-GPU placement after colocated OOM

An eight-GPU Tier-2 B200 node became available during the four-GPU colocated
recovery. Its pinned runtime, model and warm compiler cache were prepared
independently. The intended checkpoint-boundary move could not occur: the
four-GPU run failed during its third optimizer update, before a new checkpoint.
Its1,024responses,8,317,405output tokens and two completed updates are discarded.
This is a GPU-memory failure, distinct from the earlier worker deletions.

Use both prepared nodes: eight training GPUs (TP1/DP8) and four TP1 generation
engines, twelve physical GPUs across two nodes. This disaggregated allocation
previously completed a rollout and updates in resume4b before that worker was
lost. It supports the established expandable-segment allocator without native
colocation/offloading. No allocator split-size tuning is applied. This is an
operational recovery, not a controlled topology experiment.

Restore resume3checkpoint91 at94,208responses/368updates. Keep P128/G8/B256/S4,
constant LR1e-6, response cap16384, packing8192, fixed131072-response endpoint,
validation every8192 and recovery saves every1024responses. Selection stays
restricted to the initial model and32768-response multiples. Restore weights,
Adam, scheduler, trainer RNG, data position and cumulative sample/token counters.

Post-launch verification must show restored prompt order, contiguous update
indices, finite metrics, unchanged numerical hooks and durable checkpoint
archival. Different GPU reductions and fresh sampling need not reproduce an
uninterrupted trajectory. Both reservations are released after final tests.

The first twelve-GPU launch on the new node (resume6) exposed an infrastructure
placement issue: Miles sorts GPU bundles by node IP, placing the four-GPU node
first. That split training across hosts and would scatter checkpoint shards
on unshared local filesystems. Stop that attempt before accepting any work.
Resume6b explicitly places the designated eight-GPU training node first, then
uses the existing per-host rollout port allocator. Only infrastructure
placement code changes. Verify bundle ranks0–7 on the training host and8–11
on the four-GPU generation host before accepting progress.

At termination, resume6 had completed its first1,024-response rollout with
8,279,535output tokens, but no optimizer update or checkpoint. Those responses
are discarded, bringing total rewound/discarded work to11,264responses and
94,301,425output tokens, with35completed updates discarded overall. Its
337.5752seconds remain in execution cost. No checkpoint from its split
placement is accepted.
