# Planned migration to newly available Tier-2 capacity

After the DP4 continuation began, an eight-GPU B200 worker became schedulable.
The user requested more GPUs when they can shorten training. Prepare this node
while the existing run continues, then deliberately stop the named driver only
after a fresh recovery checkpoint has been archived. This is an intentional
capacity migration, not another unplanned worker loss.

The next allocation is three Tier-2 GPU nodes: eight training GPUs on one node
(TP1/DP8) and eight TP1 rollout engines across two four-GPU nodes. Sixteen GPUs
are reserved. Training stays within one node; generation remains synchronous.
Existing pilots measured faster training with DP8 than DP4, and faster
generation with eight engines than four. They do not quantify the exact saving
for this changed three-node placement. Report observed time for every segment.

Restore the latest fully archived checkpoint with weights, optimizer, scheduler,
trainer RNG and data state. Keep P128/G8/B256/S4, LR1e-6, response cap16384,
packing8192, validation cadence8192 and the 131072-response endpoint unchanged.
Recovery saves remain every4096 responses; selection is still restricted to
the initial model and multiples of32768. Verify restored counters, first update
index, constant LR and finite gradients before accepting continuation.

The pinned serving address allocator assumes equal node sizes. A scoped hook
groups TP1 engines by their actual Ray host, then delegates each group to the
unchanged native allocator with that host's engine count. It checks the resulting
addresses against actual hosts. Physical GPU IDs still come from Ray placement;
training retains its correct eight-GPU node size. No reward, loss, sampler or
optimizer code changes. A CPU check against the pinned native allocator covers
4+4, 8 and 4 engines, including unique per-host ports. Initial-model evaluation
may use the single eight-GPU node; the same host-based allocation supports it.

Archive the intentional stop event and all metrics. Generation already in
flight when the driver stops may be cancelled and is not counted as a completed
rollout. Any completed work beyond the selected checkpoint is discarded from
the learning axes and retained in cost accounting. DP changes, balancing and
restarted generation do not promise an identical trajectory. This remains one
recovered DAPO learning trajectory, not another hyperparameter trial.
