# DAPO continuation on the remaining Tier-2 allocation

The first resumed segment lost its training/head node on September 16.
Kubernetes records the container stopping at 00:37:52 UTC and the scheduler
reports it completed at 00:38:25.734 UTC. The job's final status could not be
written on the lost node. Its externally finalized duration is therefore
820.58–854.31 seconds; cost accounting uses the upper bound and labels it.
All archived metric values match persisted W&B history, including its one
optimizer update. There was no observed nonfinite-gradient or memory error.

This segment generated 1,024 responses (8,880,943 output tokens) and performed
one update before interruption. None survives checkpoint recovery. Preserve
this failed execution separately and include its time in the recovered run's
cost. The last usable checkpoint remains at 32,768 responses in main-v2.
Resume from that checkpoint again with all learning settings unchanged.

The other eight-GPU node subsequently disappeared during recovery setup, and
both eight-GPU replacements were unschedulable. Reuse the two four-GPU nodes
released by the completed GSM8K study. The resumed allocation is two Tier-2
B200 nodes: four training GPUs (TP1/DP4) and four TP1 rollout engines.
The proposed intermediate three-node allocation was never launched.

This is an operational change in parallelism and generation capacity.
The optimizer batch, LR, group size, disjoint-update count, response cap and
remaining sample budget stay fixed. Restore Megatron's DP-reshardable optimizer
checkpoint from DP8 to DP4. Verify the loaded data counters, prompt order,
step axis, constant LR and finite gradients before accepting continuation.
Different GPU reductions and restarted generation are not bitwise identical;
token balancing with a different DP size may also change which responses land
in each inner update. This is one recovered trajectory with a recorded change
in execution, not an uninterrupted control for the previous topology.
GPU-hours are calculated per segment using its actual allocation (16 before
these interruptions, eight for this continuation).

Save recovery checkpoints every 4,096 responses in this segment. The slower
eight-GPU allocation and repeated node losses justify shorter recovery gaps.
Validation remains every 8,192 responses; selection eligibility remains only
the initial model and 32,768-response multiples. Extra recovery saves cannot
change selection and their cost is included in measured execution time.

The retained trajectory still has the original 131,072-response / 512-update
endpoint and validation-selection rule. Across the two checkpoint rewinds,
7,168 completed generated responses and 61,255,986 output tokens are discarded.
The abandoned initial main-v1 attempt remains a separate additional cost.
Cause of the node deletions is not established by the available events.
