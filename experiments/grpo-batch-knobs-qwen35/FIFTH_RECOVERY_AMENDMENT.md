# DAPO continuation with colocated roles and shorter recovery gaps

The training node was terminated at06:18:42UTC on September16. The scheduler
marked it completed at06:19:08.861496UTC. The first resumed rollout and four
updates had finished, but no new checkpoint existed. Discard that1,024-response
rollout and four updates; return to the durable94,208-response checkpoint.
Raw metrics, W&B history, and bounded execution cost remain archived separately.
The observed events do not establish the cause of the pod deletion.

Use the surviving four-GPU Tier-2 B200 node with Miles' native colocated mode:
TP1/DP4 training and four TP1 generation engines share four physical GPUs.
The runtime offloads the inactive role to CPU memory. The pinned implementation
disables piecewise generation CUDA graphs in this mode. Use its standard
offload schedule without changing reward, loss, sampling or optimizer code.
The prepared4B model and8192packing target already fit a B200 in earlier
DP4 measurements. Verify checkpoint restoration and finite updates in this
placement before accepting continuation; no throughput claim precedes data.

Recovery saves now occur after every1024-response rollout. Repeated worker
losses before the previous4096-response save interval justify the extra
checkpoint cost. Validation remains every8192; selection remains restricted
to the initial model and32768-response multiples; endpoint remains131072.
P128/G8/B256/S4, constant LR1e-6, response cap16384, packing8192, and all other
learning settings remain fixed. Restore Adam, scheduler, trainer RNG and data
position; reshard the optimizer from DP8 to DP4 using Megatron's checkpoint.

A pinned warm compiler cache is restored only while the node has no GPU work.
An eight-GPU Tier-2 worker is also requested. If it becomes available, prepare
it independently and migrate only after a durable checkpoint if the remaining
work justifies the setup. Record every allocation and all setup/interruption
costs. This remains the same recovered learning trajectory, with no new
hyperparameter comparison or claim of identical stochastic generation.

Before any four-GPU colocated training started, the eight-GPU reservation
became available. Use that single eight-GPU node directly: TP1/DP8 training
and eight TP1 generation engines share eight physical B200s. The four-GPU
node only prepares a backup copy; it does not start a learning run. Release
its reservation after the new run has a durable checkpoint. This avoids a
further planned migration and retains the preceding trainer's DP size.

That eight-GPU worker was also terminated during preparation, before a new
learning run launched. Proceed with the already-prepared surviving four-GPU
node and the original colocated DP4 plan above. The eight-GPU option was a
setup attempt, not a measured training allocation. The authoritative run
configuration records four shared physical GPUs and one node.

The first four-GPU colocated attempt (resume5) failed during SGLang startup
because TorchMemorySaver rejects expandable_segments:True. It generated no
responses and applied no updates; its 72.9086 seconds are setup cost only.
Resume5b explicitly disables expandable segments for colocated roles using
both allocator environment aliases. All learning settings and checkpoint91
remain unchanged; resume5 is excluded from learning ancestry.
