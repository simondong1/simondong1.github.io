# DAPO recovery after another generation-worker loss

The four-GPU generation worker was terminated at 05:48:31 UTC on September
16, 2026. The scheduler subsequently marked it completed. These events do
not establish why the node was deleted. The training node, its checkpoints,
and the other four-GPU generation node survived.

The run had completed 95,232 responses and 372 optimizer updates. Its latest
durable checkpoint was at 94,208 responses and 368 updates. Restore that
checkpoint, including optimizer, scheduler, trainer RNG, and data state.
Discard the subsequent 1,024 responses and four updates from the learning
axes; preserve all execution records and include their cost. Partially
generated work in the interrupted next rollout is not a completed rollout.

A replacement four-GPU Tier-2 B200 node was already scheduled and is reserved
for generation. Retain the three-node allocation: eight training GPUs
(TP1/DP8), and eight TP1 generation engines across two four-GPU nodes. Reuse
the surviving private Ray service and training compiler caches. Do not stop
Ray or unrelated processes. Shut down only the interrupted study driver.

The recipe remains P128/G8/B256/S4, constant LR1e-6, response cap16384,
packing8192, recovery checkpoints every4096 responses, validation every8192,
and endpoint131072. The initial model and multiples of32768 remain the only
selection candidates. This leaves36,864 responses and144 updates. The same
host-aware serving hook is used without algorithm changes. Restarts do not
promise bitwise-identical generation.

Audit the interrupted segment against W&B. Verify the restored counters,
first prompt IDs, update368, constant learning rate, and finite gradients
before accepting the continuation. This is another segment of the same
learning trajectory, not another hyperparameter trial.

The first replacement was itself terminated during setup. The allocation
assertion stopped resume4 after15.99seconds, before any generation or optimizer
update. This is a setup failure, excluded from the learning trajectory and
reported separately. Resume4b therefore uses the surviving twelve Tier-2
B200s on two nodes: eight training GPUs (TP1/DP8) and four TP1 generation
engines. The private Ray service, trainer caches, checkpoint, response budget,
and every learning setting remain unchanged. The same serving hook supports
the mixed8+4node sizes. Report this final allocation change explicitly.
