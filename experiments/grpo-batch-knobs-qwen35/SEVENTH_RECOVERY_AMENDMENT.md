# Continue on the twelve surviving GPUs

The replacement generation worker disappeared during resume8's first rollout.
Kubernetes recorded container shutdown, and the generation manager then retried
connections to the missing worker. Its underlying deletion cause is unknown.
Stop the exact affected driver; do not stop either Ray service. No completed
rollout, optimizer update or checkpoint was recorded in this attempt. Some
individual responses may have completed inside the unfinished rollout; their
number and output-token count were not persisted. Include the attempt's full
elapsed job time in execution cost, without inventing sample counts.

After its GPU actors exited, the driver remained in kernel core-dump handling.
The controller sent SIGKILL only to that exact driver after SIGTERM did not
finish shutdown. Disable core dumps for the next driver process, retaining
ordinary logs and checkpoint evidence. This changes no GPU or learning setting.

Resume9 restores checkpoint97 from resume7: 100,352 retained responses,
780,349,493 output tokens and392 optimizer updates. Twelve Tier-2 B200s remain
available across two nodes: eight TP1/DP8 training GPUs on one node, and four
TP1 generation engines on the other. A replacement four-GPU pod is pending
and unschedulable. Continue with the surviving allocation. Additional capacity
must justify another restart with enough expected time saving to offset it.

All learning settings remain P128/G8/B256/S4, constant LR1e-6, response cap16384
and dynamic packing8192. Keep the fixed131072-response endpoint, validation
every8192 responses, recovery saves every1024 and selection only at the initial
model and32768-response multiples. There are30,720 responses and120 updates
remaining. Keep explicit training-node priority and audit the first resumed
batch and checkpoint. Restarted generation is not bitwise reproducible.

The classic S3 client with32 concurrent requests remains installed on the
training node. Resume8 made no checkpoint upload, so its native measurement
is still pending. Measure resume9's first three uploads against the preceding
10-request archives, preserving the same rollback rule and pointer-last save.
