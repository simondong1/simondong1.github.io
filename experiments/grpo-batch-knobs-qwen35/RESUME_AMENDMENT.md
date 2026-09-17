# DAPO continuation after rollout-node termination

The rollout pod for main-v2 was terminated at 00:12 UTC on September 16.
Generation then failed with HTTP 503. The trainer node survived. Kubernetes
events record container termination and replacement, but do not establish why
the pod was terminated. There were no nonfinite-gradient or GPU-memory errors.

The attempt completed 38,912 responses and 152 updates in 9,513.09 seconds.
Its latest durable checkpoint is at 32,768 responses / 128 updates. Resume
from that checkpoint, preserving weights, Adam, the constant LR scheduler,
trainer RNG, dataset position and study counters. The 6,144 responses and
52,375,043 output tokens after the checkpoint are discarded from the retained
training path; their full execution cost remains in reported job time.
Partially generated work from the failed next rollout is not fully counted by
the completed-response counter; its time remains included.

Learning settings and the final 131,072-response endpoint are unchanged.
The new segment generates 98,304 responses and performs 384 updates.
Recovery saves remain every 8,192 responses; only initial / 32,768 / 65,536 /
98,304 / 131,072 responses remain eligible for validation-only selection.
Test data does not inform any recovery or selection decision.

The replacement allocation still has two Tier-2 B200 nodes: eight training
GPUs with TP1/DP8 and eight TP1 rollout engines. The new node hosts training
and the private Ray head; the surviving node hosts rollout. A separate port
range isolates the new private Ray service without stopping managed Ray.

Keep main-v2 and main-v2-resume1 as separate, independently W&B-audited
execution segments. The analysis joins only the checkpoint ancestry. Response
and output-token learning axes exclude discarded updates; execution time and
allocated GPU-hours include them. Recovery downtime is reported separately.
Resuming does not promise identical stochastic generations to an uninterrupted
job. This remains one recovered DAPO training trajectory, not another seed.
