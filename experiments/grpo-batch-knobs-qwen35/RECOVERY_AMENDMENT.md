# DAPO recovery, before main-v2

DAPO main-v1 lost its training/head worker after 2,561 seconds and 10,240
completed training responses (40 updates), before its first scheduled saved
checkpoint. The last archived validation at 8,192 responses was 79.6875%,
versus 77.34375% initially. This interrupted attempt is preserved separately;
it is not combined with the replacement trajectory or used to choose a recipe.

Start main-v2 from the original pinned Qwen3.5-4B weights and the same seeds,
data, objective, LR, batch settings and 131,072-response budget. The surviving
eight-GPU node becomes trainer/head and the replacement eight-GPU node becomes
rollout. Both remain Tier-2 B200 nodes, TP1/DP8 training plus eight TP1 engines.

Change only recovery checkpoint frequency from 32,768 to 8,192 responses.
Validation selection eligibility remains initial / 32,768 / 65,536 / 98,304 /
131,072 responses, with ties favoring earlier. Extra recovery checkpoints are
ineligible for selection. This extra saving cost is included in measured job
time. The original main protocol still defines all learning and test choices.

The first run's last status and metrics reached S3. No saved optimizer state
was available, so this is a restart from initial weights, not a continuation
of the interrupted optimizer path. No hyperparameter selection followed the
intermediate accuracy results.

Autoscaler logs show duplicate instance records and an obsolete deletion target;
they do not by themselves establish the complete cause of the pod deletion.
The obsolete target was cleared and the eight-GPU worker ceiling increased
from 3 to 8 without requesting extra replicas. Actual allocation remains two
eight-GPU nodes plus GSM8K's two four-GPU nodes. Worker limits must not be
reduced while reservations exist.
