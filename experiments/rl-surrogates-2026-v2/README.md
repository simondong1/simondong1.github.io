# GSM8K and DAPO-Math surrogate comparison

Follow-up to [RL Surrogate Losses in 2026](https://simondong1.github.io/rl-surrogates-2026.html).
The earlier Countdown/DeepMath experiment is preserved in its own directory.

This study uses **full-parameter Qwen3.5-9B training in Miles** on two B200s.
It compares GRPO advantages with six policy surrogates: PPO clipping, CISPO,
GLM-5 token rejection, binary-TV DPPO, SAPO, and GSPO. All arms start from the
same pretrained checkpoint and fresh optimizer. This is not actor–critic PPO.

## Data

- **GSM8K:** arithmetic word problems, official training and test partitions.
  64 training problems are reserved for pilots; 7,409 remain in the training
  pool. Periodic evaluation uses a fixed 512-problem subset of the official
  1,319-problem test set.
- **DAPO-Math-17K:** competition math with numeric answers. The public file
  contains 1,791,700 rows but 17,188 unique questions. We deduplicate by
  whitespace-normalized prompt and remove 12 questions with conflicting
  labels before splitting. 64 questions are reserved for pilots, 1,024 for
  evaluation, and 16,088 for training. Periodic evaluation uses a fixed 512
  from the evaluation partition.

The grader checks the final numeric `Answer:` line or boxed answer. It accepts
equivalent decimal notation and correctly grouped thousands separators.
Reward checks the final answer; it does not verify the reasoning trace. A
response reaching its cap can receive reward if it contains a correct final
answer. Truncation is also logged separately.

`prepare_data.py` records the pinned HF revisions, excluded conflicts, exact
splits and checksums under `/tmp/rl-surrogate-2026-v2/data`.

## Controls and scope

Within each dataset, the prompt pool/order, sample count, optimizer, learning
rate, response cap, minibatch schedule, sequence-mean reduction, direct rollout
log-probability anchor, and evaluation settings are held fixed. Responses
naturally diverge as the policies change. No adapter training or quantization
is used. The text policy has 8,953,803,264 trainable parameters; unused vision
and multi-token-prediction modules are excluded from these text-only tasks.

Surrogate-specific settings are declared rather than tuned on validation:
PPO 0.2/0.2; CISPO ratio clamp 0.5–5; GLM-5 interval 0.5–5; binary-TV DPPO
threshold 0.15; SAPO temperatures 1/1.05; GSPO 0.0003/0.0004. These are not
matched trust-region strengths or reproductions of the complete lab recipes.

DAPO clipping is logged as a diagnostic; a separate full DAPO recipe would
also change sampling and other controls. Actor–critic PPO would add a critic
and change the advantage estimator, so it is outside this surrogate-only
comparison. Using DAPO-Math data does not imply using the DAPO algorithm.

## Throughput and measurements

Pilots measure dynamic sequence packing and rollout concurrency before the
main schedule is fixed. Optimizer states stay resident; final checkpoints
omit optimizer state to reduce I/O. Each arm uses the same chosen settings.
The host queue releases its two GPUs on completion, failure, or the deadline.

Every collection records training accuracy, all-correct/all-incorrect group
fractions, response lengths, truncation, generated tokens and elapsed time.
Every optimizer update records loss, gradient norm, policy-ratio histograms,
approximate KL, positive/negative gradient weights and clipping fractions.
Trainer/inference mismatch before updates is recorded separately from drift
during optimization. Miles also records full-distribution entropy during its
initial scoring pass, phase timings, and training throughput.

TensorBoard, per-example JSONL, sampled-token training dumps, GPU telemetry
and exact launch commands remain in `/tmp/rl-surrogate-2026-v2/runs`.
`export_metrics.py` exports all scalar series without discarding their axes.
The final plots will show training and held-out accuracy over updates, with
length, entropy and clipping diagnostics alongside them. Single-seed results
must be labeled exploratory; task uncertainty is not training-seed uncertainty.

## Reproduction

Model: `Qwen/Qwen3.5-9B` at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Miles: `radixark/miles` at `2ef603a8ca44410beaad8c1dda5bf85f2a9d2533`.
Image: `radixark/miles@sha256:16dc2819ed890455d54102fb88f3cdffdd210104b1707583051582af46bb6c5b`.
Run `test_surrogates.py` in that image, prepare the pinned data, then use
`run_study.py --plan study-plan.json`. Paths in `launch.py` follow the isolated
experiment container mounts. The final plan and runtime observations are
recorded separately from this initial protocol.

## Frozen main schedule

One seed (42), six surrogates per dataset, twelve jobs total. Each collection
samples 64 prompts with four responses each. Global minibatches of 16 give
16 optimizer updates per collection. The smaller minibatch than the throughput
pilot permits more updates per fixed rollout anchor; this choice is shared by
every surrogate and was made before comparing main-run scores.

| Dataset | Collections | Optimizer updates | Prompt presentations | Sampled responses | Response cap |
|---|---:|---:|---:|---:|---:|
| GSM8K | 16 | 256 | 1,024 | 4,096 | 1,536 |
| DAPO-Math-17K | 20 | 320 | 1,280 | 5,120 | 4,096 |

These are budgeted partial passes through the larger training pools, not full
epochs or claims of convergence. Each arm uses the same prompt prefix within
a dataset. Evaluate at baseline and every four collections (64 updates).
Training uses 8,192-token packing, gradient checkpointing, and two SGLang
engines with up to 128 requests each. AdamW uses learning rate 1e-6, betas
0.9/0.95, epsilon 1e-8, no weight decay, and gradient norm clipping at 1.
There is no reference KL or entropy penalty.

The corrected DAPO/GSPO pilot completed three collections with nonzero
gradients and successful weight synchronization. Warm training phases took
62.5 and 58.8 seconds for 256 responses, at global minibatch 32. The first
collection includes kernel compilation and is excluded from warm throughput.
The GSM infrastructure pilot measured 7.3/7.0-second warm training phases,
but its original Answer-line parser failed to strip EOS markers; its zero
rewards make it invalid as a learning result. That bug was corrected and
regression-tested before the DAPO pilot and every main run.

At the first DAPO pilot collection, reward was 46.1% and 51.2% of responses
reached the 4,096-token cap, all without receiving reward. Thus the main DAPO
study measures answer success under a short response budget, with substantial
truncation. The cap and grader remain identical across all algorithms.
Estimated main-queue runtime is about seven hours, bounded by the remaining
eight-hour budget after pilots. Exact commands and code hashes are captured
before execution; failure stops the queue for inspection.

## Live monitoring

`wandb_sync.py` backfills completed jobs and follows running jobs every 20
seconds. It runs separately from training, so adding W&B does not change the
frozen experiment or restart a trainer. Each job has a stable W&B run ID;
dataset, surrogate, seed, model, data manifest and source hashes are recorded
in its configuration. `--include-pilots` also imports pilots with separate
tags, including the GSM grader failure marked as excluded learning evidence.

Install `wandb==0.30.0` and `tensorboard==2.20.0` in a separate environment,
authenticate to the intended W&B server, then run:

```sh
python wandb_sync.py --root /tmp/rl-surrogate-2026-v2 \
  --base-url "$WANDB_BASE_URL" --entity "$WANDB_ENTITY" \
  --project rl-surrogates-2026 --include-pilots
```

Credentials are read from `WANDB_API_KEY` or that server's `~/.netrc` entry.
The uploader waits for credentials and retries network failures. A local
journal supports resuming interrupted uploads. `wandb-status.json` contains
run links and upload status; training status remains in `study-status.json`.

| W&B metric group | Horizontal axis | What to inspect |
|---|---|---|
| `evaluation/*` | Completed optimizer updates | Held-out accuracy and truncation |
| `train/*` | Completed optimizer updates | Gradient norm, ratio bins, clipping, KL, engine mismatch |
| `sampling/*` | Policy updates before collecting the responses | Reward, zero-variance groups, response length, truncation |
| `rollout/*`, `perf/*` | Native zero-based collection index | Entropy, phase times, throughput |
| `hardware/*` | Seconds since training started | Memory, utilization and power on GPUs 3 and 7 |

Original `eval/*` series retain Miles' native rollout index; use
`evaluation/accuracy` for comparisons over optimizer updates. Collection
records are written before training and are not treated as completed updates.
Surrogate loss magnitudes are not directly comparable across algorithms.

Completed jobs also upload a compressed artifact containing exact commands,
raw TensorBoard events, per-response reward/evaluation records, GPU telemetry,
console logs and source snapshots. Model checkpoints and training tensor dumps
stay local. W&B's automatic host/GPU monitoring is disabled in the uploader;
the logged GPU measurements come from the training queue.
