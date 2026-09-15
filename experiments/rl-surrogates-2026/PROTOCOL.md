# RL surrogates: experiment protocol

Written before comparative training results, 2026-09-14.

## Question

With the same initial model, data, sampling, GRPO advantages, sequence-mean
aggregation and optimizer, how do PPO clipping, binary-TV DPPO, and the GLM-5
async rejection surrogate affect learning and gradient retention?

This is a comparison of surrogate functions, not reproductions of entire lab
recipes. In particular DAPO's resampling, different advantage normalization,
different aggregation, and asynchronous scheduling are not silently switched on
when selecting a surrogate.

## Resources and selection

- Maximum aggregate training wall time: 8 hours on the two allocated B200s.
- Allocation: GPU UUIDs ending `e913a144b00a` and `84ecd67d38bd` (physical 3, 7).
- Full-parameter training only; no adapters. Selected candidate: Qwen/Qwen3.5-9B,
  HF revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The initial 27B
  candidate exceeds the two-GPU state budget with FP32-master Adam before
  activations; select a smaller model if the full-parameter pilot does not fit.
- Framework: radixark/miles, source revision recorded in the results manifest.
- Backend and precision are chosen from a correctness/throughput pilot before
  selecting the matched comparison duration. FSDP2, BF16 forward/backward,
  FP32 master weights, and gradient checkpointing are the initial configuration.
- Public HF datasets: Jiayi-Pan/Countdown-Tasks-3to4 and
  zwhe99/DeepMath-103K. Deterministic, deduplicated train/pilot/test splits are
  created before model evaluation. DeepMath uses integer-answer examples for
  a small, inspectable exact-answer verifier. No HF solution traces are used.

## Controls

Every arm starts anew from the same checkpoint and optimizer initialization.
Within a dataset and seed, prompts and their order, sampling settings, response
budget, group size, optimizer, update count, advantage normalization, loss
reduction and evaluation schedule are identical. Online completions may diverge
after updates; they cannot remain identical when policies change.

The primary arms use the same direct rollout-logprob anchor. PPO uses symmetric
0.2 bounds; DPPO uses probability-difference thresholds 0.15/0.15; GLM-5 uses
ratio bounds 0.5 and 5, matching the cited Mercor recipe's exposed defaults.
These thresholds are part of the algorithm specification, not matched trust
regions or separately tuned hyperparameter optima. No winner is selected by
held-out tuning. The exact effective config is exported for every arm.

Prefer matched update counts over matched wall time. Record both. Run at least
one seed for all primary arms and tasks; add a second matched seed only if the
pilot forecast fits the remaining budget. A single-seed study must be labeled
exploratory, and cannot support a general ranking or a training-seed variance
estimate. Binomial/bootstrap evaluation intervals are not seed uncertainty.

## Checks before comparative training

1. Check reward verifiers on valid, invalid and adversarial expressions.
2. Check surrogate gradients against analytic coefficients and native PPO.
3. Confirm identical initial weights and nonzero reward variance on pilot tasks.
4. Measure training/inference probability differences before optimization.
5. Confirm optimizer changes parameters and synchronization changes inference.
6. Benchmark step time and memory; lock the common duration and settings.

## Measurements

Held-out accuracy at baseline and fixed update checkpoints; training reward;
output length; truncation rate; zero-variance group rate; total wall time;
tokens generated; ratio distribution; actual nonzero-advantage gradient gating;
sampled-token surprise (not mislabeled as full-distribution entropy).

Use gate diagnostics on shared token/probability inputs to distinguish a gate
that is inactive from one that changes the update. Report negative/null results.
Do not conclude that a surrogate is universally best from these runs.

## Article order

1. One prompt, rewards, and a gradient: brief RL/PPO/GRPO recap.
2. A map of what algorithm names change.
3. PPO baseline, with a worked numerical example.
4. DAPO asymmetric clipping; CISPO bounded weight; GLM-5 rejection.
5. SAPO smooth gate; DPPO divergence gate; GSPO sequence decision.
6. Interactive common gradient comparison and aggregation caveat.
7. Dated public lab/framework evidence.
8. Reproducible measured runs, limits and compact takeaways.

The final article must contain actual measured results, source references,
reproduction commands, and an explicit account of deviations from this pilot
plan. It must pass the site's reader proofread and desktop/mobile checks before
publication.

## Locked main study (2026-09-15, before comparative runs)

Both full-parameter pilots passed forward/backward and weight synchronization.
The 256-token infrastructure smoke had all-truncated, zero-reward responses;
it is not a learning result. Longer pilot responses gave mixed reward groups.
No comparative held-out result was inspected to choose a surrogate or threshold.

- Qwen3.5-9B, FSDP2: full language-policy parameters, BF16 computation,
  FP32 master weights/reduction and Adam states, gradient checkpointing.
  The public multimodal checkpoint also contains unused vision/MTP weights;
  text-only tasks provide no vision training signal. No adapters or quantization.
- Both trainer and rollout engines remain resident; inference runs sequentially
  with training. Two one-GPU SGLang engines, static memory fraction 0.25,
  maximum 32 running requests each, decode CUDA graphs, Triton attention.
- AdamW LR 1e-6, beta1 0.9, beta2 0.95, epsilon 1e-8, weight decay 0,
  global gradient norm clipped to 1, no KL or entropy bonus.
- 32 prompts per collection, 4 samples each, global minibatch 8: 16 optimizer
  steps per collection. 32 collections = 512 optimizer steps = 4,096 sampled
  responses over 1,024 prompt presentations (two epochs of the 512-task split).
- This increases updates per fixed rollout anchor from the pilot's 4 to 16.
  The pilot's PPO gate activated only weakly; a larger shared collection lets
  the policy drift enough to test the different gates, without selecting a
  different schedule for any arm.
- Two seeds, 42 and 43; all three primary surrogates on both datasets.
- Response caps: Countdown 1,024; DeepMath 2,048. Native chat template with
  enable_thinking=False; prompts still request brief working. Temperature 1,
  top-p 1 for training; greedy held-out evaluation on 256 tasks per dataset.
- Evaluation before training and every 4 collections (64 optimizer steps).
- All runs start from the pinned original model. Advantages use Miles' GRPO
  sample-standard-deviation normalization (plus 1e-6), with no additional
  batch whitening. Sequence-mean aggregation is fixed.
- Record same-weight trainer/inference mismatch using the forward-only scores
  from before minibatch updates, separately from ratios after optimizer drift.
- Main queue, order and deadline are in study-plan.json. Code hashes are saved
  before the first run and checked before each subsequent run.

The pilots use their pilot split for both rollout and evaluation to check the
pipeline, so pilot score changes are not held-out generalization results.
The main study uses disjoint training and test splits. Two seeds remain a small,
exploratory study; differences are not evidence of a universal ranking.

### Verifier correction before restarting the matched study

Review of pilot samples found a valid Countdown expression, `40 * (24 / (18-6))`,
written with LaTeX `\frac`, receiving zero from the ASCII-only parser. The
verifier now translates explicit braced fractions (including nested fractions),
common multiplication/division macros and left/right delimiters into the same
restricted arithmetic AST, and supports unary signs. It still rejects all
function calls and requires exactly the supplied number multiset. Analytic
valid/invalid tests cover the extension. The initial PPO main attempt was
interrupted and archived; it is excluded. Every reported main run restarts from
the original checkpoint with this corrected verifier and new frozen code hashes.
The decision used pilot-response inspection, not comparisons between algorithms.

### Evaluation repeatability check during the queue

The first two Countdown initial evaluations used the identical source checkpoint,
seed 42, and greedy evaluation settings, yet differed on 38/256 task rewards
(net score difference: 2/256). Source inspection confirmed that `eval_temperature=0`
is preserved in each dataset's sampling parameters, and inference engine seeds
are derived from the configured seed. Inference was not configured for strict
batch-invariant determinism. Greedy generation is therefore not assumed bitwise
reproducible in this setup; a small numerical difference can change a token and
then its entire continuation. This observation does not identify a unique kernel
or batching cause.

No training or evaluation setting changed. All six untrained baseline evaluations
per dataset are retained as a same-checkpoint repeatability check. Plots retain
each arm's actual initial score. Final task-bootstrap intervals condition on the
realized trained models and decoded outputs; they do not include independent
training-seed or inference-runtime uncertainty.

### Shared numerical ratio guard: completed GLM run inspection

The original frozen implementation clamps log ratios to [-20,20] before
exponentiating. The first completed GLM Countdown run activated that guard on
0.00718% of total sequence-normalized token weight, across 45/512 steps. The
exporter's initial assumption that the guard would stay inactive was therefore
rejected by the recorded data.

For the selected GLM loss, this guard does not change the gradient: exp(-20) is
below 0.5, and exp(20) is above 5. Tokens reaching either guard are already
rejected by the GLM interval. Analytic extreme-ratio checks cover both advantage
signs. No training code or parameters changed. The exporter now records guard
activation and permits this inspected GLM case; an occurrence in PPO or DPPO
still requires inspection before export.

Counterfactual weight diagnostics on these extreme GLM-run inputs use the guarded
ratio and do not account for the numerical clamp's derivative saturation. They
must not be presented as exact counterfactual PPO/DPPO gradients in that region.
The article's shared-input gate table uses PPO-trained traces; guard activation
is checked separately for those runs.

### Timing revision for all DeepMath arms (2026-09-15 06:00 UTC)

The first DeepMath attempt measured a steady training phase of 37.71 seconds,
generation around 21 seconds per collection, and baseline evaluation around
36 seconds. Compared with Countdown, this forecast left only a small margin
before the fixed 11:46 UTC allocation deadline for all twelve original runs.
To preserve matched two-seed comparisons, every DeepMath arm now uses **28
collections, 448 optimizer updates, and 3,584 responses**. Countdown retains
32 collections, 512 updates, and 4,096 responses. DeepMath thus presents its
512 training tasks 1.75 times: all once, then 384 of them a second time, in the
same order for each surrogate within a seed.

The initial DeepMath attempt was stopped, archived as a runtime probe, and
excluded. Its replacement restarts from the original checkpoint with the same
frozen training/surrogate/reward/metric code and the revised common duration.
This decision used runtime measurements, not comparative algorithm scores.
All model, optimizer, sampling, grading, and surrogate parameters stay fixed.
The original plan is retained in state history, the excluded attempt is recorded,
and the revised per-run counts are explicit in study-plan.json. The exporter
checks counts against each run's declared schedule.

## Response spot-check — 2026-09-15 06:50 UTC

After DeepMath GLM-5 seed 42 completed, inspected the first three baseline-failure to endpoint-success transitions in recorded evaluation order. One written limit problem has an incorrect reference integer (the sequence has opposite even/odd subsequence limits); another output violates a requested solution-method constraint that the final-integer grader does not check; another baseline had already derived the answer but was truncated before boxing it. The exact selection and excerpts are in `grader-cases.json`. These are illustrative cases, not a prevalence estimate. Labels, scores, reward code, and scheduled arms remain frozen and unchanged. Results describe grader success under the response budget, not independently verified mathematical reasoning.

## Late log-probability outliers and truncation semantics — 2026-09-15 10:04 UTC

Countdown GLM-5 seed 43 ended at 4.296875% reward with 99.609375% cap hits. These can coexist: the frozen grader accepts a valid final box even if later text runs to the cap. Truncation itself does not force zero reward. Its initial trainer/inference tail audit found six token positions above 5 nats absolute difference out of 4,039,764, max 14.490970, in collections 20, 22, and 28 (zero based), after the reward decline was already present. No token exceeded 20 nats in this initial comparison. The source of these rare large differences was not isolated. The experiment reports this limitation; small average differences are not a proof of numerical equivalence. The selected GLM gradient remains zero at the shared log-ratio guard extremes. No reward, loss, or training control was changed.
