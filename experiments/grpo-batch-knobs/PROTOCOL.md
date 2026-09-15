# Miles four-knob study — protocol before training

Question: how should generation volume, completions per prompt, optimizer batch
size, and optimizer steps per rollout be chosen on fixed available hardware?
The outcome is a conditional selection rule, not a universal winning tuple.

## Access and hardware

Metrics were tracked in a private W&B project. The public numerical export includes learning curves, per-example outcomes, and optimizer diagnostics.

Use only tier 2. An eight-GPU request waited for capacity; a two-GPU B200 worker
was allocated promptly. The initial experiment uses one GPU for the actor and
one for SGLang, synchronously. Record exact package/repository revisions,
GPU allocation, launch arguments, dataset hashes, and timing boundaries.

## Data and model

Candidate model: Qwen/Qwen2.5-1.5B-Instruct, full parameter training. Pin its
revision before runs. The pilot must find nontrivial binary rewards and mixed
reward groups on both tasks; otherwise adjust the model or response allowance
before freezing the main comparison.

Datasets: GSM8K and Countdown-Tasks-3to4. Frozen source revisions and SHA256
hashes are in data_manifest.json. GSM8K uses official train/test, with 512
validation examples removed from train. Countdown is deduplicated by sorted
number multiset plus target before making train/validation/test splits. Its
test set has 1,024 problems. The two tasks support transfer claims across two
mathematical task types, not across all RL tasks or model scales.

Keep test sets unused for selection. Use 512-example validation sets for
learning curves. The final independent test evaluation uses all 1,319 GSM8K
test problems and all 1,024 Countdown test problems. Archive per-example
correctness, not only aggregate accuracy.

Binary exact verifiers, with no generated Python execution or format-only
reward. Evaluate accuracy and formatting failures separately. Inspect samples
for truncation and verifier disagreement before the main runs.

Pilot audit correction (before the comparison): GSM8K smoke-v4 improved strict
marker-based reward from 60.9% to 82.8%, but the initial responses already had
82.0% correctness under final-number extraction. Most apparent gain was format.
The main GSM8K verifier therefore accepts an unmarked final number, following
the flexible GSM8K extraction convention, in addition to #### and boxed
answers. Smoke-v4 is excluded from learning comparisons. Freeze the corrected
verifier for all subsequent LR/packing pilots and screening/confirmation runs.

## Controls

One synchronous generation, then disjoint optimizer mini-batches; no replay,
no async generation, no dynamic filtering. Standard GRPO advantage normalization
and the same objective, optimizer, clipping, sampling, length caps, and prompt
template in every arm. Group size 1 is not a normal GRPO comparison: disabling
the standard deviation alone does not establish a usable baseline.

Pilot LR candidates: 1e-6 and 3e-6, using the same initial weights. Freeze one
stable rate with a measurable learning signal before the main comparisons.
Do not select a different base learning rate per batch arm. Explicitly marked
square-root retuning arms are the only exceptions.

Easy speed improvements: BF16, SGLang prefix caching and CUDA graphs where
supported, token packing with a measured memory allowance, avoid unnecessary
activation recomputation if memory permits, efficient attention, CPU thread
limits, sparse checkpoints, and cached model/environment downloads. Establish
packing safety during pilots, then keep it fixed across comparisons. Never
change global batch size to solve a packing problem.

## Main screening grid

P = rollout_batch_size; G = n_samples_per_prompt; B = global_batch_size;
S = num_steps_per_rollout. Assert P*G == B*S with exact integer arithmetic.

| Arm | P | G | B | S | LR multiplier | Comparison |
|---|---:|---:|---:|---:|---:|---|
| onpolicy-small | 16 | 8 | 128 | 1 | 1 | Baseline |
| split-2 | 32 | 8 | 128 | 2 | 1 | Generation volume |
| split-4 | 64 | 8 | 128 | 4 | 1 | Generation volume |
| split-8 | 128 | 8 | 128 | 8 | 1 | More stale inner slices |
| batch-256 | 64 | 8 | 256 | 2 | 1 | Same dump, larger train batch |
| batch-256-sqrt | 64 | 8 | 256 | 2 | sqrt(2) | LR retuning |
| onpolicy-large | 64 | 8 | 512 | 1 | 1 | Same dump, single update |
| onpolicy-large-sqrt | 64 | 8 | 512 | 1 | 2 | LR retuning |
| group-4 | 128 | 4 | 128 | 4 | 1 | More prompts at equal completions |
| group-16 | 32 | 16 | 128 | 4 | 1 | More repeats at equal completions |
| group-32 | 16 | 32 | 128 | 4 | 1 | Broader exploration, fewer prompts |

First screen: seed 0, 16,384 retained/generated trajectories per arm per task.
Evaluate at equal trajectory counts (including initialization), not equal
rollout indices or optimizer steps. Screening chooses candidates; it does not
provide a multi-seed ranking. Randomize run order within task to reduce time
and shared-node load confounding.

Confirmation: baseline plus two promising distinct candidate recipes, at
32,768 trajectories per run with three fresh seeds (1, 2, 3) on both tasks.
Choose using validation accuracy/sample and accuracy/time tradeoffs; explicitly
retain uncertainty or task-specific winners. If all differences are within
noise, the conclusion is a plateau, not an invented ranking. If the pilot
timing makes this grid impractical, document a prospective budget revision
before inspecting main-comparison results.

## Evidence and accounting

Report validation learning curves vs generated trajectories, generated tokens,
optimizer updates, and elapsed active training time. Also report complete job
time including initialization/evaluation/checkpoint overhead. Prompt count and
unique prompt exposure differ across group-size arms and must be shown.

Log generation, old-logprob, training, weight-sync and evaluation durations;
generated output lengths; truncation; correct rewards; fraction of groups with
nonzero reward variance; gradient norms; clipping fractions; train/rollout
logprob discrepancy; and per-inner-step diagnostics where available.

Time-to-target thresholds are frozen from baseline pilot validation accuracy
plus 5 and 10 percentage points, provided they are below 95%. Use the first
observed evaluation crossing and report its interval between evaluations;
unreached targets are censored, not estimated by extrapolation. Primary final
comparison also includes endpoint accuracy and area under the sample-indexed
curve so a missed threshold does not erase a run.

Use paired prompt/bootstrap uncertainty for per-example differences and report
seed spread separately. Three seeds are limited evidence; do not turn small
error bars from thousands of responses into false confidence across training
seeds. Report actual hardware restrictions and all failed/aborted pilots.

## Source checks already established

- R1 v2: 8,192 generated outputs, 16 disjoint mini-batches of 512, one inner
  epoch, LR 3e-6. This is explicit in the primary paper.
- DAPO v2: 512 prompts x 16 completions, training mini-batch 512, 16 updates,
  LR 1e-6 and 20 rollout-step warmup. It also changes filtering and the loss;
  the batch shape alone does not reproduce DAPO.
- DeepSeekMath: group size **64**, training batch 1,024, one update after each
  exploration stage. Thus G=8–16 is a convenient starting range, not universal.
- GSPO: four mini-batches per rollout in its reported experiments; sequence
  rather than token ratios change what clipping diagnostics mean.
- Hunyuan 2608.29296: one global update per retained batch, no mini-batch
  subdivision or reuse; **asynchronous partial rollouts and streaming** in its
  infrastructure. It is not evidence for strict synchronization. Square-root
  LR scaling is a starting rule, not a guarantee, and its critical batch is
  workload/target dependent.
- Current TRL source defaults num_iterations=1 and separates generation batch
  size from training accumulation. A claim that >2–4 epochs universally fail
  still needs a specific source; do not state that as an established law.
- Miles source derives B by floor division when S is provided; the public
  slogan “set any three” is stronger than this implementation. Our launcher
  will validate exact divisibility and all four values before allocating work.
- With G fixed, P and B remain two independent choices; S=P*G/B follows.
- SAPO reports four mini-batches for its reasoning comparison and two for its
  multitask setting. CISPO/DPPO alter how off-policy samples contribute; these
  are stability alternatives, not evidence for a universal batch tuple.
- ScaleRL (2510.13786) compares k=1 vs 8 with 48 prompts x 16 completions per
  optimizer update (768 trajectories), and finds larger batches can win only
  later in long training. Its fixed-total-batch G=8/16/24/32 curves are similar.
  Our short-horizon, small-model study cannot establish its long-run asymptotes;
  recommendations must name the measured budget and workload.
- IsoCompute Playbook (2603.12151) and BroRL (2510.01180) examine much larger
  groups as compute increases. IsoCompute distinguishes unrestricted compute
  scaling from a fixed total batch constraint; BroRL also increases optimizer
  batch size and square-root-scales LR while holding mini-batch count fixed.
  Their larger-group results do not establish a free improvement at fixed work.
  The group-32 arm was added before any completed training run to test this
  direction conservatively within our fixed-total-completion comparison.

The blog is written and published only after measured results exist, following
the canonical repository skill. Public artifacts must exclude credentials,
internal machine identities, and private infrastructure source/configuration.


## Pilot decision log — before main screening

Corrected stochastic pilots used P64/G8/B128/S4, 4096 trajectories, seed0,
validation128, packing16384, and graph max batch512. Both learning rates
completed32 optimizer steps, with all five evaluation points and all32 train
metrics independently verified in the persisted W&B GraphQL history.

| Task | LR | Initial | Endpoint | Sample-indexed accuracy AUC |
|---|---:|---:|---:|---:|
| GSM8K | 1e-6 | 82.81% | 80.47% | 83.11% |
| GSM8K | 3e-6 | 83.59% | 82.81% | 81.74% |
| Countdown | 1e-6 | 0.78% | 15.63% | 6.93% |
| Countdown | 3e-6 | 0.78% | 14.84% | 11.33% |

Use base LR3e-6 for both tasks: it supplies faster early Countdown learning
without large clipping in either pilot. GSM8K supplies no convincing learning
rate winner in this short, small validation sample; do not claim otherwise.
Neither dataset nor model is changed. Countdown now has mixed reward groups
(~23%) and a substantial correctness improvement with the corrected verifier.
The different same-checkpoint GSM8K baselines show ordinary inference
non-repeatability; retain each run's actual initial evaluation.

Freeze primary time-to-target thresholds from the first corrected pilot's
initial accuracy: GSM8K0.8859375/0.9359375; Countdown0.0578125/0.1078125.
Thresholds remain fixed even if the512-example screening baseline differs.
Report all endpoint curves as well; targets may be unreached on GSM8K.

Packing16384 was safe (peak training allocation at most82.73GiB). A matched
8192-token pilot on each task, also with graph max512 and LR3e-6, will decide
whether that memory increase improves warm actor throughput. These packing
pilots also save a checkpoint and reload it for validation128; the test set
remains untouched. Freezing the final packing setting precedes screening.

Timing: active-phase time is the sum of measured generation, trainer work
(including old-policy log-probabilities), and recorded weight-sync durations.
It excludes initialization, evaluation, and uninstrumented orchestration.
Complete job wall time is reported separately. First-rollout compilation is
included in phase totals; warm throughput summaries explicitly exclude it.

After candidate selection, confirmations save final checkpoints. Separate
zero-training evaluation jobs load these checkpoints for the final test sets.
All configurations and candidates are fixed before those test evaluations;
test outcomes cannot alter selection or training budgets.


Final packing decision: keep8192 tokens/GPU. At LR3e-6 with the same batch
shape and seed, mean warm actor throughput (excluding rollout0) was42,396
versus38,879 tokens/s on GSM8K and22,869 versus15,936 on Countdown for8192
versus16384. The respective peak allocations were54.75/82.73GiB and
52.62/78.40GiB. These are single-pilot measurements with generated lengths
that differ; they justify a conservative setting, not a precise causal speedup.
CUDA graphs capture up to512 active requests throughout the comparison.

Four identical two-GPU B200 workers became available at tier2. Every run
still uses one actor GPU and one SGLang GPU. The22 screening runs are shuffled
with seed1729, then assigned round-robin across these workers, so hardware and
wall-time order are not confounded with a single task or batch family.
Screening budget remains16384 trajectories/run; evaluation every4096 samples
on512 validation problems. Confirmation remains32768 trajectories/run, three
fresh seeds, baseline plus two candidate recipes per task. No budget reduction.

Before screening, W&B logging additionally gains explicit train/samples and
eval/samples axes and mixed-group fractions, so dashboard comparisons align
consumed trajectories. This changes only logging, with original train/step
and eval/step values retained for audit.


Preflight complete: both saved pilot checkpoints reload successfully after
setting an unused positive LR-decay length for evaluation-only jobs. Countdown
reload reproduces24/128; GSM8K reload yields112/128 versus110/128 at save time.
All evaluation-only jobs contain zero training rows. New-worker warmups each
complete1024samples/eightupdates, with all W&B records verified. Same-seed
first-rollout prompt order is identical across LR, packing, and warmup pilots.
The screening plan is now fixed and running; no test set has been evaluated.


Source-audit clarification while screening: in the pinned native loss,
train_rollout_logprob_abs_diff compares trainer-rescored old-policy log-probs
with inference log-probs, not the changing current inner-step policy. Interpret
it as an engine/trainer mismatch diagnostic. PPO KL, clip fraction, and ESS
reflect inner-step drift. No training configuration changes follow this audit.
See SOURCE_NOTES.md for pinned source locations and the G=1 centering caveat.


Frozen-label audit during screening: a deterministic spot check of three
newly credited validation responses on each task found one incorrect GSM8K
reference (ID29cd6c85bd87ab4143da). Its30-minute head start and15/45mph speeds
imply15minutes, whereas the source reference uses20minutes and labels10.
The primary study keeps frozen labels; the issue and any sensitivity analysis
are reported separately. One identified item can change validation accuracy
by at most1/512(0.195percentage points). This small spot check does not estimate
label-error prevalence. The other two inspected GSM8K and three Countdown
answers pass arithmetic checks. See known_label_issue.json and grader cases.


Exploratory amendment after18/22screen runs completed, before any test data:
Countdown baseline P16/G8/B128/S1 peaked at16.80% validation and ended13.28%,
while its P64/G8/B512/S1 fixed-LR arm ended22.85%. One idle worker will test
P16/G8/B128/S1 at LR1e-6 for16384responses with the same seed0/validation512.
This checks dependence on LR and possible late degradation; it is not part of
the original22-arm screen and does not change those settings or results.
The control is eligible for subsequent fresh-seed confirmation if it warrants
selection on validation. lr-followup-plan.json records it prospectively.


Confirmation selection frozen after all22screen runs and the exploratory LR control completed, before any test evaluation. GSM8K: baseline, P64/G8/B512/S1 at LR3e-6, and P64/G8/B256/S2 at LR3e-6*sqrt(2). Countdown: baseline, the same P64/G8/B512/S1 candidate, and P128/G4/B128/S4 at LR3e-6. selection.json records numerical reasons and alternatives. Selection balances validation endpoint/AUC/time and distinct update families; it is not the top two endpoint values. All18fresh-seed runs retain32768responses; all candidates and budgets are fixed before held-out tests.
