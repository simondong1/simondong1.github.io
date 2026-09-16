# Qwen3.5-4B: two math learning runs

Frozen before either main learning run. All preparatory checkpoints are discarded.
Only Qwen/Qwen3.5-4B is used. Runtime and model revisions are in runtime_pin.json.

## What this experiment can answer

One training trajectory per task checks whether a literature-informed recipe
learns stably on GSM8K and DAPO-MATH-17K at a stated budget. It does not rank
four-knob alternatives, estimate seed variation, or establish an optimum.
Separate fixed-trajectory and inference pilots inform GPU allocation and
packing. Replay measurements are never counted as learning results.

## Fixed algorithm and accounting

Both runs use prompts P=128, completions/group G=8, optimizer batch B=256,
and four disjoint optimizer updates per rollout: 128*8=256*4=1024 responses.
Budget: 128 rollouts, 131072 training responses, 512 Adam updates per task.
Each generated training response is used once within its rollout. Repeated
prompts on later dataset epochs are allowed and receive fresh generation.

LR1e-6 constant; Adam betas0.9/0.98; weight decay0.1; gradient clip1.0;
GRPO centered, std-normalized group advantages; sequence-mean loss;
lower/upper policy clipping0.2/0.28; no entropy bonus or reference KL.
No dynamic filtering, async policy rollout, token reuse or inner epochs.
This is GRPO on the DAPO dataset, not the complete DAPO algorithm.

Use non-thinking chat mode while explicitly requesting step-by-step solutions;
response cap16384; rollout temperature1.0. This choice is supported by small
held-out mode/cap checks, not a claim that non-thinking always wins.
Training seed3100 and rollout seed4100. Generation and BF16 kernels can vary.

Dynamic token packing8192, TP1, PP1, CP1, no activation recomputation.
The packing allowance is a target for combining sequences; longer individual
responses remain intact. A real long-response probe passed with observed
main-rank peak allocation131.4GiB and was faster than packing16384.
SGLang TP1 per engine, radix cache disabled to enable overlap scheduling,
static memory fraction0.65, max running requests256 and CUDA graph max256.

## Hardware

Tier-2 only; capacity checked before every launch. Four separate physical
nodes supplied24 B200 GPUs during pilots. An extra eight-GPU request was
unschedulable and cancelled. Before the main launches, an allocated8-GPU
worker received SIGTERM after the worker-group limit was reduced; a replacement
is pending. GSM8K retains its eight GPUs. DAPO awaits its second eight-GPU
node, with any topology revision recorded before its launch. Planned placement
is synchronous and disaggregated:

- DAPO: two8-GPU nodes; eight training GPUs DP8, eight TP1 rollout engines.
- GSM8K: two4-GPU nodes; four training GPUs DP4, four TP1 rollout engines.

Training remains within one node. Weights cross nodes once per rollout.
Allocated GPU time includes the idle role during the other role's phase.
Startup, generation, optimization, weight transfer, evaluation and checkpoint
costs are reported separately where recorded, plus complete elapsed job time.

Evidence for allocation: on the same256 trajectories, warm training phase
means were119.14s(DP1),60.09s(DP2),30.95s(DP4),17.70s(DP8). TP2/DP4
needed25.82s on eight GPUs. DP initial gradient norms agreed within0.006%;
TP4 differed about3.2%. No claim of exact numerical equivalence is made.
A1024-response inference probe took approximately143s on four engines and
101s on eight, with different stochastic outputs. The four-engine arm shared
CPU/node resources with a probe on disjoint GPUs. These are single trials,
not repeatability estimates. Complete jobs took239.8s and235.6s because
initialization dominated this short comparison.

## Data, evaluation and selection

Use the pinned official sources and disjoint, deduplicated splits from
math_data_manifest.json:

- GSM8K: train6961, validation512, official test1319.
- DAPO: train15640, validation512, held-out test1024. The official source
  contains repeated rows; deduplication and conflicting-label exclusions are
  documented in the manifest. This held-out subset is not an official DAPO
  test benchmark and is not evidence of generalization to all math tasks.

The22-case math verifier suite must pass. Preserve native DAPO Answer:
format and GSM8K numeric answer extraction. Exclude thinking-only answers.
Validation uses one greedy response per prompt (temperature0), the same
non-thinking mode and16K cap. Evaluate before training and every8192
training responses, including the131072 endpoint. Never select on test.

Save weights, Adam, trainer RNG and data-source state every32768 responses.
Select the best *saved* checkpoint using its512-prompt validation accuracy;
resolve ties in favor of the earlier checkpoint. The initial model is also
eligible. Report the fixed-budget endpoint separately from this selection.
After training, evaluate the initial model, endpoint, and selected checkpoint
(if different) once on the full test split. Use paired prompt bootstrap
intervals for accuracy changes; sampling variability and training-seed
uncertainty are distinct. Do not treat512 optimizer steps as independent seeds.

## Monitoring, persistence and conclusions

Log W&B plus append-only local records with response/token/update axes.
Independently compare persisted W&B history with local records. Save source
hashes, resolved arguments, runtime pins and physical allocation records.
Archive live metrics and complete checkpoints to the private study S3 prefix;
checkpoint pointers are copied only after weights and continuation state.

Monitor reward, mixed-group fraction, truncation, loss/gradients, clipping,
PPO KL and ESS by position within each rollout. Trainer/rollout logprob
mismatch is an inference-engine diagnostic, not inner-update policy drift.
Nonfinite gradients, failed updates and resource failures require repair and
explicit recording. Do not silently modify the learning recipe mid-run.
Resumption preserves Adam and the constant LR schedule; it is not a claim
of bitwise identical SGLang generation after restarting processes.

The131072-response endpoint is the primary budget. Inspect late validation
trends and report whether longer training appears warranted; plateau claims
require evidence. No unplanned hyperparameter search or seed-ranking claims.

Public artifacts exclude credentials, private node identities and internal
infrastructure configuration. The article links primary sources and separates
literature heuristics, systems measurements and the two learning outcomes.
