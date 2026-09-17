# Qwen3.5-4B: GRPO batch configuration on two math datasets

These are two completed learning trajectories of one configuration, plus
systems pilots. They cannot rank batch alternatives or estimate training-seed
variation. `main_runs.json` identifies the trajectories used in the article.
The interrupted first DAPO attempt is reported separately.

The public bundle contains numerical records and training code. Trained
checkpoints remain in the study's private storage; readers can recompute the
reported analyses from the saved outcomes and rerun the pinned training recipe.

## Measured configuration

Both tasks use `128 prompts × 8 responses = 256 responses/update × 4 updates`.
Each of 128 rollouts generates 1,024 fresh responses. The budget is 131,072
responses and 512 Adam updates. LR is 1e-6; the response cap is 16,384 tokens;
the dynamic packing target is 8,192 tokens per GPU. This trains the full text backbone with GRPO and sequence-mean loss; it does not train the vision encoder or reproduce the complete DAPO algorithm.

- GSM8K: two nodes, four training B200s (TP1/DP4) plus four TP1 generation engines.
- DAPO: allocation changed during recovery. The final segment used twelve B200s across two nodes: eight training GPUs (TP1/DP8) plus four TP1 generation engines. The recovery table below records every allocation and retained checkpoint.

The runtime and model revisions are in `runtime_pin.json`. Exact measured
scientific hooks are in `measured-code/`. Native resolved settings and portable
argument lists are alongside each main run. Portable argument lists remove
private W&B routing and managed-cluster placement, and replace filesystem roots
with `{hf_model}`, `{initial_checkpoint}`, `{data}`, and `{checkpoints}`.
Supply those roots and your own cluster/tracking settings before launching the
pinned Miles runtime. Infrastructure is not included; these are the measured
arguments, not a claim that an unpinned installation will behave identically.

## Data reproduction

Use Python with `datasets`, `huggingface_hub`, and `pyarrow` installed. The
runtime already provides PyTorch and Ray; do not replace them with arbitrary
versions when reproducing GPU jobs.

```bash
python reproduce_math_data.py \
  --manifest reproduction-data-manifest.json \
  --ordered-ids ordered-split-ids.json \
  --output data
```

All six files must match the manifest's SHA256 hashes. The DAPO source is large:
1,791,700 rows representing 17,188 whitespace-normalized prompt groups.
Twelve conflicting-label groups were removed before the split. Raw original
prompt text is preserved for the selected representative of every group.

The GSM8K split uses seed 1729, and DAPO uses seed 20260915. An earlier generic
manifest field incorrectly described both with the latter seed; ordered IDs
and hashes define the measured files and the correction is recorded here.
GSM8K test is the official 1,319-question split. DAPO test is our own disjoint
1,024-question holdout, not an official benchmark.

## Numerical analysis

Install `numpy`, then run:

```bash
python analyze_main.py --root artifacts/results --data data \
  --output reconstructed-main-analysis.json
python analyze_terminal_tests.py gsm8k
python analyze_terminal_tests.py dapo
```

The programs check the batch equality, response/token/update accounting,
validation IDs, individual correctness, checkpoint eligibility, and the absence
of optimizer updates in test jobs. Prompt-level files retain IDs, correctness,
lengths and completion status; generated response text is omitted. Their hashes
therefore describe this public numerical representation. Original source-file
hashes are preserved separately and refer to the measured private run files.

`main-analysis.json` includes response, output-token, optimizer-step, elapsed
job-time and measured active-time axes. `test-analysis.json` includes the paired
20,000-resample prompt bootstrap. This interval is conditional on the actual
trained models and does not estimate training-seed variance.

`evaluation-caveats.json` records post hoc output inspection. The frozen grader
can credit a numeric answer in a response that later hits the token cap. We
therefore additionally report `completed_response_sensitivity`, treating every
unfinished response as incorrect; this changes no primary scores or checkpoint
selection. This removes six credited GSM8K endpoint answers and two selected
checkpoint answers. One incorrect validation label is documented in each task,
with the frozen source labels retained. Spot checks do not estimate label-error
prevalence. Non-thinking chat mode is a template setting and did not prevent
long reasoning or generated thinking delimiters.

On DAPO this sensitivity check removes two credited initial-model answers and
none from either trained checkpoint. The positive endpoint and selected gains
persist. The selected checkpoint's advantage over the endpoint remains
uncertain: their paired interval includes zero, and selected answers are longer.

W&B histories were independently queried and compared with local metric rows.
One discarded update from the colocated OOM attempt (step369) reached the local
metric log but not the original W&B SDK file or server. Its audit keeps
`all_match: false` and identifies the missing step. The attempt retained no
updates; all overlapping values match. The exporter permits a partial
interrupted-segment audit only when every retained value is verified at the
exact retained checkpoint boundary. Final runs and tests require full matches.
No missing W&B value was reconstructed or uploaded as original telemetry.
Public `wandb-audit.json` files retain the counts and match checks while removing
private host and account identifiers. Metrics are in `metrics.json.gz` for each
run. `sha256.json` inventories the exported files; it does not hash itself.

## Selection and recovery

Selection uses validation only at the initial model and saved checkpoints at
32,768 / 65,536 / 98,304 / 131,072 responses. Ties favor earlier checkpoints.
Initial and endpoint models are tested regardless of which checkpoint wins.
If selection chooses another checkpoint, it receives a separate test job.

GSM8K completed uninterrupted. DAPO required several recoveries; the table
summarizes retained exposure, with complete details in each run's numerical
records and the linked amendments. A segment can retain no new work while
still contributing discarded responses and elapsed time.

| DAPO segment | Allocation | Retained through | Reason for transition |
|---|---|---:|---|
| main-v2 | 8 training + 8 generation GPUs | 32,768 responses | Generation worker lost |
| resume1 | 8 + 8 | 32,768 | Training worker lost before saving |
| resume2 | 4 + 4 | 36,864 | Planned move to available capacity |
| resume3 | 8 + 4 + 4 | 94,208 | Generation worker lost |
| resume4b | 8 + 4 | 94,208 | Training worker lost before saving |
| resume5b | 4 shared GPUs, CPU offloading | 94,208 | GPU out-of-memory before saving |
| resume6 | 8 training split across two nodes, 4 generation | 94,208 | Stopped to correct placement |
| resume6b | 8 + 4, training kept on one node | 96,256 | Planned generation expansion |
| resume7 | 8 + 4 + 4 | 100,352 | Generation worker lost after durable save; no completed work lost |
| resume8 | 8 + 4 + 4 | 100,352 | Replacement lost during first rollout; no completed batch |
| resume9 | 8 + 4 | 131,072 | Completed the fixed budget on surviving workers |

Across these rewinds, **11,264 completed responses, 94,301,425 output tokens,
and 35 completed optimizer updates are discarded**. They remain in execution
cost and their original logs; they do not enter the retained learning curve.
The planned capacity moves retained all completed work. Resume8 ended during its first rollout, before any complete-batch records; individual partial responses and tokens were not persisted and are not included in these counts. Its elapsed time remains in execution cost. Recovery restored
weights, Adam, the scheduler, trainer random state, data position, and cumulative
response/token counters. Changes in reductions and sampling do not promise a
bitwise-identical path to an uninterrupted run.

An earlier main-v1 attempt lost its training node after 10,240 responses and
2,561 seconds, before a recovery checkpoint existed. Its 11.38 allocated
GPU-hours are additional cost; main-v2 restarted from initial weights. Two
startup attempts also made no responses or updates: resume4 failed its
allocation check after 15.99 seconds, and resume5 failed during engine startup
after 72.91 seconds because CPU offloading rejected expandable memory segments.
Their costs are separate from the segments above. Another eight-GPU worker
was lost during preparation before a training run began. Observed worker
deletions do not establish their underlying cause.

Disabling expandable segments allowed the colocated retry to start, but it
ran out of GPU memory during its third update. We returned to separate training
and generation GPUs with the established allocator settings. No split-size
allocator tuning was applied. A placement audit then caught Miles sorting
bundles by node IP, which split training across two unshared filesystems.
That attempt completed a rollout and no updates; its work was discarded.
The corrected placement keeps all eight trainers and checkpoint shards on one
host. Four additional generation GPUs later expanded the run to sixteen GPUs across three nodes. Two subsequent generation-worker losses led back to the twelve surviving GPUs. The learning recipe stayed fixed. A request for an eight-GPU generation replacement, which would preserve two nodes, remained unschedulable for about five minutes and was canceled while the twelve-GPU run continued.

Recovery saves progressed from every 8,192 to every 4,096 and finally every
1,024 responses as worker losses continued. Extra checkpoints never became
selection-eligible. Complete job time sums every listed segment, including
its discarded work. Elapsed time including recovery is also reported. Some
interrupted durations have infrastructure-derived bounds; cost uses their
upper bounds. Setup between jobs is additional cost.

The original frozen protocol is in `MAIN_PROTOCOL.md`. Operational changes are
recorded in `RECOVERY_AMENDMENT.md`, `RESUME_AMENDMENT.md`,
`SECOND_RESUME_AMENDMENT.md`, `CAPACITY_MIGRATION_AMENDMENT.md`,
`FOURTH_RECOVERY_AMENDMENT.md`, `FIFTH_RECOVERY_AMENDMENT.md`,
`SECOND_CAPACITY_MIGRATION_AMENDMENT.md`, and
`THIRD_CAPACITY_MIGRATION_AMENDMENT.md`, `SIXTH_RECOVERY_AMENDMENT.md`, and
`SEVENTH_RECOVERY_AMENDMENT.md`. The corresponding interruption,
placement, and migration JSON files retain exact counts and timing evidence.

The original numerical hooks are in `measured-code/`. For the corrected
heterogeneous placement, put `measured-code/training-placement` before the
original hooks on `PYTHONPATH`. Set `STUDY_TRAINING_NODE_IP` to the eight-GPU
trainer node and `STUDY_HETEROGENEOUS_ROLLOUT=1`; retain
`study_hooks.setup_worker` as Ray's worker setup hook. This orders GPU bundles
and groups TP1 serving engines by their actual hosts before allocating ports.
It does not change rewards, loss, or sampling. The earlier port-only hook
remains in `measured-code/capacity-migration` for historical reproduction.
Portable CLI arguments alone do not enable these infrastructure hooks.

## Systems measurements

`tp-dp-final.json` summarizes eight arms replaying the same 256 responses for
four updates from initial weights. Warm phase means exclude the first phase.
Three warm phases within one job are not independent benchmark repetitions.
Replay results are never counted as RL learning outcomes. Inference scaling,
mode/cap comparisons, and exact replay token counts are separate artifacts.

Complete job time includes initialization, validation, checkpoint creation and
archival. Measured active time sums completed instrumented generation, training and
recorded weight transfer; uninstrumented orchestration and incomplete phases
without an end record are excluded. Complete job time still includes them. Allocated
GPU-hours include the idle role in synchronous generation/training and use the recorded allocation with duration upper bounds for interrupted jobs. The execution table includes its listed interruptions. Separate setup failures, the abandoned main-v1 attempt, pilots, and terminal test jobs are additional costs.

Checkpoint upload concurrency increased from 10 to 32 after one verified copy
probe. The next three native archives averaged 82.06 seconds, versus 85.83
seconds for four preceding archives on the same training node. We kept the
modest change. These sequential observations do not isolate a causal speedup;
the standalone copy probe and native uploads also have different timing scopes.
`checkpoint-transfer-native-check.json` contains each timing, and the checkpoint
pointer still reaches storage only after all recovery state is uploaded.

`make_article_figures.py` uses `matplotlib` and `Pillow` to reproduce standalone
SVG plots and the social card. Pass `--systems tp-dp-final.json` and
`--main-json main-analysis.json` from this directory.

See `MAIN_PROTOCOL.md` for the original frozen protocol, the recovery amendment
for subsequent operational changes, and `SOURCE_NOTES.md` for corrections to
common literature and Miles implementation claims. Earlier Qwen2.5 experiments
remain archived in the separate `grpo-batch-knobs` directory and do not establish
a result for Qwen3.5-4B.
