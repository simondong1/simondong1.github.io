# Miles four batch knobs: reproduction artifacts

Companion to [How to Set the Four GRPO Batch Knobs](https://simondong1.github.io/grpo-batch-knobs.html).

The study trains Qwen2.5-1.5B-Instruct on GSM8K and Countdown using one
training GPU and one SGLang generation GPU per run. It tests
`prompts × responses_per_prompt = responses_per_update × updates_per_rollout`.
Every generated response is used once, with no replay or dynamic filtering.

## Evidence

- `PROTOCOL.md`: prospective comparison, pilot decisions, and declared amendments.
- `screen-plan.json`: 22 single-seed screening runs at 16,384 responses each.
- `lr-followup-plan.json`: one exploratory learning-rate control, outside that grid.
- `selection.json`, `confirmation-plan.json`: validation-based selection, then
  three fresh training seeds at 32,768 responses per arm and held-out tests.
- `terminal-test-plan.json`: the two initial-model tests and 18 final-checkpoint
  tests, with zero training updates.
- `results.json.gz`: numerical learning curves, per-example correctness,
  response lengths, prompt IDs, group rewards, and optimizer diagnostics.
- `endpoints.csv`, `summary.json`: readable endpoint summaries and uncertainty.
- `data_manifest.json`, `model_manifest.json`, `environment.json`: pinned inputs
  and the measured package/repository versions.
- `known_label_issue.json`, `label_sensitivity.json`: one identified GSM8K
  validation-label error and its post-hoc sensitivity. Primary labels stay frozen.
- `SHA256.json`: checksums of the public files.

The numerical export omits machine identities, credentials, private storage,
and private W&B URLs. It retains generated task IDs and correctness, rather than
all generated response text. Per-run `source_hashes` identify the original
frozen training files; the original infrastructure-specific launcher is replaced
here by `launch.py` plus `training_arguments.py`. Rewards, batch validation,
and logging hooks are unchanged. Portable arguments were compared with all 61
planned configurations; only storage and tracking destinations differ. See
`portable-argument-check.json` for the scope of that check.

## Read the results without GPUs

Use Python 3.12 and `requirements-analysis.txt` in a CPU analysis environment.
The exact analysis/data-preparation versions are in `analysis_environment.json`;
GPU runtime versions are recorded separately in `environment.json`.

```sh
python final_summary.py --results results.json.gz --plan confirmation-plan.json --output rebuilt-summary.json
python make_figures.py --results results.json.gz --selection selection.json --assets figures
python make_screen_figures.py --results results.json.gz --assets figures
```

Curves use the actual validation timestamps, cumulative responses, or cumulative
generation + trainer + weight-sync time. Full job time includes initialization,
validation, checkpoint saving, and orchestration; a separate final test job is
reported separately. The bars on curves are observed training-seed ranges.
The paired test bootstrap resamples problem IDs jointly across seeds. Its
interval is conditional on those three trained seeds, not a population-level
confidence interval over arbitrary training runs.

## Reproduce training

The launcher assumes a working CUDA 13.0 Miles/Megatron/SGLang environment and
two allocated GPUs. This repository does not redistribute the original runtime
image or provision GPUs. Install the pinned revisions in `environment.json`
following the projects' installation instructions. The portable launcher has
argument-parity and verifier checks; a fresh external installation has not been
benchmark-validated. Different kernels or GPU models can change results/timing.
When creating a virtual environment over a preinstalled GPU stack, preserve it
with `uv venv --seed --system-site-packages`.

1. Check out Miles at `7e03b728faf9ab861741a4a8535a72292e481cb7`, Megatron-LM
   at `87d1155be008e09ae667dac91020bb7b87238c22`, and SGLang at
   `25efacbe2c26acfed7302a322020dea82eaeff5b`. Set `MILES_PATH` and
   `MEGATRON_PATH` to the first two directories. Expose only your two allocated
   GPUs through your scheduler or container.
2. Run `python prepare_data.py` here. It uses the pinned HF revisions, creates
   the study's splits, and writes a manifest; compare its hashes with the
   published `data_manifest.json`. Dependencies: `datasets`, `huggingface_hub`.
3. Download `Qwen/Qwen2.5-1.5B-Instruct` at revision
   `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` to `model/`. Use Miles'
   `tools/convert_hf_to_torch_dist.py` with its `scripts/models/qwen2.5-1.5B.sh`
   arguments and **`--rotary-base 1000000`** to create `initial_checkpoint/`.
   `STUDY_MODEL` and `STUDY_INITIAL` can override these paths.
4. Create a local Ray cluster within your allocation, or supply `RAY_ADDRESS`
   for an existing compatible cluster. The experiment and data paths must be
   visible at the same locations to its workers. Set `STUDY_ROOT` to this
   directory if your current working directory differs.
5. Select one config from a plan and inspect its launch arguments:

   ```sh
   python - <<'PY'
   import json
   from pathlib import Path
   plan = json.loads(Path('screen-plan.json').read_text())
   config = next(c for c in plan['runs'] if c['task'] == 'gsm8k' and c['arm'] == 'split-4')
   Path('run.json').write_text(json.dumps(config, indent=2))
   PY
   python launch.py run.json --dry-run
   python launch.py run.json
   ```

   W&B defaults to offline. For online tracking, authenticate with your own
   W&B account and set `WANDB_MODE=online`, `WANDB_ENTITY`, and optionally
   `WANDB_BASE_URL`. Never put credentials in experiment configs.
6. Run configs serially on each two-GPU allocation. Give retries new names;
   existing results are not overwritten. Confirmation terminal-test configs
   in `terminal-test-plan.json` load `checkpoints/<training_run>` and contain
   zero training responses. Select and write a test config just as in step 5,
   after its corresponding confirmation run has saved its final checkpoint.
   Test data must remain unused until candidate selection and budgets are fixed.
7. Run `python validate_run.py results/RUN_NAME`. Export raw local
   records with `python analyze.py --root results --output local-results.json.gz`.

The public helper's hashes differ from the infrastructure launcher. Do not use
the archived original queue hash to validate a newly adapted launcher. Freeze
and record your own launch sources before making comparisons.

## Scope

This is full-parameter BF16 training with a 1,024-token response cap, binary
correctness, sequence-mean GRPO loss, fixed clipping 0.2/0.28, and no reference
KL penalty or entropy bonus. It compares two mathematical task types on a
small instruct model over a short horizon. It does not establish a universal
batch optimum for frontier models, long reasoning, coding, or asynchronous RL.
