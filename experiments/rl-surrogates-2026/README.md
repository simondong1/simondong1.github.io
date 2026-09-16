# RL surrogate comparison, September 2026

Companion to [RL Losses in 2026: From PPO to SAPO](https://simondong1.github.io/rl-surrogates-2026.html).

The study compares **PPO clipping, binary-TV DPPO, and GLM-5 asynchronous token
rejection**. All use the same GRPO advantages and sequence-mean reduction. This
is a surrogate ablation, not a reproduction of complete lab training recipes.

Training is **full-parameter** on the Qwen3.5-9B language policy. No adapters,
quantization, critic, reference KL, or entropy bonus. The text-only experiment
does not train the checkpoint's unused vision/MTP components.

## Files

- `PROTOCOL.md`: decisions and pilot corrections recorded before the final study.
- `study-plan.json`: exact run matrix, order, counts, and historical deadline.
- `surrogates.py`: differentiable objectives and the Miles custom-loss hook.
- `rewards.py`: restricted arithmetic and exact integer verifiers; no generated code execution.
- `prepare_data.py`, `data-manifest.json`: public dataset revisions, deduplication,
  fixed splits, and SHA-256 checksums.
- `launch.py`: one full-parameter FSDP2 run inside the container.
- `run_study.py`: sequential queue with a deadline and training-code hash checks.
- `analyze.py`: exports measured records and checks matched configurations and prompt order.
- `make_figures.py`: creates standalone learning curves and paired task summaries.
- `audit_checkpoint.py`: checks full-policy optimizer coverage and weight changes across all decoder layers.
- `audit_logprobs.py`: measures initial trainer/inference log-probability tails from every saved training rollout.
- `audit_study.py`: runs both audits on CPU for completed arms, optionally watching the queue.
- `test_surrogates.py`, `test_native.py`: analytic gradient/reward checks and comparison
  with Miles' native PPO implementation.
- `source-revisions.json`: pinned framework and recipe sources.

## Environment

Only two allocated NVIDIA B200 GPUs were exposed to the container. The model and
training workers share those GPUs; rollout and training phases are synchronous.

- Miles: `2ef603a8ca44410beaad8c1dda5bf85f2a9d2533`
- Container: `radixark/miles@sha256:16dc2819ed890455d54102fb88f3cdffdd210104b1707583051582af46bb6c5b`
- PyTorch: `2.13.0+cu130`
- Transformers: `5.12.1`
- Ray: `2.58.0`
- SGLang: `0.5.20.dev58+gaea7fb9`
- Model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`

All Hugging Face inputs are public; downloads explicitly disable implicit authentication.

## Reproduce

Allocate two GPUs and a working directory with about 200 GB for a single arm,
or about 2 TB for the full queue with optimizer checkpoints retained.
The original study used `/tmp/rl-surrogate-2026` for large artifacts. Keep this
repository checkout elsewhere and mount its experiment directory at `/experiment`.

The setup helper performs steps 1–5 below and refuses to replace an existing
container. Supply your own allocated GPU UUIDs:

```sh
python -m pip install huggingface_hub pyarrow requests matplotlib numpy
python setup.py --root /tmp/rl-surrogate-2026 --gpus GPU_UUID_0,GPU_UUID_1
```

The manual steps are included for inspection:

1. Clone Miles at the revision above into the artifact root's `miles/` directory.
2. Download the pinned HF model into the root's `model/` directory using
   `huggingface_hub.snapshot_download(..., revision=..., token=False)`.
3. Run `python prepare_data.py --out /tmp/rl-surrogate-2026/data` and compare the
   generated manifest with `data-manifest.json`. DeepMath uses only its first
   source shard, restricted to integer final answers. These are custom study
   splits, not the datasets' official benchmarks.
4. Start the pinned image with **only your two allocated GPU UUIDs**, a 64 GB
   private shared-memory segment, working directory `/work/miles`, and mounts:

   | Host path | Container path |
   |---|---|
   | artifact root | `/artifacts` |
   | pinned Miles checkout | `/work/miles` |
   | pinned model directory | `/model` (read-only) |
   | this experiment directory | `/experiment` |

   Name the container `rl-surrogates-2026` for the default controller. Do not expose
   all GPUs on a shared node. Keep `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` in its environment.
   The container's idle command can be `sleep infinity`.
5. Inside it, run the checks:

   ```sh
   PYTHONPATH=/experiment:/work/miles python /experiment/test_surrogates.py
   PYTHONPATH=/experiment:/work/miles python /experiment/test_native.py
   ```
6. From the host, launch a single primary run with:

   ```sh
   docker exec -e PYTHONPATH=/experiment:/work/miles rl-surrogates-2026 \
     python /experiment/launch.py --name countdown-ppo-s42 \
     --algorithm ppo --dataset countdown --seed 42 \
     --rollouts 32 --prompts-per-rollout 32 --max-tokens 1024 --eval-interval 4
   ```

   Replace only `--algorithm` and the run name for a surrogate comparison.
   DeepMath uses `--dataset deepmath --max-tokens 2048 --rollouts 28` in all of
   its arms: 448 updates and 3,584 responses, versus Countdown's 512 updates and
   4,096 responses. This common DeepMath duration was revised from timing
   measurements to preserve all two-seed comparisons within the allocation.
   The launcher records the entire command and argument configuration.
7. For the full queue, copy `study-plan.json` and set `deadline_utc` to a new UTC
   deadline. Run on the host:

   ```sh
   python run_study.py --plan YOUR_PLAN.json --root /tmp/rl-surrogate-2026
   ```

   The controller restarts only the named experiment container between arms.
   It stops on an error rather than silently skipping a failed run. It checks
   hashes of the model-training, surrogate, reward, and metric-hook code before
   each new arm; do not edit those files during a study.
8. Export measured records:

   ```sh
   python analyze.py --root /tmp/rl-surrogate-2026 --output results.json
   ```
9. Audit completed checkpoints and build the final figures from all 12 arms:

   ```sh
   python audit_study.py --root /tmp/rl-surrogate-2026
   python analyze.py --root /tmp/rl-surrogate-2026 --output results.json
   python make_figures.py --results results.json --assets ../../assets --summary summary.json
   ```

   The audit uses CPU loading inside the container and verifies that FP32 Adam
   state covers exactly 8,953,803,264 language-policy parameters, with weight
   changes in all 32 decoder layers. `--watch` can audit arms while the queue runs.
   Figure generation also writes `endpoints.csv` and `gate-audit.csv`.

The compact public results retain per-task IDs, rewards, lengths, and training
metrics. Local artifacts additionally retain generated responses, sampled token
IDs/log-probs, training dumps, TensorBoard events, exact commands, and full FSDP
checkpoints. Checkpoints are not stored in Git.

## Interpretation

PPO bounds are 0.8–1.2; binary-TV DPPO thresholds are 0.15/0.15; GLM-5 rejection
bounds are 0.5–5. These are different trust-region calibrations, not equally
strong regularizers or separately tuned optima. The same saved rollout log-prob
anchor is used by every arm. Six gates are audited on each run's identical token
inputs, but only its selected surrogate contributes to backpropagation.

Gate fractions are conditional on nonzero-advantage **sequence-normalized token
weight**, rather than raw token counts. This matches the fixed loss reduction.
A zero local gradient does not imply a parameter cannot move via other tokens or
optimizer momentum. Log-prob mismatch before updates is separated from policy
drift during the 16 minibatch updates.

Two training seeds and 256 test tasks per dataset make this an exploratory
comparison. The same test tasks are reused across seeds; pooling them as 512
independent examples would understate uncertainty. Report seed-specific results
and paired task differences. These holdouts were excluded from this experiment's
training split, but may have appeared in the base model's pretraining.

Greedy evaluation is not bitwise deterministic in this configuration. Every arm's
actual baseline is retained; the six same-checkpoint baseline evaluations per
dataset provide a repeatability check. Conditional task-bootstrap intervals do
not include independent training-seed or inference-runtime uncertainty. See the
protocol's repeatability note for the observation and configuration checks.

All implementations share a numerical log-ratio clamp at [-20,20]. Guard
activation is exported separately. For GLM, either guarded extreme is already
outside its rejection interval and contributes zero gradient. Counterfactual
weight diagnostics use guarded ratios and do not include the clamp's derivative
saturation; do not treat these weights as exact PPO/DPPO gradients at guarded
extremes. The protocol records the completed-run inspection and analytic checks.

`grader-cases.json` records a deterministic three-response spot-check of newly rewarded DeepMath outputs. It exposed an incorrect reference answer and a solution-method violation that the integer-only grader does not detect. The frozen comparison retains these labels for every arm. Grader success should not be read as proof of valid reasoning; the sample is too small to estimate label-error prevalence.

Truncation does not force zero reward: a valid final box can be accepted even if subsequent text reaches the cap. Also inspect log-probability tails rather than only their mean: Countdown GLM-5 seed 43 had six initial trainer/inference token differences above 5 nats among 4,039,764 positions, max 14.491, late in the already-degraded run. Their cause was not isolated. The exported records retain the tail measurements.
