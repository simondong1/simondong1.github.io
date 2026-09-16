# Scope of the new evidence

Only Qwen/Qwen3.5-4B, released March 2, 2026, is used in this study. Old
Qwen2.5 training measurements are historical and do not establish a winner
for the new model. The authorized scope is two main runs, one on GSM8K and
one on DAPO-MATH-17K, plus bounded performance pilots. Two runs cannot rank
four-knob alternatives or estimate variation across training seeds.

The [archived source audit](https://simondong1.github.io/experiments/grpo-batch-knobs/PROTOCOL.md) covers R1, DAPO,
DeepSeekMath, GSPO, SAPO, Hunyuan batch scaling, ScaleRL, BroRL, and the
IsoCompute Playbook. Those paper/source checks are reusable; their older
training results are not new-model evidence.

Runtime: Miles 7e03b728faf9ab861741a4a8535a72292e481cb7, matching the cached
SGLang 0.5.13 stack. Native Qwen3.5-4B example:
https://github.com/radixark/miles/blob/7e03b728faf9ab861741a4a8535a72292e481cb7/scripts/run-qwen3.5-4B.sh
It uses P32/G8/B256 (one update), LR 1e-6, and an 8192-token response cap.
Its broad process-kill commands are not used in this managed workspace.

Corrections to common heuristics:

- With group size G fixed, P and B remain two independent choices. S=P*G/B
  follows, subject to integer divisibility. There is not just one degree of
  freedom among all three.
- The pinned Miles parser derives B using floor division when S is given.
  It is not a general exact 'set any three' solver. Our wrapper asserts the
  exact four-knob equality and the required data-parallel divisibility.
- G=1 gives zero centered GRPO advantage; disabling standard deviation
  normalization alone does not make it a useful GRPO configuration.
- `train_rollout_logprob_abs_diff` compares trainer-rescored old-policy
  log-probabilities with rollout log-probabilities. It measures engine
  disagreement, not policy drift over inner steps. Use clip/KL/ESS for drift.
- No data filtering, sample reuse, reference KL, or asynchronous generation
  is enabled. Sequence-mean loss is held fixed. Using the DAPO dataset and
  upper clip 0.28 does not reproduce the complete DAPO algorithm.
- Hunyuan's infrastructure includes asynchronous partial rollouts. Its
  square-root LR result is not proof about every synchronous GRPO workload.
- No primary evidence establishes a universal TRL 'more than 2–4 epochs is
  unstable' rule or the supplied Turn-PPO attribution; omit those claims.

Hardware evidence:

The pinned Qwen hybrid-attention implementation duplicates its linear
attention blocks across tensor-parallel ranks (`hf_attention.py` and
`qwen3_5.py`). More TP therefore does not divide all model work or memory.
The identical-trajectory benchmarks favored data parallelism when one complete
model fit: warm training phases took 17.70s with DP8 versus 25.82s with TP2/DP4,
and 30.95s with DP4 versus 48.26s with TP2/DP2. Each is a mean of three phases
within one job, not three independent repetitions. Startup/cache state differed
across jobs, so complete short-job totals do not isolate topology.

Report complete elapsed job time separately from warm training time,
generation, weight transfer, evaluation, and checkpoint overhead. Initial
compiler failures are setup costs, not learning runs or usable speed data.

Model training scope:

The resolved runtime uses `megatron_to_hf_mode=raw`, the native GPT model
provider and the Qwen3.5 decoder specification. This trains the full text
backbone, not the vision encoder. The pinned model provider constructs
`GPTModel`; the Qwen3.5 bridge maps its decoder and embeddings to the HF
`model.language_model` weights and language-model output head. The tasks
contain text only. Source paths at the pinned Miles commit:
`miles/backends/megatron_utils/model_provider.py` and
`miles_plugins/mbridge/qwen3_5.py`.
