# Source corrections for the article

Runtime Miles commit: `7e03b728faf9ab861741a4a8535a72292e481cb7`.
Use pinned GitHub links, not the newer workspace checkout.

- `miles/utils/arguments.py:2317–2325`: with S supplied, B is calculated as
  `P * G // S`. It checks an explicitly supplied B against this floored value.
  This does not implement a general exact "set any three" solver. Our wrapper
  rejects non-divisible shapes and inconsistent four-tuples before launch.
- `miles/ray/rollout/train_data_conversion.py:108–117`: reshape rewards by G,
  subtract each group mean, optionally divide by its standard deviation.
  G=1 with this default centering gives zero policy advantage, even though
  the parser disables standard-deviation normalization. A one-response
  algorithm needs a different baseline/objective; it is not this GRPO arm.
- `miles/backends/training_utils/loss_hub/losses.py:306–325`: the logged
  `train_rollout_logprob_abs_diff` compares **old_log_probs**, rescored by the
  trainer before updates, with rollout engine log-probabilities. It does NOT
  compare the current inner-step policy with the rollout policy. In our runs
  `use_rollout_logprobs=False`; this is an inference/trainer mismatch diagnostic.
  Use the PPO ratio KL/clip/ESS metrics to assess drift across inner steps.
  Different slices can have different anchor mismatch without drift causing it.
- Native loss reduction is `sum_of_sample_mean`: equal sequence weighting,
  token-mean contribution inside each sequence. We hold it fixed; this is
  not DAPO's full token-mean/filtering recipe merely because clip-high is0.28.
- BF16 parameters, FP32 master parameters, distributed optimizer, no activation
  recomputation, and use_rollout_logprobs=False verified in actual parsed logs.

Primary paper evidence and URLs are in PROTOCOL.md/research_sources.json.
Unsupported Turn-PPO attribution and a universal TRL "more than2–4 epochs is
unstable" rule must not be asserted. Group8–16 and B512 are starting examples,
not universal SOTA prescriptions. Hunyuan uses asynchronous partial rollouts;
its one-update batches are not a reproduction of our synchronous setup.

Sample standard deviation: the pinned reward conversion uses torch.std with its
default correction=1, after group centering, then divides by std+1e-6. Group-size
comparisons therefore also change this normalization factor; they do not isolate
prompt coverage alone.
