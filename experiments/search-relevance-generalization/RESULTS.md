## Measured training results

**58 controlled quality runs complete**, plus systems checks. Results below are exploratory validation measurements from seed 173. They are not deployment recommendations or final-test results. Each row evaluates all 496,477 validation pairs unless identified as a pilot. Native five-class and native ten-class accuracies are not compared; the main table uses the registered common five-bin target.

| Reference comparison | Unique training pairs | Common-five accuracy | Common-five NLL ↓ |
|---|---:|---:|---:|
| Qwen3-0.6B, classification head | 250,000 | 79.48% | 0.5503 |
| Qwen3-0.6B, restricted token logits | 250,000 | 79.42% | 0.5512 |
| Qwen3-0.6B, full-vocabulary token | 250,000 | 79.41% | 0.5509 |
| Feature DCN, full data, two epochs | 8,953,912 | 73.21% | 0.7007 |

These reference rows are best-observed recipe comparisons with different input representations, training data counts, parameter counts, and compute. They do not isolate a single cause. The head-to-head LM comparison does freeze the backbone, examples, input serialization, batch order, and optimization recipe.

<details markdown="1"><summary>Complete experiment table</summary>

| Run | Unique pairs | Parameters | Updates | Common-five accuracy | Common-five NLL ↓ | Original MAE ↓ | Train / total seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| architecture-2m-cross4-ce10-2510u-s173 | 2,000,000 | 9,744,882 | 2,510 | 72.36% | 0.7295 | 0.7080 | 14.3 / 44.5 |
| architecture-2m-large-ce10-2510u-s173 | 2,000,000 | 56,620,878 | 2,510 | 72.42% | 0.7856 | 0.6741 | 22.2 / 59.4 |
| architecture-2m-mlp-ce10-2510u-s173 | 2,000,000 | 2,875,242 | 2,510 | 70.52% | 0.7730 | 0.7692 | 12.2 / 41.9 |
| architecture-2m-small-ce10-2510u-s173 | 2,000,000 | 1,250,062 | 2,510 | 70.96% | 0.7592 | 0.7628 | 12.7 / 41.7 |
| architecture-2m-wide-ce10-2510u-s173 | 2,000,000 | 17,763,726 | 2,510 | 72.58% | 0.7295 | 0.6933 | 13.7 / 46.0 |
| feature-2m-behavior_only-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 54.26% | 1.0944 | 1.1479 | 13.0 / 43.5 |
| feature-2m-lexical_only-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 63.32% | 0.9847 | 1.1053 | 13.7 / 43.0 |
| feature-2m-no_clicks-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.09% | 0.7352 | 0.7177 | 12.9 / 43.5 |
| feature-2m-no_corrected_behavior-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.99% | 0.7368 | 0.7218 | 12.4 / 42.9 |
| feature-2m-no_exposures-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.82% | 0.7415 | 0.7256 | 13.2 / 44.4 |
| feature-2m-no_legacy_behavior-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.31% | 0.7584 | 0.7500 | 13.4 / 44.2 |
| feature-2m-no_platform-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.08% | 0.7346 | 0.7174 | 13.0 / 43.9 |
| feature-2m-no_purchases-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.12% | 0.7339 | 0.7165 | 12.6 / 43.3 |
| feature-2m-no_similarity-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.10% | 0.7339 | 0.7177 | 13.7 / 43.5 |
| feature-2m-semantic_lexical-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 70.84% | 0.7695 | 0.7659 | 13.4 / 43.5 |
| feature-2m-semantic_only-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 59.82% | 0.9955 | 1.0234 | 13.8 / 43.2 |
| feature-2m-text_semantic_lexical-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 67.72% | 0.8470 | 0.8862 | 13.9 / 44.7 |
| label-1m-ce5-2510u-s173-r2 | 1,000,000 | 6,309,897 | 2,510 | 71.04% | 0.7995 | 0.7340 | 13.7 / 40.8 |
| label-1m-ce7-2510u-s173 | 1,000,000 | 6,309,963 | 2,510 | 71.27% | 0.7795 | 0.7221 | 13.7 / 39.7 |
| label-1m-components-2510u-s173 | 1,000,000 | 6,309,996 | 2,510 | 67.51% | 0.9357 | 0.7196 | 14.2 / 40.7 |
| label-1m-joint16-2510u-s173 | 1,000,000 | 6,310,260 | 2,510 | 71.36% | 0.7713 | 0.7178 | 12.9 / 40.3 |
| llm-qwen3-06b-classifier-250k-1ep-s173 | 250,000 | 596,057,095 | 1,954 | 79.48% | 0.5503 | 0.5361 | 353.6 / 599.4 |
| llm-qwen3-06b-restricted-250k-1ep-s173 | 250,000 | 596,049,920 | 1,954 | 79.42% | 0.5512 | 0.5416 | 353.0 / 597.7 |
| llm-qwen3-06b-token-250k-1ep-s173 | 250,000 | 596,049,920 | 1,954 | 79.41% | 0.5509 | 0.5414 | 354.0 / 601.4 |
| mix-500k-0-0-100-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 54.40% | 1.7012 | 1.1723 | 2.8 / 24.9 |
| mix-500k-0-100-0-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 65.63% | 0.8896 | 0.9339 | 3.0 / 25.2 |
| mix-500k-0-25-75-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 64.06% | 0.9928 | 0.9469 | 3.0 / 24.6 |
| mix-500k-0-50-50-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 65.41% | 0.9313 | 0.9165 | 3.0 / 25.4 |
| mix-500k-0-75-25-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 66.70% | 0.8848 | 0.8722 | 3.0 / 24.1 |
| mix-500k-100-0-0-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 68.83% | 0.8174 | 0.8344 | 3.1 / 25.4 |
| mix-500k-25-0-75-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 66.70% | 0.9181 | 0.8546 | 3.0 / 24.5 |
| mix-500k-25-25-50-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 67.72% | 0.8593 | 0.8241 | 2.8 / 24.7 |
| mix-500k-25-50-25-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 68.40% | 0.8343 | 0.8122 | 2.8 / 24.7 |
| mix-500k-25-75-0-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 67.94% | 0.8356 | 0.8517 | 2.8 / 24.6 |
| mix-500k-50-0-50-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 68.28% | 0.8500 | 0.8172 | 2.9 / 24.6 |
| mix-500k-50-25-25-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 68.96% | 0.8170 | 0.8079 | 3.0 / 24.6 |
| mix-500k-50-50-0-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 68.36% | 0.8261 | 0.8487 | 3.0 / 24.6 |
| mix-500k-70-20-10-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 69.37% | 0.8051 | 0.8055 | 3.0 / 24.6 |
| mix-500k-75-0-25-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 69.08% | 0.8160 | 0.8032 | 2.8 / 24.5 |
| mix-500k-75-25-0-ce10-500u-s173 | 500,000 | 6,310,062 | 500 | 68.73% | 0.8202 | 0.8380 | 2.9 / 24.9 |
| optimization-2m-adamw-noamsgrad-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.17% | 0.7324 | 0.7170 | 12.5 / 43.1 |
| optimization-2m-dropout0-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.11% | 0.7318 | 0.7216 | 12.9 / 43.5 |
| optimization-2m-dropout03-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.84% | 0.7422 | 0.7217 | 12.6 / 43.6 |
| optimization-2m-logweight-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.54% | 0.7510 | 0.7440 | 13.6 / 44.3 |
| optimization-2m-lr1e-3-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.41% | 0.7424 | 0.6955 | 13.5 / 45.8 |
| optimization-2m-lr1e-4-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 70.45% | 0.7754 | 0.7703 | 13.3 / 44.2 |
| optimization-2m-wd0-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.13% | 0.7332 | 0.7163 | 12.9 / 44.1 |
| optimization-2m-wd01-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.15% | 0.7321 | 0.7176 | 13.8 / 44.8 |
| size-1000000-ce10-2510u-s173 | 1,000,000 | 6,310,062 | 2,510 | 71.29% | 0.7774 | 0.7234 | 13.5 / 40.4 |
| size-1000000-ce10-2ep-s173 | 1,000,000 | 6,310,062 | 490 | 69.35% | 0.8035 | 0.8090 | 2.9 / 26.9 |
| size-2000000-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.15% | 0.7336 | 0.7164 | 12.9 / 44.4 |
| size-2000000-ce10-2ep-s173 | 2,000,000 | 6,310,062 | 978 | 70.72% | 0.7670 | 0.7712 | 5.5 / 34.7 |
| size-250000-ce10-2510u-s173 | 250,000 | 6,310,062 | 2,510 | 64.58% | 2.3308 | 0.8285 | 13.7 / 38.0 |
| size-250000-ce10-2ep-s173 | 250,000 | 6,310,062 | 124 | 64.60% | 0.9366 | 0.9532 | 1.1 / 22.6 |
| size-4000000-ce10-2510u-s173 | 4,000,000 | 6,310,062 | 2,510 | 72.34% | 0.7245 | 0.7219 | 13.6 / 50.2 |
| size-4000000-ce10-2ep-s173 | 4,000,000 | 6,310,062 | 1,954 | 71.99% | 0.7331 | 0.7364 | 9.6 / 47.4 |
| size-8953912-ce10-2510u-s173 | 8,953,912 | 6,310,062 | 2,510 | 72.45% | 0.7205 | 0.7217 | 12.6 / 71.7 |
| size-8953912-ce10-2ep-s173 | 8,953,912 | 6,310,062 | 4,374 | 73.21% | 0.7007 | 0.7077 | 22.1 / 81.0 |

</details>

Training seconds include the optimizer loop and its logging; total seconds include data loading, validation, checkpointing and private artifact upload, but end before the final tracking-client shutdown. Cached-data preparation is a separate study cost. GPU memory figures include resident datasets where used. The production model was trained on a different historical corpus/rubric, so a comparison with it does not isolate architecture.

![Measured scaling curves](experiments/search-relevance-generalization/size-scaling.png)
[SVG](experiments/search-relevance-generalization/size-scaling.svg) · [PDF](experiments/search-relevance-generalization/size-scaling.pdf)

The two curves answer different questions. Fixed updates repeat small datasets more often; fixed epochs give large datasets more updates. Query-balanced accuracy describes that sampled population and is not a substitute for a verified traffic-frequency tail metric. Points have one seed and therefore no across-seed confidence bands.

### Which inputs matter?

![Retrained feature ablations](experiments/search-relevance-generalization/feature-ablations.png)

Ablations retain the architecture while masking the registered inputs during both training and evaluation. Parameters representing removed signals can still encode constants. These are predictive-information controls, not causal estimates of clicks or purchases. Stored semantic features also have substantial missingness.

### How does source composition change quality?

![Measured source-mixture outcomes](experiments/search-relevance-generalization/source-mixtures.png)

Each dot is a completed run at a feasible mixture; there is no fitted interpolation. Pure-source controls sit at the corners. Color scales are population-specific. The 70/20/10 reference is the additional interior point.

### Precision check

The 1M-pair, 500-update FP32/BF16 systems check passed its predeclared absolute tolerance of 0.002 in common-five accuracy and NLL. BF16 minus FP32: **0.058 percentage points** accuracy and **-0.00114** NLL. This one-seed engineering check does not establish statistical equivalence.

### Language-model infrastructure check

The Qwen3-0.6B classification-head pilot trained all **596,057,095 parameters**, with nonzero gradients and detected changes in all **312 trainable tensors**. It completed 16 updates on 1,024 training examples and a fixed 2,048-pair validation probe. This confirms the execution path; it is not a generalization result. The full head comparison is separately registered at 250k pairs, one epoch, batch 128, and LR 1e-5.
