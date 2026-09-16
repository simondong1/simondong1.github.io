## Measured training results

**85 controlled quality runs complete**, plus systems checks. Results below are exploratory validation measurements; individual seeds are recorded in the complete table. They are not deployment recommendations or final-test results. Each training row evaluates all 496,477 validation pairs unless identified as a pilot. Native five-class and native ten-class accuracies are not compared; the main table uses the registered common five-bin target.

| Reference comparison | Unique training pairs | Common-five accuracy | Common-five NLL ↓ |
|---|---:|---:|---:|
| Qwen3-0.6B, classification head, 1M | 1,000,000 | 82.11% | 0.4781 |
| Qwen3-0.6B, classification head | 250,000 | 79.48% | 0.5503 |
| Qwen3-0.6B, restricted token logits | 250,000 | 79.42% | 0.5512 |
| Qwen3-0.6B, full-vocabulary token | 250,000 | 79.41% | 0.5509 |
| Qwen3-1.7B, classification head | 250,000 | 81.38% | 0.4963 |
| Feature DCN, full data, two epochs | 8,953,912 | 73.21% | 0.7007 |

The 1M-pair 0.6B classifier reaches **82.11%** shared-five accuracy and **0.4781** NLL, versus 79.48% and 0.5503 at 250k. It exceeds the matched-family 1.7B/250k validation result under a larger data and optimization budget. It consumes one million examples in 7,813 updates versus 250k in 1,954; this is not an equal-compute capacity comparison. All use one pass, LR 1e-5, full-parameter AdamW, and the same query/title format.


These reference rows are best-observed recipe comparisons with different input representations, training data counts, parameter counts, and compute. They do not isolate a single cause. The head-to-head LM comparison does freeze the backbone, examples, input serialization, batch order, and optimization recipe.

<details markdown="1"><summary>Complete experiment table</summary>

| Run | Unique pairs | Parameters | Updates | Common-five accuracy | Common-five NLL ↓ | Original MAE ↓ | Train / total seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| architecture-2m-cross4-ce10-2510u-s173 | 2,000,000 | 9,744,882 | 2,510 | 72.36% | 0.7295 | 0.7080 | 14.3 / 44.5 |
| architecture-2m-feature-transformer-ce10-2510u-s173 | 2,000,000 | 6,226,186 | 2,510 | 71.04% | 0.7835 | 0.7285 | 137.7 / 181.7 |
| architecture-2m-large-ce10-2510u-s173 | 2,000,000 | 56,620,878 | 2,510 | 72.42% | 0.7856 | 0.6741 | 22.2 / 59.4 |
| architecture-2m-mlp-ce10-2510u-s173 | 2,000,000 | 2,875,242 | 2,510 | 70.52% | 0.7730 | 0.7692 | 12.2 / 41.9 |
| architecture-2m-mlp-matched-ce10-2510u-s173 | 2,000,000 | 6,288,586 | 2,510 | 71.77% | 0.7418 | 0.7275 | 11.6 / 42.6 |
| architecture-2m-small-ce10-2510u-s173 | 2,000,000 | 1,250,062 | 2,510 | 70.96% | 0.7592 | 0.7628 | 12.7 / 41.7 |
| architecture-2m-wide-ce10-2510u-s173 | 2,000,000 | 17,763,726 | 2,510 | 72.58% | 0.7295 | 0.6933 | 13.7 / 46.0 |
| architecture-full-cross4-ce10-2ep-s173 | 8,953,912 | 9,744,882 | 4,374 | 73.40% | 0.6945 | 0.7015 | 25.0 / 83.6 |
| architecture-full-wide-ce10-2ep-s173 | 8,953,912 | 11,670,414 | 4,374 | 73.75% | 0.6848 | 0.6898 | 25.2 / 85.8 |
| confirm-full-ce10-2ep-s271 | 8,953,912 | 6,310,062 | 4,374 | 73.13% | 0.7015 | 0.7052 | 23.3 / 81.7 |
| confirm-full-ce10-2ep-s811 | 8,953,912 | 6,310,062 | 4,374 | 73.12% | 0.7008 | 0.6962 | 23.2 / 81.1 |
| confirm-qwen3-17b-classifier-250k-1ep-s271 | 250,000 | 1,720,589,319 | 1,954 | 81.47% | 0.4955 | 0.4717 | 740.3 / 1134.1 |
| confirm-qwen3-17b-classifier-250k-1ep-s811 | 250,000 | 1,720,589,319 | 1,954 | 81.45% | 0.4953 | 0.4860 | 732.8 / 1127.0 |
| epochs-2m-ce10-1ep-s173 | 2,000,000 | 6,310,062 | 489 | 69.38% | 0.8040 | 0.8080 | 2.9 / 30.6 |
| epochs-2m-ce10-4ep-s173 | 2,000,000 | 6,310,062 | 1,956 | 71.88% | 0.7368 | 0.7346 | 10.8 / 41.3 |
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
| llm-qwen3-06b-classifier-1m-1ep-s173 | 1,000,000 | 596,057,095 | 7,813 | 82.11% | 0.4781 | 0.4600 | 1415.4 / 1660.8 |
| llm-qwen3-06b-classifier-250k-1ep-s173 | 250,000 | 596,057,095 | 1,954 | 79.48% | 0.5503 | 0.5361 | 353.6 / 599.4 |
| llm-qwen3-06b-classifier-250k-2ep-s173 | 250,000 | 596,057,095 | 3,908 | 79.90% | 0.5548 | 0.4885 | 708.4 / 953.0 |
| llm-qwen3-06b-classifier-250k-frozen-lr1e-3-s173 | 250,000 | 596,057,095 | 1,954 | 50.23% | 1.2173 | 1.2822 | 114.7 / 339.3 |
| llm-qwen3-06b-classifier-250k-lr2e-5-s173 | 250,000 | 596,057,095 | 1,954 | 79.45% | 0.5494 | 0.5361 | 354.1 / 598.8 |
| llm-qwen3-06b-classifier-250k-lr5e-6-s173 | 250,000 | 596,057,095 | 1,954 | 78.72% | 0.5711 | 0.5673 | 365.0 / 609.3 |
| llm-qwen3-06b-classifier-250k-random-lr1e-4-s173-r2 | 250,000 | 596,057,095 | 1,954 | 51.57% | 1.1595 | 1.3449 | 354.2 / 603.4 |
| llm-qwen3-06b-restricted-250k-1ep-s173 | 250,000 | 596,049,920 | 1,954 | 79.42% | 0.5512 | 0.5416 | 353.0 / 597.7 |
| llm-qwen3-06b-token-250k-1ep-s173 | 250,000 | 596,049,920 | 1,954 | 79.41% | 0.5509 | 0.5414 | 354.0 / 601.4 |
| llm-qwen3-17b-classifier-250k-1ep-s173 | 250,000 | 1,720,589,319 | 1,954 | 81.38% | 0.4963 | 0.4745 | 723.5 / 1118.9 |
| mix-2m-100-0-0-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.29% | 0.7471 | 0.7497 | 12.8 / 44.3 |
| mix-2m-25-50-25-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.47% | 0.7519 | 0.7297 | 13.5 / 44.3 |
| mix-2m-50-25-25-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.79% | 0.7442 | 0.7193 | 12.6 / 43.6 |
| mix-2m-50-50-0-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.05% | 0.7539 | 0.7569 | 13.7 / 44.9 |
| mix-2m-70-20-10-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.13% | 0.7319 | 0.7194 | 12.7 / 43.3 |
| mix-2m-75-25-0-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.27% | 0.7486 | 0.7482 | 13.2 / 42.7 |
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
| optimization-2m-ce7-control-2510u-s173 | 2,000,000 | 6,309,963 | 2,510 | 72.26% | 0.7307 | 0.7142 | 13.6 / 43.3 |
| optimization-2m-ce7-emd-2510u-s173 | 2,000,000 | 6,309,963 | 2,510 | 72.25% | 0.7310 | 0.7134 | 14.5 / 45.8 |
| optimization-2m-dropout0-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.11% | 0.7318 | 0.7216 | 12.9 / 43.5 |
| optimization-2m-dropout03-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.84% | 0.7422 | 0.7217 | 12.6 / 43.6 |
| optimization-2m-logweight-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 71.54% | 0.7510 | 0.7440 | 13.6 / 44.3 |
| optimization-2m-lr1e-3-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.41% | 0.7424 | 0.6955 | 13.5 / 45.8 |
| optimization-2m-lr1e-4-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 70.45% | 0.7754 | 0.7703 | 13.3 / 44.2 |
| optimization-2m-ordinal-2510u-s173 | 2,000,000 | 6,309,771 | 2,510 | 70.02% | 0.8746 | 0.7310 | 14.8 / 44.6 |
| optimization-2m-sqrtweight-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 70.44% | 0.7793 | 0.7767 | 12.8 / 42.8 |
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

### Interpreting architecture, optimization, and labels

The full-data two-pass DCN improves shared-five accuracy from the operational production baseline's 68.74% to 73.21%. Because the old production corpus and label regime differ, this is a recipe comparison. Under fixed updates, returns diminish after a few million unique pairs; under two passes, larger datasets receive additional optimization and keep improving. The 250k fixed-update run becomes severely overconfident after approximately 41 passes, which argues against using the same update budget indiscriminately at every data size.

At 2M pairs, removing clicks or purchases individually changes accuracy little, while removing all behavior features has a larger effect. Correlated signals can replace one another: a small single-family ablation does not mean that family has no information. The semantic-only and behavior-only models both trail their combination. These observations support testing complementary branches and conditional routing.

The first capacity screen confounds architecture and parameter count. At 2M pairs and 2510 updates, a parameter-matched 6.29M MLP reaches 71.77% shared-five accuracy (NLL 0.7418), a 6.23M feature transformer reaches 71.04% (0.7835), and the 6.31M DCN reaches 72.15% (0.7335). Equal parameters do not imply equal FLOPs. The full-data width-384 follow-up reaches 73.75% (0.6848); it is an intermediate capacity point, not identical to the width-512 2M screen.

The one-factor optimizer screen shows no clear one-seed benefit from AMSGrad or stronger weight decay. LR 1e-3 improves accuracy over 3.5e-4 while worsening NLL; selecting only accuracy would conceal that confidence tradeoff. Log and square-root traffic weighting worsen aggregate validation quality in this setting. The seven-class support control performs similarly to the ten-way head. Independent title/thumbnail heads underperform their joint target, consistent with a restrictive independence assumption; this is not evidence that component labels are uninformative.

### Pretraining and adaptation controls

| Same 0.6B backbone and 250k raw-input pairs | Shared-five accuracy | Shared-five NLL ↓ |
|---|---:|---:|
| Pretrained, full post-training, LR 1e-5 | 79.48% | 0.5503 |
| Pretrained, frozen backbone/linear head, LR 1e-3 | 50.23% | 1.2173 |
| Random initialization, full training, LR 1e-4 | 51.57% | 1.1595 |

These one-pass controls use the same query/title serialization. Different declared learning rates account for different optimization scales, but each control remains a finite-budget baseline. The frozen model trains only its linear readout; the other two update every model parameter. Poor frozen/random results do not establish their fully tuned limits. The observed benefit requires both useful pretrained parameters and adaptation under the tested recipes.

### Frequency and missing-feature slices

| Model | Head proxy | Tail proxy | Zero-frequency unknown | All embeddings present | All embeddings zero |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B, classification head | 80.44% | 79.76% | 77.69% | 81.76% | 77.36% |
| Qwen3-0.6B, restricted token logits | 80.38% | 79.65% | 77.71% | 81.73% | 77.12% |
| Qwen3-0.6B, full-vocabulary token | 80.43% | 79.61% | 77.68% | 81.70% | 77.15% |
| Qwen3-1.7B, classification head | 81.99% | 81.48% | 80.35% | 83.66% | 78.98% |
| Feature DCN, full data, two epochs | 80.25% | 70.51% | 66.19% | 81.30% | 62.80% |

These are shared-five accuracies. Frequency groups use train-only positive-query median/p90 thresholds (291/22,159) in a logged 28-day query-view count. Zero counts remain unknown, not tail. This feature is constant within each query. August 12–13 source partitions are verified, while source-table event-time semantics remain a caveat; these are frequency proxies. Missing-vector groups use original availability before any retraining masks. The raw-text LLM's advantage is particularly large where cached embeddings are absent; this helps explain the aggregate difference and motivates availability-aware fusion.

### Combining feature and text models

Temperature scaling and a constant or logistic mixture are fitted on 20% of validation queries. Another 20% sets cascade thresholds; the remaining 60% measures the policies. The partitions are deterministic and query-disjoint. Earlier recipe exploration used the full validation set, so this is an exploratory internal audit, not a pristine final test.

The gate uses feature-model uncertainty, embedding availability, item dimension, and query length. It never uses the source sampling label, ground-truth relevance, or LLM outputs to decide whether to call the LLM. A cascade applies the feature model to every pair and conditionally replaces its distribution with the text model's output; a soft mixture evaluates both experts.

| Policy | LLM fraction on audit pairs | Shared-five accuracy | Shared-five NLL ↓ |
|---|---:|---:|---:|
| Feature model | 0.0% | 72.97% | 0.7060 |
| 0.6B text model | 100.0% | 79.45% | 0.5513 |
| Constant soft mixture | 100.0% | 80.00% | 0.5373 |
| Learned soft mixture | 100.0% | 80.49% | 0.5196 |
| Cascade, 25% target | 25.1% | 77.10% | 0.6080 |
| Cascade, 50% target | 49.8% | 79.34% | 0.5543 |
| Cascade, 75% target | 74.8% | 79.98% | 0.5404 |

![Conditional model use](experiments/search-relevance-generalization/cascade-quality.png)

LM call fraction is not a measured speedup: preprocessing, small-batch utilization, request grouping, and gate errors all affect serving cost. Aggregate pair policies do not yet establish complete-page ranking performance. Query bootstrap intervals and source slices are available in the [fusion artifact](experiments/search-relevance-generalization/fusion-results.json); these intervals describe query sampling for fixed checkpoints, not training-seed variance.

### Training-seed confirmation

The full-data two-pass production-size feature recipe has 3 completed seeds. Shared-five accuracy is **73.16%**, with sample standard deviation **0.049 percentage points**; mean NLL is **0.7010**. Subset selection remains fixed, so this measures optimization/initialization variability rather than independently resampled datasets.

The matched 1.7B classifier has three seeds: mean validation accuracy **81.43%**, sample SD **0.044 percentage points**, and mean NLL **0.4957**. Each checkpoint is independently deployable below 4B; these results do not use an ensemble.

### Class-token output versus a ranking score

The probability-head tables expose logits for all three LM heads. A literal one-digit response is a separate output contract: it coarsens scores and creates ties. The following diagnostic applies that same coarsening to every head; the full-token model had zero invalid emissions, so its conditional argmax agrees with its emitted label.

| Head | Single-class shared-five accuracy | Natural NDCG, discrete class | Natural NDCG, expected score |
|---|---:|---:|---:|
| classifier | 79.32% | 0.9549 | 0.9734 |
| restricted token | 79.23% | 0.9550 | 0.9736 |
| full-vocabulary token | 79.21% | 0.9546 | 0.9736 |

NDCG uses tie-averaged zero-based gains on the same 27,411 natural multi-item queries with nonzero ideal gain. Probability pooling and coarsening an argmax are different decisions, so their shared-five accuracies also differ slightly. Exposing the distribution preserves useful ranking resolution regardless of whether the head was trained as class logits or a vocabulary token.

### Measured serving cost

| Model / device | Batch 1 p50 | Batch 360 p50 | Batch 360 p95 |
|---|---:|---:|---:|
| Feature DCN, full data, two epochs / B200 | 1.13 ms | 1.16 ms | 1.33 ms |
| Feature DCN, full data, two epochs / CPU 4 threads | 0.56 ms | 8.49 ms | 9.02 ms |
| Feature DCN, width 384 / B200 | 1.13 ms | 1.17 ms | 1.31 ms |
| Feature DCN, width 384 / CPU 4 threads | 0.78 ms | 14.56 ms | 14.86 ms |
| Qwen3-0.6B, classification head / B200 | 22.31 ms | 147.56 ms | 147.72 ms |
| Qwen3-0.6B, restricted token logits / B200 | 22.42 ms | 147.56 ms | 147.64 ms |
| Qwen3-0.6B, full-vocabulary token / B200 | 22.65 ms | 148.05 ms | 148.10 ms |
| Qwen3-1.7B, classification head / B200 | 22.68 ms | 211.01 ms | 211.81 ms |

These are warm eager-PyTorch forward/scoring measurements with inputs already resident on the device, ten warmups, and 50 timed repetitions. Text rows use the median prompt length padded to 88 tokens; p95-length results, batches 32/128, and empirical p99 are in the [benchmark artifact](experiments/search-relevance-generalization/serving-benchmark.json). The full-vocabulary path includes argmax over vocabulary logits. Every feature batch is checked against a nonempty cache shard. Tokenization, encoders, cache lookup, network, request concurrency, and queueing are excluded. The shared B200 host differs from the production accelerator, and fifty samples do not characterize a production p99.

### Quality-checked inference compilation

The 0.6B classifier uses `torch.compile(mode=reduce-overhead, dynamic=True)`, FP32 weights and BF16 autocast. Compilation keeps the same trained checkpoint and output contract.

| Median-length batch | Eager p50 | Compiled p50 | Measured ratio |
|---|---:|---:|---:|
| 1 | 22.31 ms | 3.52 ms | 6.33× |
| 32 | 22.47 ms | 7.48 ms | 3.01× |
| 128 | 56.83 ms | 20.45 ms | 2.78× |
| 360 | 147.56 ms | 51.55 ms | 2.86× |

The predeclared full-validation gate **passed**: accuracy changed by **-0.0020 percentage points** and NLL by **0.000110**, within the absolute 0.002 limits on 496,477 pairs. This is an engineering tolerance, not a statistical equivalence test. The first two shape families took approximately 115 and 108 seconds including compilation; warm figures exclude that setup. Two compiled graphs and multiple CUDA-graph shapes were recorded. Full validation inference took 70.6 seconds after compilation. Serving should bucket/pad shapes and measure cold starts and concurrency explicitly. [Measurements and quality gate](experiments/search-relevance-generalization/compiled-serving.json).

### Serving check on the selected 1M checkpoint

The exact selected 0.6B/1M checkpoint separately **passed** the same full-validation quality gate. Compiled minus eager: **0.0024 percentage points** accuracy and **0.000033** NLL. Both timing paths now use this identical checkpoint: batch-1 p50 is **21.93 → 3.55 ms** and batch-360 p50 is **147.96 → 51.59 ms** at 88 padded tokens. The standalone-latency exclusions above apply. [Same-checkpoint timings and full-validation gate](experiments/search-relevance-generalization/compiled-selected-model.json).

### Larger-expert mixture

A second registered mixture combines the full-data width-384 feature model with the 1.7B text model and removes item dimension from the gate, so routing uses directly replayable inputs. On the same internal query audit, the text expert reaches **81.32%** shared-five accuracy and **0.4968** NLL; the soft mixture reaches **82.05%** and **0.4750**. This is a recipe extension with multiple changed ingredients, not a single-factor architecture claim. The two reranking experts together remain below 4B parameters. Existing embedding services are outside this parameter and latency boundary; their end-to-end footprint is not established here. [Full aggregate results](experiments/search-relevance-generalization/fusion-v2-results.json).

### Reannotation bridge for future traffic

The current combined teacher reannotated **128** deterministic validation pairs: title agreement **97.66%**, thumbnail agreement **93.75%**, exact-total agreement **92.19%**, and shared-five agreement **93.75%**. All used a stored snapshot URL, with newly fetched image bytes recorded by hash. This small bridge supports rubric continuity but does not prove identical historical prompt/model/image bytes, human agreement, or an accuracy ceiling. [Confusion matrices](experiments/search-relevance-generalization/teacher-bridge.json).

### Failed attempts retained

| Run | Recorded failure |
|---|---|
| serving-benchmark-v1 | Benchmark selected cache shard0000, which is empty; reported feature batch sizes did not reflect real rows. Discard this benchmark attempt. Quality training uses row-selected nonempty shards and is unaffected. |
| label-1m-ce5-2510u-s173 | TypeError: Got unsupported ScalarType BFloat16 |
| llm-qwen3-06b-classifier-250k-random-lr1e-4-s173 | RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float |

The CE5 probability-projection autocast issue and random-backbone dtype initialization issue were corrected and retried under new run IDs. Failed attempts are excluded from quality comparisons and retained in the ledger.

### Precision check

The 1M-pair, 500-update FP32/BF16 systems check passed its predeclared absolute tolerance of 0.002 in common-five accuracy and NLL. BF16 minus FP32: **0.058 percentage points** accuracy and **-0.00114** NLL. This one-seed engineering check does not establish statistical equivalence.

### Language-model infrastructure check

The Qwen3-0.6B classification-head pilot trained all **596,057,095 parameters**, with nonzero gradients and detected changes in all **312 trainable tensors**. It completed 16 updates on 1,024 training examples and a fixed 2,048-pair validation probe. This confirms the execution path; it is not a generalization result. The full head comparison is separately registered at 250k pairs, one epoch, batch 128, and LR 1e-5.
