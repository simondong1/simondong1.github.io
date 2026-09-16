## Measured training results

**6 controlled quality runs complete**, plus systems checks. Results below are exploratory validation measurements from seed 173. They are not deployment recommendations or final-test results. Each row evaluates all 496,477 validation pairs unless identified as a pilot. Native five-class and native ten-class accuracies are not compared; the main table uses the registered common five-bin target.

| Run | Unique pairs | Parameters | Updates | Common-five accuracy | Common-five NLL ↓ | Original MAE ↓ | Train / total seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| size-1000000-ce10-2510u-s173 | 1,000,000 | 6,310,062 | 2,510 | 71.29% | 0.7774 | 0.7234 | 13.5 / 40.4 |
| size-1000000-ce10-2ep-s173 | 1,000,000 | 6,310,062 | 490 | 69.35% | 0.8035 | 0.8090 | 2.9 / 26.9 |
| size-2000000-ce10-2510u-s173 | 2,000,000 | 6,310,062 | 2,510 | 72.15% | 0.7336 | 0.7164 | 12.9 / 44.4 |
| size-2000000-ce10-2ep-s173 | 2,000,000 | 6,310,062 | 978 | 70.72% | 0.7670 | 0.7712 | 5.5 / 34.7 |
| size-250000-ce10-2510u-s173 | 250,000 | 6,310,062 | 2,510 | 64.58% | 2.3308 | 0.8285 | 13.7 / 38.0 |
| size-250000-ce10-2ep-s173 | 250,000 | 6,310,062 | 124 | 64.60% | 0.9366 | 0.9532 | 1.1 / 22.6 |

Training seconds include the optimizer loop and its logging; total seconds include data loading, validation, checkpointing and private artifact upload, but end before the final tracking-client shutdown. Cached-data preparation is a separate study cost. GPU memory figures include resident datasets where used. The production model was trained on a different historical corpus/rubric, so a comparison with it does not isolate architecture.

![Measured scaling curves](experiments/search-relevance-generalization/size-scaling.png)
[SVG](experiments/search-relevance-generalization/size-scaling.svg) · [PDF](experiments/search-relevance-generalization/size-scaling.pdf)

The two curves answer different questions. Fixed updates repeat small datasets more often; fixed epochs give large datasets more updates. Query-balanced accuracy describes that sampled population and is not a substitute for a verified traffic-frequency tail metric. Points have one seed and therefore no across-seed confidence bands.

### Precision check

The 1M-pair, 500-update FP32/BF16 systems check passed its predeclared absolute tolerance of 0.002 in common-five accuracy and NLL. BF16 minus FP32: **0.058 percentage points** accuracy and **-0.00114** NLL. This one-seed engineering check does not establish statistical equivalence.

### Language-model infrastructure check

The Qwen3-0.6B classification-head pilot trained all **596,057,095 parameters**, with nonzero gradients and detected changes in all **312 trainable tensors**. It completed 16 updates on 1,024 training examples and a fixed 2,048-pair validation probe. This confirms the execution path; it is not a generalization result. The full head comparison is separately registered at 250k pairs, one epoch, batch 128, and LR 1e-5.
