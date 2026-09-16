## Held-out test findings

The sealed evaluation contains **500,551 pairs and 233,731 queries**. Checkpoint and analysis hashes were recorded before model scoring. The primary endpoint is common-five negative log likelihood (NLL); accuracy is secondary. These measure agreement with the Gemini rubric, not human preference or business impact.

| Frozen recipe | Common-five accuracy | Common-five NLL ↓ | Original-score MAE ↓ |
|---|---:|---:|---:|
| Operational production v67 | 68.69% | 0.8406 | 0.9335 |
| Feature DCN, 6.31M, full data | 73.04% | 0.7032 | 0.7084 |
| Feature DCN, width 384, full data | 73.64% | 0.6867 | 0.6895 |
| Qwen3-0.6B classifier, 250k | 79.46% | 0.5515 | 0.5353 |
| Qwen3-0.6B classifier, 1M | 82.04% | 0.4798 | 0.4598 |
| Qwen3-0.6B restricted token | 79.40% | 0.5526 | 0.5409 |
| Qwen3-0.6B full-vocabulary token | 79.36% | 0.5527 | 0.5406 |
| Qwen3-1.7B classifier, 250k | 81.33% | 0.4964 | 0.4735 |
| 0.6B + feature soft mixture | 80.54% | 0.5189 | 0.5143 |
| 1.7B + width-384 feature soft mixture | 82.09% | 0.4746 | 0.4652 |

The selected small-model recipe was **Qwen3-0.6B classifier, 1M**, selected solely by full-validation NLL among the registered learning-rate, epoch, and data-size candidates. No test labels tuned a model, temperature, gate, or threshold. Production v67 is an operational reference with unknown historical training overlap and a different historical rubric.

### Registered paired comparisons

Differences are treatment minus reference. Negative NLL and positive accuracy differences favor the treatment. Brackets are 95% query-cluster bootstrap intervals for fixed checkpoints; they do not include training-seed uncertainty. Holm-adjusted p-values apply to the registered family of primary NLL contrasts.

| Treatment versus reference | Δ NLL [95% interval] | Holm p | Δ accuracy, percentage points [95% interval] |
|---|---:|---:|---:|
| Width-384 versus production-size feature model | -0.0165 [-0.0175, -0.0156] | <1e-6 | 0.598 [0.524, 0.673] |
| 0.6B text versus feature model | -0.1517 [-0.1565, -0.1465] | <1e-6 | 6.419 [6.192, 6.638] |
| 1.7B versus 0.6B, matched 250k | -0.0551 [-0.0577, -0.0527] | <1e-6 | 1.865 [1.742, 1.983] |
| Validation-selected versus original 0.6B | -0.0717 [-0.0742, -0.0693] | <1e-6 | 2.574 [2.445, 2.712] |
| Classifier versus full-vocabulary token | -0.0012 [-0.0022, -0.0002] | 0.01279 | 0.101 [0.031, 0.169] |
| 1.7B/feature soft mixture versus 1.7B | -0.0218 [-0.0237, -0.0201] | <1e-6 | 0.761 [0.684, 0.843] |

The text-versus-feature contrast changes inputs, pretraining, capacity, data count, and compute together. The matched text-head comparison isolates output formulation more closely. The mixture comparison includes validation-fitted calibration and gating as part of the recipe.

### Repeated seeds and held-out slices

| Recipe, three separately deployable seeds | Mean test accuracy | Sample SD, percentage points | Mean test NLL |
|---|---:|---:|---:|
| Production-size feature DCN | 73.04% | 0.008 | 0.7026 |
| Qwen3-1.7B classifier | 81.39% | 0.078 | 0.4968 |

Each seed uses the same fixed training subset. These are not ensembles: a three-model 1.7B ensemble exceeds the sub-4B total-serving limit.

| Seed-173 recipe | Head proxy | Tail proxy | Unknown frequency | Natural | Balanced | Random 3D |
|---|---:|---:|---:|---:|---:|---:|
| Feature DCN, 6.31M, full data | 80.15% | 72.33% | 67.97% | 72.08% | 65.97% | 93.99% |
| Qwen3-0.6B classifier, 250k | 80.60% | 78.73% | 78.89% | 77.37% | 78.20% | 96.74% |
| Qwen3-0.6B classifier, 1M | 83.24% | 81.27% | 81.43% | 80.31% | 80.63% | 96.98% |
| Qwen3-1.7B classifier, 250k | 82.23% | 80.46% | 80.95% | 79.45% | 80.00% | 97.22% |
| 1.7B + width-384 feature soft mixture | 83.77% | 81.05% | 81.19% | 80.50% | 80.08% | 97.28% |

All cells use the common-five target. Frequency boundaries are frozen from training queries (positive median 291 and p90 22,159 views); zero-frequency rows remain unknown. The downloadable results include slice counts, 2D/3D metrics, original embedding availability, sampled-list ranking, and actual cascade call fractions. Balanced-pool sampled-list NDCG has too few multi-item queries to establish long-tail ranking quality.

[Complete test metrics and contrasts](experiments/search-relevance-generalization/final-test-results.json) · [Checkpoint and analysis hashes](experiments/search-relevance-generalization/finalist-registration.json)

### Item type and embedding availability

| Recipe | 2D accuracy | 3D accuracy | All embeddings present | All embeddings zero |
|---|---:|---:|---:|---:|
| Feature DCN, 6.31M, full data | 61.54% | 75.43% | 81.01% | 62.53% |
| Qwen3-0.6B classifier, 250k | 69.35% | 81.41% | 81.64% | 77.26% |
| Qwen3-0.6B classifier, 1M | 71.74% | 84.02% | 84.40% | 79.18% |
| Qwen3-1.7B classifier, 250k | 71.17% | 83.28% | 83.59% | 78.75% |
| 1.7B + width-384 feature soft mixture | 71.58% | 84.13% | 85.09% | 78.75% |

The test contains 73,709 2D, 416,248 3D, and 10,594 unknown-dimension pairs. Original embedding availability gives 223,305 all-present and 57,133 all-zero pairs. These descriptive slices differ in label/source composition; they are not causal missing-feature interventions or individual significance claims.

A scoring-loader failure occurred after the five feature/production models completed: older 0.6B checkpoints lacked a later-added model-type field. Recovery corrected type detection, verified and preserved all five completed outputs by hash, and continued with separate tracking IDs. No language-model test predictions existed before the failure. Checkpoints, data, precision, selection and analysis rules stayed fixed. The original sealed scorer and [recovery record](experiments/search-relevance-generalization/final-test-recovery.json) are retained.

## Generalization to later traffic

Frozen predictions cover **10,493 pairs in 267 complete displayed pages** from August 22 and September 8, after training-source traffic ended August 13. Two teacher content-filter responses excluded one entire 30-item request. Inverse inclusion weights account for the two-stage file/request sample; intervals resample file clusters within each date. Only eight files per date were sampled, so uncertainty remains coarse.

| Frozen recipe, both dates pooled | Accuracy %, 95% interval | NLL, 95% interval ↓ | Displayed-page NDCG@10, 95% interval |
|---|---:|---:|---:|
| Operational production v67 | 59.68 [55.35, 63.05] | 1.1130 [1.0149, 1.2333] | 0.8617 [0.8442, 0.8789] |
| Feature DCN, 6.31M, full data | 60.25 [55.02, 64.18] | 1.0739 [0.9664, 1.2120] | 0.8742 [0.8536, 0.8917] |
| Feature DCN, width 384, full data | 60.41 [55.28, 64.43] | 1.0803 [0.9670, 1.2229] | 0.8777 [0.8586, 0.8942] |
| Qwen3-0.6B classifier, 250k | 74.67 [71.29, 77.62] | 0.6684 [0.6159, 0.7258] | 0.9230 [0.9117, 0.9334] |
| Qwen3-1.7B classifier, 250k | 76.37 [73.25, 79.09] | 0.6170 [0.5728, 0.6677] | 0.9251 [0.9143, 0.9364] |
| 0.6B + feature soft mixture | 74.67 [71.05, 77.72] | 0.6631 [0.6080, 0.7239] | 0.9229 [0.9114, 0.9339] |
| 1.7B + width-384 feature soft mixture | 76.48 [73.21, 79.17] | 0.6122 [0.5662, 0.6646] | 0.9267 [0.9152, 0.9384] |

NDCG ranks the complete displayed page using tie-averaged zero-based gains; it does not evaluate unseen retrieval candidates. Logged numeric and query vectors are from request time. Item vectors use a fixed June-25 cache, with 4,863 of 10,342 unique items absent; exact deployment-cache identity is unverified. The September production export can include training after these dates and is a retrospective operational reference, not a historical counterfactual.

| Model | August 22 accuracy | September 8 accuracy | Unseen-query accuracy | Unseen-query page NDCG |
|---|---:|---:|---:|---:|
| Feature DCN, 6.31M, full data | 60.89% | 59.22% | 54.11% | 0.8091 |
| Qwen3-0.6B classifier, 250k | 72.98% | 77.34% | 72.62% | 0.8886 |
| Qwen3-1.7B classifier, 250k | 74.65% | 79.10% | 75.95% | 0.8994 |
| 1.7B + width-384 feature soft mixture | 74.76% | 79.22% | 76.08% | 0.9017 |

The unseen-query subset contains 72 requests and 2,696 pairs. The mixture's pooled paired change versus 1.7B alone is **0.113 [-0.306, 0.454] percentage points** in accuracy, **-0.0048 [-0.0082, -0.0011]** NLL, and **0.00157 [0.00003, 0.00304]** page NDCG. Accuracy is not clearly separated. These temporal comparisons and sensitivity slices are secondary, not additional multiplicity-corrected primary tests.

Image availability remains imperfect: **28** evaluable pairs explicitly used an unavailable-image placeholder, affecting **10** requests. Another 29 fetched images had catalog Pending/Blocked status and may be service placeholders despite decodable bytes. Snapshot URLs were re-fetched at study time; current-catalog fallbacks do not guarantee request-time image identity. The downloadable audit reports states and the complete-request sensitivity excluding explicit unavailable placeholders. The reannotation bridge below checks rubric continuity; it does not establish human accuracy.

[Per-date results, uncertainty, image sensitivity, and frozen cascade use](experiments/search-relevance-generalization/temporal-results.json)

The frozen cascade's cost control does not transfer unchanged: the policy calibrated for 50% LM calls routes **90.9%** of weighted future pairs to the 1.7B model; the 75% policy routes every future pair. Feature availability and uncertainty shift, so a fixed gate threshold does not enforce a fixed serving budget. Any hard per-request capacity policy needs its own evaluation.

### Supplementary 1M recipe and image-state sensitivity

The 1M training recipe was fixed before initial temporal results, but these predictions were added afterward. This is a supplemental evaluation of an unchanged recipe, without refitting or choosing hyperparameters from later traffic. Original temporal predictions and results are preserved.

| Text recipe | Pooled accuracy %, 95% interval | Pooled NLL, 95% interval ↓ | Page NDCG | Accuracy on requests with completed/snapshot images |
|---|---:|---:|---:|---:|
| Qwen3-0.6B classifier, 250k | 74.67 [71.29, 77.62] | 0.6684 [0.6159, 0.7258] | 0.9230 | 75.09% |
| Qwen3-1.7B classifier, 250k | 76.37 [73.25, 79.09] | 0.6170 [0.5728, 0.6677] | 0.9251 | 76.81% |
| Qwen3-0.6B classifier, 1M | 77.48 [74.97, 79.97] | 0.5903 [0.5378, 0.6504] | 0.9347 | 78.14% |

The 1M model's paired difference versus 1.7B is **1.113 [0.189, 2.184] percentage points** accuracy and **-0.0267 [-0.0524, 0.0005]** NLL. The stricter image sensitivity excludes whole requests containing any Pending/Blocked/unavailable state and retains 246 requests and 9,513 pairs. Successful re-fetches still do not prove historical image identity. These are secondary comparisons with coarse file-cluster uncertainty. [Supplementary results](experiments/search-relevance-generalization/temporal-1m-supplement.json).

## Reproducibility and resource accounting

The 91 completed training/pilot processes consumed **2.09 GPU-hours in optimizer loops** and **3.73 GPU-hours of per-run wall time** including their loading, evaluation, saving and uploads. The two-GPU capacity window from study start to the accounting record is **6.94 GPU-hours**; it includes data preparation, separate evaluations and idle allocation. It is not a scheduler invoice. B200 training-time comparisons do not imply A100 serving speed.

Final responses from temporal annotation and the reannotation bridge record approximately 27.23M prompt tokens and 4.43M completion tokens. This excludes unrecorded timeout usage and superseded attempts and is not a billing total. No GPU/API price was assumed. The accounting artifact retains per-run times and actual saved token counts.

W&B histories and S3 checkpoint/prediction sizes are verified separately from a successful process exit. A private source-history bundle and evidence archive retain manifests, raw update logs, failure/recovery records and exact checkpoint hashes. Six intermediate source versions were reconstructed from recorded patches/formatter versions and accepted only when their SHA-256 matched the original manifests; all 491 audited execution-file references have recoverable bytes. Raw examples, images and checkpoints remain private.

[Resource accounting](experiments/search-relevance-generalization/study-accounting.json)

Standalone safetensors exports for the selected 0.6B/1M and the 1.7B/250k classifiers include tokenizer files, model configuration, the exact input instruction and a standalone loader example. Every tensor matches the training checkpoint exactly; FP32 weight files are 2.38 GB and 6.88 GB respectively. Full optimizer checkpoints remain available separately in the private study store. [Export contracts and file hashes](experiments/search-relevance-generalization/serving-exports.json).

## What the study supports

Data composition is an objective choice. At 2M pairs and a fixed update budget, moving from natural-only to 70/20/10 natural/balanced/random improves overall validation NLL from 0.7471 to 0.7319 and random-pool NLL from 0.4747 to 0.2417, while natural NLL worsens from 0.7416 to 0.7523. More balanced coverage slightly helps the balanced population in some controls, but a universal head/tail win is not established. Heavy random sampling produces an easy negative specialist and degrades natural relevance. Preserve each source and evaluate against the intended serving population instead of optimizing a mixture's aggregate accuracy alone.

Capacity and optimization interact. Larger unique feature datasets continue improving at two passes; fixed-update gains flatten after a few million examples. The 250k model is severely overconfident after approximately 41 passes. Width-384 DCN improves full-data validation, while parameter-matched MLP and feature transformer controls do not overturn the feature DCN under their tested budget. These measurements do not fit a universal scaling exponent or establish globally optimal hyperparameters.

Raw-text adaptation and feature availability both matter. Pretrained full-parameter language models outperform the finite-budget frozen and random-initialized controls. Their aggregate advantage over the feature model is largest when frozen embeddings are unavailable. Predictive ablations do not establish causal effects of clicks, purchases, or exposure; refreshed feature coverage is the next discriminating experiment.

For serving, retain probability distributions and an expected relevance score. Matched classifier and token heads are close in quality, while emitting only a class coarsens ranking. The compact feature model occupies the lowest measured latency region. Compilation makes 0.6B more practical at the measured shapes. At matched 250k data, 1.7B improves quality; giving 0.6B one million examples offers another path and prevents a simple recommendation to scale parameters first. Compare these recipes across both held-out populations. A cascade reduces text calls, but request grouping and varying batch sizes require a full service benchmark before translating call fraction into latency savings.

### Boundaries and next experiments

The completed bounded study covers final test, temporal evaluation, serving checks, artifact verification, and publication. Earlier protocol sections retain the broader research agenda; not every candidate idea was executed. Direct multimodal post-training, end-to-end fusion heads, expert distillation, shuffled-feature counterfactuals, exhaustive data-by-capacity grids, and broad architecture families remain follow-up work. Golden and long-tail legacy collections were audited but not silently remapped across incompatible rubrics.

The next deployment decision needs a serving SLA and a human-labeled relevance bridge, followed by a separate online trial. The 898 upstream exclusions, historical production overlap, event-time semantics beyond partition dates, and temporal image/cache identity remain unresolved. No result here establishes human preference, purchase lift, or an end-to-end production latency guarantee.
