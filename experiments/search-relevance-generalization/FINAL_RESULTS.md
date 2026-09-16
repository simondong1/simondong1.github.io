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
