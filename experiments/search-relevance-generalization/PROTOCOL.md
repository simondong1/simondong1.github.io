# What Makes a Relevance Model Generalize?

## A controlled study of data composition, model capacity, supervision, and serving cost

**Simon Dong · Research protocol v1.1 · 16 September 2026**

**Status:** production replay and main data audit complete; GPU optimizer pilot passed; controlled model training in preparation. No new-model improvement is claimed yet.

## Abstract

We study whether search relevance improves through more examples, different examples, a different model, or a different use of the available signals. The dataset contains approximately ten million Gemini-labeled query–item pairs, drawn from natural search exposure, query-balanced production coverage, and random unexposed 3D items. The deployed reference is a 6.31-million-parameter deep-and-cross reranker using frozen content embeddings and historical engagement features. We will compare this reference with smaller and larger feature models, models trained from raw text, and language models below four billion total serving parameters. A central comparison holds the pretrained backbone and inputs fixed while changing the output from a generated class token to a learned classification or ordinal head. Experiments separate data volume, sampling distribution, feature information, optimization budget, and inference cost. Query-disjoint evaluation, temporal tests, source-specific diagnostics, calibration, repeated seeds, and a locked final evaluation govern model selection. This document specifies hypotheses and decision rules before model results are observed; audit observations are explicitly distinguished from planned experiments.

## 1. Research questions and falsifiable hypotheses

| ID | Question | Hypothesis and evidence that would contradict it |
|---|---|---|
| H1 | Does more unique data improve generalization? | Held-out loss and ranking quality improve with unique sample count at matched compute; a flat curve or improvement only with more updates contradicts a data-efficiency claim. |
| H2 | Does query balancing help rare queries? | Balanced coverage improves macro-query and rare-query metrics without an unacceptable common-query decline; improvement only on the balanced sampling distribution is insufficient. |
| H3 | Do unexposed random pairs teach relevance? | Moderate random additions reduce false acceptance of irrelevant 3D items; degradation on relevant random items or failure under feature masking indicates a source shortcut. |
| H4 | Does capacity unlock gains from larger datasets? | A data-by-capacity interaction persists after equal-budget optimization; gains explained only by extra hyperparameter trials do not support the hypothesis. |
| H5 | Which feature groups transfer? | Semantic inputs retain more quality on new queries/items and future traffic than behavior-only inputs; all-feature gains should survive feature-age and missingness diagnostics. |
| H6 | Does pretraining help beyond stored embeddings? | A post-trained small language model improves matched-information generalization relative to a random-initialized counterpart and the feature baseline. A text-only versus multimodal comparison does not isolate pretraining. |
| H7 | Does output formulation matter? | A classification or ordinal head matches or improves token-based quality at lower inference cost using the same backbone, input serialization, and tuning budget. |
| H8 | Can specialization improve the quality–cost tradeoff? | Late fusion or a small expert model improves complementary slices at a measured serving cost; extra parameters, ensembling compute, and source-label leakage must not explain the gain. |

The target is semantic relevance and reliable generalization. Agreement with Gemini is a measurable training target, but does not by itself establish agreement with people, user satisfaction, or causal purchase lift.

## 2. Audited production reference

The source checkout is pinned to `02e00f7ac8828a0bf64b9dca73e4312ecb767474`. Live Triton metrics identify `sr-model-v2` version **67** under the marketplace discovery service. The downloaded version-67 weights and September-14 training export have the same SHA-256 digest, `e380a876a2af912cba84d39790213c63beb345213ca36e76ec8775659c1f39f1`. Inspection of the TorchScript model gives **6,310,062 parameters**.

### 2.1 Inputs and architecture

The production configuration uses 69 numeric behavior signals: search and platform exposure, clicks and purchases at item/query/query–item levels, corrected Q2D counts, and centrality/prominence measures. Each numeric signal uses a learned four-dimensional embedding of 32 percentile bins. Two scalar similarity scores and eight query–title lexical match features accompany four dense content vectors: item text (1024), item image (1152), query text/ROME (1024), and query image-space/SigLIP (1152).

Each dense vector is reduced to 256 dimensions by a learned value network multiplied by a sigmoid gate. Concatenation produces **1310 features**. Two full-rank cross layers feed a ReLU MLP with widths **256 → 128 → 32 → 10** and dropout 0.15. The two cross matrices alone each have shape 1310 × 1310, making the cross network a substantial part of the parameter budget.

For a full-rank cross layer, the implementation family is of the form `x_(l+1) = x_0 ⊙ (W_l x_l + b_l) + x_l`. The optimized adapter matches the exported module on validation shards with maximum absolute logit error 3.82 × 10⁻⁶; all 6,310,062 production parameters are covered. The formula is not a replacement for this parity test.

The network learns from scratch **on top of pretrained frozen embeddings**. Calling the entire system “trained from scratch” would hide the contribution and serving cost of those encoders. A raw-text random-initialized model is a separate baseline.

### 2.2 Optimization and serving behavior

| Setting | Verified configuration |
|---|---|
| Objective | Ten-output cross entropy; integer total score converted to a one-hot vector |
| Optimizer | AdamW, learning rate 0.00035, weight decay 0.01, epsilon 1e-7, AMSGrad enabled |
| Schedule | Warmup 250 updates; cosine schedule configured for 2510 updates; minimum LR 0.00007 |
| Duration | Two epochs; downloaded export config also limits train batches to 1063 per epoch |
| Hardware configuration | Four A100 GPUs via DDP |
| Data batch setting | 4096; verify actual per-rank/global semantics and consumed rows before reproducing the budget |
| Validation | Every 150 training batches; a configured 40-batch cap is not equivalent to complete held-out evaluation |
| Output | Softmax over ten logits followed by expected class value on 0–9 |

The serving pipeline additionally supplies item-embedding caches and exact query–item score overrides. We will report **raw-network** quality and **complete-serving-system** quality separately. An override hit is not evidence of model generalization. The production output name includes purchase terminology for historical interface compatibility; its relevance objective must not be confused with purchase prediction.

A five-minute live sample showed approximately 14.3 ms median and 30.3 ms p99 gateway latency. Per-pod summary quantiles cannot be averaged into a fleet quantile; the reported gateway p99 comes from aggregated histogram buckets. Triton execution batch metrics near one refer to requests in this configuration, not necessarily one query–item pair. Candidate counts must be established from payload shape or inference logs.

## 3. Dataset, label semantics, and provenance

The final manifest reports **9,950,940 pairs**: **8,953,912 train**, **496,477 validation**, and **500,551 test**. The feature snapshot contains 221 columns, plus four existing golden/long-tail evaluation collections covering general and 3D items. The original labels table has 9,951,838 pairs: 898 more than the finalized feature snapshot. The join audit must explain every exclusion and show that source frequencies and split memberships survive canonicalization.

### 3.1 Three sampling sources

| Source | Meaning | Original labeled count, before final feature canonicalization |
|---|---|---:|
| Natural traffic | Pairs sampled from normal production exposure; represents the existing retrieval/ranking distribution | 6,968,322 |
| Query-balanced coverage | Production pairs sampled to give more queries representation, especially lower-frequency queries | 1,992,531 |
| Random unexposed 3D | Query–3D-item pairs not observed in the logged retrieval pool | 990,985 |

The design was approximately 70/20/10. The complete metadata audit confirms final training counts of **6,269,045 natural**, **1,793,386 query-balanced**, and **891,481 random unexposed 3D** pairs. Every final train/validation/test row joins to exactly one sidecar label with the same score and original split. The 898 upstream rows absent from the final main splits comprise 620 natural, 197 balanced, and 81 random pairs; the pipeline-level reason for each exclusion still requires tracing. The labels sidecar preserves `pool`, `pair_count`, source dates, exposure positions, random flags, and query-level split assignment. Source and logging metadata are sampling/evaluation variables, not model inputs. Random pairs retain their Gemini labels; they are not all assigned an irrelevant label.

### 3.2 The target has seven observed total-score values

The labeling prompt requests independent title and thumbnail ratings in **{1,2,3,4}**, with meanings irrelevant, partial/generic, strong, and exact/almost exact. `total_score = title_relevance + thumbnail_relevance`, giving **{2,3,4,5,6,7,8}**. The upstream contract treats a total below 4 as irrelevant. The prompt infers query type itself. It also instructs the judge to infer visual relevance from the title for broken or missing images; thumbnail labels therefore do not always represent direct visual observations.

We will preserve these original labels. Ten output units do not create ten observed relevance categories. A five-class experiment is an explicitly derived task, with a recorded mapping, rather than a remapping onto an older judge's incompatible scale.

| Output variant | Training target | Common-score interpretation |
|---|---|---|
| CE10 compatibility | Original integer total at indices 0–9; 0, 1, 9 are unobserved under the new contract | Sum of class probability × original score; report probability assigned to unsupported classes |
| CE7 support control | Original totals 2–8, indexed 0–6 internally | Expected original total |
| CE5 coarsening | `{2}`, `{3}`, `{4,5}`, `{6,7}`, `{8}` | Train-only conditional mean total within each bin, frozen before validation |
| Two CE4 heads | Original title and thumbnail ratings separately | Sum of expected component ratings |
| Joint 16-class head | Original 4 × 4 title/thumbnail pair | Sum-marginalized total distribution, preserving component dependence |
| Ordinal head | Ordered thresholds over the seven total values | Monotone cumulative probabilities and expected original total |
| Regression control | Original total | Clipped score for serving; unmodified residuals also reported |

The CE5 grouping preserves the irrelevant boundary, distinguishes both irrelevant totals, separates partial/mixed from strong totals, and reserves a class for exact agreement on both components. It is a study choice, not an assertion that five equally spaced human categories were collected. Confirmatory reporting compares all variants on the same ranking target, original-scale MAE, irrelevant detection, and derived five-class task. Native five-class and native ten-class accuracy are never directly compared as though they were the same task.

### 3.3 Data validity gates

Before model selection, audit full row counts, exact and normalized key duplicates, missing/invalid labels, total/component agreement, source join cardinality, query and pair overlap, type normalization, and source-by-label distributions. Validate array lengths and finite values for every embedding. Distinguish SQL null, zero-filled missingness, malformed vector, and valid zero similarity; zero alone is not proof of absence.

The upstream split is a stable MD5 **query-level** 90/5/5 split, which is valuable for unseen-query evaluation. Confirm that the final snapshot preserves it. Existing golden sets may overlap queries or items with training even when exact pairs were removed. Publish the overlap audit and label each set accordingly. Fit transformations, vocabularies, percentile bins, class weights, feature imputation, and calibration on the permitted training/development data only. Exclude `reason`, `raw_response`, judge-generated query classification, labels, and any answer-derived fields from prediction inputs.

**Completed audit:** all 9,950,940 main-split rows satisfy the new label contract. No normalized duplicate pairs or query/pair overlaps exist among train, validation, and test. However, the existing golden, golden-3D, long-tail, and long-tail-3D sets respectively contain **402, 418, 55, and 42 exact training pairs**; these counts are not additive unique-pair totals. Their labels span 0–8 and many components are outside the new 1–4 contract. This is evidence of an incompatible legacy rubric, not proof that those legacy labels are corrupt. Keep these sets as separate historical diagnostics, remove overlapping pairs in evaluation views, and do not use their native class accuracy or thresholds as if they were Gemini-scale labels. Establish a same-rubric bridge set before interpreting cross-rubric score calibration.

Both scalar similarities equal zero in **8,063,805 / 8,953,912 training rows (90.06%)**. Zero-filled defaults require a missingness/lineage audit; this count alone cannot distinguish a missing feature from a legitimate zero. All rows carry `ds=2026-08-13`, but that row tag alone does not establish the provenance date of every joined feature.

A source-week label does not prove that features are historical as of that week. Record the actual item, engagement, Q2D, D2Q, and embedding snapshot dates. Future-derived behavior features can invalidate a forward-generalization claim even with query-disjoint splits. If historical reconstruction is unavailable, identify the study as a retrospective feature experiment and restrict temporal claims.

## 4. Evaluation and statistical protocol

### 4.1 Evaluation populations

1. **Locked query-disjoint test:** preserve the finalized test membership after integrity auditing; use it only for the registered finalists.
2. **Validation:** all hyperparameter selection, mixture screening, calibration, threshold tuning, and early stopping.
3. **Existing golden and long-tail sets:** compatibility diagnostics; verify label origin before describing a set as human-labeled or independent of Gemini.
4. **Future traffic:** at least two nonoverlapping weeks after the source week, with both previously seen and new queries/items. Preserve historical feature cutoffs and report candidate-pool changes.
5. **Cold-start and missingness slices:** new items, unseen query–item combinations, missing each embedding/similarity family, and combinations of missing features.
6. **Independent judgment:** use an existing human set if verified. Otherwise prepare a stratified disagreement sample for human review; do not manufacture a human evaluation from another teacher pass.

Head/middle/tail buckets are fixed using exposure-frequency information available before evaluation, not training-sample count, which balancing intentionally changes. If original query-level traffic frequency is unavailable, label a pair-count-derived estimate as a proxy and do not call it exact traffic weighting. Source is crossed with head/tail and 2D/3D where sample size permits. Counts and confidence intervals accompany every slice; tiny slices remain descriptive.

### 4.2 Outcomes

**Co-primary outcomes:** macro-query NDCG@10 on the natural pool of the new Gemini evaluation split and macro-query NDCG@10 on its predeclared long-tail query slice. The legacy long-tail set is a secondary compatibility diagnostic until a same-rubric bridge is available. Report linear gain on original totals for production compatibility; also report zero-based gain `total−2`, since assigning positive gain to irrelevant items can inflate NDCG. The zero-based transform applies to the Gemini rubric only; it must not create negative gains on legacy labels. Preserve a specified tie policy. Queries with zero ideal gain under the zero-based definition are excluded from that NDCG with their count reported and are evaluated for irrelevant rejection separately.

Secondary outcomes include NDCG@1/5/20, pair-weighted and macro-query accuracy, balanced accuracy, macro-F1, per-class precision/recall, original-scale MAE, quadratic weighted kappa, ordinal error distance, multiclass log loss, Brier score, and calibration plots. NLL/Brier are comparable only on an identical support; report a shared five-bin distribution when comparing CE5 and CE10. Report false acceptance of irrelevant pairs (`total < 4`) at fixed relevant recall, AUROC/PR curves with class prevalence, and behavior on relevant random pairs. Publish confusion matrices and title/thumbnail disagreement slices.

For thresholded metrics, choose thresholds on validation at fixed operating points, then freeze them. A production score threshold is not portable across output heads without calibration. A test pool with extra random negatives is not representative production traffic; present natural traffic and challenge pools separately before any declared weighted aggregate.

### 4.3 Uncertainty and selection

Use paired query-cluster bootstrap resampling (2000 resamples) for differences on the same query lists; pairs within a query are not independent observations. Confirm finalists with at least three seeds, reporting each seed and the mean/dispersion. Seed variability and query-sampling uncertainty are different quantities and are reported separately. Temporal uncertainty also requires day/week-level variation, not only a bootstrap within one week.

Screening is exploratory. Before opening the final test, register the finalist list, model-selection rule, comparisons, and metric code hash. Apply Holm correction to the registered family of confirmatory primary comparisons. A practical improvement threshold and acceptable head-query regression margin will be set from baseline variability and serving requirements before candidate results are inspected. Until these margins are frozen, no model is declared a deployment winner. An offline success motivates a separate online experiment; it does not establish purchase lift.

## 5. Controlled experiment sequence

This is a staged design, not a Cartesian sweep across every knob. Each stage freezes the preceding controls and records exceptions. The exact number of admitted jobs depends on the allocated hardware budget and measured pilot costs.

| Stage | Changed variable | Controls | Admission criterion |
|---|---|---|---|
| A | Integrity and production reproduction | Pinned artifacts, transformations, real examples | Valid joins/splits; offline versus exported-model parity |
| B | Training systems | Same examples, initial state, batches, objective, updates | Numerical/quality parity and measured end-to-end improvement |
| C | Data size | Production architecture and audited source proportions | Completed budget-matched curves |
| D | Source mixture | Fixed unique-pair count and optimization budget | Feasible source capacities; improvement on declared validation populations |
| E | Feature groups and labels | Representative size/mixture; matched tuning budgets | Robustness and information-parity checks |
| F | Architecture and capacity | Same feature/input representation, data, and training budget | Better quality–cost frontier after controlled tuning |
| G | Small-LM post-training and output head | Same backbone revision, information, examples, tokens, tuning budget | Complete optimizer pilot and valid evaluation |
| H | Expert/fusion models and distillation | Best individual branches, disjoint calibration data | Complementary errors and useful net serving tradeoff |
| I | Confirmation | Frozen recipes, ≥3 seeds, locked test and temporal evaluations | Registered generalization and latency criteria |

### 5.1 Size scaling: distinguish data from compute

Use nested, deterministic subsets of **0.25M, 1M, 2M, 4M, and all feasible training pairs**. The full training set is 8.95M, not 10M after adding held-out data. Nested selections share pair ranks within each source and seed. Record unique queries/items and repeated exposures at every size.

Run two views. In the **fixed-update** view, hold global effective batch and optimizer updates constant, explicitly reporting repetition of small subsets. In the **fixed-epoch** view, hold passes constant, allowing larger data to consume more compute. Plot against unique pairs, examples actually consumed, optimizer updates, estimated training FLOPs, GPU-hours, and wall time. Holding epochs fixed is not holding training budget fixed. Changing batch size changes the statistical experiment unless effective batch and update schedule are preserved.

Fit empirical power laws only with enough reliable points and residual checks. Pretraining scaling exponents from language-model papers are not assumed to apply to this supervised ranking problem. Report plateaus and uncertainty rather than extrapolating a claimed optimum far beyond observed data.

### 5.2 Mixture design and capacity constraints

Begin with a 500k-unique-pair mixture screen: the 15 simplex points with source weights in increments of 0.25, plus the approximately 70/20/10 reference if distinct. This includes pure-source controls and mixtures along every edge. Reserve the same evaluation sets for every run. Follow up only the best few neighborhoods and a reference mixture at 2M and larger **feasible** sizes.

For source capacities `C_s` and desired proportions `w_s`, sampling without replacement requires `N ≤ min(C_s / w_s)` over positive weights. Approximately one million labeled random pairs does not permit an arbitrary random fraction at full scale. A requested infeasible combination is marked infeasible; it is never silently filled with duplicated examples. Oversampling is a separate explicitly labeled exposure-weighting experiment with unique count held visible.

Test random fractions 0/5/10/20% in a follow-up at a feasible size while keeping the natural:balanced ratio within the remaining mass fixed. Compare equal unique-pair weight with clipped `log1p(pair_count)` and square-root weighting, normalizing to mean one. Do not simultaneously alter sampling and loss weighting without documenting the resulting effective target distribution. Measure effective sample size `(sum w)^2 / sum(w^2)`.

### 5.3 Feature information and failure modes

| Group | Inputs and purpose |
|---|---|
| Lexical | Query/title match features; inexpensive exact and partial matching |
| Frozen semantic | Text/image embeddings and query–item similarities |
| Historical behavior | Item, query, and pair exposure/click/purchase histories, with corrected Q2D and D2Q/platform signals separated |
| Raw content | Query, title, legitimate catalog attributes; images for multimodal models |
| Hybrid | Learned semantic representation fused with structured features through a separate normalized projection |

Compare lexical-only, semantic-only, behavior-only, semantic+lexical, and all-production features. Then remove clicks, purchases, exposures, D2Q/platform, and Q2D families individually and in justified combinations. Treat behavioral signals as observational predictors; their presence does not make them causal evidence of relevance. Exclude identifiers that enable pair memorization unless explicitly studying an ID baseline.

Run both **retrained ablations** (can the remaining information learn the task?) and **inference perturbations** (how does a fitted model fail when a signal disappears?). They answer different questions. Probe missingness alone, zero/missing masks, shuffled behavior within controlled strata, age of statistics, and behavior-preserving versus content-preserving counterfactuals. A classifier that easily predicts random-source membership from missing features motivates stronger source-shortcut checks, not an automatic conclusion that relevance was learned.

### 5.4 Architecture and optimization

Start with linear/ordinal logistic and boosted-tree controls, an MLP, the production DCN, residual MLPs, and a feature-token transformer. Scale measured trainable parameters approximately around **1M, 6M, 20M, and 60M** where each architecture permits; include preprocessing and embedding reducers in the count. Sweep cross depth {0,2,4}, reducer width {128,256,512}, and MLP width/depth in staged subsets. Dense versus low-rank cross layers require matched parameter or compute comparisons, not a claim of free scaling.

For scratch models, screen LR `{1e-4, 3.5e-4, 1e-3}`, weight decay `{0, 0.01, 0.1}`, dropout `{0, 0.15, 0.3}`, and exposure budgets equivalent to `{1,2,4}` epochs using successive stages rather than their full product. Preserve a literal production recipe as the reference. Compare AdamW with and without AMSGrad; consider SGD+momentum only with an adequate independently tuned LR range. Separate optimizer quality comparisons from fused-versus-unfused implementation comparisons. Record warmup fraction, schedule length, minimum LR, clipping, precision, batch, and actual update count.

Compare CE, ordinal cumulative loss, CE plus an ordinal-distance penalty, and Huber regression on original scores. A pairwise RankNet-style or listwise extension uses within-query pairs/lists only, with an auxiliary classification term to preserve calibration. Query-balanced sampling and grouping are frozen for these loss comparisons. Focal loss is secondary: imbalance alone is not evidence it improves calibrated relevance.

### 5.5 Sub-4B language models: pretraining, head, and inputs

The initial controlled text family is **Qwen3-0.6B and Qwen3-1.7B** at pinned revisions. A roughly 3B candidate from another family is a serving-size comparison and must not be presented as a clean within-family scaling point. **Qwen3.5-0.8B/2B** and **Qwen3-VL-2B** are additional verified-access candidates for a later architecture/multimodal stage. Count all parameters required at serving, including vision and projection modules; “active parameters” alone does not satisfy the below-4B limit.

For each selected backbone, compare:

1. **Generative classification:** a minimal fixed prompt and a class-token answer; loss only on the answer. Validate tokenization of every class code in context. If codes are not all single tokens, use distinct registered codes or exact sequence likelihoods and report the cost.
2. **Restricted token scoring:** read the same LM-head class logits at the final prompt position, renormalize over permitted classes, and avoid unnecessary free decoding. This separates vocabulary-head choice from autoregressive decoding overhead.
3. **Classification head:** identical backbone/input prefix, a new K-way linear head on the final non-padding prompt representation; proper attention masks and left/right-padding checks are mandatory.
4. **Ordinal or component head:** ordered thresholds or separate title/thumbnail outputs from the same representation.

Token CE over the full vocabulary and CE over a restricted label vocabulary are different objectives. Report both rather than attributing all differences to “token versus head.” Include random-initialized matched-backbone and frozen-backbone/linear-probe controls on an affordable subset to isolate pretraining and adaptation. Main post-training uses full parameters; adapters or quantized training require an explicit separate experiment, not a silent substitution.

Text-only input contains the raw query and item title. Add catalog attributes in a controlled second format. Add images in an information-matched multimodal comparison. For numeric/embedding features, compare a hybrid projection/fusion head with carefully specified textual summaries; do not serialize thousands of floating-point embedding coordinates as ordinary text by default. A text-only LM cannot be credited with failing to use an image it never received. Re-fetching a thumbnail is logged by content hash and timestamp because the image may have changed since teacher labeling.

Screen full-parameter LR `{5e-6, 1e-5, 2e-5}`, answer-only loss, context lengths 128/256/512 subject to measured truncation, and one/two passes before scaling. Compare raw-content models on matched example sets, and separately provide best attainable quality at matched GPU-hours. Do not use teacher rationales as input. Rationale-supervised training is an optional separately costed target with short class-only serving; it is not the default serving format.

### 5.6 Mixtures of models, specialization, and distillation

Data mixtures and mixtures of experts are separate axes. First measure whether semantic and behavioral branches make complementary errors. Compare calibrated late fusion, a learned small gate, and a two/three-expert architecture with dense controls at comparable parameter and serving budgets. Expert routing may use serving-available semantics, item type, and feature availability; it may not use the offline natural/balanced/random sampling label. Record routing entropy, expert load, collapse, per-slice routing, total/active parameters, and latency.

A cascade can apply a cheap feature model to all candidates and a small LM only to uncertain/high-impact cases. Its evaluation must include gate mistakes and the full candidate-selection process. Distill a stronger sub-4B model or validated ensemble into the production-size model using train-only soft predictions, retaining the original hard labels and held-out separation. Distillation quality is measured on the original evaluation labels, not the teacher's own predictions.

## 6. Training and serving systems protocol

The scheduler reserves two B200 GPUs, approximately 46 CPU cores, and 480 GiB RAM for this devspace. Training is restricted to those two GPU UUIDs. The total study budget and serving SLA remain unspecified; stages are admitted using measured cost. Eight B200s are visible on the shared host; visibility or idleness is not allocation. Each launched job records physical UUIDs, logical indices, process/container IDs, and the allocation decision. Home storage has only approximately 26 GB free, so large reproducible caches use the spacious scratch filesystem and durable artifacts use the study's private object-store prefix. Existing users' files and processes are preserved.

### 6.1 Optimize the actual bottleneck

Profile data read/decompression, feature preprocessing, host-to-device transfer, forward/backward, optimizer, evaluation, and checkpoint writes separately. The feature snapshot is roughly 188 GB including main splits; repeatedly decoding double-precision embedding lists can dominate a small network. Project required columns, validate once, materialize contiguous float32 arrays with explicit missing masks, reuse item/query embedding caches, and precompute deterministic lexical features. Float32 conversion receives numerical parity checks; lower precision storage requires a separate validation.

Use pinned memory, bounded prefetch, persistent workers, and vectorized preprocessing when measured helpful. Cache after split/provenance checks. Benchmark one GPU before DDP for the 6M model; communication and input work may make more GPUs slower. Independent small trials may use the allocation more efficiently than a single over-distributed trial. For LMs, use length bucketing and attention-isolated packing; ordinary concatenation that lets examples attend across boundaries changes the experiment.

BF16 autocast, fused AdamW, scaled-dot-product/FlashAttention kernels, `torch.compile`, and CUDA graphs are candidates, not assumed wins. Maintain FP32 optimizer states and stable loss reductions as required. Run pilots through the first optimizer update so optimizer states and workspaces are allocated. Measure memory peaks, nonzero gradients, parameter changes, actual trainable coverage, and loss behavior. Compare eager/optimized outputs and gradients at justified tolerances, then compare held-out quality using the same optimization path. Disable a speed optimization that changes the intended objective or degrades quality beyond the registered tolerance.

### 6.2 Budget accounting and benchmarks

Record cold-start/compile time and steady-state throughput separately; report total run wall time including evaluation, checkpointing, and upload. Count **allocated** GPU-hours, not just kernel-active time. Estimate model memory from weights, master copies, gradients, optimizer states, activations and buffers, then validate the estimate by measurement. Full-parameter Adam training commonly needs far more than the BF16 weight size; a model loading successfully is not proof that training fits.

Benchmark serving at candidate batch sizes **1, 32, 128, 360**, input-length percentiles, and realistic concurrent load. Report pairs/s, queries/s with candidate count, p50/p95/p99, memory, preprocessing and encoder cost, and cold/warm cache behavior. For LM generation, include prefill and decoding. For classification heads, measure the actual full forward pass. B200 measurements characterize experiments; they do not prove the same latency on the production accelerator. Compare quantized inference only after an unquantized quality baseline, with calibration and slice checks repeated.

## 7. Reproducibility, run registry, and publication

Every run records source commit, code/config hash, data object manifest and ETags, normalization version, subset IDs, sampling seed, source weights, label mapping, input fields, architecture/parameter counts, initialization/backbone revision, trainable coverage, optimizer and schedule, update/sample/token counts, precision, hardware UUIDs, software versions, checkpoint paths, evaluation code hash, wall time, and exit state. ETags identify objects but are not universally file SHA-256 values; explicitly computed content digests are labeled accordingly.

Local JSONL logs and resumable status files are authoritative recovery records. W&B uses the explicit study project under Simon's account, with distinct audit/pilot/screen/confirmation tags. Failed runs remain visible with their failure reason; no silent recipe changes under an old run ID. Checkpoints retain model and, for exact resumption, optimizer/scheduler/RNG/sampler state. Raw data, credentials, internal endpoint metadata, and private checkpoints remain in the private artifact store.

The GitHub Pages report will expose the protocol, public methodology, aggregate results, figures, an experiment-status table, and a dated change log. It is updated after each completed stage or material plan change. Empty results stay visibly empty; planned experiments are not described as executed. Internal production source and individual query/item examples are not copied into the public repository. The final paper will include an abstract updated to actual findings, methods, results, ablations, limitations, and reproducibility materials.

## 8. Current evidence and unresolved dependencies

| Item | Status |
|---|---|
| Latest private repository | Fetched and pinned in an isolated worktree |
| Live baseline | Version 67 verified from traffic metrics; checkpoint read and hashed |
| Final dataset and provenance | Full metadata audit complete; main splits valid, source joins exact, legacy evaluation issues documented |
| Hugging Face | Public configs and a 0.6B weight download verified |
| W&B | Audit run written, finished, and remote history verified |
| GitHub Pages | Protocol and aggregate audit published; report updated after completed stages |
| Spark | Query execution and feature-table partition reads verified; future traffic exists through September 15; exit-status parsing handled in a study-specific wrapper |
| GPU allocation/budget | Two scheduler-reserved B200s verified; optimizer pilot passed on one; total study budget pending |
| Future feature cutoffs and human evaluation | To be audited before making temporal or human-quality claims |

## 9. Measured operational baseline and systems pilot

The pinned production model was evaluated on all **496,477 validation pairs**. This is an operational comparison: overlap between its historical training corpus and this evaluation population has not yet been established. The new scratch experiments preserve the audited query-disjoint split. The production reference was trained under an earlier labeling/data regime, so differences do not isolate architecture.

| Validation population | Pairs | Exact total accuracy | Balanced accuracy, supported totals | Original-scale MAE |
|---|---:|---:|---:|---:|
| All sources | 496,477 | 64.38% | 45.76% | 0.936 |
| Natural traffic | 347,924 | 63.70% | 47.21% | 0.874 |
| Query-balanced | 98,981 | 55.38% | 32.60% | 1.206 |
| Random unexposed 3D | 49,572 | 87.14% | 47.39% | 0.834 |

The shared five-bin accuracy is **68.74%**; relevant-versus-irrelevant AUROC is **0.9055**. The model assigns **5.44%** of probability mass to unsupported classes 0, 1, and 9. At a validation-tuned threshold achieving 95% relevant recall, irrelevant false acceptance is **40.67%**. These are validation operating points, not test estimates. Only **3.14%** of random-pool pairs are relevant, so its high raw accuracy does not demonstrate strong discrimination.

Natural-pool zero-based NDCG@10 is **0.9570**, over **27,411** multi-item queries with nonzero ideal gain. Of 141,510 natural queries, 111,090 are singletons. The query-balanced pool contains 98,955 queries but only **25 multi-item queries**. These sampled lists do not establish complete-result-list or robust long-tail ranking quality. Complete-list evaluation remains necessary.

Item text/image vectors are all-zero for **163,211** validation rows; query text/image vectors are all-zero for **169,370** rows. Zero-vector slices are reported explicitly without assuming every zero scalar similarity is missing.

The infrastructure pilot completed forward/backward and fused AdamW updates; all **37 parameter tensors changed**, with finite nonzero gradients. It used a repeated small training batch and therefore makes no generalization claim. On one B200, BF16 at batch 4096 reached approximately **939k examples/s** over 20 timed updates after five warmup updates, excluding input preparation; peak allocated GPU memory was 0.60 GB. FP32 at batch 512 reached approximately 110k/s and BF16 at the same batch reached 119k/s. Larger batch speed is not a matched statistical-quality comparison. Full-data loading and held-out precision comparisons are still required.

The initially available PyTorch 2.10/CUDA 13 environment failed cuBLAS matrix multiplication. A pre-existing PyTorch **2.11/CUDA 13** environment passed matrix multiplication and the optimizer pilot; the failed attempt is retained in the run journal. The feature cache preserves float32 values and row provenance; production replay and rubric/metric tests passed before training preparation.

## References

1. Wang et al. **DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems.** [arXiv:2008.13535](https://arxiv.org/abs/2008.13535). Motivation for explicit feature interactions and capacity controls.
2. Gorishniy et al. **Revisiting Deep Learning Models for Tabular Data.** [arXiv:2106.11959](https://arxiv.org/abs/2106.11959). Motivation for strong MLP/residual baselines and feature-token transformer comparisons.
3. Cao et al. **Rank consistent ordinal regression for neural networks with application to age estimation.** [arXiv:1901.07884](https://arxiv.org/abs/1901.07884). Motivation for respecting ordered labels rather than treating every class error equally.
4. Burges et al. **Learning to Rank using Gradient Descent.** [Microsoft Research](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/). Pairwise ranking objective reference.
5. Kaplan et al. **Scaling Laws for Neural Language Models.** [arXiv:2001.08361](https://arxiv.org/abs/2001.08361). Experimental scaling methodology, without assuming transfer of fitted exponents.
6. Hoffmann et al. **Training Compute-Optimal Large Language Models.** [arXiv:2203.15556](https://arxiv.org/abs/2203.15556). Motivation for jointly accounting for parameters, data, and compute; not a recipe for this supervised task.

The six publication titles and source pages were checked during protocol preparation. Repository, dataset, and checkpoint evidence is retained in the private audit manifest.
