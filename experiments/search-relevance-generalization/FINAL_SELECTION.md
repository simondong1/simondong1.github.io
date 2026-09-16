# Locked-test selection and analysis rules

Registered after exploratory validation screens and before any final-test model scoring. The final test remains locked until a machine-readable manifest records completed checkpoint hashes, calibration hashes, data hashes, and metric-code hashes.

The primary target is common-five negative log likelihood, with common-five accuracy as a secondary classification measure. Original-scale MAE, relevance AUROC, source/dimension/frequency/availability slices, and sampled-list ranking are secondary diagnostics. Native five-way and native ten-way accuracy are never substituted for this shared target.

Candidate recipes eligible for final scoring are the operational production export; the full-data two-pass production-size feature model at seeds 173/271/811; the full-data width-384 feature model; the three matched 250k Qwen3-0.6B output-head runs; the Qwen3-1.7B 250k classifier at seeds 173/271/811; and the best completed 0.6B classifier by full-validation common-five NLL among the registered learning-rate, epoch, and 1M-data runs. The raw-input format stays fixed. The 1M run is registered at LR1e-5 and one epoch, not selected using temporal outcomes.

Validation-fitted mixtures `fusion-v1-nodim` and `fusion-v2` are also eligible. Their temperatures, gate coefficients, scaling statistics, and cascade thresholds remain frozen. Their base models and total serving parameter counts are explicit. Three-seed reporting describes the variability of separate deployable checkpoints; a three-model 1.7B ensemble would violate the sub-4B total-serving limit and is not a serving candidate.

The six primary recipe contrasts are:

1. Width-384 feature model versus production-size feature model, full data/two passes/seed 173.
2. Qwen3-0.6B classifier versus production-size feature model; an operational recipe contrast with different representations, pretraining, capacity, data, and compute.
3. Qwen3-1.7B versus Qwen3-0.6B classifier, matched 250k data/one pass/seed 173; parameter and compute scaling within one model family.
4. Best validation-selected 0.6B classifier versus the original 0.6B classifier; skip if the selected checkpoint is identical.
5. 0.6B classifier versus full-vocabulary token output, matched backbone/data/order/optimization.
6. Frozen `fusion-v2` soft mixture versus its 1.7B text expert.

Use query-clustered uncertainty for paired changes on the immutable test rows. Report 95% bootstrap intervals for fixed-checkpoint effect sizes. Apply Holm correction across the six primary NLL contrasts using query-cluster-robust tests; secondary metrics and slices are exploratory. Seed variation is reported separately and is not replaced by narrow pair/query intervals. Sampling uncertainty does not remove historical-training-overlap uncertainty for the operational production model.

The temporal comparison uses the already registered frozen recipes, weighted logged requests, and complete displayed pages. It does not select training hyperparameters or calibration policies. Missing teacher labels exclude an entire affected request from ranking; coverage and image availability are reported. Date-stratified file-cluster bootstrap intervals preserve the two sampled dates and acknowledge that only eight file clusters per date were sampled. This does not establish an online ranking or purchase effect.

Serving comparison uses measured warm forward latency at candidate batches 1/32/128/360 and median/p95 prompt lengths, with CPU feature-model timing and B200 timing. Standalone forward measurements exclude RPC/queueing/cache/encoder/tokenization costs and are not an end-to-end production SLA. No serving SLA has been supplied, so the report will present the measured quality/cost frontier rather than claim that one recipe universally meets deployment requirements.

The study concludes with completed registered screens, the model-size and 1M data comparison, the three-seed production-size/1.7B confirmations, frozen final-test and temporal evaluations, serving measurements, verified private artifacts, and a published report. Further multimodal training, end-to-end hybrid heads, larger expert systems, distillation, human labeling, and online trials remain explicitly scoped follow-up work rather than implied completed experiments.
