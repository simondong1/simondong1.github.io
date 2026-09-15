"""Measured Miles arguments, with storage and W&B destinations configurable."""
import os
from pathlib import Path
from study_config import validate_config

ROOT = Path(os.environ.get("STUDY_ROOT", str(Path(__file__).resolve().parent)))
MODEL = Path(os.environ.get("STUDY_MODEL", str(ROOT / "model")))
INITIAL = Path(os.environ.get("STUDY_INITIAL", str(ROOT / "initial_checkpoint")))

def arguments(config):
    shape = validate_config(config)
    rollouts = config["samples"] // shape.trajectories
    task = config["task"]
    evaluation = config.get("eval_file", f"{task}-validation.jsonl")
    flags = [
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1", "--rollout-num-gpus", "1",
        "--num-gpus-per-node", "2", "--rollout-num-gpus-per-engine", "1",
        "--hf-checkpoint", str(MODEL), "--ref-load", str(INITIAL),
        "--no-load-optim", "--no-load-rng",
        "--swiglu", "--num-layers", "28", "--hidden-size", "1536", "--ffn-hidden-size", "8960",
        "--num-attention-heads", "12", "--use-rotary-position-embeddings", "--disable-bias-linear",
        "--add-qkv-bias", "--normalization", "RMSNorm", "--norm-epsilon", "1e-6",
        "--rotary-base", "1000000", "--group-query-attention", "--num-query-groups", "2", "--vocab-size", "151936",
        "--prompt-data", str(ROOT / "data" / f"{task}-train.jsonl"),
        "--input-key", "prompt", "--label-key", "label", "--apply-chat-template", "--rollout-shuffle",
        "--custom-rm-path", "study_rewards.reward", "--balance-data",
        "--num-rollout", str(rollouts), "--rollout-batch-size", str(shape.prompts),
        "--n-samples-per-prompt", str(shape.group), "--global-batch-size", str(shape.batch),
        "--num-steps-per-rollout", str(shape.steps),
        "--rollout-max-response-len", str(config.get("response_length", 1024)), "--rollout-temperature", "1.0",
        "--seed", str(1000 + config["seed"]), "--rollout-seed", str(2000 + config["seed"]),
        "--eval-interval", str(config["eval_every_samples"] // shape.trajectories),
        "--eval-prompt-data", task, str(ROOT / "data" / evaluation),
        "--n-samples-per-eval-prompt", "1", "--eval-temperature", "0.0", "--eval-top-p", "1.0",
        "--eval-max-response-len", str(config.get("response_length", 1024)),
        "--advantage-estimator", "grpo", "--entropy-coef", "0", "--kl-coef", "0",
        "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--optimizer", "adam", "--lr", str(config["lr"]), "--lr-decay-style", "constant",
        "--weight-decay", "0.1", "--adam-beta1", "0.9", "--adam-beta2", "0.98", "--clip-grad", "1.0",
        "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1", "--context-parallel-size", "1",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", str(config.get("max_tokens_per_gpu", 8192)),
        "--log-probs-chunk-size", "1024", "--attention-backend", "flash",
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32",
        "--sglang-mem-fraction-static", "0.65", "--sglang-attention-backend", "flashinfer",
        "--sglang-max-running-requests", "512", "--sglang-cuda-graph-max-bs", "512",
        "--sglang-chunked-prefill-size", "8192", "--sglang-decode-log-interval", "1000",
        "--custom-rollout-log-function-path", "study_hooks.rollout",
        "--custom-eval-rollout-log-function-path", "study_hooks.evaluation",
        "--use-wandb", "--wandb-host", os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai"), "--wandb-team", os.environ.get("WANDB_ENTITY", ""),
        "--wandb-project", "miles-four-knob-study", "--wandb-group", config["name"],
        "--disable-wandb-random-suffix", "--wandb-dir", str(ROOT / "wandb"),
    ]
    if config.get("save_final", False):
        flags += ["--save", str(ROOT / "checkpoints" / config["name"]), "--save-interval", str(rollouts)]
    if config.get("load"):
        flags += ["--load", config["load"]]
    if rollouts == 0:
        # Megatron still constructs an unused scheduler for eval-only jobs.
        # Its default decay length is zero, which fails initialization.
        flags += ["--lr-decay-iters", "1"]
    return flags
