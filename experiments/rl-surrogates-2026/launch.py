"""Launch one controlled Miles run inside the pinned Miles container.

The container exposes only the two allocated GPU UUIDs. No broad process-kill
commands are used. Each run has its own Ray runtime and artifact directory.
"""
import argparse
import json
import os
from pathlib import Path


def build_command(args):
    env = os.environ.copy()
    env.update(RL_SURROGATE=args.algorithm, PYTHONUNBUFFERED="1",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", WANDB_MODE="disabled",
               CUDA_DEVICE_MAX_CONNECTIONS="1", RAY_DEDUP_LOGS="0",
               PYTHONPATH="/experiment:/work/miles:/root/Megatron-LM")
    run_dir = Path("/artifacts/runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    env["TENSORBOARD_DIR"] = str(run_dir / "tensorboard")
    train_env = {"RL_SURROGATE": args.algorithm, "TENSORBOARD_DIR": env["TENSORBOARD_DIR"], "PYTHONPATH": env["PYTHONPATH"],
                 "CUDA_DEVICE_MAX_CONNECTIONS": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
    command = ["python", "/work/miles/train.py",
        "--train-backend", "fsdp", "--hf-checkpoint", "/model",
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "2", "--colocate",
        "--no-offload-train", "--no-offload-rollout",
        "--gradient-checkpointing", "--attn-implementation", "flash_attention_2",
        "--update-weight-buffer-size", "536870912", "--micro-batch-size", "1",
        "--prompt-data", f"/artifacts/data/{args.dataset}_{args.split}.jsonl",
        "--input-key", "prompt", "--label-key", "label", "--apply-chat-template", "--rollout-shuffle",
        "--apply-chat-template-kwargs", json.dumps({"enable_thinking": False}),
        "--custom-rm-path", "rewards.reward", "--num-rollout", str(args.rollouts),
        "--custom-rollout-log-function-path", "metrics.rollout",
        "--custom-eval-rollout-log-function-path", "metrics.evaluation",
        "--rollout-batch-size", str(args.prompts_per_rollout), "--n-samples-per-prompt", "4",
        "--rollout-max-response-len", str(args.max_tokens), "--rollout-temperature", "1",
        "--rollout-top-p", "1", "--global-batch-size", "8",
        "--advantage-estimator", "grpo", "--use-rollout-logprobs",
        "--loss-type", "custom_loss", "--custom-loss-function-path", "surrogates.miles_loss",
        "--entropy-coef", "0", "--kl-coef", "0", "--kl-loss-coef", "0",
        "--eps-clip", "0.2", "--eps-clip-high", "0.2",
        "--optimizer", "adam", "--lr", "1e-6", "--lr-decay-style", "constant",
        "--weight-decay", "0", "--clip-grad", "1", "--adam-beta1", "0.9", "--adam-beta2", "0.95", "--adam-eps", "1e-8",
        "--rollout-num-gpus-per-engine", "1", "--sglang-mem-fraction-static", "0.25",
        "--sglang-attention-backend", "triton",
        "--sglang-cuda-graph-max-bs", "32", "--sglang-max-running-requests", "32",
        "--sglang-context-length", str(args.max_tokens + 4096),
        "--sglang-decode-log-interval", "1000",
        "--eval-interval", str(args.eval_interval),
        "--eval-prompt-data", args.dataset, f"/artifacts/data/{args.dataset}_{args.eval_split}.jsonl",
        "--n-samples-per-eval-prompt", "1", "--eval-max-response-len", str(args.max_tokens),
        "--eval-temperature", "0", "--eval-top-p", "1",
        "--save", str(run_dir / "checkpoints"), "--save-interval", str(args.rollouts),
        "--save-debug-train-data", str(run_dir / "train_data" / "{rollout_id}_{rank}.pt"),
        "--save-debug-rollout-data", str(run_dir / "rollouts" / "{rollout_id}.pt"),
        "--save-debug-trajectory-data", str(run_dir / "trajectories" / "{rollout_id}.jsonl"),
        "--use-tensorboard", "--tensorboard-dir", str(run_dir / "tensorboard"),
        "--seed", str(args.seed), "--train-env-vars", json.dumps(train_env),
    ]
    if args.no_save:
        i = command.index("--save-interval")
        del command[i:i+2]
    if args.skip_eval:
        # Remove evaluation arguments entirely for the infrastructure smoke test.
        for flag, n in [("--eval-interval", 1), ("--eval-prompt-data", 2),
                        ("--n-samples-per-eval-prompt", 1), ("--eval-max-response-len", 1),
                        ("--eval-temperature", 1), ("--eval-top-p", 1)]:
            i = command.index(flag)
            del command[i:i+n+1]
    (run_dir / "command.json").write_text(json.dumps(command, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    return command, env


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--algorithm", choices=["ppo", "dapo", "cispo", "glm5", "sapo", "dppo"], default="ppo")
    p.add_argument("--dataset", choices=["countdown", "deepmath"], default="countdown")
    p.add_argument("--split", default="train")
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--prompts-per-rollout", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    command, env = build_command(args)
    os.execvpe(command[0], command, env)
