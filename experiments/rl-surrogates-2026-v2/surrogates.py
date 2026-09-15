"""Reference surrogates and a Miles custom-loss extension.

All functions minimize a loss. Behavior log probabilities and advantages are
constants. Training uses Miles' model, tokenizer, packing, log-prob extraction,
GRPO advantages, distributed reduction and optimizer; only the surrogate changes.
"""
import os

import torch


def token_surrogate(name, logp, logq, advantage, *, low=0.2, high=0.2):
    logq = logq.detach()
    advantage = advantage.detach()
    ratio = (logp - logq).clamp(-20, 20).exp()
    if name in ("ppo", "dapo"):
        upper = high if name == "ppo" else 0.28
        raw = -ratio * advantage
        clipped = -ratio.clamp(1 - low, 1 + upper) * advantage
        loss = torch.maximum(raw, clipped)
        gate = ~(((advantage > 0) & (ratio > 1 + upper)) |
                 ((advantage < 0) & (ratio < 1 - low)))
        weight = ratio.detach() * gate
    elif name == "glm5":
        gate = (ratio > 0.5) & (ratio < 5.0)
        weight = (ratio * gate).detach()
        loss = -weight * advantage * logp
    elif name == "cispo":
        # The 0.5/5 calibration matches GLM-5 numerically but clamps rather
        # than discarding. This is a controlled surrogate, not full M1 training.
        weight = ratio.detach().clamp(0.5, 5.0)
        gate = torch.ones_like(ratio, dtype=torch.bool)
        loss = -weight * advantage * logp
    elif name == "dppo":
        with torch.no_grad():
            diff = logp.exp() - logq.exp()
            gate = ~(((advantage > 0) & (diff > 0.15)) |
                     ((advantage < 0) & (-diff > 0.15)))
        weight = ratio.detach() * gate
        loss = -(gate * ratio * advantage)
    elif name == "sapo":
        tau = torch.where(advantage > 0, torch.ones_like(logp), torch.full_like(logp, 1.05))
        sigmoid = torch.sigmoid(tau * (ratio - 1))
        loss = -(4 / tau) * sigmoid * advantage
        weight = (4 * sigmoid * (1 - sigmoid) * ratio).detach()
        gate = weight > 0
    else:
        raise ValueError(f"Unknown token surrogate {name}")
    return loss, weight, gate


def sequence_surrogate(logp, logq, advantage, mask, lengths, low=3e-4, high=4e-4):
    """GSPO: geometric mean ratio, one directional clipping decision per response."""
    losses, weights, gates = [], [], []
    offset = 0
    for length in lengths:
        sl = slice(offset, offset + length); offset += length
        m = mask[sl].to(logp.dtype); count = m.sum().clamp_min(1)
        a = (advantage[sl].detach() * m).sum() / count
        logr = ((logp[sl] - logq[sl].detach()) * m).sum() / count
        ratio = logr.clamp(-20, 20).exp()
        loss = torch.maximum(-ratio * a, -ratio.clamp(1-low, 1+high) * a)
        gate = ~(((a > 0) & (ratio > 1+high)) | ((a < 0) & (ratio < 1-low)))
        losses.append(loss.expand(length)); weights.append((ratio.detach()*gate).expand(length))
        gates.append(gate.expand(length))
    assert offset == logp.numel()
    return torch.cat(losses), torch.cat(weights), torch.cat(gates)


def miles_loss(args, batch, logits, sum_of_sample_mean):
    """Miles custom-loss hook. CP=1; fixed GRPO/sequence-mean protocol."""
    from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
    from miles.backends.training_utils.parallel import get_parallel_state

    assert get_parallel_state().cp.size == 1
    assert not args.calculate_per_token_loss, "This experiment fixes sequence-mean aggregation"
    assert not args.use_kl_loss and args.entropy_coef == 0
    output = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
        with_entropy=False, max_seq_lens=batch.get("max_seq_lens"),
    )
    logp = torch.cat(output["log_probs"])
    logq = torch.cat(batch["rollout_log_probs"]).detach()
    advantage = torch.cat(batch["advantages"]).detach()
    mask = torch.cat(batch["loss_masks"]).to(logp.device).bool()
    logp = torch.where(mask, logp, torch.zeros_like(logp))
    logq = torch.where(mask, logq, torch.zeros_like(logq))
    advantage = torch.where(mask, advantage, torch.zeros_like(advantage))
    name = os.environ.get("RL_SURROGATE", "ppo")
    if name == 'gspo':
        per_token_loss, weights, gate = sequence_surrogate(logp, logq, advantage, mask, batch['response_lengths'])
    else:
        per_token_loss, weights, gate = token_surrogate(name, logp, logq, advantage)
    loss = sum_of_sample_mean(per_token_loss)
    with torch.no_grad():
        ratio = (logp - logq).clamp(-20, 20).exp()
        active = (advantage != 0).float()
        logs = {
            "loss": loss.detach(), "pg_loss": loss.detach(),
            "pg_clipfrac": sum_of_sample_mean((~gate).float() * active),
            "nonzero_adv_fraction": sum_of_sample_mean(active),
            "ratio_mean": sum_of_sample_mean(ratio),
            "ratio_log_clamp_fraction": sum_of_sample_mean(((logp-logq).abs() > 20).float()),
            "gradient_weight_mean": sum_of_sample_mean(weights * active),
            "rollout_logprob_abs_diff": sum_of_sample_mean((logp - logq).abs()),
            "sampled_token_surprise": sum_of_sample_mean(-logp),
            "approx_kl_behavior": sum_of_sample_mean(ratio - 1 - (logp-logq)),
            "positive_adv_fraction": sum_of_sample_mean((advantage > 0).float()),
            "negative_adv_fraction": sum_of_sample_mean((advantage < 0).float()),
            "positive_gradient_weight": sum_of_sample_mean(weights * (advantage > 0)),
            "negative_gradient_weight": sum_of_sample_mean(weights * (advantage < 0)),
            "positive_gated_fraction": sum_of_sample_mean((~gate) * (advantage > 0)),
            "negative_gated_fraction": sum_of_sample_mean((~gate) * (advantage < 0)),
        }
        if batch.get("log_probs") is not None:
            before_update = torch.cat(batch["log_probs"]).detach()
            logs["initial_engine_logprob_abs_diff"] = sum_of_sample_mean((before_update - logq).abs())
            logs['optimizer_logprob_abs_drift'] = sum_of_sample_mean((logp-before_update).abs())
            for threshold in (0.01, 0.1, 1, 5):
                logs[f'initial_engine_abs_gt_{threshold}'] = sum_of_sample_mean(((before_update-logq).abs()>threshold).float())
        for label, lower, upper in [('lt05',0,.5),('05_08',.5,.8),('08_1',.8,1),
                                    ('1_12',1,1.2),('12_2',1.2,2),('2_5',2,5),('ge5',5,float('inf'))]:
            logs['ratio_hist_'+label] = sum_of_sample_mean(((ratio>=lower)&(ratio<upper)).float())
        # Evaluate every gate on precisely the same inputs for an explanatory
        # diagnostic; only the selected surrogate contributes to backprop.
        for other in ("ppo", "dapo", "cispo", "glm5", "sapo", "dppo"):
            _, w, g = token_surrogate(other, logp, logq, advantage)
            logs[f"audit_{other}_gated"] = sum_of_sample_mean((~g).float() * active)
            logs[f"audit_{other}_weight"] = sum_of_sample_mean(w * active)
        _, w, g = sequence_surrogate(logp, logq, advantage, mask, batch['response_lengths'])
        logs['audit_gspo_gated'] = sum_of_sample_mean((~g).float() * active)
        logs['audit_gspo_weight'] = sum_of_sample_mean(w * active)
    return loss, logs
