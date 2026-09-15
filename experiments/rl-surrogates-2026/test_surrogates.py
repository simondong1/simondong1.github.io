"""Analytic gradient and verifier checks for the experimental extension."""
import math

import torch

from rewards import countdown_score, integer_score
from surrogates import token_surrogate


def gradient(name, p, q, advantage):
    lp = torch.tensor(math.log(p), dtype=torch.float64, requires_grad=True)
    lq = torch.tensor(math.log(q), dtype=torch.float64)
    a = torch.tensor(float(advantage), dtype=torch.float64)
    loss, _, _ = token_surrogate(name, lp, lq, a)
    loss.backward()
    return lp.grad.item()


def main():
    for name in ("ppo", "dapo", "cispo", "glm5", "sapo", "dppo"):
        for a in (-1, 1):
            assert abs(gradient(name, 0.2, 0.2, a) + a) < 1e-10, name
    assert gradient("ppo", 0.3, 0.2, 1) == 0
    assert abs(gradient("ppo", 0.3, 0.2, -1) - 1.5) < 1e-10
    assert gradient("glm5", 0.01, 0.001, 1) == 0
    # A completed GLM run reached the shared +/-20 log-ratio guard. Both
    # guarded and unguarded ratios remain outside GLM's rejection interval,
    # so this numerical guard adds no change to its selected gradient.
    for log_ratio in (-50, -25, 25, 50):
        logp, logq = (-1 + log_ratio, -1) if log_ratio < 0 else (-1, -1-log_ratio)
        for advantage in (-1.5, 0, 1.5):
            assert gradient("glm5", math.exp(logp), math.exp(logq), advantage) == 0
    assert abs(gradient("cispo", 0.01, 0.001, 1) + 5) < 1e-10
    assert abs(gradient("dppo", 0.01, 0.001, 1) + 10) < 1e-10
    assert gradient("dppo", 0.7, 0.5, 1) == 0
    assert abs(gradient("dppo", 0.7, 0.5, -1) - 1.4) < 1e-10
    assert -1.5 < gradient("sapo", 0.3, 0.2, 1) < 0
    assert countdown_score(r"\boxed{(10-3)*(5+2)}", [10, 3, 5, 2], 49) == 1
    assert countdown_score(r"\boxed{49}", [10, 3, 5, 2], 49) == 0
    assert countdown_score(r"\boxed{10+10}", [10, 3], 20) == 0
    assert countdown_score(r"\boxed{2**3}", [2, 3], 8) == 0
    assert countdown_score(r"\boxed{__import__('os').system('id')}", [1], 1) == 0
    assert countdown_score(r"\boxed{3/(4-4)}", [3, 4, 4], 1) == 0
    assert countdown_score(r"\boxed{40 \times \frac{24}{18 - 6}}", [40, 24, 18, 6], 80) == 1
    assert countdown_score(r"\boxed{\frac{27}{\frac{81}{3}}}", [27, 81, 3], 1) == 1
    assert countdown_score(r"\boxed{\frac{1}{0}}", [1, 0], 0) == 0
    assert countdown_score(r"\boxed{\frac{1}{__import__('os')}}", [1], 1) == 0
    assert countdown_score(r"\boxed{-3+5}", [3, 5], 2) == 1
    assert integer_score(r"wrong \boxed{2}; corrected \boxed{-13}", "-13") == 1
    assert integer_score(r"\boxed{13+0}", "13") == 0
    print("PASS: analytic surrogate gradients and deterministic reward verifiers")


if __name__ == "__main__":
    main()
