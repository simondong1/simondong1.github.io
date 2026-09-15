"""Run inside the experiment container: reference PPO versus Miles' native PPO."""
import torch
from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss
from surrogates import token_surrogate

torch.manual_seed(13)
logq = -torch.rand(1000, dtype=torch.float32) * 10
logp = (logq + torch.randn(1000) * 1.5).clamp_max(-0.01).requires_grad_()
a = torch.randn(1000)
custom, _, _ = token_surrogate('ppo', logp, logq, a)
native, _ = compute_policy_loss(logq-logp, a, 0.2, 0.2, eps_clip_c=None)
g1, = torch.autograd.grad(custom.sum(), logp, retain_graph=True)
g2, = torch.autograd.grad(native.sum(), logp)
torch.testing.assert_close(custom, native)
torch.testing.assert_close(g1, g2)
print('PASS: custom PPO loss and gradients match native Miles on 1,000 cases')
