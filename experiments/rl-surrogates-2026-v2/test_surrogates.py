import math
import torch
from rewards import score
from surrogates import token_surrogate, sequence_surrogate

for name in ['ppo','dapo','cispo','glm5','sapo','dppo']:
    for a in [-1.,1.]:
        p=torch.tensor([math.log(.2)],dtype=torch.float64,requires_grad=True)
        loss,w,g=token_surrogate(name,p,p.detach(),torch.full_like(p,a))
        loss.sum().backward();assert torch.allclose(p.grad,torch.full_like(p,-a)),name

# Heterogeneous token ratios with geometric mean one: GSPO retains every
# token while token-level PPO clips the high positive-advantage token.
q=torch.tensor([-2.,-2.,-2.,-2.],dtype=torch.float64)
p=(q+torch.tensor([.5,-.5,.3,.3],dtype=torch.float64)).requires_grad_()
a=torch.tensor([2.,2.,-1.,-1.],dtype=torch.float64)
mask=torch.ones_like(p,dtype=torch.bool)
l,w,g=sequence_surrogate(p,q,a,mask,[2,2]);(l[:2].mean()+l[2:].mean()).backward()
assert torch.allclose(p.grad,torch.tensor([-1.,-1.,math.exp(.3)/2,math.exp(.3)/2],dtype=torch.float64))
# Sequence clipping applies to the whole positive response; masked padding
# never changes geometric mean or receives a gradient.
p=torch.tensor([-1.,-1.,-100.],dtype=torch.float64,requires_grad=True)
q=torch.tensor([-1.01,-1.01,-2.],dtype=torch.float64)
a=torch.ones_like(p);mask=torch.tensor([True,True,False])
l,w,g=sequence_surrogate(p,q,a,mask,[3]);l.mean().backward();assert p.grad.abs().sum()==0
p=torch.tensor([-1.,-1.,-100.],dtype=torch.float64,requires_grad=True)
q=torch.tensor([-1.,-1.,-2.],dtype=torch.float64)
l,w,g=sequence_surrogate(p,q,a,mask,[3]);l.mean().backward()
assert torch.allclose(p.grad,torch.tensor([-.5,-.5,0.],dtype=torch.float64))
for text,answer in [('Answer: 1,024','1024'),(r'\boxed{34}','34'),('Answer: $34$','34'),('Answer: 34.0','34'),('Answer: 3\nCorrected\nAnswer: 4','4'),(r'Answer: \boxed{34}','34')]:
    assert score(text,answer)==1,(text,answer)
for text in ['I tried 34',r'\boxed{34+0}','Answer: 3,4','Answer: 34 or 35','Answer: 35']:
    assert score(text,'34')==0,text
for text in ['Answer: 34<|im_end|>','**Answer:** 34<|im_end|>','Answer: **34**<|im_end|>']:
    assert score(text,'34')==1,text
print('PASS: gradient coefficients, GSPO sequence/masking behavior, numeric verifiers')
