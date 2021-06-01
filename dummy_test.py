import torch
from torch.distributions.multivariate_normal import MultivariateNormal
# q = 1
# delta_t = 0.02
# m = 2
# Q = q * q * torch.tensor([[(delta_t**3)/3, (delta_t**2)/2],
#                                            [(delta_t**2)/2, delta_t]])
# mean = torch.zeros([m])
# distrib = MultivariateNormal(loc=mean, covariance_matrix=Q)

# eq = distrib.rsample()
# print(eq)

p = 0.1
T = 100
a = torch.bernoulli(p *torch.ones(1,T))
print(torch.nonzero(a).size(0))

# rayleigh_sigma = 1
# a = rayleigh_sigma*torch.sqrt(-2*torch.log(torch.rand(1)))
# print(a)