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
### outliers
# p = 0.1
# T = 100
# b_matrix = torch.bernoulli(p *torch.ones(T))
# print(torch.nonzero(b_matrix).size(0))
# for t in range(0, T):
#     if b_matrix[t] != 0:
#         print("outlier detect")
        # btdt = self.rayleigh_sigma*torch.sqrt(-2*torch.log(torch.rand(1)))
        # yt = torch.add(yt,btdt)
# rayleigh_sigma = 1
# n = 2
# a = rayleigh_sigma*torch.sqrt(-2*torch.log(torch.rand(n)))
# print(a)

### Pendulum high resol
import random
print(random.randint(0,7000-1))