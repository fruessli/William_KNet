import torch
from torch.distributions.multivariate_normal import MultivariateNormal
q = 1
delta_t = 0.02
m = 2
Q = q * q * torch.tensor([[(delta_t**3)/3, (delta_t**2)/2],
                                           [(delta_t**2)/2, delta_t]])
mean = torch.zeros([m])
distrib = MultivariateNormal(loc=mean, covariance_matrix=Q)

eq = distrib.rsample()
print(eq)