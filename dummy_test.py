import torch

q = 1
delta_t = 0.02
Q = q * q * torch.tensor([[(delta_t**3)/3, (delta_t**2)/2],
                                           [(delta_t**2)/2, delta_t]])
print(Q)