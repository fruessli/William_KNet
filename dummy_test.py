import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
device = torch.device('cpu')
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
# import random
# print(random.randint(0,7000-1))

### Pen traj plot
DatafolderName = 'Simulations/Pendulum/results/transfer/traj' + '/'
DataResultName = 'pen_r1_chopRTSNet_traj' 
trajs = torch.load(DatafolderName+DataResultName, map_location=device)
EKF_sample = trajs['EKF_sample'].detach().numpy()
ERTS_sample = trajs['ERTS_sample'].detach().numpy()
target_sample = trajs['target_sample'].detach().numpy()
input_sample = trajs['input_sample'].detach().numpy()
RTSNet_sample = trajs['RTSNet_sample'].detach().numpy()

diff = target_sample[0,0,:] - ERTS_sample[0,0,:]

peaks, _ = find_peaks(diff, prominence=0.31)
troughs, _ = find_peaks(-diff, prominence=0.31)
print(peaks, troughs)

for peak, trough in zip(peaks, troughs):
    plt.axvspan(peak, trough, color='red', alpha=.2)

plt.plot(np.arange(np.size(target_sample[0,:],axis=1)) , ERTS_sample[0,0,:], 'b', linewidth=0.75)
plt.show()
# DatafolderName = 'Simulations/Pendulum/results/transfer/traj' + '/'
# file_name = DatafolderName+'test_RTSNet.png'
# plt.savefig(file_name)