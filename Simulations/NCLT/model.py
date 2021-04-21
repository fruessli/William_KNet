# import autograd.numpy as np
from autograd import grad, jacobian
import torch
from torch import autograd

from parameters import F_design, H_design

def f(x):
    return torch.matmul(F_design, x)

def h(x):
    return torch.matmul(H_design, x)

def fInacc(x):
    return torch.matmul(F_mod, x)

def hInacc(x):
    return torch.matmul(H_mod, x)

def getJacobian(x, a):
    
    if(x.size()[1] == 1):
        y = torch.reshape((x.T),[x.size()[0]])

    if(a == 'ObsAcc'):
        g = h
    elif(a == 'ModAcc'):
        g = f
    elif(a == 'ObsInacc'):
        g = hInacc
    elif(a == 'ModInacc'):
        g = fInacc

    return autograd.functional.jacobian(g, y)
