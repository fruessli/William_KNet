"""# **Class: Extended RTS Smoother**
Theoretical Non-linear Linear RTS Smoother
"""
import torch
from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from model import getJacobian

class Extended_rts_smoother:

    def __init__(self, SystemModel, mode='full'):
        self.f = SystemModel.f
        self.m = SystemModel.m

        self.Q = SystemModel.Q

        self.h = SystemModel.h
        self.n = SystemModel.n

        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Full knowledge about the model or partial? (Should be made more elegant)
        if(mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif(mode == 'partial'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'


    
    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(self.m1x_posterior,self.fString), getJacobian(self.m1x_prior, self.hString))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Smoothing Gain
    def SGain(self):
        self.SG = torch.matmul(filter_sigma, self.F_T)
        filter_sigma_prior = torch.matmul(self.F, filter_sigma)
        filter_sigma_prior = torch.matmul(filter_sigma_prior, self.F_T) + self.Q
        self.SG = torch.matmul(self.SG, torch.inverse(filter_sigma_prior))

        
        self.SG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))


    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    def UpdateJacobians(self, F, H):
        self.F = F
        self.F_T = torch.transpose(F,0,1)
        self.H = H
        self.H_T = torch.transpose(H,0,1)
        #print(self.H,self.F,'\n')
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, filter_x, filter_sigma, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, self.T])
        # Pre allocate SG array
        self.SG_array = torch.zeros((self.T,self.m,self.m))
        
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0


        for t in range(0, self.T):
            yt = torch.unsqueeze(y[:, t], 1)
            xt = self.Update(yt)
            self.x[:, t] = torch.squeeze(xt)