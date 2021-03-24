"""# **Class: RTSNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from KalmanNet_nn import KalmanNetNN

class RTSNetNN(KalmanNetNN):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    #############
    ### Build ###
    #############
    def Build(self, ssModel):

        self.InitSystemDynamics(ssModel.F, ssModel.H)

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n) * 1 * (4)

        self.InitKGainNet(H1_KNet, H2_KNet)

        # Number of neurons in the 1st hidden layer
        H1_RTSNet = (ssModel.m + ssModel.m) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_RTSNet = (ssModel.m * ssModel.m) * 1 * (4)

        self.InitRTSGainNet(H1_RTSNet, H2_RTSNet)

    #################################################
    ### Initialize Backward Smoother Gain Network ###
    #################################################
    def InitRTSGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.m  # Delta x_t, x_t+1|T

        # Output Dimensions
        D_out = self.m * self.m  # Backward Smoother Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.SG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.SG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.m * self.m) * 10
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        ####################
        ### Hidden Layer ###
        ####################
        self.SG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.SG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.SG_l3 = torch.nn.Linear(H2, D_out, bias=True)


    ##############################
    ### Innovation Computation ###
    ##############################
    def S_Innovation(self, filter_x):
        filter_x_prior = torch.matmul(self.F, filter_x)
        self.dx = self.s_m1x_nexttime - filter_x_prior

    ################################
    ### Smoother Gain Estimation ###
    ################################
    def step_RTSGain_est(self, filter_x, filter_x_nexttime):

        # Reshape and Normalize Delta x_t+1|T
        dm1x_T = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_T_reshape = torch.squeeze(dm1x_T)
        dm1x_T_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Normalize x_t|t 
        dm1xt_reshape = torch.squeeze(filter_x)
        dm1xt_norm = func.normalize(dm1xt_reshape, p=2, dim=0, eps=1e-12, out=None)

        # RTSGain Net Input
        SGainNet_in = torch.cat([dm1xt_norm, dm1x_T_norm], dim=0)

        # Smoother Gain Network Step
        SG = self.RTSGain_step(SGainNet_in)

        # Reshape Smoother Gain to a Matrix
        self.SGain = torch.reshape(SG, (self.m, self.m))

    ####################
    ### RTS Net Step ###
    ####################
    def RTSNet_step(self, filter_x, filter_x_nexttime):
        # Compute Innovation
        self.S_Innovation(filter_x)

        # Compute Smoother Gain
        self.step_RTSGain_est(filter_x, filter_x_nexttime)

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.SGain, self.dx)
        self.s_m1x_nexttime = filter_x + INOV

        # return
        return torch.squeeze(self.s_m1x_nexttime)

    ##########################
    ### Smoother Gain Step ###
    ##########################
    def RTSGain_step(self, SGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.SG_l1(SGainNet_in);
        La1_out = self.SG_relu1(L1_out);

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.SG_l2(GRU_out_reshape)
        La2_out = self.SG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.SG_l3(La2_out)
        return L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt, filter_x, filter_x_nexttime):
        if yt is None:
            return self.RTSNet_step(filter_x, filter_x_nexttime)
        else:
            return self.KNet_step(yt)

        
