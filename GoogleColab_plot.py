import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from Plot import Plot_extended as Plot
from Pipeline_RTS import Pipeline_RTS as Pipeline
from Pipeline_ERTS import Pipeline_ERTS
from Pipeline_EKF import Pipeline_EKF

from Extended_sysmdl import SystemModel
from Extended_data import DecimateData
from Extended_RTSNet_nn import RTSNetNN
from Extended_KalmanNet_nn import KalmanNetNN
from Extended_data import DataGen,DataGen_LorTrue,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T

from datetime import datetime
device = torch.device('cpu')

from filing_paths import path_model, path_session
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, lambda_q_mod, lambda_r_mod, m, n,delta_t_gen,delta_t
from model import f, h

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

####################################################
### Compare RTSNet and RTS Smoother to Rotations ###
####################################################
# r2 = torch.tensor([4, 2, 1, 0.5, 0.1])
# r = torch.sqrt(r2)
# MSE_RTS_dB = torch.empty(size=[3,len(r)])

# PlotfolderName = 'Graphs' + '/'
# DatafolderName = 'Data' + '/'
# PlotResultName = 'FHrotCompare_RTSandRTSNet_Compare' 
# PlotResultName_F = 'FRotation_RTSandRTSNet_Compare'
# PlotResultName_H = 'HRotation_RTSandRTSNet_Compare'
# MSE_RTS_dB_F = torch.load(DatafolderName+PlotResultName_F, map_location=device)
# MSE_RTS_dB_H = torch.load(DatafolderName+PlotResultName_H, map_location=device)
# print(MSE_RTS_dB_H)
# Plot = Plot(PlotfolderName, PlotResultName)
# print("Plot")
# Plot.rotate_RTS_Plot_F(r, MSE_RTS_dB_F, PlotResultName_F)
# Plot.rotate_RTS_Plot_H(r, MSE_RTS_dB_H, PlotResultName_H)
# Plot.rotate_RTS_Plot_FHCompare(r, MSE_RTS_dB_F,MSE_RTS_dB_H, PlotResultName)



#############################################
### RTSNet Generalization to Large System ###
#############################################
# DatafolderName = 'RTSNet' + '/'
# DataResultName = 'pipeline_RTSNet_10x10.pt'
# RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet_10x10")
# RTSNet_Pipeline = torch.load(DatafolderName+DataResultName, map_location=device)
# RTSNet_Pipeline.modelName = "RTSNet 10x10"

# DatafolderName = 'Data' + '/'
# DataResultName = '10x10_KFandRTS' 
# KFandRTS_10x10 = torch.load(DatafolderName+DataResultName, map_location=device)
# MSE_KF_linear_arr = KFandRTS_10x10['MSE_KF_linear_arr']
# MSE_KF_dB_avg = KFandRTS_10x10['MSE_KF_dB_avg']
# MSE_RTS_linear_arr = KFandRTS_10x10['MSE_RTS_linear_arr']
# MSE_RTS_dB_avg = KFandRTS_10x10['MSE_RTS_dB_avg']

# print("Plot")
# RTSNet_Pipeline.PlotTrain_RTS(MSE_KF_linear_arr, MSE_KF_dB_avg, MSE_RTS_linear_arr, MSE_RTS_dB_avg)



########################
### Nonlinear RTSNet ###
########################
# DatafolderName = 'EKNet' + '/'
# DataResultName = 'pipeline_EKNet_NCLT_r1q1.pt'
# ModelResultName = 'model_EKNet.pt'
# KNet_Pipeline = Pipeline_EKF(strTime, "EKNet", "EKNet_nclt")
# # KNet_Pipeline.setssModel(sys_model)
# KNet_model = KalmanNetNN()
# # KNet_model = torch.load(DatafolderName+ModelResultName, map_location=device)
# KNet_Pipeline.setModel(KNet_model)
# KNet_Pipeline = torch.load(DatafolderName+DataResultName, map_location=device)


# DatafolderName = 'Simulations/Lorenz_Atractor/results' + '/'
# DataResultName = 'pipeline_ERTSNet_3k_unchop.pt'
# ModelResultName = 'model_ERTSNet_3k_unchop.pt'
# RTSNet_Pipeline = Pipeline_ERTS(strTime, "ERTSNet", "ERTSNet")
# RTSNet_model = RTSNetNN()
# # RTSNet_model = torch.load(DatafolderName+ModelResultName, map_location=device)
# RTSNet_Pipeline.setModel(RTSNet_model)
# RTSNet_Pipeline = torch.load(DatafolderName+DataResultName, map_location=device)

# # DatafolderName = 'Data' + '/'
# DataResultName = 'EKFandERTS_lor_3k_unchop' 
# EKFandERTS = torch.load(DatafolderName+DataResultName, map_location=device)
# MSE_test_baseline_dB_avg_dec = EKFandERTS['MSE_test_baseline_dB_avg_dec']
# MSE_EKF_linear_arr = EKFandERTS['MSE_EKF_linear_arr']
# MSE_EKF_dB_avg = EKFandERTS['MSE_EKF_dB_avg']
# MSE_ERTS_linear_arr = EKFandERTS['MSE_ERTS_linear_arr']
# MSE_ERTS_dB_avg = EKFandERTS['MSE_ERTS_dB_avg']

# print("Plot")
# PlotfolderName = DatafolderName

# ERTSNet_Plot = Plot(PlotfolderName,RTSNet_Pipeline.modelName)
# ERTSNet_Plot.NNPlot_epochs_KF_RTS(RTSNet_Pipeline.N_Epochs, RTSNet_Pipeline.N_B, 
#                       MSE_EKF_dB_avg, MSE_ERTS_dB_avg,
#                       KNet_Pipeline.MSE_test_dB_avg,KNet_Pipeline.MSE_cv_dB_epoch, KNet_Pipeline.MSE_train_dB_epoch,
#                       RTSNet_Pipeline.MSE_test_dB_avg,RTSNet_Pipeline.MSE_cv_dB_epoch,RTSNet_Pipeline.MSE_train_dB_epoch)

# #KNet_Pipeline.PlotTrain_KF(MSE_EKF_linear_arr, MSE_EKF_dB_avg)

# #ERTSNet_Plot.NNPlot_trainsteps(RTSNet_Pipeline.N_Epochs, MSE_EKF_dB_avg, MSE_ERTS_dB_avg,
#                       RTSNet_Pipeline.MSE_test_dB_avg, RTSNet_Pipeline.MSE_cv_dB_epoch, RTSNet_Pipeline.MSE_train_dB_epoch)
# RTSNet_Pipeline.PlotTrain_RTS(MSE_EKF_linear_arr, MSE_EKF_dB_avg, MSE_ERTS_linear_arr, MSE_ERTS_dB_avg)


## Plot Trajectories Lor
# DatafolderName = DatafolderName
# DataResultName = 'lor_traj_3k_unchop' 
# trajs = torch.load(DatafolderName+DataResultName, map_location=device)
# EKF_sample = trajs['EKF_sample']
# ERTS_sample = trajs['ERTS_sample']
# target_sample = trajs['target_sample']
# input_sample = trajs['input_sample']
# RTSNet_sample = trajs['RTSNet_sample']

# titles = ["True Trajectory","Observation", "Extended RTS", "EKF","RTSNet"]
# input = [target_sample, input_sample,ERTS_sample,EKF_sample, RTSNet_sample]
# ERTSNet_Plot = Plot(DatafolderName,DataResultName)
# ERTSNet_Plot.plotTrajectories(input,3, titles,DatafolderName+'traj_3k_unchop.png')

## Plot Trajectories Pen
DatafolderName = 'Simulations/Pendulum/results/traj' + '/'
DataResultName = 'pen_r0q1_traj' 
trajs = torch.load(DatafolderName+DataResultName, map_location=device)
EKF_sample = trajs['EKF_sample']
ERTS_sample = trajs['ERTS_sample']
target_sample = trajs['target_sample']
input_sample = trajs['input_sample']

titles = ["Noise Free"]#,"Observation"]#,"EKF","RTS"]#, "RTSNet"]
input = [target_sample]#, input_sample]#,EKF_sample, ERTS_sample]#, RTSNet_sample]
ERTSNet_Plot = Plot(DatafolderName,DataResultName)
ERTSNet_Plot.plotTrajectories(input,4, titles,DatafolderName+'pen_theta.png')



#############################
### Lorenz Data Load Test ###
#############################
# DatafolderName = 'Simulations/Lorenz_Atractor/data' + '/'
# data_gen = 'data_gen_3k.pt'
# data_gen_file = torch.load(DatafolderName+data_gen, map_location=device)

# ### test chopped traj
# [true_sequence] = data_gen_file['All Data']
# print(true_sequence.size())
# [train_target, train_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_E, h, lambda_r_mod, offset=0)
# # [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long)
# print(train_target.size(),train_input.size())
# train_target_sample = torch.reshape(train_target[37,:,:],[1,m,T])
# train_input_sample = torch.reshape(train_input[37,:,:],[1,m,T])
# titles = ["True Trajectory","Observation"]#, "Vanilla RNN"]
# input = [train_target_sample,train_input_sample]#, input_sample_dec, EKF_sample, knet_sample]#, rnn_sample]
# ERTSNet_Plot = Plot(DatafolderName,data_gen)
# ERTSNet_Plot.plotTrajectories(input,3, titles, DatafolderName+'test_train_start.png')

# print(data_gen_file.keys())
# [true_sequence] = data_gen_file['target_sample']
# [observations] = data_gen_file['input_sample']
# [ekf] = data_gen_file['EKF_sample']
# [erts] = data_gen_file['ERTS_sample']
# [rtsnet] = data_gen_file['RTSNet_sample']
# true_sequence = torch.unsqueeze(true_sequence, 0)
# observations= torch.unsqueeze(observations, 0)
# ekf = torch.unsqueeze(ekf, 0)
# erts = torch.unsqueeze(erts, 0)
# rtsnet = torch.unsqueeze(rtsnet, 0)
# print(true_sequence.size(),observations.size(),erts.size(),rtsnet.size())
# true_dec = DecimateData(true_sequence, t_gen = 1e-5,t_mod = 0.02, offset=0)
# print(true_dec.size())

# titles = ["True Decimated Trajectory"]
# input = [true_dec]
# ERTSNet_Plot = Plot(DatafolderName,data_gen)
# ERTSNet_Plot.plotTrajectories(input,3, titles, DatafolderName+'True Decimated Trajectory.png')

# titles = ["True Trajectory","Observation", "Extended RTS", "EKF","RTSNet"]#, "Vanilla RNN"]
# input = [true_sequence,observations,erts,ekf,rtsnet]#, input_sample_dec, EKF_sample, knet_sample]#, rnn_sample]
# ERTSNet_Plot = Plot(DatafolderName,data_gen)
# ERTSNet_Plot.plotTrajectories(input,3, titles, DatafolderName+'traj_3k.png')

# titles = ["True Trajectory","Observation", "Extended RTS", "RTSNet"]#, "Vanilla RNN"]
# input = [true_sequence,observations,erts,rtsnet]#, input_sample_dec, EKF_sample, knet_sample]#, rnn_sample]
# ERTSNet_Plot = Plot(DatafolderName,data_gen)
# ERTSNet_Plot.plotTrajectories(input,3, titles, DatafolderName+'traj_30.png')