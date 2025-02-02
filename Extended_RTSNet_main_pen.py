import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from EKF_test import EKFTest
from Extended_RTS_Smoother_test import S_Test
from Extended_sysmdl import SystemModel
from Extended_data import DataGen, DataLoader_GPU, DataGen_True,Decimate_and_perturbate_Data
from Extended_data import N_E, N_CV, N_T
from Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipeline_EKF import Pipeline_EKF

from Extended_RTSNet_nn import RTSNetNN
from Extended_KalmanNet_nn import KalmanNetNN

from datetime import datetime

from Plot import Plot_extended as Plot

from filing_paths import path_model, path_session
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, lambda_q_mod, lambda_r_mod, m, n,delta_t_gen,delta_t
from model import f, h, f_gen

if torch.cuda.is_available():
   device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   device = torch.device("cpu")
   print("Running on the CPU")


print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)


####################
### Design Model ###
####################
sys_model = SystemModel(f, lambda_q_mod, h, lambda_r_mod, T, T_test, m, n,'pendulum')
sys_model.InitSequence(m1x_0, m2x_0)

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Pendulum/results/traj' + '/'
dataFileName = 'data_pen_highresol_q1e-5.pt'
offset = 0
print("Start Data Gen")
# DataGen_True(sys_model,dataFolderName + dataFileName, T_test)
[input_sample, target_sample] = torch.load(dataFolderName + dataFileName, map_location=device)
[test_target, test_input] = Decimate_and_perturbate_Data(target_sample, delta_t_gen, delta_t, N_T, h, lambda_r_mod, offset)
[train_target, train_input] = Decimate_and_perturbate_Data(target_sample, delta_t_gen, delta_t, N_E, h, lambda_r_mod, offset)
[cv_target, cv_input] = Decimate_and_perturbate_Data(target_sample, delta_t_gen, delta_t, N_CV, h, lambda_r_mod, offset)

# MSE Baseline
print("Evaluate Baseline")
loss_fn = nn.MSELoss(reduction='mean')
MSE_test_baseline_arr = torch.empty(N_T)
for i in range(0, N_T):
   MSE_test_baseline_arr[i] = loss_fn(test_input[i, :, :], test_target[i, :, :]).item()
MSE_test_baseline_avg = torch.mean(MSE_test_baseline_arr)
MSE_test_baseline_dB_avg_dec = 10 * torch.log10(MSE_test_baseline_avg)
print(MSE_test_baseline_dB_avg_dec)

#######################################
### Evaluate Extended Kalman Filter ###
#######################################
print("Evaluate Extended Kalman Filter")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
print(MSE_EKF_dB_avg)

# PlotfolderName = 'Graphs' + '/'
# PlotResultName = 'EKF_his'  
# Plot = Plot(PlotfolderName, PlotResultName)
# print("Plot")
# Plot.EKFPlot_Hist(MSE_EKF_linear_arr)

######################################
### Evaluate Extended RTS Smoother ###
######################################
print("Evaluate RTS Smoother")
[MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(sys_model, test_input, test_target)
print(MSE_ERTS_dB_avg)


### Save results

DatafolderName = 'Data' + '/'
DataResultName = 'EKFandERTS_pen_r1q0.3' 
torch.save({
            'MSE_EKF_linear_arr': MSE_EKF_linear_arr,
            'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
            'MSE_ERTS_linear_arr': MSE_ERTS_linear_arr,
            'MSE_ERTS_dB_avg': MSE_ERTS_dB_avg,
            }, DatafolderName+DataResultName)

### Save trajectories

DatafolderName = 'ERTSNet' + '/'
DataResultName = 'pen_r1q0.3_traj' 
EKF_sample = torch.reshape(EKF_out[0,:,:],[1,m,T_test])
ERTS_sample = torch.reshape(ERTS_out[0,:,:],[1,m,T_test])
target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
torch.save({
            'EKF_sample': EKF_sample,
            'ERTS_sample': ERTS_sample,
            'target_sample': target_sample,
            'input_sample': input_sample,
            }, DatafolderName+DataResultName)


##############################
###  Compare KF and RTS    ###
##############################
# r2 = torch.tensor([4, 2, 1, 0.5, 0.25])
# # r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
# r = torch.sqrt(r2)
# q = r
# MSE_KF_RTS_dB = torch.empty(size=[2,len(r)])
# dataFileName = ['data_lor_r4q4.pt','data_lor_r2q2.pt','data_lor_r1q1.pt','data_lor_r.5q.5.pt','data_lor_r.25q.25.pt']
# for rindex in range(0, len(r)):
#    #Model
#    Q = (torch.squeeze(q[rindex])**2) * torch.eye(m)
#    R = (torch.squeeze(r[rindex])**2) * torch.eye(m)
#    SysModel_design = SystemModel(f, Q, h, R, T, T_test)  
#    SysModel_design.InitSequence(m1x_0, m2x_0)
#    #Generate and load data
#    DataGen(SysModel_design, dataFolderName + dataFileName[rindex], T, T_test)
#    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName[rindex])
#    #Evaluate KF and RTS
#    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(SysModel_design, test_input, test_target)
#    [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg] = S_Test(SysModel_design, test_input, test_target)
#    MSE_KF_RTS_dB[0,rindex] = MSE_EKF_dB_avg
#    MSE_KF_RTS_dB[1,rindex] = MSE_ERTS_dB_avg

# PlotfolderName = 'Graphs' + '/'
# PlotResultName = 'Nonlinear_KFandRTS'  
# Plot = Plot(PlotfolderName, PlotResultName)
# print("Plot")
# Plot.KF_RTS_Plot(r, MSE_KF_RTS_dB)


######################
### EKNet Pipeline ###
######################
# print("Evaluate KNet")
# modelFolder = 'EKNet' + '/'
# KNet_Pipeline = Pipeline_EKF(strTime, "EKNet", "EKNet")
# KNet_Pipeline.setssModel(sys_model)
# KNet_model = KalmanNetNN()
# KNet_model.Build(sys_model, infoString = 'fullInfo')
# KNet_Pipeline.setModel(KNet_model)
# KNet_Pipeline.setTrainingParams(n_Epochs=200, n_Batch=30, learningRate=1E-3, weightDecay=5E-6)

# KNet_Pipeline.model = torch.load(modelFolder+"model_EKNet.pt")

# KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
# KNet_Pipeline.NNTest(N_T, test_input, test_target)
# KNet_Pipeline.save()
# print("Plot")
# KNet_Pipeline.PlotTrain_KF(MSE_EKF_linear_arr, MSE_EKF_dB_avg, MSE_ERTS_linear_arr, MSE_ERTS_dB_avg)




########################
### ERTSNet Pipeline ###
########################
print("Evaluate RTSNet")
modelFolder = 'ERTSNet' + '/'
RTSNet_Pipeline = Pipeline(strTime, "ERTSNet", "ERTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_model = RTSNetNN()
RTSNet_model.Build(sys_model, infoString = 'fullInfo')
RTSNet_Pipeline.setModel(RTSNet_model)
RTSNet_Pipeline.setTrainingParams(n_Epochs=200, n_Batch=30, learningRate=1E-2, weightDecay=5E-5)

# RTSNet_Pipeline.model = torch.load(modelFolder+"model_ERTSNet_lor_r1q1.pt")

RTSNet_Pipeline.NNTrain(train_input, train_target, cv_input, cv_target)
[RTSNet_MSE_test_linear_arr, RTSNet_MSE_test_linear_avg, RTSNet_MSE_test_dB_avg, RTSNet_test] = RTSNet_Pipeline.NNTest(test_input, test_target)
RTSNet_Pipeline.save()

# Save trajectories

DatafolderName = 'ERTSNet' + '/'
DataResultName = 'pen_r1q0.3_traj' 
EKF_sample = torch.reshape(EKF_out[0,:,:],[1,m,T_test])
ERTS_sample = torch.reshape(ERTS_out[0,:,:],[1,m,T_test])
target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
RTSNet_sample = torch.reshape(RTSNet_test[0,:,:],[1,m,T_test])
torch.save({
            'EKF_sample': EKF_sample,
            'ERTS_sample': ERTS_sample,
            'target_sample': target_sample,
            'input_sample': input_sample,
            'RTSNet_sample': RTSNet_sample,
            }, DatafolderName+DataResultName)


# print("Plot")
# RTSNet_Pipeline.PlotTrain_RTS(MSE_EKF_linear_arr, MSE_EKF_dB_avg, MSE_ERTS_linear_arr, MSE_ERTS_dB_avg)


