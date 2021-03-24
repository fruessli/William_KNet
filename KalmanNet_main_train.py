import torch

from KalmanNet_sysmdl import SystemModel
from KalmanNet_data import DataGen, DataLoader
from KalmanNet_data import F, H, T, m1_0, m2_0

from KalmanFilter_test import KFTest
from RTS_Smoother_test import S_Test
from KalmanNet_data import N_E, N_CV, N_T

from Pipeline import Pipeline
from KalmanNet_nn import KalmanNetNN
from datetime import datetime

from DataAnalysis import DataAnalysis
from Plot import Plot

#if torch.cuda.is_available():
#    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
#    print("Running on the GPU")
#else:
#    device = torch.device("cpu")
#    print("Running on the CPU")



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
r = 1
q = 1

SysModel_design = SystemModel(F, q, H, r, T)
SysModel_design.InitSequence(m1_0, m2_0)

###################################
### Data Loader (Generate Data) ###
###################################
print("Start Gen Data")
dataFolderName = 'Data' + '\\'
dataFileName = 'data_ssr_10x1_r1_T1_10000.pt'
# DataGen(SysModel_design, dataFolderName + dataFileName)
print("Data Load")
[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)


##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter")
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(SysModel_design, test_input, test_target)

##############################
### Evaluate RTS Smoother ###
##############################
print("Evaluate RTS Smoother")
[MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg] = S_Test(SysModel_design, test_input, test_target)

##############################
###  Compare KF and RTS    ###
##############################
# r = torch.tensor([2, 1, 0.5, 0.1])
# r = torch.sqrt(r)
# q = r
# MSE_KF_RTS_dB = torch.empty(size=[2,len(r)])
# for rindex in range(0, len(r)):
#     #Generate and load data
#     SysModel_design = SystemModel(F, torch.squeeze(q[rindex]), H, torch.squeeze(r[rindex]), T)  
#     SysModel_design.InitSequence(m1_0, m2_0)
#     DataGen(SysModel_design, dataFolderName + dataFileName)
#     [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
#     #Evaluate KF and RTS
#     [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(SysModel_design, test_input, test_target)
#     [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg] = S_Test(SysModel_design, test_input, test_target)
#     MSE_KF_RTS_dB[0,rindex] = MSE_KF_dB_avg
#     MSE_KF_RTS_dB[1,rindex] = MSE_RTS_dB_avg

# PlotfolderName = 'Graphs' + '\\'
# modelName = 'Linear_KFandRTS'  
# Plot = Plot(PlotfolderName, modelName)
# print("Plot")
# Plot.KF_RTS_Plot(r, MSE_KF_RTS_dB)



##########################
### KalmanNet Pipeline ###
##########################

KNet_Pipeline = Pipeline(strTime, "KNet", "KalmanNet")
KNet_Pipeline.setssModel(SysModel_design)
KNet_model = KalmanNetNN()
KNet_model.Build(SysModel_design)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(n_Epochs=10, n_Batch=50, learningRate=1E-3, weightDecay=5E-6)
KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
KNet_Pipeline.NNTest(N_T, test_input, test_target)
KNet_Pipeline.PlotTrain(MSE_KF_linear_arr, MSE_KF_dB_avg)
KNet_Pipeline.save()

matlab_import = DataAnalysis()
matlab_import.main(MSE_KF_dB_avg)


