import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

from EKF_test import EKFTest
from Extended_RTS_Smoother_test import S_Test
from Extended_sysmdl import SystemModel
from Extended_data import DataGen, DataLoader
from Extended_data import N_E, N_CV, N_T

from datetime import datetime

from Plot import Plot_extended as Plot

from filing_paths import path_model, path_session
import sys
sys.path.insert(1, path_model)
from parameters import T, m1x_0, m2x_0, Q, R, m, n
from model import f, h

if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   cpu0 = torch.device("cpu")
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
T_test = T
# sys_model = SystemModel(f, Q, h, R, T, T_test)
# sys_model.InitSequence(m1x_0, m2x_0)


###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Data' + '/'
# dataFileName = 'data_EKF.pt'
# print("Start Data Gen")
# DataGen(sys_model,dataFolderName + dataFileName, T, T_test)
# print("Finish Data Gen")
# print("Data Load")
# [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
#######################################
### Evaluate Extended Kalman Filter ###
#######################################
# print("Evaluate Extended Kalman Filter")
# [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
# print(MSE_EKF_dB_avg)

# PlotfolderName = 'Graphs' + '/'
# PlotResultName = 'EKF_his'  
# Plot = Plot(PlotfolderName, PlotResultName)
# print("Plot")
# Plot.EKFPlot_Hist(MSE_EKF_linear_arr)

######################################
### Evaluate Extended RTS Smoother ###
######################################
# print("Evaluate RTS Smoother")
# [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg] = S_Test(sys_model, test_input, test_target)
# print(MSE_ERTS_dB_avg)

##############################
###  Compare KF and RTS    ###
##############################
r2 = torch.tensor([4, 2, 1, 0.5, 0.25])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2)
q = r
MSE_KF_RTS_dB = torch.empty(size=[2,len(r)])
dataFileName = ['data_lor_r4q4.pt','data_lor_r2q2.pt','data_lor_r1q1.pt','data_lor_r.5q.5.pt','data_lor_r.25q.25.pt']
for rindex in range(0, len(r)):
   #Model
   Q = (torch.squeeze(q[rindex])**2) * torch.eye(m)
   R = (torch.squeeze(r[rindex])**2) * torch.eye(m)
   SysModel_design = SystemModel(f, Q, h, R, T, T_test)  
   SysModel_design.InitSequence(m1x_0, m2x_0)
   #Generate and load data
   DataGen(SysModel_design, dataFolderName + dataFileName[rindex], T, T_test)
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName[rindex])
   #Evaluate KF and RTS
   [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(SysModel_design, test_input, test_target)
   [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg] = S_Test(SysModel_design, test_input, test_target)
   MSE_KF_RTS_dB[0,rindex] = MSE_EKF_dB_avg
   MSE_KF_RTS_dB[1,rindex] = MSE_ERTS_dB_avg

PlotfolderName = 'Graphs' + '/'
PlotResultName = 'Nonlinear_KFandRTS'  
Plot = Plot(PlotfolderName, PlotResultName)
print("Plot")
Plot.KF_RTS_Plot(r, MSE_KF_RTS_dB)

