import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from EKF_test import EKFTest
from Extended_RTS_Smoother_test import S_Test
from Extended_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
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
from model import f, h, fInacc, hInacc

if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   cuda0 = torch.device("cpu")
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

######################################
###  Compare EKF, RTS and RTSNet   ###
######################################
offset = 0
DatafolderName = 'Simulations/Lorenz_Atractor/data' + '/'
data_gen_short = 'data_gen_3k.pt'

r2 = torch.tensor([1,0.01,0.0001])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2)
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)

q2 = torch.mul(v,r2)
q = torch.sqrt(q2)

#####
# test Jonas
# r2 = torch.tensor([0.01])
# q2 = torch.tensor([1E-4])
# r = torch.sqrt(r2)
# q = torch.sqrt(q2)
#####

MSE_dB = torch.empty(size=[5,len(r)])
traj_resultName = ['partial_lorH1_r1.pt','partial_lorH1_r0.01.pt','partial_lorH1_r1E-4.pt']
dataFileName = ['data_lor_r1q0.1_T30.pt','data_lor_r0.01q0.001.pt','data_lor_r1e-4q1e-5.pt']

for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q[rindex]**2))
   #Model
   sys_model = SystemModel(f, q[rindex], h, r[rindex], T, T_test, m, n,"Lor")
   sys_model.InitSequence(m1x_0, m2x_0)

   sys_model_partial = SystemModel(fInacc, q[rindex], h, r[rindex], T, T_test, m, n,"Lor")
   sys_model_partial.InitSequence(m1x_0, m2x_0)
   
   #Generate and load data
   print("Start Data Gen")
   DataGen(sys_model, DatafolderName + dataFileName[rindex], T, T_test)
   print("Data Load")
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(DatafolderName + dataFileName[rindex])  
   
   # ### test Jonas's code
   # test_target = torch.load(DatafolderName+"GT_-40dB_discrete")
   # test_input = torch.load(DatafolderName+"identity20_-40_discrete")
   # test_target = test_target[N_T,m,0:T_test]
   # test_input = test_input[N_T,m,0:T_test]
   
   #Evaluate EKF true
   [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
   #Evaluate EKF partial
   [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partial, test_input, test_target)
   #Evaluate RTS true
   [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(sys_model, test_input, test_target)
   #Evaluate RTS partial
   [MSE_ERTS_linear_arr_partial, MSE_ERTS_linear_avg_partial, MSE_ERTS_dB_avg_partial, ERTS_out_partial] = S_Test(sys_model_partial, test_input, test_target)
   MSE_dB[0,rindex] = MSE_EKF_dB_avg
   MSE_dB[1,rindex] = MSE_EKF_dB_avg_partial 
   MSE_dB[2,rindex] = MSE_ERTS_dB_avg
   MSE_dB[3,rindex] = MSE_ERTS_dB_avg_partial
   print("MSE_EKF True [dB]: ", MSE_EKF_dB_avg)
   print("MSE_EKF Partial [dB]: ", MSE_EKF_dB_avg_partial)
   print("MSE_RTS True [dB]: ", MSE_ERTS_dB_avg)
   print("MSE_RTS Partial [dB]: ", MSE_ERTS_dB_avg_partial)

   # print("Evaluate RTSNet with partial info")
   # modelFolder = 'ERTSNet' + '/'
   # RTSNet_Pipeline = Pipeline(strTime, "ERTSNet", "ERTSNet")
   # RTSNet_Pipeline.setssModel(sys_model_partial)
   # RTSNet_model = RTSNetNN()
   # RTSNet_model.Build(sys_model_partial, infoString = 'partialInfo')
   # RTSNet_Pipeline.setModel(RTSNet_model)
   # RTSNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=150, learningRate=1e-3, weightDecay=1e-9)

   # # RTSNet_Pipeline.model = torch.load(modelFolder+"model_ERTSNet_lor_r1q1.pt")

   # RTSNet_Pipeline.NNTrain(train_input, train_target, cv_input, cv_target)
   # [RTSNet_MSE_test_linear_arr, RTSNet_MSE_test_linear_avg, RTSNet_MSE_test_dB_avg, RTSNet_test] = RTSNet_Pipeline.NNTest(test_input, test_target)
   
   # MSE_dB[4,rindex] = RTSNet_MSE_test_dB_avg
   # print("MSE RTSNet Partial [dB]: ", RTSNet_MSE_test_dB_avg)
   
   # Save trajectories
   DatafolderName = 'ERTSNet' + '/'
   DataResultName = traj_resultName[rindex]
   EKF_sample = torch.reshape(EKF_out[0,:,:],[1,m,T_test])
   ERTS_sample = torch.reshape(ERTS_out[0,:,:],[1,m,T_test])
   target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
   input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
   # RTSNet_sample = torch.reshape(RTSNet_test[0,:,:],[1,m,T_test])
   torch.save({
               'EKF_sample': EKF_sample,
               'ERTS_sample': ERTS_sample,
               'target_sample': target_sample,
               'input_sample': input_sample,
               # 'RTSNet_sample': RTSNet_sample,
               }, DatafolderName+DataResultName)


MSE_ResultName = 'Partial_lorH1_MSE' 
torch.save(MSE_dB,DatafolderName + MSE_ResultName)

   





