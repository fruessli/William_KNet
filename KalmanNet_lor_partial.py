import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from EKF_test import EKFTest
from Extended_RTS_Smoother_test import S_Test
from Extended_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
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
from parameters import T, T_test, m1x_0, m2x_0, m, n,delta_t_gen,delta_t
from model import f, h, fInacc, hInacc, fRotate

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
data_gen = 'data_gen.pt'
# data_gen_file = torch.load(DatafolderName+data_gen, map_location=cuda0)
# [true_sequence] = data_gen_file['All Data']

r2 = torch.tensor([0.1])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2)
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)

q2 = torch.mul(v,r2)
q = torch.sqrt(q2)

# MSE_dB = torch.empty(size=[2,len(r)])
traj_resultName = ['traj_lor_obsmis_rq1030_T1000_NT1000.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
dataFileName = ['data_lor_v20_rq1030_T1000_NT1000.pt']#,'data_lor_v20_r1e-2_T100.pt','data_lor_v20_r1e-3_T100.pt','data_lor_v20_r1e-4_T100.pt']
for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q[rindex]**2))
   #Model
   sys_model = SystemModel(f, q[rindex], h, r[rindex], T, T_test, m, n,"Lor")
   sys_model.InitSequence(m1x_0, m2x_0)

   # sys_model_partialf = SystemModel(fRotate, q[rindex], h, r[rindex], T, T_test, m, n,"Lor")
   # sys_model_partialf.InitSequence(m1x_0, m2x_0)

   sys_model_partialh = SystemModel(f, q[rindex], hInacc, r[rindex], T, T_test, m, n,"Lor")
   sys_model_partialh.InitSequence(m1x_0, m2x_0)
   
   #Generate and load data DT case
   # print("Start Data Gen")
   # T = 1000
   # DataGen(sys_model, DatafolderName + dataFileName[rindex], T, T_test)
   print("Data Load")
   print(dataFileName[rindex])
   [train_input_long, train_target_long, cv_input_long, cv_target_long, test_input, test_target] =  torch.load(DatafolderName + dataFileName[rindex],map_location=cuda0)  
   print("trainset long:",train_target_long.size())
   print("testset:",test_target.size())
   T = 100
   [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
   [cv_target, cv_input] = Short_Traj_Split(cv_target_long, cv_input_long, T)
   print("trainset chopped:",train_target.size())
   
   #Generate and load data Decimation case (chopped)
   # print("Data Gen")
   # [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, r[rindex], offset)
   # print(test_target.size())
   # [train_target_long, train_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_E, h, r[rindex], offset)
   # [cv_target_long, cv_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_CV, h, r[rindex], offset)

   # [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
   # [cv_target, cv_input] = Short_Traj_Split(cv_target_long, cv_input_long, T)

   #Evaluate EKF true
   [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
   # #Evaluate EKF partial
   [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialh, test_input, test_target)
   #Evaluate RTS true
   # [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(sys_model, test_input, test_target)
   # #Evaluate RTS partial
   # [MSE_ERTS_linear_arr_partial, MSE_ERTS_linear_avg_partial, MSE_ERTS_dB_avg_partial, ERTS_out_partial] = S_Test(sys_model_partial, test_input, test_target)
   # MSE_dB[0,rindex] = MSE_EKF_dB_avg
   # MSE_dB[1,rindex] = MSE_EKF_dB_avg_partial 
   # # MSE_dB[2,rindex] = MSE_ERTS_dB_avg
   # # MSE_dB[3,rindex] = MSE_ERTS_dB_avg_partial
   # print("MSE_EKF True [dB]: ", MSE_EKF_dB_avg)
   # print("MSE_EKF Partial [dB]: ", MSE_EKF_dB_avg_partial)
   # print("MSE_RTS True [dB]: ", MSE_ERTS_dB_avg)
   # print("MSE_RTS Partial [dB]: ", MSE_ERTS_dB_avg_partial)
   
   # Save results

   DatafolderName = 'Data' + '/'
   DataResultName = 'EKF_obsmis_rq1030_T1000_NT1000' 
   torch.save({'MSE_EKF_linear_arr': MSE_EKF_linear_arr,
               'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
               'MSE_EKF_linear_arr_partial': MSE_EKF_linear_arr_partial,
               'MSE_EKF_dB_avg_partial': MSE_EKF_dB_avg_partial,
               }, DatafolderName+DataResultName)

   # KNet without model mismatch
   # modelFolder = 'KNet' + '/'
   # KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
   # KNet_Pipeline.setssModel(sys_model)
   # KNet_model = KalmanNetNN()
   # KNet_model.Build(sys_model)
   # KNet_Pipeline.setModel(KNet_model)
   # KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=100, learningRate=5e-3, weightDecay=1e-4)

   # # KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")

   # KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
   # [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
   # KNet_Pipeline.save()
   
   # KNet with model mismatch
   modelFolder = 'KNet' + '/'
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
   KNet_Pipeline.setssModel(sys_model_partialh)
   KNet_model = KalmanNetNN()
   KNet_model.Build(sys_model_partialh)
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=50, learningRate=1e-3, weightDecay=1e-9)

   # KNet_Pipeline.model = torch.load(modelFolder+"model_KalmanNet.pt")

   KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
   [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
   KNet_Pipeline.save()

   # Save trajectories
   trajfolderName = 'EKNet' + '/'
   DataResultName = traj_resultName[rindex]
   # EKF_sample = torch.reshape(EKF_out[0,:,:],[1,m,T_test])
   # ERTS_sample = torch.reshape(ERTS_out[0,:,:],[1,m,T_test])
   target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
   input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
   KNet_sample = torch.reshape(KNet_test[0,:,:],[1,m,T_test])
   torch.save({
               # 'EKF_sample': EKF_sample,
               # 'ERTS_sample': ERTS_sample,
               'target_sample': target_sample,
               'input_sample': input_sample,
               'KNet_sample': KNet_sample,
               }, trajfolderName+DataResultName)

   ## Save histogram
   # MSE_ResultName = 'Partial_MSE_KNet' 
   # torch.save(KNet_MSE_test_dB_avg,trajfolderName + MSE_ResultName)

   





