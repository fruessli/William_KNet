import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from Plot import Plot_RTS as Plot
from Pipeline_RTS import Pipeline_RTS as Pipeline
from datetime import datetime
device = torch.device('cpu')
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
r2 = torch.tensor([4, 2, 1, 0.5, 0.1])
r = torch.sqrt(r2)
MSE_RTS_dB = torch.empty(size=[3,len(r)])

PlotfolderName = 'Graphs' + '/'
DatafolderName = 'Data' + '/'
PlotResultName = 'FHrotCompare_RTSandRTSNet_Compare' 
PlotResultName_F = 'FRotation_RTSandRTSNet_Compare'
PlotResultName_H = 'HRotation_RTSandRTSNet_Compare'
MSE_RTS_dB_F = torch.load(DatafolderName+PlotResultName_F, map_location=device)
MSE_RTS_dB_H = torch.load(DatafolderName+PlotResultName_H, map_location=device)
print(MSE_RTS_dB_H)
Plot = Plot(PlotfolderName, PlotResultName)
print("Plot")
Plot.rotate_RTS_Plot_F(r, MSE_RTS_dB_F, PlotResultName_F)
Plot.rotate_RTS_Plot_H(r, MSE_RTS_dB_H, PlotResultName_H)
Plot.rotate_RTS_Plot_FHCompare(r, MSE_RTS_dB_F,MSE_RTS_dB_H, PlotResultName)



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
