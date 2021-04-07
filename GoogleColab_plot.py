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
# r = torch.tensor([4, 2, 1, 0.5, 0.1])
# r = torch.sqrt(r)
# MSE_RTS_dB = torch.empty(size=[3,len(r)])

# PlotfolderName = 'Graphs' + '/'
# DatafolderName = 'Data' + '/'
# PlotResultName = 'HRotation_RTSandRTSNet_Compare' 
# MSE_RTS_dB = torch.load(DatafolderName+PlotResultName, map_location=device)

# Plot = Plot(PlotfolderName, PlotResultName)
# print("Plot")
# Plot.rotate_RTS_Plot(r, MSE_RTS_dB)

####################
### RTSNet 10x10 ###
####################
DatafolderName = 'RTSNet' + '/'
DataResultName = 'pipeline_RTSNet.pt'
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline = torch.load(DatafolderName+DataResultName, map_location=device)

DatafolderName = 'Data' + '/'
DataResultName = '10x10_KFandRTS' 
KFandRTS_10x10 = torch.load(DatafolderName+DataResultName, map_location=device)
MSE_KF_linear_arr = KFandRTS_10x10['MSE_KF_linear_arr']
MSE_KF_dB_avg = KFandRTS_10x10['MSE_KF_dB_avg']
MSE_RTS_linear_arr = KFandRTS_10x10['MSE_RTS_linear_arr']
MSE_RTS_dB_avg = KFandRTS_10x10['MSE_RTS_dB_avg']

print("Plot")
RTSNet_Pipeline.PlotTrain_RTS(MSE_KF_linear_arr, MSE_KF_dB_avg, MSE_RTS_linear_arr, MSE_RTS_dB_avg)
