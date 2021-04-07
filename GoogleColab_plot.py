import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from Plot import Plot_RTS as Plot

####################################################
### Compare RTSNet and RTS Smoother to Rotations ###
####################################################
# r = torch.tensor([4, 2, 1, 0.5, 0.1])
# r = torch.sqrt(r)
# MSE_RTS_dB = torch.empty(size=[3,len(r)])

# PlotfolderName = 'Graphs' + '/'
# DatafolderName = 'Data' + '/'
# PlotResultName = 'HRotation_RTSandRTSNet_Compare' 
# MSE_RTS_dB = torch.load(DatafolderName+PlotResultName, map_location=torch.device('cpu'))

# Plot = Plot(PlotfolderName, PlotResultName)
# print("Plot")
# Plot.rotate_RTS_Plot(r, MSE_RTS_dB)

####################
### RTSNet 10x10 ###
####################
DatafolderName = 'RTSNet' + '/'
DataResultName = 'model_RTSNet'
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.load(DatafolderName+DataResultName)

