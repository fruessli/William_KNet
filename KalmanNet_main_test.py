from KalmanNet_data import F_design, H_design, m1x_0_design, m2x_0_design, N_T

from KalmanFilter_test import KFTest
from KalmanNet_test import NNTest
from KalmanNet_sysmdl import SystemModel
from Plot import KNPlot_test

from datetime import datetime

import numpy as torch
import torch

print("Kalman Net Test - Trained Models")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

####################
### Design Model ###
####################
r_design = 1
q_design = r_design*1

T_ARR = [10]

for T in T_ARR:

    SysModel_design = SystemModel(F_design, q_design, H_design, r_design, T)
    SysModel_design.InitSequence(m1x_0_design, m2x_0_design)

    SysModel_design.GenerateBatch(N_T)
    test_input = SysModel_design.Input
    test_target = SysModel_design.Target

    print("T = ", T)

    # Test Kalman Filter with design parameters
    [MSE_KF_design_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(SysModel_design, test_input, test_target)

    # Test Kalman Net with design parameters
    [MSE_KN_linear_arr, MSE_KN_linear_avg, MSE_KN_dB_avg] = NNTest(SysModel_design, test_input, test_target)