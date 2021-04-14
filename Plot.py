import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cpu0 = torch.device("cpu")
   print("Running on the CPU")

# Legend
Klegend = ["Train", "CV", "Test", "Kalman Filter"]
RTSlegend = ["Train", "CV", "Test", "RTS Smoother","Kalman Filter"]
# Color
KColor = ['ro', 'yo', 'g-', 'b-','r-']

class Plot:
    
    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName

    def NNPlot_epochs(self, N_Epochs_plt, MSE_KF_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):

        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=Klegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=Klegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName)


    def NNPlot_Hist(self, MSE_KF_data_linear_arr, MSE_KN_linear_arr):

        fileName = self.folderName + 'plt_hist_dB'

        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))
        sns.distplot(10 * torch.log10(MSE_KN_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = self.modelName)
        #sns.distplot(10 * torch.log10(MSE_KF_design_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter - design')
        sns.distplot(10 * torch.log10(MSE_KF_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'r', label = 'Kalman Filter')

        plt.title("Histogram [dB]",fontsize=32)
        plt.legend(fontsize=32)
        plt.savefig(fileName)

    def KFPlot(res_grid):

        plt.figure(figsize = (50, 20))
        x_plt = [-6, 0, 6]

        plt.plot(x_plt, res_grid[0][:], 'xg', label='minus')
        plt.plot(x_plt, res_grid[1][:], 'ob', label='base')
        plt.plot(x_plt, res_grid[2][:], '+r', label='plus')
        plt.plot(x_plt, res_grid[3][:], 'oy', label='base NN')

        plt.legend()
        plt.xlabel('Noise', fontsize=16)
        plt.ylabel('MSE Loss Value [dB]', fontsize=16)
        plt.title('Change', fontsize=16)
        plt.savefig('plt_grid_dB')

        print("\ndistribution 1")
        print("Kalman Filter")
        print(res_grid[0][0], "[dB]", res_grid[1][0], "[dB]", res_grid[2][0], "[dB]")
        print(res_grid[1][0] - res_grid[0][0], "[dB]", res_grid[2][0] - res_grid[1][0], "[dB]")
        print("KalmanNet", res_grid[3][0], "[dB]", "KalmanNet Diff", res_grid[3][0] - res_grid[1][0], "[dB]")

        print("\ndistribution 2")
        print("Kalman Filter")
        print(res_grid[0][1], "[dB]", res_grid[1][1], "[dB]", res_grid[2][1], "[dB]")
        print(res_grid[1][1] - res_grid[0][1], "[dB]", res_grid[2][1] - res_grid[1][1], "[dB]")
        print("KalmanNet", res_grid[3][1], "[dB]", "KalmanNet Diff", res_grid[3][1] - res_grid[1][1], "[dB]")

        print("\ndistribution 3")
        print("Kalman Filter")
        print(res_grid[0][2], "[dB]", res_grid[1][2], "[dB]", res_grid[2][2], "[dB]")
        print(res_grid[1][2] - res_grid[0][2], "[dB]", res_grid[2][2] - res_grid[1][2], "[dB]")
        print("KalmanNet", res_grid[3][2], "[dB]", "KalmanNet Diff", res_grid[3][2] - res_grid[1][2], "[dB]")

    def NNPlot_test(MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg,
               MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg):


        N_Epochs_plt = 100

        ###############################
        ### Plot per epoch [linear] ###
        ###############################
        plt.figure(figsize = (50, 20))

        x_plt = range(0, N_Epochs_plt)

        # KNet - Test
        y_plt3 = MSE_test_linear_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_linear_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend()
        plt.xlabel('Number of Training Epochs', fontsize=16)
        plt.ylabel('MSE Loss Value [linear]', fontsize=16)
        plt.title('MSE Loss [linear] - per Epoch', fontsize=16)
        plt.savefig('plt_model_test_linear')

        ###########################
        ### Plot per epoch [dB] ###
        ###########################
        plt.figure(figsize = (50, 20))

        x_plt = range(0, N_Epochs_plt)

        # KNet - Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend()
        plt.xlabel('Number of Training Epochs', fontsize=16)
        plt.ylabel('MSE Loss Value [dB]', fontsize=16)
        plt.title('MSE Loss [dB] - per Epoch', fontsize=16)
        plt.savefig('plt_model_test_dB')

        ########################
        ### Linear Histogram ###
        ########################
        plt.figure(figsize=(50, 20))
        sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
        sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
        plt.title("Histogram [Linear]")
        plt.savefig('plt_hist_linear')

        fig, axes = plt.subplots(2, 1, figsize=(50, 20), sharey=True, dpi=100)
        sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label='KalmanNet', ax=axes[0])
        sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='b', label='Kalman Filter', ax=axes[1])
        plt.title("Histogram [Linear]")
        plt.savefig('plt_hist_linear_1')

        ####################
        ### dB Histogram ###
        ####################

        plt.figure(figsize=(50, 20))
        sns.distplot(10 * torch.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
        sns.distplot(10 * torch.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
        plt.title("Histogram [dB]")
        plt.savefig('plt_hist_dB')


        fig, axes = plt.subplots(2, 1, figsize=(50, 20), sharey=True, dpi=100)
        sns.distplot(10 * torch.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet', ax=axes[0])
        sns.distplot(10 * torch.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter', ax=axes[1])
        plt.title("Histogram [dB]")
        plt.savefig('plt_hist_dB_1')

        print('End')


        # KF_design_MSE_mean_dB = 10 * torch.log10(torch.mean(MSE_KF_design_linear_arr))
        # KF_design_MSE_median_dB = 10 * torch.log10(torch.median(MSE_KF_design_linear_arr))
        # KF_design_MSE_std_dB = 10 * torch.log10(torch.std(MSE_KF_design_linear_arr))
        # print("kalman Filter - Design:",
        #       "MSE - mean", KF_design_MSE_mean_dB, "[dB]",
        #       "MSE - median", KF_design_MSE_median_dB, "[dB]",
        #       "MSE - std", KF_design_MSE_std_dB, "[dB]")
        
        # KF_data_MSE_mean_dB = 10 * torch.log10(torch.mean(MSE_KF_data_linear_arr))
        # KF_data_MSE_median_dB = 10 * torch.log10(torch.median(MSE_KF_data_linear_arr))
        # KF_data_MSE_std_dB = 10 * torch.log10(torch.std(MSE_KF_data_linear_arr))
        # print("kalman Filter - Data:",
        #       "MSE - mean", KF_data_MSE_mean_dB, "[dB]",
        #       "MSE - median", KF_data_MSE_median_dB, "[dB]",
        #       "MSE - std", KF_data_MSE_std_dB, "[dB]")
        
        # KN_MSE_mean_dB = 10 * torch.log10(torch.mean(MSE_KN_linear_arr))
        # KN_MSE_median_dB = 10 * torch.log10(torch.median(MSE_KN_linear_arr))
        # KN_MSE_std_dB = 10 * torch.log10(torch.std(MSE_KN_linear_arr))
        
        # print("kalman Net:",
        #       "MSE - mean", KN_MSE_mean_dB, "[dB]",
        #       "MSE - median", KN_MSE_median_dB, "[dB]",
        #       "MSE - std", KN_MSE_std_dB, "[dB]")


class Plot_RTS(Plot):

    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName

    def NNPlot_epochs(self, N_MiniBatchTrain_plt, BatchSize, MSE_KF_dB_avg, MSE_RTS_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):
        N_Epochs_plt = np.floor(N_MiniBatchTrain_plt/BatchSize).astype(int) # number of epochs
        
        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=RTSlegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=RTSlegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=RTSlegend[2])

        # RTS
        y_plt4 = MSE_RTS_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=RTSlegend[3])

        # KF
        y_plt5 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt5, KColor[4], label=RTSlegend[4])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName)


    def NNPlot_Hist(self, MSE_KF_linear_arr, MSE_RTS_data_linear_arr, MSE_RTSNet_linear_arr):

        fileName = self.folderName + 'plt_hist_dB'
        fontSize = 32
        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))
        sns.distplot(10 * torch.log10(MSE_RTSNet_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = self.modelName)
        sns.distplot(10 * torch.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
        sns.distplot(10 * torch.log10(MSE_RTS_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'r', label = 'RTS Smoother')

        plt.title(self.modelName + ":" +"Histogram [dB]",fontsize=fontSize)
        plt.legend(fontsize=fontSize)
        plt.savefig(fileName)

    def KF_RTS_Plot(self, r, MSE_KF_RTS_dB):
        fileName = self.folderName + 'KF_RTS_Compare_dB'
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r**2)

        plt.plot(x_plt, MSE_KF_RTS_dB[0,:], '-gx', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, KF')
        plt.plot(x_plt, MSE_KF_RTS_dB[1,:], '--bo', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTS')

        plt.legend(fontsize=32)
        plt.xlabel(r'Noise $\frac{1}{r^2}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        plt.title('Comparing Kalman Filter and RTS Smoother', fontsize=32)
        plt.grid(True)
        plt.savefig(fileName)

    def rotate_RTS_Plot_F(self, r, MSE_RTS_dB,rotateName):
        fileName = self.folderName + rotateName
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r**2)

        plt.plot(x_plt, MSE_RTS_dB[0,:], '-r^', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=0^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB[1,:], '-gx', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB[2,:], '-bo', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTSNet ($\mathbf{F}_{\alpha=10^\circ}$)')

        plt.legend(fontsize=16)
        plt.xlabel(r'Noise $\frac{1}{r^2}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)
        plt.savefig(fileName)  

    def rotate_RTS_Plot_H(self, r, MSE_RTS_dB,rotateName):
        fileName = self.folderName + rotateName
        magnifying_glass, main_H = plt.subplots(figsize = [25, 10])
        # main_H = plt.figure(figsize = [25, 10])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_H.plot(x_plt, NoiseFloor, '--r', linewidth=2, markersize=12, label=r'Noise Floor')
        main_H.plot(x_plt, MSE_RTS_dB[0,:], '-y^', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB] , 2x2, RTS Smoother ($\mathbf{H}_{\alpha=0^\circ}$)')
        main_H.plot(x_plt, MSE_RTS_dB[1,:], '-gx', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=10^\circ}$)')
        main_H.plot(x_plt, MSE_RTS_dB[2,:], '-bo', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{H}_{\alpha=10^\circ}$)')

        main_H.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-20, 15))
        main_H.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)

        ax2 = plt.axes([.15, .15, .27, .27]) 
        x1, x2, y1, y2 = -0.5, 0.5, -7.5, 10
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.plot(x_plt, NoiseFloor, '--r', linewidth=2, markersize=12, label=r'Noise Floor')
        ax2.plot(x_plt, MSE_RTS_dB[0,:], '-y^', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB] , 2x2, RTS Smoother ($\mathbf{H}_{\alpha=0^\circ}$)')
        ax2.plot(x_plt, MSE_RTS_dB[1,:], '-gx', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=10^\circ}$)')
        ax2.plot(x_plt, MSE_RTS_dB[2,:], '-bo', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{H}_{\alpha=10^\circ}$)')
        ax2.grid(True)
        plt.savefig(fileName)    

    def rotate_RTS_Plot_FHCompare(self, r, MSE_RTS_dB_F,MSE_RTS_dB_H,rotateName):
        fileName = self.folderName + rotateName
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r)

        plt.plot(x_plt, MSE_RTS_dB_F[0,:], '-r^', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=0^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_F[1,:], '-gx', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_F[2,:], '-bo', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTSNet ($\mathbf{F}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_H[0,:], '--r^', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=0^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_H[1,:], '--gx', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_H[2,:], '--bo', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTSNet ($\mathbf{H}_{\alpha=10^\circ}$)')

        plt.legend(fontsize=16)
        plt.xlabel(r'Noise $\frac{1}{r^2}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)
        plt.savefig(fileName)  

class Plot_extended(Plot_RTS):
    def EKFPlot_Hist(self, MSE_EKF_linear_arr):   
        fileName = self.folderName + 'plt_hist_dB'
        fontSize = 32
        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))       
        sns.distplot(10 * np.log10(MSE_EKF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Extended Kalman Filter')
        plt.title(self.modelName + ":" +"Histogram [dB]",fontsize=fontSize)
        plt.legend(fontsize=fontSize)
        plt.savefig(fileName)

    def KF_RTS_Plot(self, r, MSE_KF_RTS_dB):
        fileName = self.folderName + 'KF_RTS_Compare_dB'
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r**2)

        plt.plot(x_plt, MSE_KF_RTS_dB[0,:], '-gx', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, KF')
        plt.plot(x_plt, MSE_KF_RTS_dB[1,:], '--bo', label=r'$\frac{q^2}{r^2}=0$ [dB], 2x2, RTS')

        plt.legend(fontsize=32)
        plt.xlabel(r'Noise $\frac{1}{r^2}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        plt.title('Comparing Kalman Filter and RTS Smoother', fontsize=32)
        plt.grid(True)
        plt.savefig(fileName)
