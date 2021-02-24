import torch
import scipy.io

class DataAnalysis:

    def main(self, MSE_KF_dB_avg):

        fName_rnn = 'RNN' + '\\' + 'pipeline_rnn.pt'
        pipe_rnn = torch.load(fName_rnn)

        fName_mb_rnn = 'MB_RNN' + '\\' + 'pipeline_mb_rnn.pt'
        pipe_mb_rnn = torch.load(fName_mb_rnn)

        fName_mb_rnn_diff = 'MB_RNN_diff' + '\\' + 'pipeline_mb_rnn_diff.pt'
        pipe_mb_rnn_diff = torch.load(fName_mb_rnn_diff)

        fName_KNet = 'KNet' + '\\' + 'pipeline_KalmanNet.pt'
        pipe_KNet = torch.load(fName_KNet)

        self.setData(KF=MSE_KF_dB_avg,
                     rnn_test=pipe_rnn.MSE_test_dB_avg, rnn_cv=pipe_rnn.MSE_cv_dB_epoch,
                     mb_rnn_test=pipe_mb_rnn.MSE_test_dB_avg, mb_rnn_cv=pipe_mb_rnn.MSE_cv_dB_epoch,
                     mb_rnn_diff_test=pipe_mb_rnn_diff.MSE_test_dB_avg, mb_rnn_diff_cv=pipe_mb_rnn_diff.MSE_cv_dB_epoch,
                     KNet_test=pipe_KNet.MSE_test_dB_avg, KNet_cv=pipe_KNet.MSE_cv_dB_epoch)

        matlabFolderName = 'Matlab' + '\\'
        torch.save(self, matlabFolderName + 'myMatlab.pt')
        myMatlab = torch.load(matlabFolderName + 'myMatlab.pt')

        matDict = dict(KF=myMatlab.KF, KNet_cv=myMatlab.KNet_cv, KNet_test=myMatlab.KNet_test,
                       mb_rnn_diff_cv=myMatlab.mb_rnn_diff_cv, mb_rnn_diff_test=myMatlab.mb_rnn_diff_test,
                       mb_rnn_cv=myMatlab.mb_rnn_cv, mb_rnn_test=myMatlab.mb_rnn_test,
                       rnn_cv=myMatlab.rnn_cv, rnn_test=myMatlab.rnn_test)

        scipy.io.savemat(matlabFolderName + 'myMatlab.mat', matDict)


    def setData(self, KF, rnn_test, rnn_cv, mb_rnn_test, mb_rnn_cv, mb_rnn_diff_test, mb_rnn_diff_cv, KNet_test, KNet_cv):

        self.KF = KF
        self.rnn_test = rnn_test
        self.rnn_cv = rnn_cv
        self.mb_rnn_test = mb_rnn_test
        self.mb_rnn_cv = mb_rnn_cv
        self.mb_rnn_diff_test = mb_rnn_diff_test
        self.mb_rnn_diff_cv = mb_rnn_diff_cv
        self. KNet_test = KNet_test
        self.KNet_cv = KNet_cv

