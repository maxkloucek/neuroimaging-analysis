import numpy as np
from scipy import signal
from scipy.interpolate import UnivariateSpline

# not sure whether to z-score or not?
def binarize(trajectory, TH=0):
    # this seems like crap...
    # I'm slightly worried about the way I've binarized the data..
    # oh well! that's an issue for later!
    t_len, ROI_len = trajectory.shape
    mu = []  # global signal mean for each time point
    sigma = []  # std of global signal for each time point
    for t in range(0, t_len):
        mu.append(np.mean(trajectory[t, :]))
        sigma.append(np.std(trajectory[t, :]))
    mu = np.array(mu)
    sigma = np.array(sigma)

    z = np.zeros(t_len*ROI_len).reshape(t_len, ROI_len)
    S = np.zeros(t_len*ROI_len).reshape(t_len, ROI_len)
    for t in range(0, t_len):
        for ROI in range(0, ROI_len):
            z[t, ROI] = (trajectory[t, ROI] - mu[t]) / sigma[t]

    for t in range(0, t_len):
        for ROI in range(0, ROI_len):
            if z[t, ROI] >= TH:
                S[t, ROI] = 1
            else:  # i.e. < 0
                S[t, ROI] = -1
    return z, S

def binarize2(raw_trajectory, TH=0):
    B, N = raw_trajectory.shape
    spin_trajectory = np.zeros((B, N))
    spin_trajectory[raw_trajectory >= 0] = 1
    spin_trajectory[raw_trajectory <= 0] = -1
    return spin_trajectory


def overlap(trajectory):
    # print(trajectory.shape)
    B, N = trajectory.shape
    auto_corr = 0
    for i in range(0, N):
        data = trajectory[:, i]
        # normalising or z-scoring it!
        # EXP_si = np.mean(data)
        # STD_si = np.std(data)
        data = (data - np.mean(data))  # / STD_si
        ac = signal.correlate(
            data, data, mode='full', method='auto')

        ac = ac[int(ac.size/2):]
        auto_corr += ac
    auto_corr /= N
    lags = np.arange(0, auto_corr.size)
    auto_corr_norm = auto_corr/auto_corr.max()
    ac_spline = UnivariateSpline(
        lags, auto_corr_norm-np.exp(-1), s=0)
    ac_roots = ac_spline.roots()
    correlation_time = ac_roots[0]
    auto_corr /= auto_corr.max()
    # auto_corr = auto_corr / auto_corr[0]
    return auto_corr, correlation_time