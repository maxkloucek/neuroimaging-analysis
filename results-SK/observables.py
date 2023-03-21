# defining each oth the things I can measure
import numpy as np
from pyplm.utilities import tools
from scipy import signal
from scipy.interpolate import UnivariateSpline

def calc_m(trajectory):
    si_avrg = np.mean(trajectory, axis=0)
    return np.mean(si_avrg)

def calc_q(trajectory):
    si_avrg = np.mean(trajectory, axis=0)
    si_avrg_sqr = si_avrg ** 2
    return np.mean(si_avrg_sqr)

def calc_C2(trajectory):
    cij = np.cov(trajectory.T)
    N, _ = cij.shape
    C = np.sum(cij ** 2)
    C = C / N
    return C

def calc_tau(trajectory):
    B, N = trajectory.shape
    auto_corr = 0
    for i in range(0, N):
        data = trajectory[:, i]
        # normalising or z-scoring it!
        # EXP_si = np.mean(data)
        # STD_si = np.std(data)
        # data = (data - np.mean(data))  # / STD_si
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
    # auto_corr /= auto_corr.max()
    # return auto_corr, correlation_time
    return correlation_time


def recon_error_nguyen(true_model, inferred_model):
    true_params = tools.triu_flat(true_model, k=0)
    infr_params = tools.triu_flat(inferred_model, k=0)
    infr_params = np.nan_to_num(infr_params, nan=np.nanmax(infr_params))
    numerator = np.sum((infr_params - true_params) ** 2)
    denominator = np.sum(true_params ** 2)
    return np.sqrt(numerator / denominator), numerator, denominator