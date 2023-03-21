# aim:
# usng the saturation curves T^*(B) we fit the
# limit temperature T^{*}(B -> inf) and plot this as a function
# of T^0
import os
import glob
import h5py
import numpy as np
from scipy.optimize import curve_fit

import load
from inference import tools


def T_true_v_inferred(iB_max):
    Jrep_root = '/Users/mk14423/Desktop/PaperData'
    JRps, JRobs, samples = load.load_N200_Jrepeats_Bscaling(Jrep_root)
    print(samples[iB_max])
    iB_min = 1
    iTs = np.arange(0, 21)
    T_limits = np.zeros((iTs.shape))
    for iT in iTs:
        Bs, Ts, Ts_err = load.get_curves_fixedT(JRps, JRobs, samples, iT, iB_min, iB_max)
        Tlim_Btilde = saturation_fit(Bs, Ts)
        # print(Tlim_Btilde[0])
        T_limits[iT] = Tlim_Btilde[0]
    T_theory = JRps['T'][0, :]
    T_limits[T_limits <= 0] = 0
    # print(T_limits)
    return T_theory, T_limits


def saturation_fit(Bs, Ts):
    popt, _ = curve_fit(tools.arctan, Bs, Ts)
    Tlim = (popt[0] * np.pi) / 2
    Btilde = 1 / popt[1]
    return np.array([Tlim, Btilde])


def overwrite_correction(iB_max, iT, T_fit):
    root = '/Users/mk14423/Desktop/PaperData/N200_saturation_correction'
    Ts = [
        0.5, 0.575, 0.65, 0.725, 0.8, 0.875, 0.95, 1.025, 1.1,
        1.175, 1.25, 1.325, 1.4, 1.475, 1.55, 1.625, 1.7, 1.775,
        1.85, 1.925, 2.0]
    Bs = [
        '1e3_1', '2e3_1', '3e3_1', '4e3_1', '5e3_1', '6e3_1', '7e3_1',
        '8e3_1', '9e3_1', '1e4_1', '2e4_1', '3e4_1', '4e4_1', '5e4_1',]
    fnames = (
        f'B{Bs[iB_max]}/T_{Ts[iT]:.3f}' +
        '-h_0.000-J_*-Jstd_1.000.hdf5')

    fnames = os.path.join(root, fnames)
    print(fnames)
    files = np.sort(glob.glob(fnames))
    for file in files:
        with h5py.File(file, 'a') as fin:
            mod_plm = fin['InferredModel'][()]
            p_plm = tools.triu_flat(mod_plm)
            T_plm = 1 / (np.std(p_plm) * (200 ** 0.5))
            saturation_correction = T_fit / T_plm
            # print(f'{T_fit:.3f}, {T_plm:.3f}, {saturation_correction:.3f}')
            correction = fin['correction'][0]
            print(correction, saturation_correction)
            fin['correction'][0] = saturation_correction
