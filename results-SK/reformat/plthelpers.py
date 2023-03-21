import os
import numpy as np
import load
import h5py
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

from inference import tools


def get_kajrimuraSubSamples():
    dataroot = (
        '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/' +
        'noMM_subsample_noL1')
    # model_files = [
    #     'noMM.hdf5',
    #     # 'noMM_matched_subsamples/noMM_subsample0.hdf5',
    #     # 'noMM_matched_subsamples/noMM_subsample1.hdf5',
    #     # 'MM.hdf5'
    # ]
    # lbls = ['noMM', 'ss-noMM', 'MM', 'plm', 'correction']
    # lbls = ['noMM', 'MM', 'ss-plm', 'ss-correction']
    B, T_plm, T_cor = load.kajimura_get_temps(dataroot)
    return B, T_plm, T_cor


def kajimura_load_all_subsampled_models(
        root='/Users/mk14423/Desktop/PaperData/Kajimura_analysis/' +
        'noMM_subsample_noL1'):
    nSubSamples = np.arange(4, 41)
    dataDirectories = [
        os.path.join(root, 'ss{:d}'.format(nSS)) for nSS in nSubSamples]
    subsampled_parameters_list = []
    for iD, dir in enumerate(dataDirectories):
        models = kajimura_load_subsampled_models(dir)
        subsampled_parameters_list.append(models)
    nSubSamples = nSubSamples * 236
    return nSubSamples, subsampled_parameters_list


def kajimura_load_subsampled_models(dir):
    files = [
        os.path.join(dir, f) for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f))]

    with h5py.File(files[0], 'r') as fin:
        N, _ = fin['InferredModel'].shape
    models = np.zeros((len(files), N, N))
    for iF, file in enumerate(files):
        with h5py.File(file, 'r') as fin:
            models[iF] = fin['InferredModel'][()]
    return models


# --------------------------------------------------------------------------- #
# these are the functions that are more generally applicable; load all params

def kajimura_load_all_subsampled_parameters(
        root='/Users/mk14423/Desktop/PaperData/Kajimura_analysis/' +
        'noMM_subsample_noL1'):
    nSubSamples = np.arange(4, 41)
    dataDirectories = [
        os.path.join(root, 'ss{:d}'.format(nSS)) for nSS in nSubSamples]
    subsampled_parameters_list = []
    for iD, dir in enumerate(dataDirectories):
        parameters = kajimura_load_subsampled_parameters(dir)
        subsampled_parameters_list.append(parameters)
    nSubSamples = nSubSamples * 236
    return nSubSamples, subsampled_parameters_list


def kajimura_load_subsampled_parameters(dir):
    files = [
        os.path.join(dir, f) for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f))]

    with h5py.File(files[0], 'r') as fin:
        N, _ = fin['InferredModel'].shape
    # Np = int(N + (N * (N - 1)) / 2)
    Np = int((N * (N - 1)) / 2)
    parameters = np.zeros((len(files), Np))
    for iF, file in enumerate(files):
        with h5py.File(file, 'r') as fin:
            # models[iF] = fin['InferredModel'][()]
            parameters[iF] = tools.triu_flat(fin['InferredModel'][()], k=1)
    return parameters


# --------------------------------------------------------------------------- #


def SK_load_all_sample_parameters(iT):
    root = '/Users/mk14423/Desktop/PaperData'
    JRparams, JRfull_obs, Bs = load.load_N200_Jrepeats_Bscaling(root)
    # ok I have inferred sigs already; I'm going to trust I did
    # this for parameters, and not models!
    sigmas = JRfull_obs['infrSig']
    sigmas = sigmas[:, :, iT]
    # so iT selects the temperature or whatever!
    # I don't remember which index fixes temperature...
    # Ts = JRparams['T'][:, iT] the second!
    sigmas = np.mean(sigmas, axis=1)
    return Bs, sigmas

# --------------------------------------------------------------------------- #


def kajimura_convert_to_sigma(parameter_list):
    sigma_for_subsample = []
    for parameters in parameter_list:
        # print(parameters.shape)
        simgas = np.std(parameters, axis=1)
        sigma_for_subsample.append(np.mean(simgas))
    return np.array(sigma_for_subsample)


def get_SK_B_T(iT):
    root = '/Users/mk14423/Desktop/PaperData'
    JRparams, JRfull_obs, Bs = load.load_N200_Jrepeats_Bscaling(root)
    T_theory = JRparams['T'][0, :]
    T_curves = []
    T_curves_std = []

    for i, B in enumerate(Bs):
        obs = JRfull_obs[i]
        Ts_inf = 1 / (obs['infrSig'] * 200 ** 0.5)
        T_curves.append(np.mean(Ts_inf, axis=0))
        T_curves_std.append(np.std(Ts_inf, axis=0))

    T_curves = np.array(T_curves)
    T_curves_std = np.array(T_curves_std)

    # so T_curves shape is (14, 21)
    # i.e. I want iT not iB!

    # Bs = Bs[Bcut:Bmax]
    # T_curves = T_curves[Bcut:Bmax, Tcut:Tmax]
    # T_curves_std = T_curves_std[Bcut:Bmax, Tcut:Tmax]
    # T_theory = T_theory[Tcut:Tmax]
    print(T_theory[iT])
    return Bs, T_curves[:, iT]


# this selects in revers!
# select the fit_ic with best r2!
def invB_T_lin_fit_varyingCut(x, y):
    r2s = []
    # print(x.size, y.shape)
    fit_ics = np.arange(x.size, 5, -1)
    for fit_ic in fit_ics:
        fit_params, _ = curve_fit(tools.linear, x[-fit_ic:], y[-fit_ic:])
        y_fit = tools.linear(x[-fit_ic:], *fit_params)
        r2s.append(r2_score(y[-fit_ic:], y_fit))
    r2s = np.array(r2s)
    fit_ic = int(fit_ics[r2s == r2s.max()])

    fit_params, _ = curve_fit(tools.linear, x[-fit_ic:], y[-fit_ic:])
    fit_x = np.linspace(0, x[0], 200)
    fit_y = tools.linear(fit_x, *fit_params)
    print(fit_ic, 1/x[fit_ic], fit_y[0], fit_params, r2s.max())
    return fit_x, fit_y, fit_params


def fit_linear_decreasing(xs, ys):
    r2s = []
    # print(x.size, y.shape)
    fit_ics = np.arange(xs.size, 5, -1)
    for fit_ic in fit_ics:
        fit_params, _ = curve_fit(tools.linear, xs[-fit_ic:], ys[-fit_ic:])
        y_fit = tools.linear(xs[-fit_ic:], *fit_params)
        r2s.append(r2_score(ys[-fit_ic:], y_fit))
    r2s = np.array(r2s)
    fit_ic = int(fit_ics[r2s == r2s.max()])

    fit_params, _ = curve_fit(tools.linear, xs[-fit_ic:], ys[-fit_ic:])
    fit_x = np.linspace(0, xs[0], 200)
    fit_y = tools.linear(fit_x, *fit_params)
    # print(fit_ic, 1/xs[fit_ic], fit_y[0], fit_params, r2s.max())
    return fit_x, fit_y, fit_params


def fit_linear_increaseing(xs, ys):
    r2s = []
    # print(x.size, y.shape)
    n_min_points = 3
    fit_thresholds = np.arange(0, xs.size - n_min_points, 1)
    for th in fit_thresholds:
        xs_th = xs[th:]
        ys_th = ys[th:]
        fit_params, _ = curve_fit(tools.linear, xs_th, ys_th)
        y_fit = tools.linear(xs_th, *fit_params)
        r2s.append(r2_score(ys_th, y_fit))
    r2s = np.array(r2s)
    print(r2s)
    best_th = int(fit_thresholds[r2s == r2s.max()])

    fit_params, _ = curve_fit(tools.linear, xs[best_th:], ys[best_th:])
    # fit_x = np.linspace(xs[0], xs[-1], 200)
    fit_x = np.linspace(0, 5e4, 200)
    fit_y = tools.linear(fit_x, *fit_params)
    print(best_th, r2s.max(), fit_y[0], fit_params)
    return fit_x, fit_y, fit_params
