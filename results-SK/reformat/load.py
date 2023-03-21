import os
import glob
from time import perf_counter
import h5py
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import inference.analysis.new as analysis
from inference import tools

# this is to load all the files and put them in one thing?
def load_full_dataset(globRuns, splitchar='B'):
    weights = []
    full_obs = 0

    outpath = os.path.normpath(globRuns[0] + os.sep + os.pardir)
    BSCALING_outpath = os.path.join(outpath, 'B-SCALINGOBS.hdf5')
    if os.path.exists(BSCALING_outpath) is False:
        print('NO SCALINGS OBS FOUND')
        for c, run in enumerate(globRuns):
            run_dirs = glob.glob(run)
            # print(globRuns)
            # print(run_dirs)
            nSamples = run_dirs[0].split('_')
            # print(nSamples)
            nSamples = nSamples[2].split(splitchar)[1]
            # print(nSamples)
            nSamples = float(nSamples)
            # for _ in range(0, len(run_dirs)):
            weights.append(nSamples)
            params, obs = load_repeats(run_dirs)
            obs = obs[np.newaxis, ...]
            if c == 0:
                full_obs = np.copy(obs)
            else:
                full_obs = np.append(full_obs, obs, axis=0)
        weights = np.array(weights)
        with h5py.File(BSCALING_outpath, 'w') as fout:
            param_dset = fout.create_dataset("params", data=params)
            param_dset[()] = params
            obs_dset = fout.create_dataset("full_obs", data=full_obs)
            obs_dset[()] = full_obs
            weights_dset = fout.create_dataset("weights", data=weights)
            weights_dset[()] = weights
    else:
        # print('SACLING OBS FOUND, reading ./SCALINGOBS.hdf5...')
        with h5py.File(BSCALING_outpath, 'r') as fin:
            params = fin['params'][()]
            full_obs = fin['full_obs'][()]
            weights = fin['weights'][()]
    return params, full_obs, weights


def load_repeats(run_directories, returnISFs=False):
    observables = []
    print(run_directories)
    pd = analysis.PhaseDiagram(run_directories)
    # pd.calculate(obs_kwrds=['basic', 'tau', 'pca', 'tail_weight', 'syserr'])
    # pd.calculate(obs_kwrds=['basic', 'tau', 'syserr'])
    for repID in range(0, len(run_directories)):
        params, obs, ISFs = pd.load_run(repID)
        obs = analysis.set_dimensions(obs, 0, None, 0, None)
        observables.append(obs)
    observables = np.array(observables)
    params = analysis.set_dimensions(params, 0, None, 0, None)
    return params, observables


# LOADING SPECIFIC DATASETS #
def load_PD_fixedB(runs, root='/Users/mk14423/Desktop/PaperData'):
    runs = [os.path.join(root, run) for run in runs]
    print(runs)
    # pd = analysis.PhaseDiagram(runs)
    # pd.calculate(obs_kwrds=['basic', 'tau', 'syserr'])
    # params, obs, stds = pd.averages()

    params, observables = load_repeats(runs)
    print(params.shape, observables.shape)
    keys = observables.dtype.names
    obs = np.zeros_like(observables)[0, :, :]
    for key in keys:
        obs[key] = np.mean(observables[key], axis=0)
    return params, obs


def load_N200_Bfixed_obs(root='/Users/mk14423/Desktop/PaperData'):
    runs = [
            'B1e4_Nscaling/N200_1',
            'B1e4_Nscaling/N200_2',
            'B1e4_Nscaling/N200_3',
        ]
    runs = [os.path.join(root, run) for run in runs]
    print(runs)
    # pd = analysis.PhaseDiagram(runs)
    # pd.calculate(obs_kwrds=['basic', 'tau', 'syserr'])
    # params, obs, stds = pd.averages()

    params, observables = load_repeats(runs)
    print(params.shape, observables.shape)
    keys = observables.dtype.names
    obs = np.zeros_like(observables)[0, :, :]
    for key in keys:
        obs[key] = np.mean(observables[key], axis=0)
    return params, obs


def load_N200_Jrepeats_Bscaling(root='/Users/mk14423/Desktop/PaperData'):
    runs = [
        'N200_J0.1_Bscaling/B1e3_*',
        'N200_J0.1_Bscaling/B2e3_*',
        'N200_J0.1_Bscaling/B3e3_*',
        'N200_J0.1_Bscaling/B4e3_*',
        'N200_J0.1_Bscaling/B5e3_*',
        'N200_J0.1_Bscaling/B6e3_*',
        'N200_J0.1_Bscaling/B7e3_*',
        'N200_J0.1_Bscaling/B8e3_*',
        'N200_J0.1_Bscaling/B9e3_*',
        'N200_J0.1_Bscaling/B1e4_*',
        'N200_J0.1_Bscaling/B2e4_*',
        'N200_J0.1_Bscaling/B3e4_*',
        'N200_J0.1_Bscaling/B4e4_*',
        'N200_J0.1_Bscaling/B5e4_*',
    ]
    runs = [os.path.join(root, run) for run in runs]
    params, full_obs, Bs = load_full_dataset(runs, 'Bscaling/B')
    full_obs = full_obs[:, 0, :, :]
    return params, full_obs, Bs


def load_trail_params(
        root='/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated'):
    runs = [
        'B1e3_1/T_1.250-h_0.000-J_0.500-Jstd_1.000.hdf5',
        'B2e3_1/T_1.250-h_0.000-J_0.500-Jstd_1.000.hdf5',
        'B1e4_1/T_1.250-h_0.000-J_0.500-Jstd_1.000.hdf5'
    ]

    runs = [os.path.join(root, run) for run in runs]
    # print(runs)
    params_true = []
    params_infr = []
    params_corr = []
    for run in runs:
        with h5py.File(run, 'r') as fin:
            # print(list(fin.keys()))
            mod_true = fin['InputModel'][()]
            mod_infr = fin['InferredModel'][()]
            correction = fin['correction'][()]
            mod_corr = mod_infr / correction[0]

            param_true = tools.triu_flat(mod_true, k=1)
            param_infr = tools.triu_flat(mod_infr, k=1)
            param_corr = tools.triu_flat(mod_corr, k=1)

            params_true.append(param_true)
            params_infr.append(param_infr)
            params_corr.append(param_corr)
            # mod corrected? would probably be good to have as well!
    return np.array(params_true), np.array(params_infr), np.array(params_corr)


def load_trail_params_JR(
        root='/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated'):
    runs = [
        'B1e3_1/T_1.250-h_0.000-J_*-Jstd_1.000.hdf5',
        'B2e3_1/T_1.250-h_0.000-J_*-Jstd_1.000.hdf5',
        'B1e4_1/T_1.250-h_0.000-J_*-Jstd_1.000.hdf5',
        # 'B5e4_1/T_1.250-h_0.000-J_*-Jstd_1.000.hdf5',
    ]
    extra = (
        '/Users/mk14423/Desktop/PaperData/N200_J0.1_Bscaling/B5e4_1/' +
        'T_1.250-h_0.000-J_*-Jstd_1.000.hdf5')
    runs = [os.path.join(root, run) for run in runs]
    runs.append(extra)
    print(np.array(runs))
    params_true = []
    params_infr = []
    params_corr = []
    for run in runs:
        files = glob.glob(run)
        ps_true = []
        ps_infr = []
        ps_corr = []
        for file in files:
            with h5py.File(file, 'r') as fin:

                mod_true = fin['InputModel'][()]
                mod_infr = fin['InferredModel'][()]
                param_true = tools.triu_flat(mod_true, k=1)
                param_infr = tools.triu_flat(mod_infr, k=1)
                ps_true = ps_true + list(param_true)
                ps_infr = ps_infr + list(param_infr)

                if extra not in runs:
                    correction = fin['correction'][()]
                    mod_corr = mod_infr / correction[0]
                    param_corr = tools.triu_flat(mod_corr, k=1)
                    ps_corr = ps_corr + list(param_corr)

        params_true.append(np.array(ps_true))
        params_infr.append(np.array(ps_infr))
        if extra not in runs:
            params_corr.append(np.array(ps_corr))
    if extra not in runs:
        return np.array(params_true), np.array(params_infr), np.array(params_corr)
    else:
        return np.array(params_true), np.array(params_infr)


def average_over_Jrepeats(params, full_obs, iB):
    inTemps = params['T'][iB, :]
    repObs = full_obs[iB]
    print(params.shape, full_obs.shape, iB, repObs.shape)
    # outObs = np.nanmean(repObs)
    outMeans, outStds = npNamed_meanstd(repObs, axis=0)
    return inTemps, outMeans, outStds


# this is not great; always collapses onto 0th axis...
def npNamed_meanstd(array, axis):
    keys = array.dtype.names
    means = np.zeros_like(array)[0]
    stds = np.zeros_like(array)[0]

    for key in keys:
        means[key] = np.mean(array[key], axis=axis)
        stds[key] = np.std(array[key], axis=axis)
    return means, stds


# calculates the average of observalbes across a number of runs.
def crossrunAverage(runs, obsKwrds=['trueMu', 'trueSig', 'chiSG']):
    print(runs)
    weights = []
    dt_superObs = np.dtype(
        {
            'names': obsKwrds,
            'formats': [(float)]*len(obsKwrds)
            })
    superObs = np.zeros((len(runs), 441), dtype=dt_superObs)
    musigChi = np.zeros((441), dtype=dt_superObs)
    print(superObs.shape)
    for c, run in enumerate(runs):
        fixedB_rundirs = glob.glob(run)
        print(fixedB_rundirs)
        pd = analysis.PhaseDiagram(fixedB_rundirs)
        _, _, autoCorrs = pd.load_run(runID=0)
        _, nSamples = autoCorrs.shape
        params, obs, stds = pd.averages()
        for kwrd in obsKwrds:
            superObs[kwrd][c, :] = obs[kwrd]
        weights.append(nSamples)
    # super_obs = np.array(super_obs)
    # print(superObs.shape, weights)
    # I'm not sold on this yet... anyway...
    for kwrd in obsKwrds:
        musigChi[kwrd] = np.average(superObs[kwrd], axis=0, weights=weights)
    return musigChi, weights


def crossrunsChi_interpolate(musigChi_cravrg, mu_infr, sig_infr):
    # print(muTrueKwrd, sigTrueKwrd, muInfrKwrd, sigInfrKwrd, chiKwrd)
    mu_true = musigChi_cravrg['trueMu']
    sigma_true = musigChi_cravrg['trueSig']
    # mu_infr = obs[muInfrKwrd]
    # sigma_infr = obs[sigInfrKwrd]
    chi = musigChi_cravrg['chiSG']

    x = musigChi_cravrg['trueMu'].reshape(-1, 1)
    y = musigChi_cravrg['trueSig'].reshape(-1, 1)
    x_interp = mu_infr.reshape(-1, 1)
    y_interp = sig_infr.reshape(-1, 1)

    z = chi
    nStatepoints = z.size
    L = int(np.sqrt(nStatepoints))
    z = z.reshape(L, L, order='C')
    # z = gaussian_filter(z, sigma=SIGMA)
    z = z.reshape(nStatepoints, order='C')
    xy = np.hstack((x, y))
    xy_interp = np.hstack((x_interp, y_interp))
    interpolator = RBFInterpolator(
        xy, z,
        kernel='thin_plate_spline',
        # smoothing=0.01
        # epsilon=10
        )
    z_interp = interpolator(xy_interp)
    lim = np.NaN
    z_interp[mu_infr <= mu_true.min()] = lim
    z_interp[mu_infr >= mu_true.max()] = lim
    z_interp[sig_infr <= sigma_true.min()] = lim
    z_interp[sig_infr >= sigma_true.max()] = lim
    z_interp[z_interp < 0] = lim
    # this should not be able to be less than 0!!
    print(np.nanmin(z_interp), np.nanmax(z_interp))
    return z_interp


def fe2D(obs, muTrueKwrd, sigTrueKwrd, muInfrKwrd, sigInfrKwrd, chiKwrd):
    # print(muTrueKwrd, sigTrueKwrd, muInfrKwrd, sigInfrKwrd, chiKwrd)
    mu_true = obs[muTrueKwrd]
    sigma_true = obs[sigTrueKwrd]
    mu_infr = obs[muInfrKwrd]
    sigma_infr = obs[sigInfrKwrd]
    chi = obs[chiKwrd]

    x = obs[muTrueKwrd].reshape(-1, 1)
    y = obs[sigTrueKwrd].reshape(-1, 1)
    x_interp = obs[muInfrKwrd].reshape(-1, 1)
    y_interp = obs[sigInfrKwrd].reshape(-1, 1)

    z = chi
    nStatepoints = z.size
    L = int(np.sqrt(nStatepoints))
    z = z.reshape(L, L, order='C')
    # z = gaussian_filter(z, sigma=SIGMA)
    z = z.reshape(nStatepoints, order='C')
    xy = np.hstack((x, y))
    xy_interp = np.hstack((x_interp, y_interp))
    interpolator = RBFInterpolator(
        xy, z,
        kernel='thin_plate_spline',
        # smoothing=0.01
        # epsilon=10
        )
    z_interp = interpolator(xy_interp)
    # z_err = abs(z_interp - z) / abs(z)
    lim = np.NaN
    z_interp[mu_infr < mu_true.min()] = lim
    z_interp[mu_infr > mu_true.max()] = lim
    z_interp[sigma_infr < sigma_true.min()] = lim
    z_interp[sigma_infr > sigma_true.max()] = lim
    z_interp[z_interp < 0] = lim
    return z_interp


# This doesn't quite work... it needs a way to map
# so far haven't used obs_kwrd!
def modify_observable(run_dirs, func, *args):
    pd = analysis.PhaseDiagram(run_dirs)
    for i in range(0, len(run_dirs)):
        params, os, _ = pd.load_run(runID=i)
        # os = analysis.set_dimensions(os, 0, None, 0, None)
        z = func(os, *args)
        if i == 0:
            new_obs = np.copy(z)
        else:
            new_obs = np.vstack((new_obs, z))

    if len(run_dirs) == 1:
        means = new_obs
        stds = np.zeros_like(new_obs)
    else:
        means = np.nanmean(new_obs, axis=0)
        stds = np.nanstd(new_obs, axis=0)
    return means, stds


# load kajimura data!
# --- temp helpers --- #
def temp_conveter(model):
    N, _ = model.shape
    Js = tools.triu_flat(model, k=1)
    temp = 1 / (np.std(Js) * (N ** 0.5))
    return temp


def kajimura_calc_mean_temps(dir):
    # print(dir)
    files = [
        os.path.join(dir, f) for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f))]
    T_plms = []
    T_cors = []
    corrects = []
    for file in files:
        with h5py.File(file, 'r') as fin:
            mod_PLM = fin['InferredModel'][()]
            correction = fin['correction'][()]
            mod_COR = mod_PLM / correction[0]
            T_plms.append(temp_conveter(mod_PLM))
            T_cors.append(temp_conveter(mod_COR))
            corrects.append(correction[0])
    T_plm = [np.mean(T_plms), np.std(T_plms)]
    T_cor = [np.mean(T_cors), np.std(T_cors)]
    cor = [np.mean(corrects), np.std(corrects)]
    return T_plm, T_cor, cor
# --- temp helpers --- #


def kajimura_get_temps(root):
    nSubSamples = np.arange(4, 41)
    dataDirectories = [
        os.path.join(root, 'ss{:d}'.format(nSS)) for nSS in nSubSamples]

    ssTs_plm = np.zeros((nSubSamples.size, 2))
    ssTs_cor = np.zeros((nSubSamples.size, 2))
    # ssCors = np.zeros((nSubSamples.size, 2))

    # ssC2s_plm = np.zeros((nSubSamples.size, 2))
    # ssC2s_cor = np.zeros((nSubSamples.size, 2))
    # ssC2s_target = np.zeros((nSubSamples.size, 2))

    for iD, dir in enumerate(dataDirectories):
        T_plm, T_cor, cor = kajimura_calc_mean_temps(dir)
        # C2_plm, C2_cor, C2_target = correction_getMeanCorrelations(dir)

        ssTs_plm[iD, :] = T_plm
        ssTs_cor[iD, :] = T_cor
        # ssCors[iD, :] = cor

        # ssC2s_plm[iD, :] = C2_plm
        # ssC2s_cor[iD, :] = C2_cor
        # ssC2s_target[iD, :] = C2_target
    nSubSamples = nSubSamples * 236
    return nSubSamples, ssTs_plm, ssTs_cor


def kajimura_get_models(root, mod_fnmaes):
    fnames = [os.path.join(root, fname) for fname in mod_fnmaes]

    mods_plm = []
    mods_cor = []
    nSamples = []

    for fname in fnames:
        with h5py.File(fname, 'r') as fin:
            B, N = fin['configurations'].shape
            mod_infr = fin['InferredModel'][()]
            correction = fin['correction'][()]
            mod_corr = mod_infr / correction[0]
            mods_plm.append(mod_infr)
            mods_cor.append(mod_corr)
            nSamples.append(B)
    return np.array(nSamples), np.array(mods_plm), np.array(mods_cor)


# for the varying B included fit
def get_curves_fixedT(params, full_obs, Bs, iT, Bmin, Bmax):
    T_theory = params['T'][0, :]
    T_curves = []
    T_curves_std = []

    for i, B in enumerate(Bs):
        obs = full_obs[i]
        Ts_inf = 1 / (obs['infrSig'] * 200 ** 0.5)
        T_curves.append(np.mean(Ts_inf, axis=0))
        T_curves_std.append(np.std(Ts_inf, axis=0))

    T_curves = np.array(T_curves)
    T_curves_std = np.array(T_curves_std)

    Bs = Bs[Bmin:Bmax]
    T_curves = T_curves[Bmin:Bmax, iT]
    T_curves_std = T_curves_std[Bmin:Bmax, iT]
    T_theory = T_theory[Bmin:Bmax]
    return Bs, T_curves, T_curves_std


def rescaleHelper_Bminvar_fit(Bs, Ts, ax, color):
    func = tools.arctan
    popt, _ = curve_fit(tools.arctan, Bs, Ts)
    # xfit = np.linspace(Bs.min(), Bs.max(), 100)
    Bfit = np.linspace(1e3, 5e4, 100)
    Tfit = func(Bfit, *popt)
    ax.plot(Bfit, Tfit, ls='--', marker=',', color=color)

    # ax.plot(
    #     xfit,
    #     yfit,
    #     marker=',',
    #     c=cTs[iT],
    #     # label=lbl
    #     )
    # popts.append(popt)
    # r2s.append(r2)
    # popts = np.array(popts)
    # pass
    Tlim = (popt[0] * np.pi) / 2
    Btilde = 1 / popt[1]
    r2 = r2_score(Ts, func(Bs, *popt))

    return np.array([Tlim, Btilde, r2])


# def load_N200_Jrepeats_Bscaling(root='/Users/mk14423/Desktop/PaperData'):
#     runs = [
#         'N200_J0.1_Bscaling/B1e3_*',
#         'N200_J0.1_Bscaling/B2e3_*',
#         'N200_J0.1_Bscaling/B3e3_*',
#         'N200_J0.1_Bscaling/B4e3_*',
#         'N200_J0.1_Bscaling/B5e3_*',
#         'N200_J0.1_Bscaling/B6e3_*',
#         'N200_J0.1_Bscaling/B7e3_*',
#         'N200_J0.1_Bscaling/B8e3_*',
#         'N200_J0.1_Bscaling/B9e3_*',
#         'N200_J0.1_Bscaling/B1e4_*',
#         'N200_J0.1_Bscaling/B2e4_*',
#         'N200_J0.1_Bscaling/B3e4_*',
#         'N200_J0.1_Bscaling/B4e4_*',
#         'N200_J0.1_Bscaling/B5e4_*',
#     ]


def rescaleHelper_get_models(iT, iB_max, iR=1):
    # root = '/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated'
    root = '/Users/mk14423/Desktop/PaperData/N200_J0.1_Bscaling'
    # runs = [
    #     'B1e3_1/T_1.250-h_0.000-J_0.500-Jstd_1.000.hdf5',
    #     'B2e3_1/T_1.250-h_0.000-J_0.500-Jstd_1.000.hdf5',
    #     'B1e4_1/T_1.250-h_0.000-J_0.500-Jstd_1.000.hdf5'
    # ]
    Ts = [
        0.5, 0.575, 0.65, 0.725, 0.8, 0.875, 0.95, 1.025, 1.1,
        1.175, 1.25, 1.325, 1.4, 1.475, 1.55, 1.625, 1.7, 1.775,
        1.85, 1.925, 2.0]
    Bs = [
        '1e3_1', '2e3_1', '3e3_1', '4e3_1', '5e3_1', '6e3_1', '7e3_1',
        '8e3_1', '9e3_1', '1e4_1', '2e4_1', '3e4_1', '4e4_1', '5e4_1',]
    Rs = [0.1 * i for i in range(0, 21)]

    fname = (
        f'B{Bs[iB_max]}/T_{Ts[iT]:.3f}' +
        f'-h_0.000-J_{Rs[iR]:.3f}-Jstd_1.000.hdf5')

    fname = os.path.join(root, fname)

    if root == '/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated':
        with h5py.File(fname, 'r') as fin:
            # print(list(fin.keys()))
            mod_true = fin['InputModel'][()]
            mod_infr = fin['InferredModel'][()]
            correction = fin['correction'][()]
            mod_corr = mod_infr / correction[0]
        return [mod_true, mod_infr, mod_corr]
    elif root == '/Users/mk14423/Desktop/PaperData/N200_J0.1_Bscaling':
        with h5py.File(fname, 'r') as fin:
            # print(list(fin.keys()))
            mod_true = fin['InputModel'][()]
            mod_infr = fin['InferredModel'][()]
        return [mod_true, mod_infr]
    else:
        print('You messed up!')


import inference.scripts.optimizeT_funcs as opT


def kajimura_sweep_calc_obs(root, fname):
    npz_fout = root + 'noMMchis.npz'
    data = np.load(npz_fout)['C2s']
    print(data)
    data = np.load(npz_fout)['qs']
    print(data)
    # exit()
    with h5py.File(root + fname, 'r') as fin:
        print(fin['ChiSweep'].keys())
        temps = fin['ChiSweep/alphas'][()]
    nStatepoints = temps.size
    qs = np.zeros((nStatepoints, 2))
    C2s = np.zeros((nStatepoints, 2))

    # I also want to calculate q!
    with h5py.File(root + fname, 'r') as fin:
        for i in range(0, nStatepoints):
            t0 = perf_counter()
            trajectories = fin['ChiSweep/trajectories'][i, :, :, :]
            print(trajectories.shape)
            q_reps = np.array([opT.q(traj) for traj in trajectories])
            qs[i, 0] = np.mean(q_reps)
            qs[i, 1] = np.std(q_reps)

            C2_reps = np.array([opT.correlation(traj) for traj in trajectories])
            C2s[i, 0] = np.mean(C2_reps)
            C2s[i, 1] = np.std(C2_reps)
            t1 = perf_counter()
            print(f'time take = {t1-t0:.3}')

    # print(qs[:, 0])
    # print(qs[:, 1])
    # print('-------')
    # print(C2s[:, 0])
    # print(C2s[:, 1])
    np.savez(npz_fout, true_temps=temps, C2s=C2s, qs=qs)
