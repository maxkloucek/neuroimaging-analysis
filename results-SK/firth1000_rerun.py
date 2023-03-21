import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import shapiro, kstest, anderson
# shaprio, small samples, kstest starndrd normal, anderson any normal
# see https://stackoverflow.com/questions/7903977/implementing-a-kolmogorov-smirnov-test-in-python-scipy
from scipy.stats import norm
from scipy.optimize import curve_fit
from tqdm import tqdm

from pyplm.utilities.hdf5io import write_models_to_hdf5, write_configurations_to_hdf5
from pyplm.utilities.metadataio import get_metadata_df
from pyplm.pipelines import data_pipeline
from pyplm.utilities.tools import triu_flat
from pyplm.plotting import mkfigure

# plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')

def _calc_temp(model):
    N, _ = model.shape
    Js = triu_flat(model)
    T = 1 / (np.std(Js) * (N ** 0.5))
    return T

def _calc_C2(trajectory):
    cij = np.cov(trajectory.T)
    N, _ = cij.shape
    C = np.sum(cij ** 2)
    C = C / N
    return C


def std_test(params, std_factor=10):
    pmin = np.nanmin(params)
    pmax = np.nanmax(params)
    pmean = np.nanmean(params)
    pmedian = np.nanmedian(params)
    pstd = np.nanstd(params)
    std10_min = pmedian - (pstd * std_factor)
    std10_max = pmedian + (pstd * std_factor)
    # print(iM, pmin, std10_min)
    # print(iM, pmax, std10_max)
    if (pmin < std10_min) or (pmax > std10_max):
        return False
    else:
        return True

def  anderson_test(params, i_sigLevel=2):
    statistic, critVals, sigLevels = anderson(params, dist='norm')
    critVal = critVals[i_sigLevel]
    sigLevel = sigLevels[i_sigLevel]
    # check which way round acutally makese sense!
    if statistic > critVal:
        separation = True
    else:
        separation = False


def _identify_separation(models):
    separation_mask = np.zeros(len(models), dtype=bool)
    for iM, model in enumerate(models):
        params = triu_flat(model)
        # plt.hist(params, bins='auto')
        # plt.show()
        pmin = np.nanmin(params)
        pmax = np.nanmax(params)
        pmean = np.nanmean(params)
        pmedian = np.nanmedian(params)
        pstd = np.nanstd(params)
        table = [
            [pmin, pmax, pmean, pmedian, pstd]
            ]
        headers = ["min", "max", "mean", "median", "s.d"]
        # print(tabulate(table, headers, tablefmt='grid'))
        # limit = 
        std10_min = pmedian - (pstd * 10)
        std10_max = pmedian + (pstd * 10)
        # print(iM, pmin, std10_min)
        # print(iM, pmax, std10_max)
        if (pmin < std10_min) or (pmax > std10_max):
            separation_mask[iM] = False
        else:
            separation_mask[iM] = True
    # will delte separated points!

    return separation_mask


def _transform_to_pipe():
    root = '/Users/mk14423/Desktop/Data/N200_J0.1_optimizeT/B1e3_1/'
    out_file = './B1000_firth.hdf5'
    # group = 'infMod_is_firth'
    group = 'small_test'
    Js = np.linspace(0, 2, 21)
    Js = [f'{J:.3f}' for J in Js]
    Ts = np.linspace(0.5, 2, 21)
    Ts = [f'{T:.3f}' for T in Ts] 

    Ts = [Ts[0]]
    Js = Js[0:2]

    md = np.zeros((len(Ts) * len(Js), 2))
    trajectories = np.zeros((len(Ts) * len(Js), 1000, 200))
    # print(md.shape, trajectories.shape)

    i = 0
    for T in Ts:
        for iD, J in enumerate(Js):
            file = 'T_' + T + '-h_0.000-J_' + J + '-Jstd_1.000.hdf5'
            # print(file)
            file = root + file
            print(file)
            with h5py.File(file, 'r') as f:
                # print(f.keys())
                configs = f['configurations'][1000:, :]
                # in_mod = f['InputModel'][()]
                # configs = f['configurations'][1000:, :]
                # configs = f['configurations'][1000::10, :]
            trajectories[i, :, :] = configs
            md[i, 0] = T
            md[i, 1] = iD
            i+=1
    md = md.astype(str)

    print(trajectories.shape, md.shape)
    write_configurations_to_hdf5(out_file, group, trajectories, md)


def _make_analysis_df_old_data():
    root = '/Volumes/IT047719/InvIsInf-Data/N200_J0.1_optimizeT/0_updated/B1e3_1/'
    group = 'infMod_is_firth'
    Js = np.linspace(0, 2, 21)
    Js = [f'{J:.3f}' for J in Js]
    Ts = np.linspace(0.5, 2, 21)
    Ts = [f'{T:.3f}' for T in Ts]
    # so 
    # Ts = Ts[0:4]
    # Js = Js[0:2]
    # what do I want to include..?
    #T, iD,
    nR = 6
    nDatapoints = len(Ts) * len(Js) * nR
    print(nDatapoints)
    data_array = np.zeros((nDatapoints, 9))
    print(data_array.shape)
    i = 0
    for T in tqdm(Ts):
        for iD, J in enumerate(Js):
            file = 'T_' + T + '-h_0.000-J_' + J + '-Jstd_1.000.hdf5'
            file = root + file
            with h5py.File(file, 'r') as f:
                # print(f.keys())
                # print(f['ChiRecalc']['cor'].shape)
                # print(f['ChiRecalc']['plm'].shape)
                # print(f['ChiRecalc']['tru'].shape)

                trajs_tru = f['ChiRecalc']['tru'][()]
                trajs_plm = f['ChiRecalc']['plm'][()]
                trajs_cor = f['ChiRecalc']['cor'][()]
                
                # print(f['correction'][()])
                # this is for the 3 temperatures!
                mod_in = f['InputModel'][()]
                mod_plm = f['InferredModel'][()]
                C2_factor = f['correction'][0]
                mod_C2 = mod_plm / C2_factor

            T0 = _calc_temp(mod_in)
            T_plm = _calc_temp(mod_plm)
            T_C2 = _calc_temp(mod_C2)
            # print(i, T0, T_plm, T_C2)
            for k in range(0, nR):
                data_array[i, 0] = T
                data_array[i, 1] = iD
                data_array[i, 2] = k
                data_array[i, 3] = T0
                data_array[i, 4] = T_plm
                data_array[i, 5] = T_C2

                C2_0 = _calc_C2(trajs_tru[k, :, :])
                C2_PLM = _calc_C2(trajs_plm[k, :, :])
                C2_C2 = _calc_C2(trajs_cor[k, :, :])
                data_array[i, 6] = C2_0
                data_array[i, 7] = C2_PLM
                data_array[i, 8] = C2_C2

                i+=1

    headers = ['T', 'iD', 'iR', 'T_0', 'T_PLM', 'T_C2', 'C2_0', 'C2_PLM', 'C2_C2']
    df = pd.DataFrame(data=data_array, columns=headers)
    print(df)
    df_name = group + '_old_aDF'
    df.to_hdf('./B1000_firth.hdf5', df_name)


def _make_analysis_df():
    file = './B1000_firth.hdf5'
    group = 'infMod_is_firth'
    with h5py.File(file, 'r') as f:
        g = f[group]
        print(g.keys())
        md = g['configurations_metadata'].asstr()[()].astype(float)
        firth_mods = g['inferredModels'][:, :, :]
        trajectories = g['sweep-trajectories'][:, 0, :, :]
        # print(trajectories.shape)

    # print(md.shape, firth_mods.shape)
    nMods, _, _ = firth_mods.shape
    nMods, nReps, _, _ = trajectories.shape
    nDatapoints = nMods * nReps
    obs = np.zeros((nDatapoints, 5))
    print(nMods, nReps, nDatapoints)
    print(obs.shape)
    print(trajectories.shape)
    # ooohh cause now we've got the reps as well,
    # that's why it will get confusing...

    # ok nice they are matched!
    k = 0
    for i in tqdm(range(0, nMods)):
        mod = firth_mods[i, :, :]
        T_FIR = _calc_temp(mod)
        T = md[i, 0]
        iD = md[i, 1]
        for j in range(0, nReps):
            obs[k, 0] = T
            obs[k, 1] = iD
            obs[k, 2] = j
            obs[k, 3] = T_FIR
            C2 = _calc_C2(trajectories[i, j, :, :])
            obs[k, 4] = C2
            k+=1

    headers = ['T', 'iD', 'iR', 'T_FIR', 'C2_FIR']
    df = pd.DataFrame(data=obs, columns=headers)
    print(df)

    df_name = group + '_aDF'
    df.to_hdf(file, df_name)
    # print(md)
    # there should surely be a better way to do this!
    # we get everything together here!!

def run_firth():
    file = './B1000_firth.hdf5'
    group = 'infMod_is_firth'
    # group = 'small_test'
    # this is the frith run, then I need to make some C2s!!!
    with h5py.File(file, 'r') as f:
        print(f[group].keys())
    pipeline = data_pipeline(file, group)
    # pipeline.correct_firth(mod_name='inferredModels')
    # pipeline.correct_C2()
    pipeline.ficticiousT_sweep(np.array([1]), 10000, 6)

def _sep_check_df():
    file = './B1000_firth.hdf5'
    group = 'infMod_is_firth'
    with h5py.File(file, 'r') as f:
        g = f[group]
        print(g.keys())
        md = g['configurations_metadata'].asstr()[()].astype(float)
        firth_mods = g['inferredModels'][0:, :, :]
        # trajectories = g['sweep-trajectories'][:, 0, :, :]
    print(firth_mods.shape)

    firth_seps = []
    for i in range(0, 21):
        # for mod in firth_mods:
        params = []
        for j in range(0, 21):
            # params
            iMod = (i*21)+j
            # print(iMod)
            mod = firth_mods[iMod, :, :]
            ps = triu_flat(mod)
            params.append(ps)
        params = np.array(params).ravel()

        separation = std_test(params)
        firth_seps.append(separation)

    root = '/Volumes/IT047719/InvIsInf-Data/N200_J0.1_optimizeT/0_updated/B1e3_1/'
    Js = np.linspace(0, 2, 21)
    Js = [f'{J:.3f}' for J in Js]
    Ts = np.linspace(0.5, 2, 21)
    Ts = [f'{T:.3f}' for T in Ts]
    plm_mods = np.zeros((441, 200, 200))
    i = 0
    for T in Ts:
        for iD, J in enumerate(Js):
            file = 'T_' + T + '-h_0.000-J_' + J + '-Jstd_1.000.hdf5'
            file = root + file
            with h5py.File(file, 'r') as f:
                mod_plm = f['InferredModel'][()]
            plm_mods[i, :, :] = mod_plm
            i+=1
            print(i)

    plm_seps = []
    for i in range(0, 21):
        # for mod in firth_mods:
        params = []
        for j in range(0, 21):
            # params
            iMod = (i*21)+j
            # print(iMod)
            mod = plm_mods[iMod, :, :]
            ps = triu_flat(mod)
            params.append(ps)
        params = np.array(params).ravel()

        separation = std_test(params)
        plm_seps.append(separation)
        print(separation)

    plt.plot(plm_seps, label='plm')
    plt.plot(firth_seps, label='firth')
    plt.axhline(separation, ls='--', c='k', marker=',')
    plt.legend()
    plt.show()
    df = pd.DataFrame({
        'sep_PLM': plm_seps,
        'sep_FIR': firth_seps
    })
    print(df)
    df_name = group + '_sepDF'
    file = './B1000_firth.hdf5'
    df.to_hdf(file, df_name)
    # ok awesome!

# sick this works :)!
def plot_filtered_separation(ax, means, errs, obskey, sepkey, **pltargs):

    # this is the over the top, only filtered data!
    ebar = ax.errorbar(
        x=means['T'][means[sepkey] == True],
        y=means[obskey][means[sepkey] == True],
        yerr=errs[obskey][errs[sepkey] == True], zorder=10,
        **pltargs
    )
    # this is the underlying, all data!
    ax.errorbar(
        x=means['T'],
        y=means[obskey],
        yerr=errs[obskey],
        zorder=0,
        alpha=0.5,
        ls='--',
        c=ebar[0].get_color(),
    )

def analyse_firth():
    file = './B1000_firth.hdf5'
    group = 'infMod_is_firth'
    df_firth = pd.read_hdf(file, group + '_aDF')
    df_old = pd.read_hdf(file, group + '_old_aDF')
    df_sep = pd.read_hdf(file, group + '_sepDF')

    df = df_old.merge(df_firth) 
   
    # df = df.groupby(['T'], as_index=True).mean().reset_index()
    df = df.groupby(['T', 'iD'], as_index=True).mean().reset_index()
    print(df)
    means = df.groupby(['T'], as_index=True).mean().reset_index()
    # errs = df.groupby(['T'], as_index=True).std().reset_index()
    errs = df.groupby(['T'], as_index=True).sem().reset_index()
    means = pd.concat([means, df_sep], axis=1)
    errs = pd.concat([errs, df_sep], axis=1)
    print(means)


    figw, figh = plt.rcParams['figure.figsize']
    figh = 1.7 * figh
    fig, ax = mkfigure(nrows=2, ncols=1, sharex=True, figsize=(figw,figh))
    plot_filtered_separation(ax[0, 0], means, errs, 'T_PLM', 'sep_PLM', label='plm')
    plot_filtered_separation(ax[0, 0], means, errs, 'T_C2', 'sep_PLM', label='C2')
    plot_filtered_separation(ax[0, 0], means, errs, 'T_FIR', 'sep_FIR', label='firth')
    ax[0, 0].legend(loc='upper center', fontsize=9, ncol=3, framealpha=1)
    ax[0, 0].plot(means['T'], means['T'], c='k', marker=',')
    ax[0, 0].set(xlim=[means['T'].min(), means['T'].max()], ylim=[0, 2])


    plot_filtered_separation(ax[1, 0], means, errs, 'C2_PLM', 'sep_PLM')
    plot_filtered_separation(ax[1, 0], means, errs, 'C2_C2', 'sep_PLM')
    plot_filtered_separation(ax[1, 0], means, errs, 'C2_FIR', 'sep_FIR')
    ax[1, 0].errorbar(means['T'], means['C2_0'], errs['C2_0'], c='k', marker=',')
    ax[1, 0].set(xlim=[means['T'].min(), means['T'].max()], ylim=[0, None])


    # inset axes....
    axins = ax[1, 0].inset_axes([0.4, 0.4, 0.57, 0.57])
    plot_filtered_separation(axins, means, errs, 'C2_PLM', 'sep_PLM')
    plot_filtered_separation(axins, means, errs, 'C2_C2', 'sep_PLM')
    plot_filtered_separation(axins, means, errs, 'C2_FIR', 'sep_FIR')
    axins.errorbar(means['T'], means['C2_0'], errs['C2_0'], c='k', marker=',')

    x1, x2, y1, y2 = 1.2, 2, 1, 5.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax[1, 0].indicate_inset_zoom(axins, edgecolor="black")


    ax[0, 0].set(ylabel=r'$T^{*}$')
    ax[1, 0].set(ylabel=r'$C^2$', xlabel=r'$T$')
    plt.show()

    exit()
    fig, ax = plt.subplots()
    x = df['T']
    y = (df['T_PLM'] - df['T_0']) / df['T_0']
    ax.plot(x, y)
    y = (df['T_C2'] - df['T_0']) / df['T_0']
    ax.plot(x, y)
    y = (df['T_FIR'] - df['T_0']) / df['T_0']
    ax.plot(x, y)
    plt.show()

    fig, ax = plt.subplots()
    x = df['T']
    y = (df['C2_PLM'] - df['C2_0']) / df['C2_0']
    ax.plot(x, y)
    y = (df['C2_C2'] - df['C2_0']) / df['C2_0']
    ax.plot(x, y)
    y = (df['C2_FIR'] - df['C2_0']) / df['C2_0']
    ax.plot(x, y)
    plt.show()

def test_high_T_scaling():
    # thus theres a competion, and a natrual limit at the critical regime.
    # sampling correalted configs measn that they will be good predictors.
    file = './B1000_firth.hdf5'
    group = 'infMod_is_firth'
    df_firth = pd.read_hdf(file, group + '_aDF')
    df_old = pd.read_hdf(file, group + '_old_aDF')
    df_sep = pd.read_hdf(file, group + '_sepDF')

    df = df_old.merge(df_firth) 
   
    # df = df.groupby(['T'], as_index=True).mean().reset_index()
    df = df.groupby(['T', 'iD'], as_index=True).mean().reset_index()
    print(df)
    means = df.groupby(['T'], as_index=True).mean().reset_index()
    # errs = df.groupby(['T'], as_index=True).std().reset_index()
    errs = df.groupby(['T'], as_index=True).sem().reset_index()
    means = pd.concat([means, df_sep], axis=1)
    errs = pd.concat([errs, df_sep], axis=1)
    print(means)

    # #  / means['T_0']
    # means['T_PLM'] = (means['T_PLM'] - means['T_0'])
    # means['T_C2'] = (means['T_C2'] - means['T_0'])
    # means['T_FIR'] = (means['T_FIR'] - means['T_0'])
    fig, ax = plt.subplots()
    plot_filtered_separation(ax, means, errs, 'T_PLM', 'sep_PLM')
    plot_filtered_separation(ax, means, errs, 'T_C2', 'sep_PLM')
    plot_filtered_separation(ax, means, errs, 'T_FIR', 'sep_FIR')

    cols = ['#4053d3', '#ddb310', '#b51d14', '#00beff', '#fb49b0', '#00b25d', '#cacaca']

    x = means['T'][means['sep_PLM'] == True]
    y = means['T_PLM'][means['sep_PLM'] == True]
    popt, cov = curve_fit(invx, x, y)
    print('PLM', popt)
    xs = means['T']
    # xs = np.linspace(0.1, 100)
    ax.plot(xs, invx(xs, *popt), ls='-', marker=',', c='k', zorder=200)

    y = means['T_C2'][means['sep_PLM'] == True]
    popt, cov = curve_fit(invx, x, y)
    print('C2', popt)
    ax.plot(xs, invx(xs, *popt), ls='-', marker=',', c='k', zorder=200)

    y = means['T_FIR'][means['sep_PLM'] == True]
    popt, cov = curve_fit(invx, x, y)
    print('FIR', popt)
    ax.plot(xs, invx(xs, *popt), ls='-', marker=',', c='k', zorder=200)

    ax.set(ylim=[0, None], xlabel=r'$T$', ylabel=r'$T^{*}$')
    ax.plot(xs, xs, ls='-', marker=',', c='k')
    plt.show()
    

    fig, ax = plt.subplots()
    x = means['T_PLM'][means['sep_PLM'] == True]
    y = means['T_FIR'][means['sep_PLM'] == True]
    # how do I encapsulate this behaviour..?
    popt, cov = curve_fit(linx, x, y)
    ax.plot(x, y)
    ax.plot(x, linx(x, *popt), c='k', marker=',', ls='-')
    plt.show()
    # PLM [-2.27860055  2.64923123  0.41102163]
    # C2 [ 0.35940626  0.19764177 -1.58128131]
    # FIR [-12.74796951  13.39600603   0.06579226]

   
# def invx(x, a, b, gamma):
#     y = (a / (x ** gamma)) + b
#     return y

def invx(x, a, b, gamma):
    y = (a / (x ** gamma)) + b
    return y

def linx(x, a, b):
    y = a + (b * x)
    return y

# _transform_to_pipe()
# run_firth()
# _make_analysis_df_old_data()
# _make_analysis_df()
# _sep_check_df()
# analyse_firth()
test_high_T_scaling()


# ok great, next up to make the recalc run!



def santiy_check_is_input_data_ok():
    # THE REAL DATA IS ON THE HARDDRIVE! KEEP THAT IN MIND, I THINK
    # I WANT TO COPY EVERYTHING FROM THE HARDDRIVE!
    f_local = '/Users/mk14423/Desktop/Data/N200_J0.1_optimizeT/B1e3_1/T_0.500-h_0.000-J_0.000-Jstd_1.000.hdf5'
    f_harddrive = '/Volumes/IT047719/InvIsInf-Data/N200_J0.1_optimizeT/0_updated/B1e3_1/T_0.500-h_0.000-J_0.000-Jstd_1.000.hdf5'
    with h5py.File(f_local, 'r') as f:
        confgis_local = f['configurations'][()]
    with h5py.File(f_harddrive, 'r') as f:
        confgis_hd = f['configurations'][()]
    print(np.all(confgis_hd == confgis_local)) # TRUE!
# santiy_check_is_input_data_ok()
   
