import numpy as np
import h5py
import helpers
import matplotlib.pyplot as plt
from pyplm.utilities.hdf5io import write_models_to_hdf5, write_configurations_to_hdf5
from pyplm.utilities.metadataio import get_metadata_df
from tabulate import tabulate

from pyplm.pipelines import data_pipeline
from pyplm.utilities.tools import triu_flat
from pyplm.plotting import mkfigure
from scipy.ndimage import gaussian_filter1d
plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
CAT_COLS = [
        '#4053d3', '#ddb310', '#b51d14',
        '#00beff', '#fb49b0', '#00b25d', '#cacaca'
    ]


def analyse_test():
    file = './test_dataset.hdf5'
    grp = 'initial_test'
    with h5py.File(file, 'r') as fin:
        g = fin[grp]
        print(g.keys())
        temps = get_metadata_df(g, 'inputModels')
        # plm_md = get_metadata_df(g, 'inferredModels')
        # C2_md = get_metadata_df(g, 'correctedModels')
        # firth_md = get_metadata_df(g, 'firthcorModels')
        # print(plm_md)
        # print(C2_md)
        # print(firth_md)
        print(temps)
        mod_in = g['inputModels'][0, :, :]
        mod_plm = g['inferredModels'][0, :, :]
        mod_C2 = g['correctedModels'][0, :, :]
        mod_firth = g['firthcorModels'][0, :, :]
    models = np.array([mod_in, mod_plm, mod_C2, mod_firth])
    fig, ax = plt.subplots()
    cols = ['k', CAT_COLS[0], CAT_COLS[1], CAT_COLS[2]]
    lables = ['true', 'plm', r'$C^2$', 'firth']
    bins = np.linspace(models.min(), models.max(), 100)
    # should I measure the errors? 
    # for now dont worry about it; I guess I can measure
    # soemthing about the fisher infromation?
    # let's just do the inference for now!
    for i, mod in enumerate(models):
        temp = helpers._calc_temp(mod)
        print(temp)
        params = triu_flat(mod)
        n, x = np.histogram(params, bins=bins, density=True)
        x = x[:-1]
        n = gaussian_filter1d(n, sigma=1)
        line, = ax.plot(
            x, n, marker=',', c=cols[i], label=lables[i]
        )
    ax.legend()
    plt.show()

def analyse_lowest_10():
    file = './test_dataset.hdf5'
    grp = 'lowest_10'
    with h5py.File(file, 'r') as fin:
        g = fin[grp]
        print(g.keys())
        temps = get_metadata_df(g, 'inputModels')
        temps = temps.to_numpy()[:, 0]
        print(temps.shape) # this should be as array!
        mods_in = g['inputModels'][:, :, :]
        mods_plm = g['inferredModels'][:, :, :]
        # mod_C2 = g['correctedModels'][0, :, :]
        mods_firth = g['firthcorModels'][:, :, :]
    temps = temps[0:10]
    T0s = np.zeros(len(mods_in))
    Tplm = np.zeros(len(mods_in))
    Tfith = np.zeros(len(mods_in))
    print(T0s.shape, Tplm.shape, Tfith.shape)
    labels = ['true', 'plm', r'$C^2$', 'firth']
    for i in range(0, len(mods_in)):
        T0s[i] = helpers._calc_temp(mods_in[i])
        Tplm[i] = helpers._calc_temp(mods_plm[i])
        Tfith[i] = helpers._calc_temp(mods_firth[i])
    fig, ax = plt.subplots()
    ax.plot(temps, temps, marker=',', ls='--', c='k', label='theory')
    ax.plot(temps, T0s, label='input')
    ax.plot(temps, Tplm, label='plm')
    ax.plot(temps, Tfith, label='firth')
    ax.set(xlim=[temps.min(), temps.max()])
    ax.legend()
    plt.show()


def identify_separation(models):
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

def analyse_full():
    # let it iterate to 200 for firth for this! before only 100!
    file = './test_dataset.hdf5'
    grp = 'all_T_rep0'
    with h5py.File(file, 'r') as fin:
        g = fin[grp]
        print(g.keys())
        temps = get_metadata_df(g, 'inputModels')
        temps = temps.to_numpy()[:, 0]
        print(temps.shape) # this should be as array!
        mods_in = g['inputModels'][:, :, :]
        mods_plm = g['inferredModels'][:, :, :]
        mods_C2 = g['correctedModels'][:, :, :]
        mods_firth = g['firthcorModels'][:, :, :]
    # temps = temps[0:10]
    T0s = np.zeros(len(mods_in))
    Tplm = np.zeros(len(mods_in))
    T_C2s = np.zeros(len(mods_in))
    Tfith = np.zeros(len(mods_in))

    # mask = identify_separation(mods_firth)
    # print(mask)
    # exit()
    mask = identify_separation(mods_plm)

    # let's plot the max and the min as well to see!
    # that's something useufl maybe...?

    print(T0s.shape, Tplm.shape, Tfith.shape)
    labels = ['true', 'plm', r'$C^2$', 'firth']
    for i in range(0, len(mods_in)):
        T0s[i] = helpers._calc_temp(mods_in[i])
        Tplm[i] = helpers._calc_temp(mods_plm[i])
        T_C2s[i] = helpers._calc_temp(mods_C2[i])
        Tfith[i] = helpers._calc_temp(mods_firth[i])
    fig, ax = plt.subplots()
    x = temps
    ax.plot(x, temps, marker=',', ls='--', c='k', label='theory')
    ax.plot(x, T0s, label='input', marker='.')
    ax.plot(x, Tplm, label='plm', marker='.')
    ax.plot(x, T_C2s, label=r'$C^2$', marker='.')
    ax.plot(x, Tfith, label='firth', marker='.')
    # ax.set(xlim=[x.min(), x.max()], ylim=[x.min(), x.max()])
    ax.legend()
    plt.show()

    # let's plot the difference!
    # I think plotting it vs T0 makes more sense acutally :)!
    fig, ax = plt.subplots()
    x = T0s
    # ax.plot(x, (T0s - T0s), marker=',', ls='--', c='k')
    # ax.plot(x, , label='input', marker='.')
    ax.plot(x, ((Tplm - T0s)/T0s), label='plm', marker='o')
    ax.plot(x[mask], ((T_C2s - T0s)/T0s)[mask], label=r'$C^2$', marker='o')
    ax.plot(x, ((Tfith - T0s)/T0s), label='firth', marker='o')
    ax.set(xlim=[x.min(), x.max()], ylim=[None, 0.0])
    ax.set(xlabel=r'$T^{0}$', ylabel=r'$\left( T^{*} - T^{0} \right) / T^{0}$')

    axins = ax.inset_axes([0.5, 0.25, 0.47, 0.47])
    axins.plot(x, ((Tplm - T0s)/T0s), label='plm', marker='o')
    axins.plot(x[mask], ((T_C2s - T0s)/T0s)[mask], label=r'$C^2$', marker='o')
    axins.plot(x, ((Tfith - T0s)/T0s), label='firth', marker='o')
    axins.set_xlim(0.68, 2)
    axins.set_ylim(-0.18, 0)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.legend()
    plt.show()

    return
    # I could ahve this look better by making a difference?
    mods = [mods_in, mods_plm, mods_firth]
    # let's have a look at the 3 lowest temperature models
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    for row in range(0, 3):
        for col in range(0, 3):
            vmin = mods_in[row, :, :].min()
            vmax = mods_in[row, :, :].max()
            ax[row, col].matshow(mods[col][row][:, :], vmin=vmin, vmax=vmax)
    plt.show()

    # not sure I want to use this stuff...
    # error distirbutions...? i.e. one minus the other?
    plm_errors = mods_plm - mods_in
    firth_errors = mods_firth - mods_in
    Tchoice = 0

    plm_errors = plm_errors[Tchoice, :, :]
    firth_errors = firth_errors[0, :, :]
    mean = np.mean(firth_errors)
    std = np.std(firth_errors)
    central_bins = np.linspace(mean - (20*std), mean + (20*std), 200)
    bins = central_bins
    print(bins)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    params = triu_flat(plm_errors)
    n, x = np.histogram(params, bins=bins, density=True)
    x = x[:-1]
    n = gaussian_filter1d(n, sigma=1)
    line, = ax.plot(x, n, marker=',')
    params = triu_flat(firth_errors)
    n, x = np.histogram(params, bins=bins, density=True)
    x = x[:-1]
    n = gaussian_filter1d(n, sigma=1)
    line, = ax.plot(x, n, marker=',')
    plt.show()


def analyse_full_subsampled():
    # let it iterate to 200 for firth for this! before only 100!
    file = './test_dataset.hdf5'
    grp = 'all_T_rep0_subsampled1000'
    with h5py.File(file, 'r') as fin:
        g = fin[grp]
        print(g.keys())
        temps = get_metadata_df(g, 'inputModels')
        temps = temps.to_numpy()[:, 0]
        print(temps.shape) # this should be as array!
        mods_in = g['inputModels'][:, :, :]
        mods_plm = g['inferredModels'][:, :, :]
        mods_C2 = g['correctedModels'][:, :, :]
        mods_firth = g['firthcorModels'][:, :, :]
    # temps = temps[0:10]
    T0s = np.zeros(len(mods_in))
    Tplm = np.zeros(len(mods_in))
    T_C2s = np.zeros(len(mods_in))
    Tfith = np.zeros(len(mods_in))

    mask = identify_separation(mods_firth)
    print(mask)
    # exit()
    mask = identify_separation(mods_plm)
    print(mask)
    # let's plot the max and the min as well to see!
    # that's something useufl maybe...?

    print(T0s.shape, Tplm.shape, Tfith.shape)
    labels = ['true', 'plm', r'$C^2$', 'firth']
    for i in range(0, len(mods_in)):
        T0s[i] = helpers._calc_temp(mods_in[i])
        Tplm[i] = helpers._calc_temp(mods_plm[i])
        T_C2s[i] = helpers._calc_temp(mods_C2[i])
        Tfith[i] = helpers._calc_temp(mods_firth[i])
    # fig, ax = plt.subplots()
    # x = temps
    # ax.plot(x, temps, marker=',', ls='--', c='k', label='theory')
    # ax.plot(x, T0s, label='input', marker='.')
    # ax.plot(x, Tplm, label='plm', marker='.')
    # ax.plot(x, T_C2s, label=r'$C^2$', marker='.')
    # ax.plot(x, Tfith, label='firth', marker='')
    # # ax.set(xlim=[x.min(), x.max()], ylim=[x.min(), x.max()])
    # ax.legend()
    # plt.show()

    # let's plot the difference!
    # I think plotting it vs T0 makes more sense acutally :)!
    fig, ax = plt.subplots()
    x = T0s
    # ax.plot(x, (T0s - T0s), marker=',', ls='--', c='k')
    # ax.plot(x, , label='input', marker='.')
    ax.plot(x, ((Tplm - T0s)/T0s), label='plm', marker='s', ls='--')
    ax.plot(x[mask], ((T_C2s - T0s)/T0s)[mask], label=r'$C^2$', marker='s', ls='--')
    ax.plot(x, ((Tfith - T0s)/T0s), label='firth', marker='s', ls='--')
    ax.set(xlim=[x.min(), x.max()], ylim=[None, 0.0])
    ax.set(xlabel=r'$T^{0}$', ylabel=r'$\left( T^{*} - T^{0} \right) / T^{0}$')

    # axins = ax.inset_axes([0.5, 0.25, 0.47, 0.47])
    # axins.plot(x, ((Tplm - T0s)/T0s), label='plm', marker='.')
    # # axins.plot(x[mask], ((T_C2s - T0s)/T0s)[mask], label=r'$C^2$', marker='.')
    # axins.plot(x, ((Tfith - T0s)/T0s), label='firth', marker='.')
    # axins.set_xlim(0.68, 2)
    # axins.set_ylim(-0.18, 0)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    # ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.legend()
    plt.show()

def _get_temps_helper(group):
    file = './test_dataset.hdf5'
    # group = 'all_T_rep0'
    # group = 'all_T_rep0_subsampled1000'
    with h5py.File(file, 'r') as fin:
        g = fin[group]
        # print(g.keys())
        temps = get_metadata_df(g, 'inputModels')
        temps = temps.to_numpy()[:, 0]
        # print(temps.shape) # this should be as array!
        mods_in = g['inputModels'][:, :, :]
        mods_plm = g['inferredModels'][:, :, :]
        mods_C2 = g['correctedModels'][:, :, :]
        mods_firth = g['firthcorModels'][:, :, :]
    # temps = temps[0:10]
    T0s = np.zeros(len(mods_in))
    Tplm = np.zeros(len(mods_in))
    T_C2s = np.zeros(len(mods_in))
    Tfith = np.zeros(len(mods_in))

    mask = identify_separation(mods_plm)
    # print(mask)
    # print(T0s.shape, Tplm.shape, Tfith.shape)
    for i in range(0, len(mods_in)):
        T0s[i] = helpers._calc_temp(mods_in[i])
        Tplm[i] = helpers._calc_temp(mods_plm[i])
        T_C2s[i] = helpers._calc_temp(mods_C2[i])
        Tfith[i] = helpers._calc_temp(mods_firth[i])
    return T0s, Tplm, T_C2s, Tfith, mask

def combo_plot():
    fig, ax = plt.subplots()
    T0s, Tplm, TC2, Tfir, mask = _get_temps_helper('all_T_rep0')
    x = T0s
    # ax.plot(x, (T0s - T0s), marker=',', ls='--', c='k')
    # ax.plot(x, , label='input', marker='.')
    ax.plot(x, ((Tplm - T0s)/T0s), label='plm', marker='o', ls='-', c=CAT_COLS[0])
    ax.plot(x[mask], ((TC2 - T0s)/T0s)[mask], label=r'$C^2$', marker='o', ls='-', c=CAT_COLS[1])
    ax.plot(x, ((Tfir - T0s)/T0s), label='firth', marker='o', ls='-', c=CAT_COLS[2])

    T0s, Tplm, TC2, Tfir, mask = _get_temps_helper('all_T_rep0_subsampled1000')
    ax.plot(x, ((Tplm - T0s)/T0s), label='plm', marker='s', ls='--', c=CAT_COLS[0])
    ax.plot(x[mask], ((TC2 - T0s)/T0s)[mask], label=r'$C^2$', marker='s', ls='--', c=CAT_COLS[1])
    ax.plot(x, ((Tfir - T0s)/T0s), label='firth', marker='s', ls='--', c=CAT_COLS[2])

    ax.set(xlim=[x.min(), x.max()], ylim=[None, 0.0])
    ax.set(xlabel=r'$T^{0}$', ylabel=r'$\left( T^{*} - T^{0} \right) / T^{0}$')
    ax.legend(ncol=2)
    plt.show()
 
def firth_double_check():
    file = './test_dataset.hdf5'
    with h5py.File(file, 'r') as fin:
        # g = fin['all_T_rep0']
        g = fin['all_T_rep0_subsampled1000']
        # print(g.keys())
        temps = get_metadata_df(g, 'inputModels')
        temps = temps.to_numpy()[:, 0]
        # print(temps.shape) # this should be as array!
        mods_in = g['inputModels'][:, :, :]
        mods_plm = g['inferredModels'][:, :, :]
        mods_C2 = g['correctedModels'][:, :, :]
        mods_firth = g['firthcorModels'][:, :, :]
    
    # # 4 PANNEL
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax = ax.ravel()
    # i_choices = [0, 1, 2, 15]
    # for iPlot, i in enumerate(i_choices):
    #     vmin = mods_in[i].min()
    #     vmax = mods_in[i].max()
    #     diff_plm = mods_plm[i] - mods_in[i]
    #     diff_firth = mods_firth[i] - mods_in[i]
    #     # let's still keep the min max :)!
    #     bins = np.linspace(vmin, vmax, 100)

    #     params = triu_flat(mods_in[i])
    #     n, x = np.histogram(params, bins=bins, density=True)
    #     x = x[:-1]
    #     n = gaussian_filter1d(n, sigma=1)
    #     ax[iPlot].plot(x, n, marker=',', ls='-', c='k')

    #     params = triu_flat(mods_plm[i])
    #     n, x = np.histogram(params, bins=bins, density=True)
    #     x = x[:-1]
    #     n = gaussian_filter1d(n, sigma=1)
    #     line, = ax[iPlot].plot(x, n, marker=',', ls='-', label='plm')

    #     params = triu_flat(mods_C2[i])
    #     n, x = np.histogram(params, bins=bins, density=True)
    #     x = x[:-1]
    #     n = gaussian_filter1d(n, sigma=1)
    #     line, = ax[iPlot].plot(x, n, marker=',', ls='--', c=line.get_color(), label='C2')

    #     params = triu_flat(mods_firth[i])
    #     n, x = np.histogram(params, bins=bins, density=True)
    #     x = x[:-1]
    #     n = gaussian_filter1d(n, sigma=1)
    #     line, = ax[iPlot].plot(x, n, marker=',', ls=':', c=line.get_color(), label='firth')
    #     text = f'T={temps[i]:.2}'
    #     ax[iPlot].text(0.1, 0.8, s=text, ha='left',
    #             va='bottom', transform=ax[iPlot].transAxes)
    # h, l = ax[iPlot].get_legend_handles_labels()
    # # fig.legend(handles=h, labels=l, loc='center', ncol=3, framealpha=1)
    # fig.supxlabel(r'$J_{ij}$')
    # fig.supylabel(r'$PDF$')
    # plt.show()

    # 2 PANNEL!
    # fig, ax = plt.subplots(nrows=2, ncols=1)
    # ax = ax.ravel()
    # i_choices = [0, 15]
    # for iPlot, i in enumerate(i_choices):
    #     vmin = mods_in[i].min()
    #     vmax = mods_in[i].max()
    #     diff_plm = mods_plm[i] - mods_in[i]
    #     diff_firth = mods_firth[i] - mods_in[i]
    #     # let's still keep the min max :)!
    #     bins = np.linspace(vmin, vmax, 100)

    #     params = triu_flat(mods_in[i])
    #     n, x = np.histogram(params, bins=bins, density=True)
    #     x = x[:-1]
    #     n = gaussian_filter1d(n, sigma=1)
    #     ax[iPlot].plot(x, n, marker=',', ls='-', c='k')

    #     params = triu_flat(mods_plm[i])
    #     n, x = np.histogram(params, bins=bins, density=True)
    #     x = x[:-1]
    #     n = gaussian_filter1d(n, sigma=1)
    #     line, = ax[iPlot].plot(x, n, marker=',', ls='-', label='plm')

    #     params = triu_flat(mods_C2[i])
    #     n, x = np.histogram(params, bins=bins, density=True)
    #     x = x[:-1]
    #     n = gaussian_filter1d(n, sigma=1)
    #     line, = ax[iPlot].plot(x, n, marker=',', ls='--', c=line.get_color(), label='C2')

    #     params = triu_flat(mods_firth[i])
    #     n, x = np.histogram(params, bins=bins, density=True)
    #     x = x[:-1]
    #     n = gaussian_filter1d(n, sigma=1)
    #     line, = ax[iPlot].plot(x, n, marker=',', ls=':', c=line.get_color(), label='firth')
    #     text = f'T={temps[i]:.2}'
    #     ax[iPlot].text(0.1, 0.8, s=text, ha='left',
    #             va='bottom', transform=ax[iPlot].transAxes)
    # h, l = ax[iPlot].get_legend_handles_labels()
    # ax[iPlot].set_xlabel(r'$J_{ij}$')
    # ax[iPlot].legend(fontsize=9, framealpha=1)
    # # fig.legend(handles=h, labels=l, loc='center', ncol=3, framealpha=1)
    # # fig.supxlabel(r'$J_{ij}$')
    # fig.supylabel(r'$PDF$')
    # plt.show()

    # let's try something like this...
    i_choices = [1, 15]
    figw, figh = plt.rcParams['figure.figsize']
    figh = 1.8 * figh
    fig, ax = plt.subplots(nrows=3, ncols=2, squeeze=False, figsize=(figw,figh))
    print(ax.shape)
    lbls = ['(a)', '(b)']
    for i, iM in enumerate(i_choices):
        print(iM)
        vmin = mods_in[iM].min()
        vmax = mods_in[iM].max()
        
        #
        ax[0, i].matshow(
            mods_plm[iM][0:50, 0:50], cmap='viridis',
            # vmin=vmin, vmax=vmax,
            )
        ax[1, i].matshow(
            mods_firth[iM][0:50, 0:50], cmap='viridis',
            # vmin=vmin, vmax=vmax,
            )
        
        ax[0, i].set_xticklabels([])
        ax[0, i].set_yticklabels([])
        ax[1, i].set_xticklabels([])
        ax[1, i].set_yticklabels([])
        bins = np.linspace(vmin, vmax, 100)

        params = triu_flat(mods_in[iM])
        n, x = np.histogram(params, bins=bins, density=True)
        x = x[:-1]
        n = gaussian_filter1d(n, sigma=1)
        ax[2, i].plot(x, n, marker=',', ls='-', c='k')

        params = triu_flat(mods_plm[iM])
        n, x = np.histogram(params, bins=bins, density=True)
        x = x[:-1]
        n = gaussian_filter1d(n, sigma=1)
        line, = ax[2, i].plot(x, n, marker=',', ls='-', label='plm')

        # params = triu_flat(mods_C2[iM])
        # n, x = np.histogram(params, bins=bins, density=True)
        # x = x[:-1]
        # n = gaussian_filter1d(n, sigma=1)
        # line, = ax[2, i].plot(x, n, marker=',', ls='--', label='C2')

        params = triu_flat(mods_firth[iM])
        n, x = np.histogram(params, bins=bins, density=True)
        x = x[:-1]
        n = gaussian_filter1d(n, sigma=1)
        line, = ax[2, i].plot(x, n, marker=',', ls='--', c=CAT_COLS[2], label='firth')
        text = f'{lbls[i]} T = {temps[iM]:.2}'
        print(text, np.abs(mods_plm[iM]).max(), np.abs(mods_firth[iM]).max())
        ax[0, i].set(title=text)
        # ax[2, i].text(0.1, 0.8, s=text, ha='left',
        #         va='bottom', transform=ax[2, i].transAxes)
    ax[2, 0].legend(fontsize='8', framealpha=1)

    ax[0, 0].set(ylabel=r'PLM')
    ax[1, 0].set(ylabel=r'Firth')
    ax[2, 0].set(xlabel=r'$J_{ij}$', ylabel=r'PDF')
    ax[2, 1].set(xlabel=r'$J_{ij}$')
    plt.show()


def checkC2_sim():
    file = './test_dataset.hdf5'
    # with h5py.File(file, 'r') as fin:
    #     g = fin['all_T_rep0']
    #     # g = fin['all_T_rep0_subsampled1000']
    #     # print(g.keys())
    #     temps = get_metadata_df(g, 'inputModels')
    #     temps = temps.to_numpy()[:, 0]
    #     # print(temps.shape) # this should be as array!
    #     mods_in = g['inputModels'][:, :, :]
    #     mods_plm = g['inferredModels'][:, :, :]
    #     mods_C2 = g['correctedModels'][:, :, :]
    #     mods_firth = g['firthcorModels'][:, :, :]
    #     confurations = g['configurations'][:, :, :]

    i = 7
    group = 'checkC2'
    with h5py.File(file, 'r') as f:
        print(f[group]['configurations'].shape)
        print(f[group].keys())


    # configs = np.array([confurations[i]])
    # configs_md = np.array([temps[i]], dtype=str)
    # write_configurations_to_hdf5(file, group, configs, configs_md)
   
    # mods = np.array([mods_plm[i], mods_C2[i], mods_firth[i]])
    # mods_md = np.array(['plm', 'C2', 'firth'], dtype=str)
    # write_models_to_hdf5(file, group, "inferredModels", mods, mods_md)

    pipeline = data_pipeline(file, group)
    alphas = np.array([1])
    pipeline.ficticiousT_sweep(alphas, 10000, 24)


def checkC2_recorrect():
    file = './test_dataset.hdf5'
    group = 'checkC2'
    with h5py.File(file, 'r') as fin:
        g = fin['all_T_rep0']
        confurations = g['configurations'][:, :, :]
    # i = 7
    # configs = np.array([confurations[i], confurations[i], confurations[i]])
    # configs_md = np.array([1.025, 1.025, 1.025], dtype=str)
    # write_configurations_to_hdf5(file, group, configs, configs_md)

    pipeline = data_pipeline(file, group)
    pipeline.correct_C2()
    

def calc_C2(trajectory):
    cij = np.cov(trajectory.T)
    N, _ = cij.shape
    C = np.sum(cij ** 2)
    C = C / N
    return C

def calc_simple_error(model_true, model_infr):
    p_true = triu_flat(model_true, k=0)
    p_infr = triu_flat(model_infr, k=0)
    top = np.sum((p_infr - p_true) ** 2)
    bottom = np.sum(p_true ** 2)
    err = np.sqrt(top/bottom)
    # err = np.sum(np.abs(model_infr - model_true)) / np.sum(np.abs(model_true))
    # err = np.mean(np.abs(model_infr - model_true))
    return err


def checkC2_measure():
    file = './test_dataset.hdf5'
    group = 'checkC2'
    # 'inferredModels': plm, C2, firth!!!!
    # 'inputModels': true, true, true !!!!
    # 'correctedModels': C2, C2, firthC2 !
    with h5py.File(file, 'r') as f:
        print(f[group].keys())
        in_configs = f[group]['configurations'][()]
        print(in_configs.shape)
        out_mods = f[group]['inferredModels'][()]
        out_configs = f[group]['sweep-trajectories'][()]
        firth_recor_mod = f[group]['correctedModels'][2, :, :]
        in_mod = f[group]['inputModels'][0, :, :]
    # I want to look at the smallest errors as well!!
    firth_recor_mod = np.array([firth_recor_mod])
    print(out_mods.shape, firth_recor_mod.shape)
    out_mods = np.append(out_mods, firth_recor_mod, axis=0)
    print(out_mods.shape)

    # aha nevermind my previous comment; I was taking the wrong corrected model!
    print(in_configs.shape)
    print(out_configs.shape)
    temps = [helpers._calc_temp(mod) for mod in out_mods]
    print(temps, helpers._calc_temp(in_mod))
    # well... what does all of this mean for me ultimatley?
    # I wnat to repreat this a few more times to make sure...
    # then I want to correct the firth model and see what that looks like?
    T = 1.025
    errs = [calc_simple_error(in_mod, mod) for mod in out_mods]

    # [mods_plm[i], mods_C2[i], mods_firth[i]
    C2_in = np.mean([calc_C2(cs) for cs in in_configs])
    C2_plms = [calc_C2(cs) for cs in out_configs[0, 0, :, :, :]]
    C2_C2s = [calc_C2(cs) for cs in out_configs[1, 0, :, :, :]]
    C2_firths = [calc_C2(cs) for cs in out_configs[2, 0, :, :, :]]
    C2_firths_re = [
        4.533811407066329, 4.521754455761394, 4.501407870088133, 4.533090381486594]
    suscepts = [np.mean(C2_plms), np.mean(C2_C2s), np.mean(C2_firths), np.mean(C2_firths_re)]

    # print(C2_in, np.mean(C2_plms), np.mean(C2_C2s), np.mean( C2_firths), np.mean(C2_firths_re)) 
    x = ['PLM', r'PLM $\rightarrow$ C2 ', 'Firth', r'Firth $\rightarrow$ C2']
    
    # fig, ax = mkfigure(nrows=3, ncols=1, sharex=True)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=False)
    yerr = [np.std(C2_plms), np.std(C2_C2s), np.std(C2_firths), 0.038031]
    ax[1, 0].bar(x, suscepts, yerr=yerr, align='center', alpha=0.8, ecolor='black', capsize=10, ec='k')
    ax[1, 0].axhline(C2_in, ls='--', c='k', marker=',')
    ax[1, 0].set(ylim=[0.75 * np.max(suscepts), None])
    ax[1, 0].set(ylabel=r'$C^2$')

    ax[0, 0].bar(x, temps, align='center', alpha=0.8, ecolor='black', capsize=10, ec='k')
    ax[0, 0].axhline(T, ls='--', c='k', marker=',')
    ax[0, 0].set(ylim=[0.75 * T, None])
    ax[0, 0].set(ylabel=r'$T^{*}$')
    ax[1, 0].set(xlabel='Inference method')

    # ax[2, 0].bar(x, errs, align='center', alpha=0.8, ecolor='black', capsize=10, ec='k')
    # ax[2, 0].set(ylim=[0.75 * np.max(errs), None])
    # ax[2, 0].set(xlabel='Inference method', ylabel=r'$\varepsilon$')
    # ax.axhline(T, ls='--', c='k', marker=',')

    plt.show()
    print(x)
    print('C2', suscepts)
    print('T', temps)
    print('e', errs)

    # let's check matrix similarites
    # true - firth: firth closest
    a = in_mod
    d1 = np.linalg.norm(a - out_mods[0])
    d2 = np.linalg.norm(a - out_mods[1])
    d3 = np.linalg.norm(a - out_mods[2])
    d4 = np.linalg.norm(a - out_mods[3])
    print('dist IN-X', d1, d2, d3, d4)


def checkC2_percentage_bars():
    file = './test_dataset.hdf5'
    group = 'checkC2'
    # 'inferredModels': plm, C2, firth!!!!
    # 'inputModels': true, true, true !!!!
    # 'correctedModels': C2, C2, firthC2 !
    with h5py.File(file, 'r') as f:
        print(f[group].keys())
        in_configs = f[group]['configurations'][()]
        print(in_configs.shape)
        out_mods = f[group]['inferredModels'][()]
        out_configs = f[group]['sweep-trajectories'][()]
        firth_recor_mod = f[group]['correctedModels'][2, :, :]
        in_mod = f[group]['inputModels'][0, :, :]
    # I want to look at the smallest errors as well!!
    firth_recor_mod = np.array([firth_recor_mod])
    print(out_mods.shape, firth_recor_mod.shape)
    out_mods = np.append(out_mods, firth_recor_mod, axis=0)
    print(out_mods.shape)

    # aha nevermind my previous comment; I was taking the wrong corrected model!
    print(in_configs.shape)
    print(out_configs.shape)
    temps = np.array([helpers._calc_temp(mod) for mod in out_mods])
    print(temps, helpers._calc_temp(in_mod))
    # well... what does all of this mean for me ultimatley?
    # I wnat to repreat this a few more times to make sure...
    # then I want to correct the firth model and see what that looks like?
    T = 1.025
    errs = [calc_simple_error(in_mod, mod) for mod in out_mods]

    # [mods_plm[i], mods_C2[i], mods_firth[i]
    C2_in = np.mean([calc_C2(cs) for cs in in_configs])
    C2_plms = [calc_C2(cs) for cs in out_configs[0, 0, :, :, :]]
    C2_C2s = [calc_C2(cs) for cs in out_configs[1, 0, :, :, :]]
    C2_firths = [calc_C2(cs) for cs in out_configs[2, 0, :, :, :]]
    C2_firths_re = [
        4.533811407066329, 4.521754455761394, 4.501407870088133, 4.533090381486594]
    suscepts = [np.mean(C2_plms), np.mean(C2_C2s), np.mean(C2_firths), np.mean(C2_firths_re)]
    suscepts = np.array(suscepts)
    suscepts = (suscepts - C2_in) / C2_in
    # print(C2_in, np.mean(C2_plms), np.mean(C2_C2s), np.mean( C2_firths), np.mean(C2_firths_re)) 
    x = ['PLM', r'PLM $\rightarrow$ C2 ', 'Firth', r'Firth $\rightarrow$ C2']
    
    # fig, ax = mkfigure(nrows=3, ncols=1, sharex=True)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=False)
    yerr = np.array([np.std(C2_plms), np.std(C2_C2s), np.std(C2_firths), 0.038031])
    yerr = yerr / C2_in
    ax[1, 0].bar(x, suscepts, yerr=yerr, align='center', alpha=0.8, ecolor='black', capsize=10, ec='k')

    # ax[0, 0].axhline(C2_in, ls='--', c='k', marker=',')
    # ax[0, 0].set(ylim=[0.75 * np.max(suscepts), None])
    ax[1, 0].set(ylabel=r'$\%$ error $C^2$')

    temps = (temps - T) / T
    ax[0, 0].bar(x, temps, align='center', alpha=0.8, ecolor='black', capsize=10, ec='k')
    # ax[0, 0].axhline(T, ls='--', c='k', marker=',')
    # ax[0, 0].set(ylim=[0.75 * T, None])
    ax[0, 0].set(ylabel=r'$\%$ error $T^{*}$')
    ax[1, 0].set(xlabel='Inference method')

    # ax[2, 0].bar(x, errs, align='center', alpha=0.8, ecolor='black', capsize=10, ec='k')
    # ax[2, 0].set(ylim=[0.75 * np.max(errs), None])
    # ax[2, 0].set(xlabel='Inference method', ylabel=r'$\varepsilon$')
    # ax.axhline(T, ls='--', c='k', marker=',')
    plt.show()
    print('------------')
    print(suscepts * 100)
    print(yerr* 100)
    print(temps* 100)

def recorrect_lowT():
    file = './test_dataset.hdf5'
    i = 0
    group = 'lowT_recorr' # this has firth as the inferred model!!
    # print(i, group)
    # with h5py.File(file, 'r') as fin:
    #     g = fin['all_T_rep0']
    #     # g = fin['all_T_rep0_subsampled1000']
    #     # print(g.keys())
    #     temps = get_metadata_df(g, 'inputModels')
    #     temps = temps.to_numpy()[:, 0]
    #     mods_in = g['inputModels'][i, :, :]
    #     # mods_plm = g['inferredModels'][:, :, :]
    #     # mods_C2 = g['correctedModels'][:, :, :]
    #     # mods_firth = g['firthcorModels'][:, :, :]
    #     confurations = g['configurations'][i, :, :]
   

    # configs = np.array([confurations])
    # configs_md = np.array([temps[i]], dtype=str)
    # print(configs.shape, configs_md)
    # write_configurations_to_hdf5(file, group, configs, configs_md)
   
    # mods = np.array([mods_in])
    # mods_md = np.array(['true'], dtype=str)
    # write_models_to_hdf5(file, group, "inputModels", mods, mods_md)
    # print(mods.shape, mods_md)

    # with h5py.File(file, 'r') as f:
    #     print(f[group].keys())
    pipeline = data_pipeline(file, group)
    pipeline.correct_firth(mod_name='inferredModels')
    pipeline.correct_C2()

plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')

# analyse_test()
# analyse_lowest_10()
# analyse_full()
# analyse_full_subsampled()
# combo_plot()
# firth_double_check()

# checkC2_sim()
# checkC2_measure()
checkC2_percentage_bars()
# checkC2_recorrect()

# recorrect_lowT()

# pick some models see if it worked..?

# OK I want to check what the C2s are!!
# so how do I do that...?

# file = './test_dataset.hdf5'
# grp = 'lowest_10'
# grp = 'initial_test'
# grp = 'all_T_rep0'
# grp = 'all_T_rep0_subsampled1000'

# temps, mods, trajs = helpers.load_const_mu_models()
# temps = np.array(temps, dtype=str)
# print(mods.shape, temps.shape, trajs.shape)
# write_models_to_hdf5(file, grp, 'inputModels', mods, temps)
# write_configurations_to_hdf5(file, grp, trajs, temps)


# pipeline = data_pipeline(file, grp)
# pipeline.infer()
# pipeline.correct_firth()
# pipeline.correct_C2()



# loading and 
# with h5py.File(file, 'r') as fin:
#     g = fin[grp]
#     # df = get_metadata_df(g, 'inputModels')
#     # that's how you acess that!
