import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import kajimura_helpers as helpers

from pyplm.plotting import mkfigure
from pyplm.utilities import tools
from inference.scripts.paper2022May import load

COLORS_CAT = plt.rcParams['axes.prop_cycle'].by_key()['color']


def coupling_distributions():
    plm_model, cor_model = helpers.load_model('noMM.hdf5')
    N, _ = plm_model.shape
    noMM_params_plm = tools.triu_flat(plm_model)
    params_cor = tools.triu_flat(cor_model)

    plm_model, cor_model = helpers.load_model('MM.hdf5')
    MM_params_plm = tools.triu_flat(plm_model)

    params= np.array([noMM_params_plm, MM_params_plm])
    bins = np.linspace(params.min(), params.max(), 200)

    noMM_T = 1/(np.std(noMM_params_plm) * (N**0.5))
    MM_T = 1/(np.std(MM_params_plm) * (N**0.5))
    print(noMM_T, MM_T)
    'noMM'
    fig, ax = plt.subplots()
    lbl = r'no-MM, $T^{*}$' + f'={noMM_T:.3f}'
    helpers.distribution(ax, noMM_params_plm, bins=bins, marker=',', lw=2, label=lbl)
    lbl = r'MM, $T^{*}$' + f'={MM_T:.3f}'
    helpers.distribution(ax, MM_params_plm, bins=bins, marker=',', lw=2, label=lbl)
    ax.set(
        xlabel=r'$J_{ij}$',
        ylabel=r'$PDF$',
        ylim=[5e-4, None], yscale='log')
    ax.legend()
    plt.show()

def show_noMM_model():
    plm_model, cor_model = helpers.load_model('noMM.hdf5')
    fig, ax = plt.subplots()
    mat = ax.matshow(cor_model, cmap='cividis')

    
    # ax.axvline(0, c='k', marker=',')
    # ax.axhline(0, c='k', marker=',')
    ax.axhline(359.5, c='k', marker=',')
    ax.axvline(359.5, c='k', marker=',')
    # shouldn't it be 360/2 = 180..?
    ax.axhline(179.5, c='k', marker=',', ls='--')
    ax.axvline(179.5, c='k', marker=',', ls='--')
    ax.set(xlabel=r'$i$', ylabel=r'$j$')
    cbar = plt.colorbar(mat)
    cbar.set_label(r'$\theta_{ij}^{*}$', rotation=360)

    N, _ = cor_model.shape
    nticks = 4
    ticks = np.linspace(0, N, nticks)
    ticks = [0, 180, 360, N]
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
    ax.yaxis.set_major_locator(plt.FixedLocator(ticks))
    ax.set(
        xlim=[ax.get_xticks()[0], ax.get_xticks()[-1]],
        ylim=[ax.get_yticks()[-1], ax.get_yticks()[0]],
        xlabel='j',
        ylabel='i'
        )
    plt.show()


def noMM_subsampling():
    dataroot = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/'
    l1_condition = 'noMM_subsample_noL1'
    # l1_condition = 'noMM_subsample_yesL1'
    subsamples, temps_plm, temps_corr = load.kajimura_get_temps(
        dataroot + l1_condition
    )
    fig, ax = plt.subplots()

    print(subsamples.shape)
    cut = 20
    xc = subsamples[cut]
    print(xc)
    invB_cut = 1/(xc)
    # plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisaside.mplstyle')
    # fig2, ax2 = plt.subplots()
    # helpers.liner_saturation_checker(ax2, subsamples, temps_plm[:, 0], invB_cut, label='PLM')
    # helpers.liner_saturation_checker(ax2, subsamples, temps_corr[:, 0], invB_cut, label='SC')
    # ax2.legend(fontsize=9)
    # plt.savefig('/Users/mk14423/Documents/tempfigs/kajimura_invB_aside.png')
    # plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')

    l1 = ax.errorbar(
            x= subsamples,
            y=temps_plm[:, 0], yerr=temps_plm[:, 1],
            c=COLORS_CAT[0],
            label='PLM'
            )
    l2 = ax.errorbar(
        x=subsamples,
        y=temps_corr[:, 0], yerr=temps_corr[:, 1],
        c=COLORS_CAT[1],
        label='SC'
        )

    xs = subsamples[cut:]
    ys = temps_plm[cut:, 0]
    popt, _ = curve_fit(tools.arctan, xs, ys)
    r2 = r2_score(ys, tools.arctan(xs, *popt))
    T_lim = (popt[0] * np.pi) / 2
    B_tilde = (1 / popt[1])
    print('arctan: ', T_lim, B_tilde, r2)
    # xfit = np.linspace(0, 1e5, 100)
    xfit = np.linspace(subsamples.min(), subsamples.max(), 100)
    # xfit = xs
    yfit = tools.arctan(xfit, *popt)
    ax.plot(
        xfit, yfit, c='k', ls='--', marker=',',
        lw=2, zorder=500)

    ax.set(xlabel=r'$B$', ylabel=r'$T^{*}$', ylim=[0, 1.55])
    # lbls = ['noMM', 'MM', 'ss-plm', 'ss-correction']

    # ax.plot(
    #         9440, load.temp_conveter(noMM_cor),
    #         marker='X', ms=10, zorder=500,
    #         ls='none',
    #         # label=lbls[i]
    #         )
    # ax.plot(
    #         4248, load.temp_conveter(MM_cor),
    #         marker='X', ms=10, zorder=500,
    #         ls='none',
    #         # label=lbls[i]
    #         )
    noMM_plm, noMM_cor = helpers.load_model('noMM.hdf5')
    MM_plm, MM_cor = helpers.load_model('MM.hdf5')
    l3, = ax.plot(
            9440, load.temp_conveter(noMM_plm),
            marker='X', ms=10, zorder=500,
            ls='none', label='no-MM', c=COLORS_CAT[2]
            # label=lbls[i]
            )
    l4, = ax.plot(
            4248, load.temp_conveter(MM_plm),
            marker='X', ms=10, zorder=500,
            ls='none', label='MM', c=COLORS_CAT[3]
            # label=lbls[i]
            )
    ax.legend(handles=[l1,l2,l3,l4], ncol=2)
    
    # # '---- adding L1 regularisation ----'
    # subsamples, temps_plm, temps_corr = load.kajimura_get_temps(
    #         dataroot + 'noMM_subsample_yesL1'
    #         )
    # ax.errorbar(
    #     x=subsamples,
    #     y=temps_plm[:, 0], yerr=temps_plm[:, 1],
    #     marker='o', zorder=0,
    #     label = 'ss-pm-L1'
    #     )
    # # not sure I want to show this anyway...?
    plt.show()


def model_similarity(save=False):
    # let's also include the whole no-mindfullness, the matched subsample noMM
    # and the thingy temps!
    dataroot = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/'
    model_files = [
        'noMM.hdf5',
        'noMM_matched_subsamples/noMM_subsample0.hdf5',
        # 'noMM_matched_subsamples/noMM_subsample1.hdf5',
        'MM.hdf5'
    ]

    choice_samples, choice_plm, choice_cor = load.kajimura_get_models(
        dataroot, model_files)

    # plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    # FIGW = plt.rcParams.get('figure.figsize')[0]
    # fig, axs = splots.mkfigure(
    #     nrows=2, ncols=1,
    #     figsize=(FIGW, FIGW * (5/6) * 2)
    #     )
    # axs = axs.ravel()
    # for i in range(0, len(choice_samples)):
    #     axs[i].matshow(choice_plm[i])
    # plt.show()
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    FIGW = plt.rcParams.get('figure.figsize')[0]
    fig, axs = mkfigure(
        nrows=2, ncols=1,
        figsize=(FIGW, FIGW * (5/6) * 2),
        sharex=True,
        )
    axs = axs.ravel()
    noMM_mod = choice_plm[0]
    noMMss_mod = choice_plm[1]
    MM_mod = choice_plm[2]
    # setting everything within a core to 0?
    print(np.std(noMM_mod))

    # noMM_mod[np.abs(noMM_mod) <= np.std(noMM_mod)] = 0
    # noMMss_mod[np.abs(noMMss_mod) <= np.std(noMMss_mod)] = 0
    # MM_mod[np.abs(MM_mod) <= np.std(MM_mod)] = 0

    # noMM_mod = choice_cor[0]
    # noMMss_mod = choice_cor[1]
    # MM_mod = choice_cor[2]

    noMM_params = tools.triu_flat(noMM_mod, k=0)
    noMMss_params = tools.triu_flat(noMMss_mod, k=0)
    MM_params = tools.triu_flat(MM_mod, k=0)

    print(pearsonr(MM_params, noMM_params))
    print(pearsonr(MM_params, noMMss_params))

    axs[0].plot(MM_params, noMM_params, ls='none', alpha=0.2, c=COLORS_CAT[0])
    axs[0].plot(MM_params, MM_params, marker=',', c='k')
    axs[1].plot(MM_params, noMMss_params, ls='none', alpha=0.2, c=COLORS_CAT[0])
    axs[1].plot(MM_params, MM_params, marker=',', c='k')
    plt.show()


def noMM_sweep_Tf(save=False):
    root = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/'
    npz_fout = root + 'noMMchis.npz'
    Ts = np.load(npz_fout)['true_temps']
    C2s = np.load(npz_fout)['C2s']
    qs = np.load(npz_fout)['qs']
    # plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    fig, ax = mkfigure()
    ax = ax[0, 0]
    ax2 = ax.twinx()
    # the shift of the thingy shown as well?
    ax.errorbar(
        x=Ts, y=qs[:, 0], yerr=qs[:, 1] / np.sqrt(60), c=COLORS_CAT[0])
    ax2.errorbar(
        x=Ts, y=C2s[:, 0], yerr=C2s[:, 1] / np.sqrt(60), c=COLORS_CAT[1])

    ax2.plot([1, 1.1277], [10.695, 5.359], zorder=250, marker=',', c='k', lw='2')
    l1, = ax2.plot([1], [10.695], marker='X', ls='none', ms=10, zorder=251, c=COLORS_CAT[2], label='no-MM PLM')
    l2, = ax2.plot([1.1277], [5.359], marker='X', ls='none', ms=10, zorder=251, c=COLORS_CAT[5], label='no-MM SC')
    # Tf_MM = 0.985/1.331
    # l3, = ax2.plot([Tf_MM], [5.359], marker='X', ls='none', ms=10, zorder=251, c=COLORS_CAT[3], label='MM PLM')
    # not sure if to include this...?
    # would probably have to do another sweep to find this out...?

    # ax2.annotate(
    #     "",
    #     xy=(1.1277, 5.359),
    #     xytext=(1, 10.695),
    #     zorder=250,
    #     arrowprops={
    #         'arrowstyle': '->',
    #         'linewidth': 2.5,
    #         'shrinkA': 0,
    #         'shrinkB': 0,
    #         # 'headwidth': 6,
    #         # 'length_includes_head': True,
    #         },
    #     )
    ax.yaxis.label.set_color(COLORS_CAT[0])
    ax2.yaxis.label.set_color(COLORS_CAT[1])
    ax.set(ylabel=r'$q$', xlabel=r'$T_{f}$')
    ax2.set(ylabel=r'$C^{2}$')
    ax2.legend(handles=[l1, l2])
    plt.show()


def correlations():
    # I WONDER IF THE CIJS LOOK THE SAME... HAVE A QUICK LITTLE GO!
    fpath = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/noMM.hdf5'
    with h5py.File(fpath, 'r') as fin:
        input_trajectory = fin['configurations'][()]
    Cij_input = np.cov(input_trajectory.T)
    # Cij_input = np.corrcoef(input_trajectory.T)
    # they are so similar it doesn't really matter...
    print(helpers.calc_C2(input_trajectory))
    correlations_input = tools.triu_flat(Cij_input)

    i_crit = 11
    i_plm = 17 # or 17 or 18
    i_cor = 21 # 21 is the one corresponding to the correction!

    selections = [i_crit, i_plm, i_cor, -1]
    labels=[
        r'Critical, $T_{f}=0.8$',
        r'PLM, $T_{f}=1.0$',
        r'SC, $T_{f}=1.1$',
        r'Para, $T_{f}=1.8$']
    # selections = [0, 10, -1]
    N, _ = Cij_input.shape
    Cijs = np.zeros((4, N, N))
    # Cijs[0] = Cij_input

    fig, ax = plt.subplots()
    ax.plot(correlations_input, correlations_input, ls='-', c='k', marker=',', zorder=10)
    for iS in range(0, len(selections)):
        mc_trajectory = helpers.get_example_trajecotry(selections[iS])
        Cij = np.cov(mc_trajectory.T)
        # Cij = np.corrcoef(mc_trajectory.T)
        # print(np.mean(Cij - np.cov(mc_trajectory.T)))
        correlations = tools.triu_flat(Cij)
        C2 = helpers.calc_C2(mc_trajectory)
        print(selections[iS], C2)
        ax.plot(
            correlations_input, correlations, ls='none',
            markeredgewidth=0.0,
            alpha=0.8, zorder=1,
            label=labels[iS]
            )
        np.fill_diagonal(Cij, 0)
        Cijs[iS] = Cij
    ax.set(
        xlabel=r'$C_{ij}$ no-MM dataset',
        ylabel=r'$C_{ij}$ mc simulations',
        xlim=[correlations_input.min(), correlations_input.max()]
    )
    ax.legend(framealpha=1.0)
    plt.show()
    
    fig, ax = mkfigure(nrows=2, ncols=2, sharex=True, sharey=True)
    np.fill_diagonal(Cij_input, 0)
    delta = 0.1
    # let's say everything over 0.1 is a "connection"
    Cijs[Cijs <= delta] = 0
    Cijs[Cijs > delta] = 1
    print(delta)
    vmin = Cij_input.min()
    vmax = Cij_input.max()
    ax = ax.ravel()
    # Cijs = Cijs[:, 360:, 360:]
    # how should we fix them model...?
    for iA in range(0, ax.size):
        im = ax[iA].matshow(
            Cijs[iA], cmap='cividis',
            # , vmin=vmin, vmax=vmax
            )
        # fig.colorbar(im, ax=ax[iA])
        # ticks = [0, 180, 360]
        ax[iA].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[iA].yaxis.set_major_locator(plt.MaxNLocator(4))
    ax[2].xaxis.tick_bottom()
    ax[3].xaxis.tick_bottom()

    # fig.suptitle(r'$A_{ij}$')
    fig.supxlabel(r'$j$')
    fig.supylabel(r'$i$')
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, ax=ax.ravel().tolist(), label=r'$C_{ij}$')
    plt.show()


def example_trajectories():
    # I WONDER IF THE CIJS LOOK THE SAME... HAVE A QUICK LITTLE GO!
    # Cij_input = np.cov(input_trajectory.T)
    # Cij_input = np.corrcoef(input_trajectory.T)
    # they are so similar it doesn't really matter...
    # print(helpers.calc_C2(input_trajectory))
    # correlations_input = tools.triu_flat(Cij_input)

    i_crit = 11
    i_plm = 17 # or 17 or 18
    i_cor = 21 # 21 is the one corresponding to the correction!

    selections = [2, i_crit, i_plm, i_cor, -1]
    labels=[
        r'SG, $T_{f}=0.5$',
        r'Critical, $T_{f}=0.8$',
        r'PLM, $T_{f}=1.0$',
        r'SC, $T_{f}=1.1$',
        r'Para, $T_{f}=1.8$']
    # selections = [0, 10, -1]
    print(labels)
    start = 2000
    roi_lim = 180
    length = roi_lim * 2
    # aspec ratio = 2 at the moment, let's try changing!
    # nrows=len(selections) + 1
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    ax = ax.ravel()

    fpath = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/noMM.hdf5'
    with h5py.File(fpath, 'r') as fin:
        input_trajectory = fin['configurations'][()]
    input_trajectory = input_trajectory[start:start+length, 0:roi_lim]

    ax[0].matshow(input_trajectory.T)
    ax[0].set(title='Brain Signal')
    # ax.plot(correlations_input, correlations_input, ls='-', c='k', marker=',', zorder=10)
    for iS in range(0, len(selections)):
        # fig, ax = plt.subplots()
        mc_trajectory = helpers.get_example_trajecotry(selections[iS])
        
        mc_trajectory = mc_trajectory[start:start+length, 0:roi_lim]
        print(iS, mc_trajectory.shape)
        ax[iS+1].matshow(mc_trajectory.T)
        ax[iS+1].set(title=labels[iS])
    for a in ax:
        # a.xaxis.set_major_locator(plt.MaxNLocator(3))
        # a.yaxis.set_major_locator(plt.MaxNLocator(3))
        a.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labeltop=False, labelbottom=False, labelleft=False)
    # ax[iA].yaxis.set_major_locator(plt.MaxNLocator(4))
    ax[-1].xaxis.tick_bottom()
    ax[-2].xaxis.tick_bottom()
    fig.supxlabel(r'time, $t \rightarrow$')
    fig.supylabel(r'$\leftarrow i$')
    plt.show()