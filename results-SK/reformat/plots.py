import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import load
import subplots as splots

import inference.analysis.new as analysis
from inference import tools


def figOverview(root='/Users/mk14423/Desktop/PaperData', save=False):
    JRparams, JRfull_obs, Bs = load.load_N200_Jrepeats_Bscaling(root)
    iB = 9
    print(Bs[iB] / 1e4)
    inTemps, outJRMeans, outJRStds = load.average_over_Jrepeats(
        JRparams, JRfull_obs, iB)

    N200params, N200obs, = load.load_N200_Bfixed_obs(root)
    print(N200obs.dtype.names)
    # load the B1e4 repeats now!
    # fig, ax = plt.subplots()
    # splots.phase_diagram(fig, ax, N200obs, N200params, 'e')
    # plt.show()

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-2col.mplstyle')
    FIGW = plt.rcParams.get('figure.figsize')[0]
    print(FIGW)
    # exit()
    # this basically always depends! ah that's so annoying!!
    # need to change this everytime I want a differnt figure layout...
    fig = plt.figure(figsize=(13, 5), constrained_layout=True)
    # fig = plt.figure(figsize=(FIGW, FIGW / 2.5))
    spec = fig.add_gridspec(nrows=2, ncols=5)
    ax_pd = fig.add_subplot(spec[0:, 1:3])
    axs = []
    axs.append(fig.add_subplot(spec[0, 0]))
    axs.append(fig.add_subplot(spec[1, 0]))
    axs.append(fig.add_subplot(spec[0:, 3:]))

    cbar = splots.phase_diagram(
        fig, axs[0], N200params, N200obs, 'q', contour=False, cbar=True)
    cbar.ax.set_title(r'$q$')  # fontsize=ls
    cbar = splots.phase_diagram(
        fig, axs[1], N200params, N200obs, 'chiSG', contour=False, cbar=True)
    cbar.ax.set_title(r'$C^{2}$')  # fontsize=ls
    cbar = splots.phase_diagram(
        fig, ax_pd, N200params, N200obs, 'e', setmax=2.5)
    cbar.ax.set_title(r'$\varepsilon_{\gamma}$')  # fontsize=ls

    atwin = axs[2].twinx()
    cols = plt.cm.get_cmap('cividis')(np.linspace(0, 1, 3))
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    l1 = splots.error_bars(
        axs[2], inTemps, outJRMeans, outJRStds, 'e', setmax=2,
        c=cols[0], label=r'$\varepsilon _{\gamma}$'
    )
    l2 = splots.error_bars(
        atwin, inTemps, outJRMeans, outJRStds, 'tau', setmax=10,
        c=cols[2], label=r'$\tau$'
    )
    l3 = splots.error_bars(
        atwin, inTemps, outJRMeans, outJRStds, 'chiSG', setmax=10,
        c=cols[1], label=r'$C^{2}$'
    )
    atwin.legend(handles=[l1, l2, l3])

    axs[2].set(
        xlabel=r'$T^{0}$',
        ylabel=r'$\varepsilon_{\gamma}$',
        xlim=[0.8, None],
        # yscale='log'
    )
    atwin.set(
        ylabel=r'$C^{2}$, $\tau$',
        ylim=[0.8, None],
        # yscale='log'
    )

    c = 'peru'
    ax_pd.axvline(
        0.1, ymin=-0.1, ymax=1.1, marker=',', color=c)
    splots.connect(
        fig, ax_pd, axs[2], [0.1, 1.2], [0, 1], [0, 0], color=c)
    if save is True:
        print('saving pdf...')
        plt.savefig(
            os.path.join(root, "figOverview.pdf"),
            dpi=600, bbox_inches='tight')
    plt.show()


def figSaturation(root='/Users/mk14423/Desktop/PaperData', save=False):
    JRparams, JRfull_obs, Bs = load.load_N200_Jrepeats_Bscaling(root)
    fig, ax = plt.subplots()
    Ts, popts, r2s = splots.JR_Tsaturation(fig, ax, JRparams, JRfull_obs, Bs)

    plt.close()
    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
    ax = ax.ravel()

    splots.JR_T0vsTinfr(fig, ax[1], JRparams, JRfull_obs, Bs, 'infrSig')
    splots.JR_Tsaturation(
        fig, ax[0], JRparams, JRfull_obs, Bs, Bcut=2, Tcut=6, Tmax=18)

    sm = 1.5
    l2 = splots.JR_Tstauration_fitparamscaling(
        ax[2], Ts, popts, r2s,
        setmax=sm
    )

    ax2 = ax[2].twinx()
    iBs = [3, 9, -1]
    iB = -1
    lines = [l2]
    for iB in iBs:
        lbl = r'$B=$' + '{:d}'.format(int(Bs[iB] / 1000)) + r'$\times 10^{3}$'
        print(Bs[iB] / 1e4)
        inTemps, outJRMeans, outJRStds = load.average_over_Jrepeats(
            JRparams, JRfull_obs, iB)
        line = splots.error_bars(
            ax2, inTemps, outJRMeans, outJRStds, 'e',
            setmax=sm,
            normbyMin=True,
            marker='X',
            label=lbl,
        )
        lines.append(line)
    # lines.append(l2)
    ax[2].legend(handles=lines)
    ax[2].set(ylabel=r'$\tilde{B}$')
    ax2.set(ylabel=r'$\varepsilon_{\gamma} / \varepsilon_{\gamma}^{max}$')
    # fig.subplots_adjust(wspace=0.5, hspace=0.1)
    fig.tight_layout(pad=0)
    if save is True:
        print('saving pdf...')
        plt.savefig(
            os.path.join(root, "figSaturation.pdf"),
            dpi=600, bbox_inches='tight')
    plt.show()


def figSaturation_2pannel(root='/Users/mk14423/Desktop/PaperData', save=False):
    JRparams, JRfull_obs, Bs = load.load_N200_Jrepeats_Bscaling(root)
    fig, ax = plt.subplots()
    Ts, popts, r2s = splots.JR_Tsaturation(fig, ax, JRparams, JRfull_obs, Bs)
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax = ax.ravel()

    splots.JR_Tsaturation_normalised(
        # ax[0], JRparams, JRfull_obs, Bs, Bcut=2, Tcut=6, Tmax=18
        ax[0], JRparams, JRfull_obs, Bs, Bcut=2, Tcut=6, Tmax=None
        )
    ax[0].set(xscale='log')

    sm = 1.5
    l2 = splots.JR_Tstauration_fitparamscaling(
        ax[1], Ts, popts, r2s,
        setmax=sm
    )

    ax2 = ax[1].twinx()
    iBs = [3, 9, -1]
    iB = -1
    lines = [l2]
    for iB in iBs:
        lbl = r'$B=$' + '{:d}'.format(int(Bs[iB] / 1000)) + r'$\times 10^{3}$'
        print(Bs[iB] / 1e4)
        inTemps, outJRMeans, outJRStds = load.average_over_Jrepeats(
            JRparams, JRfull_obs, iB)
        line = splots.error_bars(
            ax2, inTemps, outJRMeans, outJRStds, 'e',
            setmax=sm,
            normbyMin=True,
            marker='X',
            label=lbl,
        )
        lines.append(line)
    # lines.append(l2)
    ax[1].legend(handles=lines)
    ax[1].set(ylabel=r'$\tilde{B}$')
    ax2.set(ylabel=r'$\varepsilon_{\gamma} / \varepsilon_{\gamma}^{max}$')
    # fig.subplots_adjust(wspace=0.5, hspace=0.1)
    fig.tight_layout(pad=0)
    if save is True:
        print('saving pdf...')
        plt.savefig(
            os.path.join(root, "figSaturation_2pannel.pdf"),
            dpi=600, bbox_inches='tight')
    plt.show()


def figTcurves_and_distro(save=False):

    fig, ax = plt.subplots(2, 1)
    # fig, ax = splots.mkfigure(nrows=2, ncols=1)
    # ax = ax.ravel()

    root = '/Users/mk14423/Desktop/PaperData'
    JRparams, JRfull_obs, Bs = load.load_N200_Jrepeats_Bscaling(root)
    splots.JR_T0vsTinfr(fig, ax[0], JRparams, JRfull_obs, Bs, 'infrSig')

    root = '/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated'
    params_true, params_infr = load.load_trail_params_JR(root)

    nBs, nDatapoints = params_true.shape
    nbins = 250
    lbls = [
        r'$B = 1 \times 10^{3}$',
        r'$B = 2 \times 10^{3}$',
        r'$B = 1 \times 10^{4}$',
        r'$B = 5 \times 10^{4}$',
    ]
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fitdata = params_true.ravel()
    A, x = splots.distribution(
                ax[1], fitdata, nbins,
                c=cols[0],
                marker=',',
                label=r'$true$',
                zorder=50
                )

    for iB in range(0, nBs):
        fitdata = params_infr[iB]
        A, x = splots.distribution(
                        ax[1], fitdata, nbins,
                        c=cols[iB + 1],
                        marker=',',
                        label=lbls[iB]
                        )

    ax[1].set(
        # xscale='log',
        yscale='log',
        # xlim=[0.05, 1],
        ylim=[2 * 1e-3, None]
        )
    ax[1].set(ylabel=r'$P \left( J_{ij} ^ {*} \right)$')
    ax[1].set(xlabel=r'$J_{ij} ^ {*}$')
    ax[1].legend()
    # ax[iT, 1].set(xscale='log', yscale='log')

    # fig.tight_layout(pad=0)
    if save is True:
        print('saving pdf...')
        plt.savefig(
            os.path.join(root, "figTcurves_and_distro.pdf"),
            )
    plt.show()


def figDistribution(
        root='/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated',
        save=False):
    # how should I do this exactly... let's just pick a specific file and
    # see how distro looks!
    # maybe I should plot the two distros seperately?
    # params_true, params_infr, params_corr = load.load_trail_params(root)
    params_true, params_infr, params_corr = load.load_trail_params_JR(root)

    print(params_true.shape, params_infr.shape)

    nBs, nDatapoints = params_true.shape
    nbins = 250
    # don't know how to best present the two tails.
    fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    ax = ax.reshape(3, 1)
    # fig, ax = plt.subplots(3, 2, figsize=(6, 8))

    # for iB in range(0, nBs):
    #     params = params_true[iB, :]
    #     splots.loglog2side(ax[0, 0], ax[0, 1], params, nbins)
    # fig.tight_layout(pad=0)

    # for a in ax.ravel():
    #     # a.set_xticklabels([])
    #     a.set_yticklabels([])

    # plt.subplots_adjust(
    #     wspace=0,
    #     # hspace=0
    #     )
    # plt.show()
    # exit()

    lbls = [
        r'$B = 1 \times 10^{3}$',
        r'$B = 2 \times 10^{3}$',
        r'$B = 1 \times 10^{4}$',
    ]

    for iB in range(0, nBs):
        p_params_true = params_true[iB, params_true[iB] >= 0]
        p_params_infr = params_infr[iB, params_infr[iB] >= 0]
        p_params_corr = params_corr[iB, params_corr[iB] >= 0]

        n_params_true = params_true[iB, params_true[iB] < 0]
        n_params_infr = params_infr[iB, params_infr[iB] < 0]
        n_params_corr = params_corr[iB, params_corr[iB] < 0]

        print(p_params_true.shape)
        p_params = [p_params_true, p_params_infr, p_params_corr]
        n_params = [n_params_true, n_params_infr, n_params_corr]
        params = [params_true[iB], params_infr[iB], params_corr[iB]]
        # print(params_true.shape)
        cols1 = plt.cm.get_cmap('cividis')(np.linspace(0, 1, 3))
        # cols2 = plt.cm.get_cmap('plasma')(np.linspace(0, 1, 3))

        for iP in range(0, len(p_params)):
            # A, x = splots.distribution(
            #     ax[iB, 0], p_params[iP], nbins,
            #     label=None, c=cols1[iP], marker=',')
            # splots.distribution(
            #     ax[iB, 0], -n_params[iP], nbins,
            #     label=None, c=cols1[iP], ls='--', marker=',')
            fitdata = params[iP]
            A, x = splots.distribution(
                ax[iB, 0], fitdata, nbins,
                label=None, c=cols1[iP], marker=',')

            xs = np.linspace(x.min(), x.max(), 250)
            gauss = tools.gaussian(
                    xs, A.max(), np.mean(fitdata), np.std(fitdata))
            print(xs.shape, gauss.shape)
            ax[iB, 0].plot(xs, gauss, ls='-', marker=',', c='k')

        ax[iB, 0].annotate(
            lbls[iB], xy=(0.05, 0.05), xycoords='axes fraction',
            size=10, ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='w'))

        # leg = ax[iT, 0].legend(loc='lower left')
        # leg.legendHandles[0].set_color('white')
        ax[iB, 0].set(
            # xscale='log',
            yscale='log',
            # xlim=[0.05, 1],
            ylim=[1e-3, None]
            )
        ax[1, 0].set(ylabel=r'$P \left( J_{ij} ^ {*} \right)$')
        ax[2, 0].set(xlabel=r'$J_{ij} ^ {*}$')
        # ax[iT, 1].set(xscale='log', yscale='log')

    fig.tight_layout(pad=0)
    if save is True:
        print('saving pdf...')
        plt.savefig(
            os.path.join(root, "figDistribution4.pdf"),
            dpi=600, bbox_inches='tight')
    plt.show()


def figDistribution_single(
        root='/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated',
        save=False):

    # params_true, params_infr, params_corr = load.load_trail_params(root)
    # params_true, params_infr, params_corr = load.load_trail_params_JR(root)
    params_true, params_infr = load.load_trail_params_JR(root)

    print(params_true.shape, params_infr.shape)

    nBs, nDatapoints = params_true.shape
    nbins = 250

    fig, ax = plt.subplots(figsize=(6, 5))
    lbls = [
        r'$B = 1 \times 10^{3}$',
        r'$B = 2 \times 10^{3}$',
        r'$B = 1 \times 10^{4}$',
        r'$B = 5 \times 10^{4}$',
    ]
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fitdata = params_true.ravel()
    A, x = splots.distribution(
                ax, fitdata, nbins,
                c=cols[0],
                marker=',',
                label=r'$true$',
                zorder=50
                )
    # ax.hist(
    #         fitdata, nbins, color=cols[0], density=True, alpha=1,
    #         label=r'$true$',
    #         )
    # xs = np.linspace(x.min(), x.max(), 250)
    # gauss = tools.gaussian(
    #         xs, A.max(), np.mean(fitdata), np.std(fitdata))
    # ax.plot(xs, gauss, ls='-', marker=',', c='k')

    for iB in range(0, nBs):
        fitdata = params_infr[iB]
        # params = [params_true[iB], params_infr[iB], params_corr[iB]]
        # print(params_true.shape)
        # cols1 = plt.cm.get_cmap('cividis')(np.linspace(0, 1, 3))
        # cols2 = plt.cm.get_cmap('plasma')(np.linspace(0, 1, 3))
        # ax.hist(
        #     fitdata, nbins, color=cols[iB+1], density=True, alpha=1,
        #     label=lbls[iB],
        #     )
        A, x = splots.distribution(
                        ax, fitdata, nbins,
                        c=cols[iB + 1],
                        marker=',',
                        label=lbls[iB]
                        )
        # xs = np.linspace(x.min(), x.max(), 250)
        # gauss = tools.gaussian(
        #         xs, A.max(), np.mean(fitdata), np.std(fitdata))
        # ax.plot(xs, gauss, ls='-', marker=',', c='k')

        # ax.annotate(
        #     lbls[iB], xy=(0.05, 0.05), xycoords='axes fraction',
        #     size=10, ha='left', va='bottom',
        #     bbox=dict(boxstyle='round', fc='w'))

        # leg = ax[iT, 0].legend(loc='lower left')
        # leg.legendHandles[0].set_color('white')
    ax.set(
        # xscale='log',
        yscale='log',
        # xlim=[0.05, 1],
        ylim=[2 * 1e-3, None]
        )
    ax.set(ylabel=r'$P \left( J_{ij} ^ {*} \right)$')
    ax.set(xlabel=r'$J_{ij} ^ {*}$')
    ax.legend()
    # ax[iT, 1].set(xscale='log', yscale='log')

    fig.tight_layout(pad=0)
    if save is True:
        print('saving pdf...')
        plt.savefig(
            os.path.join(root, "figDistribution_plm_fillplus.pdf"),
            dpi=600, bbox_inches='tight')
    plt.show()
    # fig2
    fig, ax = plt.subplots(figsize=(3, 2.5))

    lbls = [
        r'$true$',
        r'$plm$',
        r'$correction$',
    ]
    # cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cols = plt.cm.get_cmap('cividis')(np.linspace(0, 1, 3))
    iB = 0
    params = [params_true[iB], params_infr[iB], params_corr[iB]]

    for iP, param in enumerate(params):
        fitdata = param
        ax.hist(
            fitdata, nbins,
            color=cols[iP], density=True, alpha=1,
            label=lbls[iP],
            )
        A, x = splots.distribution(
                        ax, fitdata, nbins,
                        c=cols[iP],
                        marker=',',
                        # label=lbls[iP]
                        )
        xs = np.linspace(x.min(), x.max(), 250)
        gauss = tools.gaussian(
                xs, A.max(), np.mean(fitdata), np.std(fitdata))
        ax.plot(xs, gauss, ls='-', marker=',', c='k')

    ax.set(
        # xscale='log',
        yscale='log',
        # xlim=[0.05, 1],
        ylim=[2 * 1e-3, None]
        )
    ax.set(ylabel=r'$P \left( J_{ij} ^ {*} \right)$')
    ax.set(xlabel=r'$J_{ij} ^ {*}$')
    ax.legend(prop={'size': 5.5})
    # ax[iT, 1].set(xscale='log', yscale='log')

    fig.tight_layout(pad=0)
    if save is True:
        print('saving pdf...')
        plt.savefig(
            os.path.join(root, "figDistribution_B1e3cor_fillplus.pdf"),
            dpi=600, bbox_inches='tight')
    plt.show()


def figCorrectionTC2(
        root='/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated',
        save=False,
        TorC2='C2'):

    files_B1e3 = [
        np.load(root + '/B1e3_1_temps.npz'),
        np.load(root + '/B1e3_1_corrs.npz')]
    files_B2e3 = [
        np.load(root + '/B2e3_1_temps.npz'),
        np.load(root + '/B2e3_1_corrs.npz')]
    files_B1e4 = [
        np.load(root + '/B1e4_1_temps.npz'),
        np.load(root + '/B1e4_1_corrs.npz')]

    T_files = [files_B1e3[0], files_B2e3[0], files_B1e4[0]]
    C_files = [files_B1e3[1], files_B2e3[1], files_B1e4[1]]


    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    fig, ax = splots.mkfigure(ncols=1, nrows=1)

    # fig.set_size_inches(w=FIGW, h=FIGW * GR)
    # fig.set_size_inches(w=FIGW, h=FIGW * GR)
    print(fig.get_size_inches())
    if TorC2 == 'C2':
        splots.C2T_npzplot(ax, C_files)
        ax.set(xlabel=r'$T^{0}$', ylabel=r'$\mathcal{C}^{2}$', yscale='linear')
    elif TorC2 == 'T':
        splots.C2T_npzplot(ax, T_files)
        ax.set(xlabel=r'$T^{0}$', ylabel=r'$T^{*}$', yscale='linear')
    else:
        print('invalid TorC2 choice')
        return None

    titles = [
        r'$B=1 \times 10^{3}$',
        r'$B=2 \times 10^{3}$',
        r'$B=1 \times 10^{4}$']

    # funky legend stuff...
    ncols = 1
    h, ls = ax.get_legend_handles_labels()
    ph = [plt.plot([], marker="", ls="")[0]]*3
    handles = (
        h[0:1:1] + ph[0:1] + h[1:3:1] + ph[1:2] +
        h[3:5:1] + ph[2:3] + h[5:7:1]
    )
    labels = (
        ls[0:1:1] + titles[0:1] + ls[1:3:1] +
        titles[1:2] + ls[3:5:1] + titles[2:3] + ls[5:7:1]
    )
    leg = ax.legend(handles, labels, ncol=ncols, loc='upper right', prop={'size': 8})
    for vpack in leg._legend_handle_box.get_children():
        x = vpack.get_children()
        hpacks = [x[1], x[4], x[7]]
        for hpack in hpacks:
            hpack.get_children()[0].set_width(0)

    # fig.tight_layout(pad=0)

    if save is True:
        print('saving pdf...')
        plt.savefig(
            os.path.join("/Users/mk14423/Dropbox/Apps/Overleaf/InverseIsingInference_paperMBK/small_draft/Figures/main-nice", "figTError.png"),
            )
    plt.show()


def figKajimuraDemonstration(
        root='/Users/mk14423/Desktop/PaperData/Kajimura_analysis',
        save=False,
        ):

    runs = [
        'noMM.hdf5',
        'noMM_matched_subsamples/noMM_subsample0.hdf5',
        # 'noMM_subsample1.hdf5'
        'MM.hdf5'
    ]
    lbls = ['noMM', 'ss-noMM', 'MM']
    # lbls = ['noMM', 'MM']
    # lbls = ['ss-noMM', 'MM']
    params_infr, params_corr = load.load_kajimura_parameters(runs, root)
    # np.array(params_infr), np.array(params_corr)
    # print(y.shape, z.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    # params = [params_infr, params_corr]
    nbins = 200
    cols = plt.cm.get_cmap('cividis')(np.linspace(0, 1, len(runs)))
    cuts = np.linspace(0.05, 0.2, 10)
    ms = np.zeros((len(runs), cuts.size))

    for iR in range(0, len(runs)):
        # for iP in range(0, len(params)):
        A, xs = splots.distribution(
            ax, params_infr[iR], nbins,
            ls='-',
            label=lbls[iR],
            # c=cols[iR],
            marker=',')
        # xs = np.linspace(x.min(), x.max(), 250)
        gauss = tools.gaussian(
            xs, A.max(), np.mean(params_infr[iR]), np.std(params_infr[iR]))
        # print(xs.shape, gauss.shape)
        T = 1 / (np.std(params_infr[iR]) * (399 ** 0.5))
        print(lbls[iR], T)
        # ax.plot(xs, gauss, ls='-', marker=',', c='k')
        # A_tail = A[A >= gauss]
        # x_tail = xs[A >= gauss]
        # A_tail = A_tail[A_tail > 0]
        # x_tail = x_tail[A_tail > 0]

        # for iC, cut in enumerate(cuts):
        #     x_tail = xs[xs >= cut]
        #     A_tail = A[xs >= cut]

        #     x_tail = x_tail[A_tail > 0]
        #     A_tail = A_tail[A_tail > 0]
        #     # ax.plot(x_tail, A_tail)
        #     func, popts = tools.curve_fit1D(
        #         np.log10(x_tail), np.log10(A_tail), tools.linear)
        #     # y = mx + c
        #     yfit = func(np.log10(x_tail), popts[0], popts[1])
        #     Afit = 10 ** (yfit)
        #     plt.plot(
        #         x_tail, Afit, marker=',',
        #         label='cut = {:.2f} : m = {:.2f}'.format(cut, popts[0]), alpha=0.5)
        #     ms[iR, iC] = popts[0]
        # print(ms)
        # splots.distribution(
        #     ax, params_corr[iR], nbins,
        #     ls='--',
        #     label=lbls[iR], c=cols[iR], marker=',')
    ax.set(
        # xscale='log',
        yscale='log',
        ylim=[3 * 1e-3, None],
        )
    ax.legend()
    ax.set(xlabel=r'$J_{ij} ^ {*}$', ylabel=r'$P \left( J_{ij} ^ {*} \right)$')
    fig.tight_layout(pad=0)
    if save is True:
        print('saving pdf...')
        plt.savefig(
            os.path.join(root, "figKajimuraDistros_inset.pdf"),
            dpi=600, bbox_inches='tight')
    plt.show()
    # fig, ax = plt.subplots()
    # ax.plot(cuts, ms[0, :], label=lbls[0])
    # ax.plot(cuts, ms[1, :], label=lbls[1])
    # ax.set(xlabel='cutoff', ylabel='m')
    # plt.legend()
    # plt.show()


def figKajimuraSubsampling(
        root='/Users/mk14423/Desktop/PaperData/Kajimura_analysis',
        save=False,
        ):
    pass
