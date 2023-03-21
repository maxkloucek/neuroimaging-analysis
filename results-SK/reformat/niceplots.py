import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from inference import tools
import load
import subplots as splots
import plthelpers as plth
import corr2helpers

# FIGDIR = (
#     '/Users/mk14423/Dropbox/Apps/Overleaf/InverseIsingInference_paperMBK/'
#     + 'small_draft/Figures/main-nice')
FIGDIR = '/Users/mk14423/Documents/tempfigs'
COLORS_CAT = [
    '#4053d3', '#ddb310', '#b51d14',
    '#00beff', '#fb49b0', '#00b25d', '#cacaca']


def PhaseDiagramOverview(root='/Users/mk14423/Desktop/PaperData', save=False):
    N200params, N200obs, = load.load_N200_Bfixed_obs(root)
    print(N200obs.dtype.names)

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-2col.mplstyle')
    FIGW = plt.rcParams.get('figure.figsize')[0]
    fig, axs = splots.mkfigure(nrows=1, ncols=3, figsize=(FIGW, FIGW / 3))
    axs = axs.ravel()

    cbar = splots.phase_diagram(
        fig, axs[0], N200params, N200obs,
        'q', contour=False, cbar=True,
        # xlabel=r'$\mu / \sigma$',
        ylabel=r'$T^{0} / \sigma ^{0}$'
        )
    cbar.ax.set_title(r'$q$')  # fontsize=ls
    cbar = splots.phase_diagram(
        fig, axs[1], N200params, N200obs, 'chiSG', contour=False, cbar=True,
        xlabel=r'$\mu ^{0} / \sigma ^{0}$',
        # ylabel=r'$T^{0} / \sigma$'
        )
    cbar.ax.set_title(r'$C^{2}$')  # fontsize=ls
    cbar = splots.phase_diagram(
        fig, axs[2], N200params, N200obs, 'e',
        setmax=1.5,
        contour=True,
        # xlabel=r'$\mu / \sigma$',
        # ylabel=r'$T^{0} / \sigma$'
        )
    cbar.ax.set_title(r'$\varepsilon$')  # fontsize=ls
    # c = 'peru'
    c = COLORS_CAT[4]
    index_min = np.unravel_index(
        np.nanargmin(N200obs['e']), N200obs['e'].shape)
    Jmin = N200params['J'][index_min]
    Tmin = N200params['T'][index_min]
    axs[2].plot(Jmin, Tmin, c=COLORS_CAT[0])
    axs[2].axvline(
        0.1, ymin=-0.1, ymax=1.1, marker=',', color=c, linewidth=2.0)
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'phase-diagram-overview.png'))
    plt.show()


def PhaseDiagramOverview_squareTau(root='/Users/mk14423/Desktop/PaperData', save=False):
    N200params, N200obs, = load.load_N200_Bfixed_obs(root)
    print(N200obs.dtype.names)

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    figsize = plt.rcParams.get('figure.figsize')
    print(figsize)
    # exit()
    fig, axs = splots.mkfigure(
        nrows=2, ncols=2,
        # figsize=(FIGW, FIGW * 0.8)
        sharex=True, sharey=True)
    axs = axs.ravel()

    cbar = splots.phase_diagram(
        fig, axs[0], N200params, N200obs,
        'q', contour=False, cbar=True,
        # xlabel=r'$\mu / \sigma$',
        # ylabel=r'$T^{0} / \sigma ^{0}$'
        )
    cbar.ax.set_title(r'$q$')  # fontsize=ls

    cbar = splots.phase_diagram(
        fig, axs[1], N200params, N200obs, 'chiSG', contour=False, cbar=True,
        # xlabel=r'$\mu ^{0} / \sigma ^{0}$',
        # ylabel=r'$T^{0} / \sigma$'
        )
    cbar.ax.set_title(r'$C^{2}$')  # fontsize=ls

    cbar = splots.phase_diagram(
        fig, axs[2], N200params, N200obs, 'tau',
        setmax=100,
        contour=False,
        # xlabel=r'$\mu / \sigma$',
        # ylabel=r'$T^{0} / \sigma$'
        )
    cbar.ax.set_title(r'$\tau$')  # fontsize=ls

    cbar = splots.phase_diagram(
        fig, axs[3], N200params, N200obs, 'e',
        setmax=1.5,
        contour=True,
        # xlabel=r'$\mu / \sigma$',
        # ylabel=r'$T^{0} / \sigma$'
        )
    cbar.ax.set_title(r'$\varepsilon$')

    c = COLORS_CAT[4]
    index_min = np.unravel_index(
        np.nanargmin(N200obs['e']), N200obs['e'].shape)
    Jmin = N200params['J'][index_min]
    Tmin = N200params['T'][index_min]
    axs[3].plot(Jmin, Tmin, c=COLORS_CAT[0])
    axs[3].axvline(
        0.1, ymin=-0.1, ymax=1.1, marker=',', color=c, linewidth=2.0)

    fig.supxlabel(r'$\mu$')
    fig.supylabel(r'$T$')

    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'phase-diagram-overview-square.png'))
    plt.show()


def PhaseDiagramOverview_allN(
        dataroot='/Users/mk14423/Desktop/PaperData',
        obsname='tau', save=False):
    runsN50 = [
            'B1e4-Nscaling/N50_1',
            'B1e4-Nscaling/N50_2',
            'B1e4-Nscaling/N50_3',
            'B1e4-Nscaling/N50_4',
            'B1e4-Nscaling/N50_5',
            'B1e4-Nscaling/N50_6',
        ]
    runsN100 = [
            'B1e4-Nscaling/N100_1',
            'B1e4-Nscaling/N100_2',
            'B1e4-Nscaling/N100_3',
            'B1e4-Nscaling/N100_4',
            'B1e4-Nscaling/N100_5',
            'B1e4-Nscaling/N100_6',
        ]
    runsN200 = [
            'B1e4-Nscaling/N200_1',
            'B1e4-Nscaling/N200_2',
            'B1e4-Nscaling/N200_3',
        ]
    runsN400 = [
            'B1e4-Nscaling/N400_1',
            'B1e4-Nscaling/N400_2',
            'B1e4-Nscaling/N400_3',
        ]
    runsN800 = [
            'B1e4-Nscaling/N800_1',
        ]
    runsNx = [runsN50, runsN100, runsN200, runsN400, runsN800]
    # runsNx = [runsN50, runsN100]

    # plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-2col.mplstyle')
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    FIGW = plt.rcParams.get('figure.figsize')[0]
    print(FIGW)
    nN = len(runsNx)
    # fig, axs = splots.mkfigure(
        # nrows=nN, ncols=3, figsize=(FIGW, (FIGW/4) * nN), sharex=True, sharey=True)
    fig, axs = plt.subplots(
        nrows=nN, ncols=3, figsize=(FIGW, (FIGW/4) * nN), sharex=True, sharey=True
    )
    # axs = axs.ravel()
    print(axs.shape)
    print(nN)
    minimum_obs = np.zeros((nN, 3))
    Ns = [r'$N=50$', r'$N=100$', r'$N=200$', r'$N=400$', r'$N=800$']
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)']
    for iR in range(0, nN):
        params, obs, = load.load_PD_fixedB(runsNx[iR], dataroot)
        cbar = splots.phase_diagram(
            fig, axs[iR, 0], params, obs,
            'q', contour=False, cbar=True,
            # xlabel=r'$\mu / \sigma$',
            ylabel=letters[iR] + ' '+ Ns[iR]
            )
        if iR <= 0:
            cbar.ax.set_title(r'$q$')  # fontsize=ls
        cbar = splots.phase_diagram(
            fig, axs[iR, 1], params, obs, 'chiSG', contour=False, cbar=True,
            # xlabel=r'$\mu / \sigma$',
            # ylabel=r'$T^{0} / \sigma$'
            )
        if iR <= 0:
            cbar.ax.set_title(r'$C^{2}$')  # fontsize=ls
        cbar = splots.phase_diagram(
            fig, axs[iR, 2], params, obs, 'e', setmax=1.5, contour=True,
            # xlabel=r'$\mu / \sigma$',
            # ylabel=r'$T^{0} / \sigma$'
            )
        # axs[iR, 2].plot(params['T'])

        index_min = np.unravel_index(np.nanargmin(obs['e']), obs['e'].shape)
        Emin = np.nanmin(obs['e'])
        Jmin = params['J'][index_min]
        Tmin = params['T'][index_min]
        minimum_obs[iR] = [Emin, Jmin, Tmin]
        axs[iR, 2].plot(Jmin, Tmin)
        if iR <= 0:
            cbar.ax.set_title(r'$\varepsilon$')  # fontsize=ls

        c = COLORS_CAT[4]
        axs[iR, 2].axvline(
            0.1, ymin=-0.1, ymax=1.1, marker=',', color=c, linewidth=2.0)
    print('------')
    print(minimum_obs)
    print('------')
    # shared_sp = fig.add_subplot(111, frameon=False)
    # shared_sp.set(xlabel=r'$\mu$', ylabel=r'$T$')
    fig.supxlabel(r'$\mu$')
    fig.supylabel(r'$T$')
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'phase-diagram-overview-allN.png'))
    plt.show()

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    Ns = np.array([50, 100, 200, 400, 800])
    # Nparams = np.array([N + ((N * (N-1)) / 2) for N in Ns])
    fig, ax = splots.mkfigure(nrows=1, ncols=1)
    # x = np.sqrt(Ns)
    x = Ns
    y = minimum_obs[:, 0]  # * np.sqrt(Ns)
    ax.plot(x, y, ls='none', color=COLORS_CAT[0])
    ax.set(xlabel=r'$N$', ylabel=r'$\varepsilon _{\mathrm{min}}$')

    popt, pcov = curve_fit(power, x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    yfit = power(xs, *popt)

    score = r2_score(y, power(x, *popt))
    print(popt, score)
    lbl = (
        r'$\varepsilon_{\mathrm{min}}(N) = A N ^\gamma$'
        # + ': $R^2 =$'
        # + f'{score:.3}'
        )
    ax.plot(xs, yfit, marker=',', c='k', ls='--', label=lbl)
    ax.legend()
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'emin-scaling-allN.png'))
    plt.show()


# this is not a polynomial!!
def power(x, A, B):
    return A * (x ** (B))


def BtildeNscaling_convtoB(
        dataroot='/Users/mk14423/Desktop/PaperData',
        save=False):
    runsN50 = [
            'B1e4-Nscaling/N50_1',
            'B1e4-Nscaling/N50_2',
            'B1e4-Nscaling/N50_3',
            'B1e4-Nscaling/N50_4',
            'B1e4-Nscaling/N50_5',
            'B1e4-Nscaling/N50_6',
        ]
    runsN100 = [
            'B1e4-Nscaling/N100_1',
            'B1e4-Nscaling/N100_2',
            'B1e4-Nscaling/N100_3',
            'B1e4-Nscaling/N100_4',
            'B1e4-Nscaling/N100_5',
            'B1e4-Nscaling/N100_6',
        ]
    runsN200 = [
            'B1e4-Nscaling/N200_1',
            'B1e4-Nscaling/N200_2',
            'B1e4-Nscaling/N200_3',
        ]
    runsN400 = [
            'B1e4-Nscaling/N400_1',
            'B1e4-Nscaling/N400_2',
            'B1e4-Nscaling/N400_3',
        ]
    runsN800 = [
            'B1e4-Nscaling/N800_1',
        ]
    runsNx = [runsN50, runsN100, runsN200, runsN400, runsN800]
    # runsNx = [runsN50, runsN100]
    nN = len(runsNx)
    minimum_obs = np.zeros((nN, 3))
    Ns = [r'$N=50$', r'$N=100$', r'$N=200$', r'$N=400$', r'$N=800$']
    for iR in range(0, nN):
        params, obs, = load.load_PD_fixedB(runsNx[iR], dataroot)
        index_min = np.unravel_index(np.nanargmin(obs['e']), obs['e'].shape)
        Emin = np.nanmin(obs['e'])
        Jmin = params['J'][index_min]
        Tmin = params['T'][index_min]
        minimum_obs[iR] = [Emin, Jmin, Tmin]

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    Ns = np.array([50, 100, 200, 400, 800])
    # Nparams = np.array([N + ((N * (N-1)) / 2) for N in Ns])
    fig, ax = splots.mkfigure(nrows=1, ncols=1)
    ax2 = ax.twinx()

    x = Ns
    y = minimum_obs[:, 0]
    y2 = np.copy(y)

    B_tilde_at_emin = 1055.392
    y_200 = y2[2]
    y2 = (y2 / y_200) * B_tilde_at_emin
    y2 /= 1e3
    print('------')
    print(x)
    print('emin', y)
    print('Btilde', y2)
    print('------')
    ax.plot(x, y, ls='none', color=COLORS_CAT[0])
    ax2.plot(x, y2, ls='none', color=COLORS_CAT[0])

    popt, pcov = curve_fit(power, x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    yfit = power(xs, *popt)
    score = r2_score(y, power(x, *popt))
    print(popt, score)
    # lbl = (
    #     r'$\tilde{B}(\varepsilon = \varepsilon _{\mathrm{min}}, N)' +
    #     r'= A N ^\gamma$'
    #     )
    lbl = (r'$\varepsilon _{\mathrm{min}}(N) = A N ^\gamma$')
    ax.plot(xs, yfit, marker=',', c='k', ls='--', label=lbl)

    # ylbl = r'$\tilde{B}(\varepsilon = \varepsilon _{\mathrm{min}}, N)$'
    ylbl = r'$\varepsilon _{\mathrm{min}}$'
    ax.set(
        ylim=[y.min() * 0.97, y.max() * 1.03],
        xlabel=r'$N$',
        ylabel=ylbl)

    ax2.set(
        ylim=[y2.min() * 0.97, y2.max() * 1.03],
        xlabel=r'$N$', ylabel=r'$\tilde{B} (\times 10^{3})$')

    ax.legend()
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'emin-Btilde-scaling-allN.png'))
    plt.show()


def FixedN_observablesCut(root='/Users/mk14423/Desktop/PaperData', save=False):
    JRparams, JRfull_obs, Bs = load.load_N200_Jrepeats_Bscaling(root)
    # print(Bs)
    inTemps, outJRMeans, outJRStds = load.average_over_Jrepeats(
        JRparams, JRfull_obs, iB=9)
    print(outJRMeans.shape)

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    fig, axs = splots.mkfigure()
    ax = axs

    atwin = axs.twinx()
    axs = atwin
    atwin = ax
    # cols = plt.cm.get_cmap('cividis')(np.linspace(0, 1, 3))
    # cols = plt.cm.get_cmap('Set1')(np.arange(0, 10))
    # cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # cols = ['#003f5c', '#bc5090', '#ffa600']

    l1 = splots.error_bars(
        axs, inTemps, outJRMeans, outJRStds, 'e', setmax=2,
        label=r'$\varepsilon$',
        c=COLORS_CAT[2]
    )
    l2 = splots.error_bars(
        atwin, inTemps, outJRMeans, outJRStds, 'tau', setmax=10,
        c=COLORS_CAT[1], label=r'$\tau$'
    )
    l3 = splots.error_bars(
        atwin, inTemps, outJRMeans, outJRStds, 'chiSG', setmax=10,
        c=COLORS_CAT[0], label=r'$C^{2}$'
    )
    atwin.legend(handles=[l3, l2, l1])

    axs.set(
        xlabel=r'$T$',
        ylabel=r'$\varepsilon$',
        xlim=[0.8, None],
        # xlim=[0.6, None],
        # yscale='log'
    )
    atwin.set(
        xlabel=r'$T$',
        ylabel=r'$C^{2}$, $\tau$',
        ylim=[0.8, None],
        # yscale='log'
    )

    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'N200-obs-cut.png'))
    plt.show()


def InferredT_and_distribution(save=False):

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    FIGW = plt.rcParams.get('figure.figsize')[0]
    # maybe I want an inest here, it would make the distros quite small!
    fig, axs = splots.mkfigure(
        nrows=1, ncols=1,
        # figsize=(FIGW * 2, FIGW * (5/6))
        )
    axin = axs.inset_axes([0.15, 0.65, 0.3, 0.3])
    axs = [axin, axs]
    
    root = '/Users/mk14423/Desktop/PaperData'
    JRparams, JRfull_obs, Bs = load.load_N200_Jrepeats_Bscaling(root)
    splots.JR_T0vsTinfr(fig, axs[1], JRparams, JRfull_obs, Bs, 'infrSig')

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
    # cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cols = COLORS_CAT
    fitdata = params_true.ravel()
    print('-Distros come from:-')
    print(1 / (np.std(params_true) * 200 ** 0.5))
    A, x = splots.distribution(
                axs[0], fitdata, nbins,
                c='k',
                marker=',',
                label=r'$true$',
                zorder=50,
                linewidth=1,
                ls='--'
                )

    for iB in range(0, nBs):
        fitdata = params_infr[iB]
        print(1 / (np.std(fitdata) * 200 ** 0.5))
        A, x = splots.distribution(
                        axs[0], fitdata, nbins,
                        c=cols[iB],
                        marker=',',
                        label=lbls[iB],
                        linewidth=2,
                        )

    axs[0].set(
        # xscale='log',
        yscale='log',
        # xlim=[0.05, 1],
        ylim=[2.3 * 1e-6, None]
        )
    axs[0].set_ylabel(r'$P \left( J_{ij} ^ {*} \right)$', fontsize=10, labelpad=0)
    axs[0].set_xlabel(r'$J_{ij} ^ {*}$', fontsize=10, labelpad=0)
    # axs[1].legend()
    # ax[iT, 1].set(xscale='log', yscale='log')

    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'infrT-and-distros.png'))
    plt.show()
    plt.show()


def TSaturation_and_fit_inset(root='/Users/mk14423/Desktop/PaperData', save=False):
    JRparams, JRfull_obs, Bs = load.load_N200_Jrepeats_Bscaling(root)
    fig, ax = plt.subplots()
    Ts, popts, r2s = splots.JR_Tsaturation(fig, ax, JRparams, JRfull_obs, Bs)
    print('----- Btildes ------')
    for Ttrue, Btilde in zip(Ts, popts[:, 1]):
        print(Ttrue, 1 / Btilde)
    print('----- ------- ------')
    plt.close()
    # plt.show()
    # yep I want to split this!

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    # plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisaside.mplstyle')
    FIGW = plt.rcParams.get('figure.figsize')[0]
    # ---------- #
    fig, axs = splots.mkfigure(nrows=1, ncols=1)
    axs = [axs]
    By, Tx, Tcols = splots.JR_Tsaturation_normalised(
        axs[0], JRparams, JRfull_obs, Bs, Bcut=2, Tcut=None, Tmax=None
        )
    axs[0].set(xscale='log')
    # plt.savefig(os.path.join(FIGDIR, 'fixed-mu-Tinfr-vs-B-lowT.png'))
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'T-saturation-inset-nofix.png'))
    plt.show()
    # ---------- #

    fig, axs = splots.mkfigure(nrows=1, ncols=1)
    axs = [0, axs]
    sm = 1.5
    l2 = splots.JR_Tstauration_fitparamscaling(
        axs[1], Ts, popts, r2s,
        setmax=sm
    )
    # this is some extra stuff!
    print(By / 1e3)
    # print(Tx)
    for i in range(0, len(By)):
        axs[1].plot(
            Tx[i], By[i] / 1e3, c=Tcols[i], marker='X', ms=9,
            zorder=1000)

    # axs[1].xaxis.label.set_size(10)
    # axs[1].yaxis.label.set_size(10)
    ax.tick_params(axis='both', which='major') # , labelsize=10
    ax.tick_params(axis='both', which='minor') # , labelsize=8
    # axs[1].xaxis.labelpad = -3
    # axs[1].yaxis.labelpad = -1

    # iBs = [-1, -2, -3]
    iBs = [-1]
    i = 1
    Bmin = np.min(1/popts[3:, 1])
    print(Bmin)

    for iB in iBs:
        # lbl = r'$B=$' + '{:d}'.format(int(Bs[iB] / 1000)) + r'$\times 10^{3}$'
        # print(Bs[iB] / 1e4)
        lbl = r'$\varepsilon/ \varepsilon _{min}$'
        inTemps, outJRMeans, outJRStds = load.average_over_Jrepeats(
            JRparams, JRfull_obs, iB)
        # print(outJRStds)
        l1 = splots.error_bars(
            axs[1],
            # ax2,
            inTemps, outJRMeans, outJRStds, 'e',
            setmax=sm,
            normbyMin=True,
            marker='o',
            label=lbl,
            # ls='none',
            color=COLORS_CAT[0],
        )
        i += 1

    lines = [l1, l2]
    axs[1].legend(handles=lines)
    # axs[1].set(ylabel=r'$\tilde{B} (\times 10^{3})$')

    lbl = r'$\tilde{B}(\times 10^{3})$,  '
    lbl += r'$\varepsilon/ \varepsilon _{min}$'
    axs[1].set(ylabel=lbl, xlim=[0.65, 2.05])
    # ax2.set(ylabel=r'$\varepsilon/ \varepsilon _{min}$')
    # ax2.set(yscale='log')
    plt.show()


def TSaturation_vary_includedB(
        root='/Users/mk14423/Desktop/PaperData', save=False):
    JRps, JRobs, samples = load.load_N200_Jrepeats_Bscaling(root)

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    FIGW = plt.rcParams.get('figure.figsize')[0]
    fig, axs = splots.mkfigure(
        nrows=1, ncols=1,
        figsize=(FIGW, FIGW * (5/6))
        )
    axs = [axs]
    # iT Bmin = ? 12 I think!
    iT = 8
    iB_min = 1
    iB_max = len(samples)
    # iB_max = 8
    print(samples[iB_max - 1])

    Bs, Ts, Ts_err = load.get_curves_fixedT(JRps, JRobs, samples, iT, 0, None)
    axs[0].errorbar(
        x=Bs, y=Ts, yerr=Ts_err,
        ls='none', marker='o', color=COLORS_CAT[3])
    Tlim_Btilde_r2 = load.rescaleHelper_Bminvar_fit(Bs, Ts, axs[0], color='k')
    print(Tlim_Btilde_r2)

    T_theory = JRps['T'][0, iT]

    print(f'-------- T0 = {T_theory} -------- ')
    cm = plt.cm.get_cmap('cividis')
    cols = cm(np.linspace(0, 1, 1 + iB_max - iB_min))

    measures = np.zeros((iB_max - iB_min - 3, 3))
    i = 0

    for iB in range(iB_min + 3, iB_max):
        Bs, Ts, Ts_err = load.get_curves_fixedT(
            JRps, JRobs, samples, iT, iB_min, iB)
        Tlim_Btilde_r2 = load.rescaleHelper_Bminvar_fit(
            Bs, Ts, axs[0], color=cols[iB])
        measures[i] = Tlim_Btilde_r2
        i+= 1
        print(Tlim_Btilde_r2)
    axs[0].set(xlabel=r'$B$', ylabel=r'$T^{*}$')
    plt.show()
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax = ax.ravel()
    ax[0].plot(measures[:, 0])
    ax[0].set(ylabel=r'$T_{inf}$')
    ax[1].plot(measures[:, 1])
    ax[1].set(ylabel=r'$\tilde{B}$')
    ax[2].plot(measures[:, 2])
    ax[2].set(ylabel=r'$r^2$')
    plt.show()


# do the rescaling
# check if the model I chose aligns with what I expect!
# not sure what my story is here...
def TSaturation_rescale_test(
        root='/Users/mk14423/Desktop/PaperData', save=False):
    JRps, JRobs, samples = load.load_N200_Jrepeats_Bscaling(root)
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    FIGW = plt.rcParams.get('figure.figsize')[0]
    fig, axs = splots.mkfigure(
        nrows=2, ncols=1,
        figsize=(FIGW, FIGW * (5/6) * 2)
        )
    # axs = [axs]
    # iT Bmin = ? 12 I think!
    iT = -1
    iB_min = 1
    iB_max = 9
    print(samples[iB_min], samples[iB_max])
    # might make me life easier if I chose B_max = 1e4 (can compare)

    Bs, Ts, Ts_err = load.get_curves_fixedT(JRps, JRobs, samples, iT, 1, None)
    axs[0].errorbar(
        x=Bs, y=Ts, yerr=Ts_err,
        ls='none', marker='o', color=COLORS_CAT[3])
    Tlim_Btilde_r2 = load.rescaleHelper_Bminvar_fit(Bs, Ts, axs[0], color='k')

    Bs, Ts, Ts_err = load.get_curves_fixedT(
        JRps, JRobs, samples, iT, iB_min, iB_max)
    Tlim_Btilde_r2 = load.rescaleHelper_Bminvar_fit(
        Bs, Ts, axs[0], color='r')
    # print(Tlim_Btilde_r2)
    axs[0].set(xlabel=r'$B$', ylabel=r'$T^{*}$')

    mods = load.rescaleHelper_get_models(iT, iB_max)

    print('-----')
    T_theory = JRps['T'][0, iT]
    T_plm = 1 / (np.std(tools.triu_flat(mods[1], k=1)) * 200 ** 0.5)
    T_fit = Tlim_Btilde_r2[0]
    lim_factor = T_fit / T_plm
    print(f'T-0   = {T_theory:.3f}')
    print(f'T-PLM = {T_plm:.3f}')
    if len(mods) == 3:
        T_cor = 1 / (np.std(tools.triu_flat(mods[2], k=1)) * 200 ** 0.5)
        print(f'T-COR = {T_cor:.3f}')
    print(f'T-FIT = {T_fit:.3f}')
    print(f'T-FIT / T-PLM = {lim_factor:.3f}')
    print('-----')
    mods.append(mods[1]/lim_factor)
    nbins = 100
    cols = ['k', COLORS_CAT[0], COLORS_CAT[1], COLORS_CAT[2]]

    for iM, mod in enumerate(mods):
        # print(1 / (np.std(tools.triu_flat(mod, k=1)) * 200 ** 0.5))
        A, x = splots.distribution(
                    axs[1],
                    tools.triu_flat(mod, k=1),
                    nbins,
                    c=cols[iM],
                    marker=',',
                    # ls='none',
                    label=r'$true$',
                    zorder=50
                    )

    # p_true = tools.triu_flat(mods[0], k=0)
    # for iM, mod in enumerate(mods[1:]):
    #     p_infr = tools.triu_flat(mod, k=0)
    #     print(1 / (np.std(p_infr) * (200 ** 0.5)))
    #     pcc = pearsonr(p_true, p_infr)
    #     print(pcc)
    #     axs[1].plot(p_true, p_infr, alpha=0.2, c=COLORS_CAT[iM], ls='none')

    n_true, bin_edges = np.histogram(tools.triu_flat(mods[0], k=1), bins=nbins)
    n_plm, _ = np.histogram(tools.triu_flat(mods[1], k=1), bins=bin_edges)
    err_plm = np.mean(np.abs(n_plm - n_true))
    print(f'err-plm = {err_plm:.3f}')

    if len(mods) == 4:
        n_cor, _ = np.histogram(tools.triu_flat(mods[2], k=1), bins=bin_edges)
        err_cor = np.mean(np.abs(n_cor - n_true))
        print(f'err-cor = {err_cor:.3f}')

    n_cor2, _ = np.histogram(tools.triu_flat(mods[-1], k=1), bins=bin_edges)
    err_cor2 = np.mean(np.abs(n_cor2 - n_true))
    print(f'err-cor2 = {err_cor2:.3f}')
    # axs[1].plot(bin_edges[0:-1], (n_plm - n_true) / n_true)
    # axs[1].plot(bin_edges[0:-1], (n_cor2 - n_true) / n_true)
    plt.show()
    # I want to do a predcity thingy for temps at Bmax = something?


def cor2_infr_temp(save=False):
    # iB_maxs = [7, 8, 9, 10, 11, 12, 13, 14]
    # iB_maxs = [7, 8, 9, 10, 11, 14]
    # I don't know where the recalc has gone, I probably deleted
    # something I shouldn't have!

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    FIGW, FIGH = plt.rcParams.get('figure.figsize')
    fig, ax = splots.mkfigure(
        nrows=2, ncols=1,
        figsize=(FIGW, FIGH * 1.2),
        sharex=True,
        )
    ax = ax.ravel()
    T_file_C2cor = (
        '/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated'
        + '/B1e4_1_temps.npz'
        )
    lines1 = splots.npz_funcofT(
        ax[0], T_file_C2cor,
        ['tru', 'plm', 'cor'],
        [COLORS_CAT[0], COLORS_CAT[1]],
        )
    T_file_satcor = (
        '/Users/mk14423/Desktop/PaperData/N200_saturation_correction'
        + '/B1e4_1_temps.npz'
        )
    lines2 = splots.npz_funcofT(
        ax[0], T_file_satcor,
        ['cor2'],
        [COLORS_CAT[2]],

        )

    C2_file_C2cor = (
        '/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated'
        + '/B1e4_1_corrs.npz'
    )
    splots.npz_funcofT(
        ax[1], C2_file_C2cor,
        ['tru', 'plm', 'cor'],
        [COLORS_CAT[0], COLORS_CAT[1]],
        labels=['PLM', 'cor-SC']
        )

    C2_file_satcor = (
        '/Users/mk14423/Desktop/PaperData/N200_saturation_correction'
        + '/B1e4_1_corrs.npz'
    )
    splots.npz_funcofT(
        ax[1], C2_file_satcor,
        ['cor2'], [COLORS_CAT[2]],
        ['cor-SS']
        )
    lines = lines1.append(lines2[0])
    ax[1].legend(handles=lines)
    ax[0].set(
                # xlabel=r'$T^{0} / \sigma^{0}$',
                ylabel=r'$T^{*}$', yscale='linear')
    ax[1].set(
                xlabel=r'$T$',
                ylabel=r'$C^{2}$', yscale='linear')
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'correction-comparison.png'))
    plt.show()


def Correction_T_C2(
        root='/Users/mk14423/Desktop/PaperData/N200_J0.1_optimizeT/0_updated',
        save=False):

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

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    FIGW, FIGH = plt.rcParams.get('figure.figsize')
    fig, axs = splots.mkfigure(
        nrows=2, ncols=1,
        figsize=(FIGW, FIGH * 1.2),
        sharex=True,
        )
    axs = axs.ravel()
    obsnames = ['T', 'C2']
    for iO in range(0, len(obsnames)):
        ax = axs[iO]
        name = obsnames[iO]
        if name == 'C2':
            splots.C2T_npzplot(ax, C_files, npzheaders=['plm', 'cor'])
            ax.set(
                xlabel=r'$T$',
                ylabel=r'$C^{2}$', yscale='linear')
            # ax.legend()
            # --------- legend stuff ------ #
            titles = [
                r'$B=1 \times 10^{3}$',
                r'$B=2 \times 10^{3}$',
                r'$B=1 \times 10^{4}$']
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
            # --------- ---------- ------ #
        elif name == 'T':
            splots.C2T_npzplot(ax, T_files, npzheaders=['plm', 'cor'])
            ax.set(
                xlim=[0.5, 2],
                # xlabel=r'$T$',
                ylabel=r'$T^{*}$', yscale='linear')
    # axs[0].axhline(0.65)
    # axs[0].axhline(0.5)

    # axs[1].axvline(1.367)
    # axs[1].axvline(1.02)
    # axs[1].axvline(0.78)
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'correction-T-C2.png'))
    plt.show()


def KajimuraSubSampling(save=False):

    # this should really be a subplot, but anyway!
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    FIGW = plt.rcParams.get('figure.figsize')[0]
    fig, axs = splots.mkfigure(
        nrows=2, ncols=1,
        figsize=(FIGW, FIGW * (5/6) * 2)
        )
    # axs = [axs]
    # axs.append(axs[0].inset_axes([0.48, 0.1, 0.5, 0.5]))
    axs = axs.ravel()
    axs = np.array([axs[1], axs[0]])
    dataroot = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/'

    l1_conditions = ['noMM_subsample_noL1', 'noMM_subsample_yesL1']
    # l1_conditions = ['noMM_subsample_noL1']
    markers = ['o', 'd']
    for iC in range(0, 1):
        model_files = [
            'noMM.hdf5',
            # 'noMM_matched_subsamples/noMM_subsample0.hdf5',
            # 'noMM_matched_subsamples/noMM_subsample1.hdf5',
            'MM.hdf5'
        ]
        # lbls = ['noMM', 'ss-noMM', 'MM', 'plm', 'correction']
        lbls = ['noMM', 'MM', 'ss-plm', 'ss-correction']
        subsamples, temps_plm, temps_corr = load.kajimura_get_temps(
            # dataroot + 'noMM_subsample_yesL1'
            dataroot + l1_conditions[iC]
            )

        choice_samples, choice_plm, choice_cor = load.kajimura_get_models(
            dataroot, model_files)
        for i in range(0, len(choice_samples)):
            # print(choice_samples[i])
            print('---infrred temps')
            print(load.temp_conveter(choice_plm[i]))
            axs[0].plot(
                choice_samples[i], load.temp_conveter(choice_plm[i]),
                marker='X', ms=10, c=COLORS_CAT[i + 2], zorder=500,
                ls='none',
                label=lbls[i]
                )
            # axs[0].plot(
            #     choice_samples[i], load.temp_conveter(choice_cor[i]),
            #     marker='X', ms=10, c=COLORS_CAT[i + 2], zorder=500,
            #     ls='none',
            #     )

        axs[0].errorbar(
            x=subsamples,
            y=temps_plm[:, 0], yerr=temps_plm[:, 1],
            # ls='none',
            marker=markers[iC],
            # ms=4,
            color=COLORS_CAT[0],
            label=lbls[-2],
            )
        axs[0].errorbar(
            x=subsamples,
            y=temps_corr[:, 0], yerr=temps_corr[:, 1],
            # ls='none',
            marker=markers[iC],
            # ms=4,
            color=COLORS_CAT[1],
            label=lbls[-1],
            )

        cut = 10
        xs = subsamples[cut:]
        # ys = temps_plm[cut:, 0]
        ys_both = [temps_plm[cut:, 0]]
        # ys_both = [temps_plm[cut:, 0], temps_corr[cut:, 0]]
        for ys in ys_both:
            popt, pcov = curve_fit(tools.arctan, xs, ys)
            r2 = r2_score(ys, tools.arctan(xs, *popt))
            T_lim = (popt[0] * np.pi) / 2
            B_tilde = (1 / popt[1])
            print(r2, T_lim, B_tilde)
            # xfit = np.linspace(0, 1e5, 100)
            xfit = np.linspace(subsamples.min(), subsamples.max(), 100)
            # xfit = xs
            yfit = tools.arctan(xfit, *popt)
            axs[0].plot(
                xfit, yfit, c='k', ls='--', marker=',',
                lw=2, zorder=500)

        for i in range(0, len(choice_samples)):
            splots.distribution(
                axs[1], tools.triu_flat(choice_plm[i], k=0),
                nbins=125, marker=',', c=COLORS_CAT[i + 2]
            )

    # '---- adding L1 regularisation ----'
    subsamples, temps_plm, temps_corr = load.kajimura_get_temps(
            dataroot + l1_conditions[1]
            )
    axs[0].errorbar(
        x=subsamples,
        y=temps_plm[:, 0], yerr=temps_plm[:, 1],
        marker='o', color=COLORS_CAT[-1], zorder=0,
        label = 'ss-pm-L1'
        )

    axs[0].set(xlabel=r'$B$', ylabel=r'$T^{*}$', ylim=[0, 1.55])
    axs[0].legend()
    axs[1].set(
        yscale='log',
        xlabel=r'$\theta_{ji} ^{*}$',
        ylabel=r'$P(\theta_{ji} ^{*})$'
        )
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'kajimura-subsampling.png'))
    plt.show()


def Kajimura_model_similarity(save=False):
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
    fig, axs = splots.mkfigure(
        nrows=2, ncols=1,
        figsize=(FIGW, FIGW * (5/6) * 2),
        sharex=True,
        )
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


def KajimuraSweep(save=False):
    root = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/'
    npz_fout = root + 'noMMchis.npz'
    Ts = np.load(npz_fout)['true_temps']
    C2s = np.load(npz_fout)['C2s']
    qs = np.load(npz_fout)['qs']
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    fig, ax = splots.mkfigure()
    ax2 = ax.twinx()
    # the shift of the thingy shown as well?
    ax.errorbar(
        x=Ts, y=qs[:, 0], yerr=qs[:, 1] / np.sqrt(60), c=COLORS_CAT[6])
    ax2.errorbar(
        x=Ts, y=C2s[:, 0], yerr=C2s[:, 1] / np.sqrt(60), c=COLORS_CAT[5])

    ax2.plot([1, 1.1277], [10.695, 5.359], zorder=250, marker=',', c='k', lw='2')
    ax2.plot([1], [10.695], marker='*', ms=12, zorder=251, c=COLORS_CAT[2])
    ax2.plot([1.1277], [5.359], marker='*', ms=12, zorder=251, c=COLORS_CAT[1])
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
    ax.set(ylabel=r'$q$', xlabel=r'$\alpha$')
    ax2.set(ylabel=r'$C^{2}$')
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'kajimura-sweep.png'))
    plt.show()


def KajimuraShowModel(save=False):
    dataroot = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/'
    model_files = [
        'noMM.hdf5',
        # 'noMM_matched_subsamples/noMM_subsample0.hdf5',
        # 'noMM_matched_subsamples/noMM_subsample1.hdf5',
        # 'MM.hdf5'
    ]
    choice_samples, choice_plm, choice_cor = load.kajimura_get_models(
        dataroot, model_files)
    print(choice_plm.shape, choice_cor.shape)
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-2col.mplstyle')
    fig, ax = splots.mkfigure()
    mat = ax.matshow(choice_cor[0], cmap='cividis')
    ax.xaxis.tick_bottom()
    # ax.imshow(choice_cor[0], interpolation="none")

    ax.axhline(358.5, c='k', marker=',')
    # ax.axvline(0, c='k', marker=',')

    # ax.axhline(0, c='k', marker=',')
    ax.axvline(358.5, c='k', marker=',')

    ax.axhline(359/2, c='k', marker=',', ls='--')
    ax.axvline(359/2, c='k', marker=',', ls='--')
    ax.set(xlabel=r'$i$', ylabel=r'$j$')
    cbar = plt.colorbar(mat)
    cbar.set_label(r'$\theta_{ij}$', rotation=360)
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'kajimura-modshow-cor.png'))
    plt.show()


def KajimuraShowCij(save=False):
    dataroot = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/'
    model_files = [
        'noMM.hdf5',
        # 'noMM_matched_subsamples/noMM_subsample0.hdf5',
        # 'noMM_matched_subsamples/noMM_subsample1.hdf5',
        # 'MM.hdf5'
    ]

    fpath = dataroot + model_files[0]
    print(fpath)
    with h5py.File(fpath, 'r') as fin:
        trajectory = fin['configurations'][()]
    # Cij = np.cov(trajectory.T)
    Cij = np.corrcoef(trajectory.T)
    np.fill_diagonal(Cij, 0)

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-2col.mplstyle')
    fig, ax = splots.mkfigure()
    mat = ax.matshow(Cij, cmap='cividis')
    ax.xaxis.tick_bottom()
    # ax.imshow(choice_cor[0], interpolation="none")

    ax.axhline(358.5, c='k', marker=',')
    # ax.axvline(0, c='k', marker=',')

    # ax.axhline(0, c='k', marker=',')
    ax.axvline(358.5, c='k', marker=',')

    ax.axhline(359/2, c='k', marker=',', ls='--')
    ax.axvline(359/2, c='k', marker=',', ls='--')
    ax.set(xlabel=r'$i$', ylabel=r'$j$')
    cbar = plt.colorbar(mat)
    cbar.set_label(r'$C_{ij}$', rotation=360)

    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'kajimura-modshow-cij.png'))
    plt.show()


def rework_1overB(save=False):

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
    fig, ax = splots.mkfigure()

    B, T, _ = plth.get_kajrimuraSubSamples()
    x = 1/B
    T = T[:, 0]
    fit_x, fit_T, fps = plth.invB_T_lin_fit_varyingCut(x, T)
    plt.plot(
        x, T / fps[1],
        # ls='none'
        label='noMM-fMRI'
        )
    plt.plot(fit_x, fit_T / fps[1], marker=',', c='k', ls='--')

    B, T = plth.get_SK_B_T(iT=20)
    x = 1/B
    fit_x, fit_T, fps = plth.invB_T_lin_fit_varyingCut(x, T)
    plt.plot(
        x, T / fps[1],
        # ls='none'
        label=r'SK $: T^{0}/ \sigma^{0} = 2.0$'
        )
    plt.plot(fit_x, fit_T / fps[1], marker=',', c='k', ls='--')

    B, T = plth.get_SK_B_T(iT=12)
    x = 1/B
    fit_x, fit_T, fps = plth.invB_T_lin_fit_varyingCut(x, T)
    plt.plot(
        x, T / fps[1],
        # ls='none',
        label=r'SK $: T^{0}/ \sigma^{0} = 1.4$'
        )
    plt.plot(fit_x, fit_T / fps[1], marker=',', c='k', ls='--')


    # B, T = plth.get_SK_B_T(iT=8)
    # x = 1/B
    # fit_x, fit_T = plth.invB_T_lin_fit_varyingCut(x, T)
    # plt.plot(
    #     x, T,
    #     # ls='none'
    #     )
    # plt.plot(fit_x, fit_T, marker=',', c='k', ls='--')

    ax.set(xlabel=r'$1/B$', ylabel=r'$T^{*} / c$')
    ax.set(ylim=[0.5, None])
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend()
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'saturation-inverseB-normed.png'))
    plt.show()


def var_B_plot(ax, xs, ys, label=None):
    # ys = sigmas * (N_kajimura ** 0.5)
    # fit_xs, fit_ys, fps = plth.fit_linear_increaseing(xs, ys)
    # # x, T / fps[1],
    xs = xs * 1e4
    fit_xs, fit_ys, fps = plth.fit_linear_decreasing(xs, ys)
    # m , c
    # print(fps * 1e4)
    Btilde = fps[0] * 1e4
    T = 1/np.sqrt(fps[1])
    print(Btilde, T)
    # this will now have to be the linar_decraseing one!
    ax.plot(xs, ys, label=label)
    # # plt.plot(fit_x, fit_T / fps[1], marker=',', c='k', ls='--')
    ax.plot(fit_xs, fit_ys, marker=',', c='k', ls='--')


def rework_var_and_B(save=False):

    # plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')

    fig, ax = splots.mkfigure()

    Bs, param_list = plth.kajimura_load_all_subsampled_parameters()
    sigmas = plth.kajimura_convert_to_sigma(param_list)
    N_kajimura = 399
    sigmas = sigmas * (N_kajimura ** 0.5)
    var_B_plot(ax, 1/Bs, sigmas ** 2, 'noMM-fMRI')

    Bs, sigmas = plth.SK_load_all_sample_parameters(iT=12)
    N_SK = 200
    sigmas = sigmas * (N_SK ** 0.5)
    var_B_plot(ax, 1/Bs, sigmas ** 2, r'SK: $T^{0}/ \sigma^{0} = 1.4$')

    Bs, sigmas = plth.SK_load_all_sample_parameters(iT=20)
    N_SK = 200
    sigmas = sigmas * (N_SK ** 0.5)
    var_B_plot(ax, 1/Bs, sigmas ** 2, r'SK: $T^{0}/ \sigma^{0} = 2.0$')

    ax.set(xlabel=r'$1/B (\times 10^{-4})$', ylabel=r'$\sigma ^{2} _{J} N$')
    ax.set(ylim=[0.2, 2], xlim=[0, 10.05])
    # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(loc='upper right')
    # FIGDIR = '/Users/mk14423/Dropbox/Apps/Overleaf/thesis/thesis/chapter-corrections/figures'
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'sigma-B.png'))
    plt.show()


def C2_B_fakedata(save=False):
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisaside.mplstyle')
    fig, ax = splots.mkfigure()
    # Bs = np.arange(10, 10000, 100)
    Bs = np.logspace(1.0, 5.0, num=50)
    Bs = Bs.astype(int)
    C2s = np.zeros(Bs.shape)
    N = 200
    reps = 6
    for i, B in enumerate(Bs):
        C2 = 0
        for rep in range(0, reps):
            trajectory = np.random.randint(0, 2, (B, N))
            trajectory[trajectory == 0] = -1
            cij = np.cov(trajectory.T)
            # print(trajectory.shape)
            # print(cij.shape)
            N, _ = cij.shape
            C = np.sum(cij ** 2)
            C = C / N
            C2 += C
        C2 /= reps
        C2s[i] = C2
    # print(C2s)
    # yep somethings wrong because otherwise they flucutate!
    # oooooo cause it's making C2s an integer! silly silly!
    ax.plot(Bs, C2s, marker=',')
    ax.set(xlabel=r'$B$', ylabel=r'$C^2$', xscale='log', yscale='log')
    # ax.set(ylim=[1, 1.05])
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'C2B-fakedata.png'))
    plt.show()
