import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from pyplm.utilities import tools
from pyplm import plotting as plmplt

FIGDIR = '/Users/mk14423/Library/CloudStorage/Dropbox/Apps/Overleaf/thesis/thesis/1-results-sk-plm/figures/error-examples'


def error_sum_vs_i(phases, true_mods, infr_mods, save=False):
    catcols = [plmplt.category_col(i) for i in range(0, 6)]
    # Sum of error on each row
    fig, ax = plmplt.mkfigure(nrows=1, ncols=1, sharex=True)
    for iP in range(0, len(phases)):
        true_model = true_mods[iP]
        infr_model = infr_mods[iP]
        error_model = np.abs(infr_model - true_model)

        true_vs_i = np.sum(np.abs(true_mods[iP]), axis=0)
        infr_vs_i = np.sum(np.abs(infr_mods[iP]), axis=0)
        error_vs_i = np.sum(error_model, axis=0)

        # ax[0, 0].plot(true_vs_i, label=phases[iP], c=catcols[iP])
        # ax[1, 0].plot(infr_vs_i, label=phases[iP], c=catcols[iP])
        ax[0, 0].plot(error_vs_i, label=phases[iP], c=catcols[iP])

        # ax.plot(error_vs_i, label=phases[iP], c=catcols[iP])
        ylbl = r'$ \Sigma _{j} \left( \left| \theta_{ij}^{*} - \theta_{ij}^{0} \right| \right)$'
    for ax in ax.ravel():
        ax.set(yscale='log', xlabel=r'$i$', ylabel=ylbl)
    plt.legend()
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'error-sum-vs-i.png'))
    plt.show()


# somethin is wrong; showing up wrong way round I think!
def models_and_correlations(phases, true_mods, infr_mods, save=False):
    for iP in range(0, len(phases)):
        fig = plt.figure()
        gs = fig.add_gridspec(4, 4)
        ax0 = fig.add_subplot(gs[0:2, 0:2])
        ax1 = fig.add_subplot(gs[0:2, 2:])
        ax2 = fig.add_subplot(gs[2:, :])
        ax = np.array([ax0, ax1, ax2])
        ax = np.reshape(ax, (1, 3))
        plmplt.add_subplot_labels(ax)

        true_model = true_mods[iP]
        infr_model = infr_mods[iP]
 
        true_params = tools.triu_flat(true_model)
        infr_params = tools.triu_flat(infr_model)

        true_max = np.nanmax(true_model)
        true_min = np.nanmin(true_model)

        ax[0, 0].matshow(true_model)
        ax[0, 1].matshow(infr_model, vmin=true_min, vmax=true_max)

        ax[0, 2].plot(true_params, infr_params, c=plmplt.category_col(iP), ls='none', alpha=0.5)
        ax[0, 2].plot(true_params, true_params, c='k', marker=',')
        
        ax[0, 0].xaxis.set_ticks_position('bottom')
        ax[0, 0].set(xlabel=r'$j$', ylabel=r'$i$')

        ax[0, 1].xaxis.set_ticks_position('bottom')
        ax[0, 1].set(xlabel=r'$j$', ylabel=r'$i$')
        
        ax[0, 2].set(xlabel=r'$\theta_{ij}^{0}$', ylabel=r'$\theta_{ij}^{*}$')
        ax[0, 2].set(ylim=[true_min, true_max])

        if save is True:
            plt.savefig(
                os.path.join(FIGDIR, 'model-and-correlation-' + phases[iP] + '.png'))
        plt.show()



# somethin is wrong; showing up wrong way round I think!
def models_correlations_distributions(phases, true_mods, infr_mods, save=False):
    for iP in range(0, len(phases)):
        fig, ax = plmplt.mkfigure(nrows=2, ncols=2)
        # fig = plt.figure()
        # gs = fig.add_gridspec(4, 4)
        # ax0 = fig.add_subplot(gs[0:2, 0:2])
        # ax1 = fig.add_subplot(gs[0:2, 2:])
        # ax2 = fig.add_subplot(gs[2:, :])
        # ax = np.array([ax0, ax1, ax2])
        # ax = np.reshape(ax, (1, 3))
        # plmplt.add_subplot_labels(ax)

        true_model = true_mods[iP]
        infr_model = infr_mods[iP]
        N, _ = true_model.shape
 
        true_params = tools.triu_flat(true_model, k=1)
        infr_params = tools.triu_flat(infr_model, k=1)

        true_max = np.nanmax(true_model)
        true_min = np.nanmin(true_model)

        # PRINT OBSERVABLES: MU, SIGMA, R2, ERROR!
        print('---------------')
        m = (np.mean(true_params) / np.std(true_params)) * (N ** 0.5)
        print(m)
        print('---------------')

        mu_true = np.mean(true_params) * N
        sigma_true = np.std(true_params) * (N**0.5)
        
        mu_infr = np.mean(infr_params) * N
        sigma_infr = np.std(infr_params) * (N**0.5)

        mu_true = mu_true / sigma_true
        mu_infr = mu_infr / sigma_infr

        r2 = r2_score(true_params, infr_params)
        error = np.sqrt(np.sum((infr_params - true_params) ** 2) / np.sum(true_params ** 2))
        print(f'------------ {phases[iP]} ------------')
        print(f'mu_T:  {mu_true:.3f}, mu_I:  {mu_infr:.3f}')
        print(f'tem_T: {1/sigma_true:.3f}, tem_I: {1/sigma_infr:.3f}')
        print(f'r2:    {r2:.3f}, err:   {error:.3f}')
        print(
            f'mu_hT: {np.mean(np.diag(true_model)/sigma_true):.3f}, mu_hI {np.mean(np.diag(infr_model/sigma_infr)):.3f}')
        ax[0, 0].matshow(true_model)
        ax[0, 1].matshow(infr_model, vmin=true_min, vmax=true_max)

        ax[1, 0].plot(true_params, infr_params, c=plmplt.category_col(iP), ls='none', alpha=0.5)
        ax[1, 0].plot(true_params, true_params, c='k', marker=',')

        # twin = ax[1, 1].twinx()
        twin = ax[1, 1]
        # nbins = 25
        nbins = np.linspace(true_params.min(), true_params.max(), 30)
        plmplt.histogram(ax[1, 1], true_params, nbins, c='k')
        plmplt.histogram(twin, infr_params, nbins, c=plmplt.category_col(iP))
        
        ax[0, 0].xaxis.set_ticks_position('bottom')
        ax[0, 0].set(xlabel=r'$j$', ylabel=r'$i$')

        ax[0, 1].xaxis.set_ticks_position('bottom')
        ax[0, 1].set(xlabel=r'$j$', ylabel=r'$i$')
        
        ax[1, 0].set(xlabel=r'$\theta_{ij}^{0}$', ylabel=r'$\theta_{ij}^{*}$')
        ax[1, 0].set(ylim=[true_min, true_max])

        # relabel to Jij!!
        ax[1, 1].set(xlabel=r'$J_{ij}$', ylabel=r'$P(J_{ij})$')
        ax[1, 1].set(xlim=[true_min, true_max])

        if save is True:
            plt.savefig(
                os.path.join(FIGDIR, 'mod-corr-dist-' + phases[iP] + '.png'))
        plt.show()


def models_error_distributions(phases, true_mods, infr_mods, save=False):
    fig, ax = plmplt.mkfigure(nrows=2, ncols=1)
    for iP in range(0, len(phases)):

        true_model = true_mods[iP]
        infr_model = infr_mods[iP]
        N, _ = true_model.shape
 
        true_params = tools.triu_flat(true_model, k=1)
        infr_params = tools.triu_flat(infr_model, k=1)

        true_hs = np.diag(true_model)
        infr_hs = np.diag(infr_model)

        true_max = np.nanmax(true_model)
        true_min = np.nanmin(true_model)

        # ax[0, 0].matshow(true_model)
        # ax[0, 1].matshow(infr_model, vmin=true_min, vmax=true_max)

        # ax[1, 0].plot(true_params, infr_params, c=plmplt.category_col(iP), ls='none', alpha=0.5)
        # ax[1, 0].plot(true_params, true_params, c='k', marker=',')

        # I need to use the same bins I think!!!
        # that only seems fair!!


        ax[0, 0].axvline(np.mean(true_hs), c='k', marker=',')
        plmplt.histogram(
            ax[0, 0], np.abs(infr_hs - true_hs), int(N/2),
            c=plmplt.category_col(iP), marker='o', ls='-',
            label=phases[iP])
        # ax[0, 0].set(xlabel=r'$|\Delta h_{i}|$', ylabel=r'$P(|\Delta h_{i}|)$')
        ax[0, 0].set(
            xlabel=r'$\left| h_{i}^{*} - h_{i}^0 \right|$',
            ylabel=r'$P \left( \left| h_{i}^{*} - h_{i}^0 \right| \right)$')
        # ax[1, 0].set(xlim=[true_min, true_max])
        ax[0, 0].set(xlim=[5e-5, 50], xscale='log', yscale='log')
    

        nbins = 200
        ax[1, 0].axvline(0, c='k', marker=',')
        plmplt.histogram(
            ax[1, 0], np.abs(infr_params - true_params), nbins,
            c=plmplt.category_col(iP), marker='o', ls='-',
            label=phases[iP])
        ax[1, 0].set(
            xlabel=r'$\left| J_{ij}^{*} - J_{ij}^0 \right|$',
            ylabel=r'$P\left( \left| J_{ij}^{*} - J_{ij}^0 \right| \right)$')
        # ax[iP, 1].set(xlim=[true_min, true_max])
        ax[1, 0].set(xlim=[5e-5, 50], xscale='log', yscale='log')
    ax[0, 0].legend()
    if save is True:
        plt.savefig(
            os.path.join(FIGDIR, 'error-dists' + '.png'))
    plt.show()


from scipy.interpolate import UnivariateSpline
from scipy.signal import correlate
def overlap(trajectory):
    # print(trajectory.shape)
    B, N = trajectory.shape
    auto_corr = 0
    for i in range(0, N):
        data = trajectory[:, i]
        # normalising or z-scoring it!
        # EXP_si = np.mean(data)
        # STD_si = np.std(data)
        # data = (data - np.mean(data))  # / STD_si
        data = (data)  # / STD_si
        ac = correlate(
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


def autocorrs(trajectories, phases):
    fig, ax = plt.subplots()
    # some-sort of averaging..?
    i_sorted = [0, 3, 2, 1]
    catcols = [plmplt.category_col(i) for i in range(0, 6)]
    # catcols = [catcols[0], ]
    for i in i_sorted:
        traj = trajectories[i]
        acs = []
        taus = []
        # print(traj.shape)
        for j in range(0, 6):
            start = int(j * 10000)
            stop = int((j+1) * 10000)
            t = traj[start:stop, :]
            # print(i, j, t.shape)
            ac, tau = overlap(t)
            acs.append(ac)
            taus.append(tau)
        acs = np.array(acs)
        taus = np.array(taus)
        ac = np.mean(acs, axis=0)
        taus = np.mean(taus, axis=0)
        if tau < 1:
            tau = 1
        tau = round(tau)
        lbl = f'{phases[i]}'  # + r'$\tau=$' + f' {tau}'
        print(phases[i], tau)
        # np.arange(0, 10000) / tau,
        ax.plot(ac, marker=',', label=lbl, c=catcols[i])
    ax.set(xlabel=r'$\Delta t$', ylabel=r'$C_t (\Delta t)$', xscale='log')
    ax.legend(loc='upper right')
    plt.show()
