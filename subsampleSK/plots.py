import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# really these should all be individual things that load their data that
# I can re-organsie, but anyway, it doesn't matter.

import subplots

from pyplm.utilities import tools

FIGDIR = '/Users/mk14423/Dropbox/Apps/Overleaf/thesis/thesis/1-results-sk-plm/figures/varying-N'

def subsampling_plot(ax, df, fitting_th, iG, label):
    COLORS_CAT = [
        '#4053d3', '#ddb310', '#b51d14',
        '#00beff', '#fb49b0', '#00b25d', '#cacaca'
    ]
    print(df)
    iDs = df['iD'].unique()
    for iD in iDs:
        # subplots.saturation_split_iD(ax, df, 'T', iD, ls='none', alpha=0.1, c=COLORS_CAT[iD])
        subplots.saturation_split_iD(ax, df, 'T', iD, ls='none', alpha=0.25, c='grey')
    popts = subplots.saturation_fit(
        ax, df, 'T', fitting_th,
        c=COLORS_CAT[iG], ls='-', marker=',', lw=2, label=label)
    return popts


def N_emin_Btilde(Ns, emins, Btildes, save=False):
    fig, ax = plt.subplots()
    Btildes = Btildes / 1e3
    ax2 = ax.twinx()
    emins *= np.sqrt(Ns)
    T_trues = np.array([1.1, 1.175, 1.1, 1.1, 1.25])
    emins /= (T_trues)
    print('--- emin ---')
    # l1 = subplots.plot_with_fit(ax, Ns, emins, tools.linear, '#4053d3', r'$\varepsilon N ^{1/2}$')
    l1 = subplots.plot_with_fit(
        ax, Ns, emins, tools.linear,
        '#4053d3', r'$\varepsilon N ^{1/2}$'
    )
    print('-- Btilde --')
    # l2 = subplots.plot_with_fit(ax2, Ns, Btildes, tools.linear, '#ddb310', r'$\tilde{B}$')
    l2 = subplots.plot_with_fit(
        ax2, Ns, Btildes, tools.linear, '#ddb310',
        r'$\tilde{B}$'
    )

    ax.set(
        xlabel=r'$N$',
        ylabel=r'$ \varepsilon N ^{1/2}$',
        # ylabel=r'$\tilde{B}$'
        # xscale='log',
        # yscale='log'
        )
    ax2.set_yticks(
        np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))

    ax2.set(
        ylabel=r'$ \tilde{B} (\times 10^3)$',
        # yscale='log'
    )
    print('-----')
    ax.legend(handles=[l1,l2])
    # inset_ax = ax.inset_axes([0.58, 0.12, 0.39, 0.39])
    # inset_ax.tick_params(axis='both', which='both', labelsize=10)
    # inset_ax.xaxis.label.set_size(10)
    # inset_ax.yaxis.label.set_size(10)
    # subplots.N_error(inset_ax)
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'error-Btilde-vs-N.png'))
    plt.show()
    fig, ax = plt.subplots()
    subplots.N_error(ax)
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'error-vs-N.png'))
    plt.show()

    fig, ax = plt.subplots()
    print('-- Corr --')
    subplots.plot_with_fit(
        ax, emins, Btildes,
        tools.linear, '#4053d3', None,
    )
    ax.set(
        # xlabel=r'$ \varepsilon _{\mathrm{min}} N ^{1/2}$',
        xlabel=r'$ \varepsilon N ^{1/2}$',
        ylabel=r'$ \tilde{B} (\times 10^3)$',
        )
    if save is True:
        plt.savefig(os.path.join(FIGDIR, 'error-vs-Btilde.png'))
    plt.show()
