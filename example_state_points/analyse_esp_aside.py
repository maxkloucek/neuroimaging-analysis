import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from pyplm.utilities import tools
from pyplm import plotting as plmplt

FIGDIR = '/Users/mk14423/Dropbox/Apps/Overleaf/thesis/thesis/results-sk-plm/figures'


def correlation_ferro(phases, true_mods, infr_mods, save=False):
    iP = 1
    print(phases[iP])
    fig, ax = plmplt.mkfigure(nrows=1, ncols=1)
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

    true_params = tools.triu_flat(true_model, k=0)
    infr_params = tools.triu_flat(infr_model, k=0)

    true_max = np.nanmax(true_model)
    true_min = np.nanmin(true_model)

    true_params = tools.triu_flat(true_model, k=1)
    infr_params = tools.triu_flat(infr_model, k=1)
    ax[0, 0].plot(true_params, infr_params, c=plmplt.category_col(iP), ls='none', alpha=0.5)
    ax[0, 0].plot(true_params, true_params, c='k', marker=',')
    # ax[0, 0].set(xlabel=r'$\theta_{ij}^{0}$', ylabel=r'$\theta_{ij}^{*}$')
    # ax[0, 0].set(ylim=[true_min, true_max])
    if save is True:
        plt.savefig(
            os.path.join(FIGDIR, 'ferro-corr-aside.png'))
    plt.show()
