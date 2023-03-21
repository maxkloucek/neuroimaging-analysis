import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d

from pyplm.plotting import mkfigure
from pyplm.utilities import tools
from inference.scripts.paper2022May import load

# COLORS_CAT = plt.rcParams['axes.prop_cycle'].by_key()['color']
# FIGDIR = '/Users/mk14423/Documents/tempfigs'


def load_model(file=None, directory='/Users/mk14423/Desktop/PaperData/Kajimura_analysis/'):
    if file==None:
        print("Possible options are: 'noMM.hdf5', 'MM.hdf5'")
        exit()
    with h5py.File(directory + file, 'r') as f:
        print(f.keys())
        print(f['configurations'].shape)
        plm_model = f['InferredModel'][()]
        cor_model = plm_model / f['correction'][()][0]
        # print(1/ np.std(cor_model), 1/ np.std(plm_model))
        # f['InferredModel'][()]
    return plm_model, cor_model


def distribution(ax, parameters, bins, **pltargs):
    n, x = np.histogram(parameters, bins, density=True)
    x = x[:-1]
    n = gaussian_filter1d(n, sigma=1)
    ax.plot(
        x, n,
        **pltargs,
    )
    return n, x

def liner_saturation_checker(ax2, samples, temperatures, xcut, label):
    x = 1/samples
    y = 1/temperatures
    ymax = 2
    x = x[y < ymax]
    y = y[y < ymax]

    # fig2, ax2 = plt.subplots()
    ax2.plot(x, y, zorder=1, ls='none', label=label)
    # print('------')
    # print(xcut)
    popt, _ = curve_fit(tools.p1, x[x < xcut], y[x < xcut])
    # print(popt)
    print('linear: ', 1/popt[0], popt[1],r2_score(y[x < xcut], tools.p1(x[x < xcut], *popt)))
    ax2.plot(x, tools.p1(x, *popt), ls='-', c='k', marker=',', zorder=2)
    ax2.set(xlabel=r'$B^{-1}$', ylabel=r'$\sigma^{*} N^{1/2}$')
    ax2.set(xlim=[x.min(), 0.00035], ylim=[y.min(), y.max()])

def calc_C2(trajectory):
    cij = np.cov(trajectory.T)
    N, _ = cij.shape
    C = np.sum(cij ** 2)
    C = C / N
    return C

def get_example_trajecotry(i_sweep):
    fpath = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/noMM.hdf5'
    with h5py.File(fpath, 'r') as fin:
        # input_trajectory = fin['configurations'][()]
        # print(fin['ChiSweep'].keys())
        # Tfs = fin['ChiSweep']['alphas'][()]
        # print(Tfs)
        # print(Tfs[i_plm], Tfs[i_cor])
        temp = fin['ChiSweep']['alphas'][i_sweep]
        print(f'loading:{i_sweep}, T={temp}')
        example_trajecotry = fin['ChiSweep']['trajectories'][i_sweep, 0, :, :]

        # plm_trajectories = fin['ChiSweep']['trajectories'][i_plm, 0, :, :]
        # cor_trajectories = fin['ChiSweep']['trajectories'][i_cor, 0, :, :]
        # print(input_trajectory.shape, plm_trajectories.shape, cor_trajectories.shape)
    return example_trajecotry

# def reshape_trjactorie():
# def KajimuraShowCij(save=False):
#     dataroot = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/'
#     model_files = [
#         'noMM.hdf5',
#         # 'noMM_matched_subsamples/noMM_subsample0.hdf5',
#         # 'noMM_matched_subsamples/noMM_subsample1.hdf5',
#         # 'MM.hdf5'
#     ]

#     fpath = dataroot + model_files[0]
#     print(fpath)
#     with h5py.File(fpath, 'r') as fin:
#         trajectory = fin['configurations'][()]
#     # Cij = np.cov(trajectory.T)
#     Cij = np.corrcoef(trajectory.T)
#     np.fill_diagonal(Cij, 0)

#     plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-2col.mplstyle')
#     fig, ax = mkfigure()
#     mat = ax.matshow(Cij, cmap='cividis')
#     ax.xaxis.tick_bottom()
#     # ax.imshow(choice_cor[0], interpolation="none")

#     ax.axhline(358.5, c='k', marker=',')
#     # ax.axvline(0, c='k', marker=',')

#     # ax.axhline(0, c='k', marker=',')
#     ax.axvline(358.5, c='k', marker=',')

#     ax.axhline(359/2, c='k', marker=',', ls='--')
#     ax.axvline(359/2, c='k', marker=',', ls='--')
#     ax.set(xlabel=r'$i$', ylabel=r'$j$')
#     cbar = plt.colorbar(mat)
#     cbar.set_label(r'$C_{ij}$', rotation=360)

#     if save is True:
#         plt.savefig(os.path.join(FIGDIR, 'kajimura-modshow-cij.png'))
#     plt.show()
