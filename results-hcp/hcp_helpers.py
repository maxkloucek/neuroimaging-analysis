import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d

from pyplm.plotting import mkfigure
from pyplm.utilities import tools
from inference.scripts.paper2022May import load
from figures import analysis

# COLORS_CAT = plt.rcParams['axes.prop_cycle'].by_key()['color']
# FIGDIR = '/Users/mk14423/Documents/tempfigs'


def load_grouped_model(
        file='/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'):
    with h5py.File(file, 'r') as f:
        print(f.keys())
        g = f['grouped']
        print(g.keys())
        print(g['configurations'].shape)
        plm_model = g['inferredModels'][0, :, :]
        # cor_model = plm_model / f['correction'][()][0]
        # print(1/ np.std(cor_model), 1/ np.std(plm_model))
        # f['InferredModel'][()]
    return plm_model


def load_noMM_model():
    file = '/Users/mk14423/Desktop/PaperData/Kajimura_analysis/noMM.hdf5'
    with h5py.File(file, 'r') as f:
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
    line, = ax.plot(
        x, n,
        **pltargs,
    )
    return n, x, line

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

def get_example_trajecotry(i_sweep, i_rep=0):
    fpath ='/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    with h5py.File(fpath, 'r') as fin:
        g = fin['grouped']
        # print(g.keys())
        # print(g['sweep-trajectories'].shape)
        # print(g['sweep-alphas'][()])
        temp = g['sweep-alphas'][i_sweep]
        print(f'loading:{i_sweep}, T={temp}')
        # print(g['sweep-trajectories'].shape)
        example_trajecotry = g['sweep-trajectories'][0, i_sweep, i_rep, :, :]

    return example_trajecotry, temp



def get_example_threshold_trajecotries(i_th, i_rep = 0):
    fpath ='/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    with h5py.File(fpath, 'r') as fin:
        g = fin['grouped']
        # print(g.keys())
        sym_grp = g['sweepTH_symmetric']
        pos_grp = g['sweepTH_positive']
        # print(sym_grp.keys())
        # print(pos_grp.keys())
        # sym_deltas = sym_grp['parameters'][()] # these are the same i'm sure of it!
        # pos_deltas = pos_grp['parameters'][()]
        # print(sym_grp['trajectories'].shape)
        deltas = sym_grp['parameters'][()]
        th = deltas[i_th]
        print(i_th, th)
        sym_traj = sym_grp['trajectories'][i_th, i_rep, :, :]
        pos_traj = pos_grp['trajectories'][i_th, i_rep, :, :]

        base_mod = g['inferredModels'][0, :, :]
        sym_mod = np.copy(base_mod)
        sym_mod[np.abs(sym_mod) <= th] = 0   

        pos_mod = np.copy(base_mod)
        pos_mod[pos_mod <= th] = 0

    return deltas, sym_mod, sym_traj, pos_mod, pos_traj

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

def configuration_similarity_matrix(trajectory):
    # I need to define some "distance" which I minimize...
    # where the distance depends on the vectors!
    B, N = trajectory.shape

    # test_times = np.arange(0, 7) * 100
    # fig, ax = plt.subplots()
    # times = np.arange(0, B)
    # for t in test_times:
    #     dp_trajectory = cs_dot_prod_traj(trajectory, t)
    #     dp_plot_trajecotry = gaussian_filter1d(dp_trajectory, sigma=2)
    #     ax.plot(times, dp_plot_trajecotry, marker=',', label=t)
    # ax.legend()
    # plt.show()
    
    # n points by thingies...?
    # how do I interperet this...?
    # and are these states that I visit the same...?
    dot_matrix = np.zeros((B, B))
    for t in range(0, B):
        dot_trajectory = cs_dot_prod_traj(trajectory, t)
        dot_matrix[t, :] = dot_trajectory
    return dot_matrix


def configuration_similarity_running_avrg(trajectory, window_length=2):
    B, N = trajectory.shape
    print(B, N)
    # let's do summs instead of means!
    # let's see what q would be for 1 sample
    window_traj = sliding_window_view(trajectory, window_shape=window_length, axis=0)
    print(window_traj.shape)
    si_avrg = np.mean(window_traj, axis=2)
    si_avrg_sqr = si_avrg ** 2
    Qs = np.sum(si_avrg_sqr, axis=1)

    # Ok i've learend ms do not characterise.
    # Ms = np.mean(window_traj, axis=2)
    # Ms = np.sum(Ms, axis=1)
    # Ms = np.sum(trajectory, axis=1)
    # print(Ms.shape)

    return Qs


def configuration_similarity_local_dot(trajectory):
    B, N = trajectory.shape
    print(B, N)
    # this is interesting I think!!
    local_overlaps = np.zeros(B - 1)
    for t in range(0, B - 1):
        local_dot = np.dot(trajectory[t, :], trajectory[t + 1, :])
        local_overlaps[t] = local_dot

    # let's do summs instead of means!
    # let's see what q would be for 1 sample
    # window_traj = sliding_window_view(trajectory, window_shape=window_length, axis=0)
    # print(window_traj.shape)
    # si_avrg = np.mean(window_traj, axis=2)
    # si_avrg_sqr = si_avrg ** 2
    # Qs = np.sum(si_avrg_sqr, axis=1)


    # Ok i've learend ms do not characterise.
    # Ms = np.mean(window_traj, axis=2)
    # Ms = np.sum(Ms, axis=1)
    # Ms = np.sum(trajectory, axis=1)
    # print(Ms.shape)

    return local_overlaps
    

def cs_dot_prod_traj(trajectory, t_choice):
    B, N = trajectory.shape
    dp = np.zeros(B)
    for t in range(0, B):
        dp[t] = np.dot(trajectory[t_choice, :], trajectory[t, :])
    dp = dp / N
    return dp