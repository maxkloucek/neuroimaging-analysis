import h5py
import numpy as np
import matplotlib.pyplot as plt
from figures import analysis
from figures import wrangle

def sweep_plot(save=False):
    # file = '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5'
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    # file = '/Volumes/IT047719/HCP_analyised/HCP_rsfmri_added_data.hdf5'
    group = 'grouped'
    trajecotries = analysis.SweepAnalysis(file, group, ['m', 'q', 'C2'])
    # trajecotries.calculate_observables(0)
    obs = trajecotries.load_analysed_data()
    obs = obs.groupby(['alpha'], as_index=True).mean()
    obs = obs.reset_index()
    
    print(obs)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=False)
    print(ax.shape)
    x = obs['alpha']
    # anyway the conclusion is no grithiffs; that's good to know though!
    # x = np.arange(0, len(obs['alpha']))
    ax[0, 0].plot(x, obs['m'], ls='none')
    ax[0, 0].plot(x, obs['q'], ls='none')
    ax[1, 0].plot(x, obs['C2'], ls='none')
    plt.show()


def autocorrelations_set_temp():
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    group = 'grouped'
    iT = 27
    # overview of a few temps
    with h5py.File(file, 'r') as fin:
        parameter = fin[group]['sweep-alphas'][iT]
        trajectory_ds = fin[group]['sweep-trajectories']
        nMods, nParameters, nReps, B, N = trajectory_ds.shape
        # let's just pick some...
        trajectories = trajectory_ds[0, iT, :, :, :]
    # let's make a plot for every temperature instead!!
    print(iT, trajectories.shape)
    fig, ax = plt.subplots()
    cols = plt.get_cmap('plasma')(np.linspace(0, 1, len(trajectories)))
    ac_array = np.zeros((12, 9999))
    for i in range(0, len(trajectories)):
        traj = trajectories[i, :, :]
        ac, tau = wrangle.overlap(traj)
        print(i, parameter, tau)
        ac_array[i, :] = ac[1:]
        ax.plot(ac[1:], label=f'{parameter:.3f}', marker=',', c=cols[i])
    ax.plot(np.mean(ac_array, axis=0), marker=',', c='k', ls='--')
    plt.show()

def _calc_ac_over_repeats(trajectories):
    # print(trajectories.shape[1] - 1)
    ac_array = np.zeros((trajectories.shape[0], trajectories.shape[1] - 1))
    for i in range(0, len(trajectories)):
        traj = trajectories[i, :, :]
        ac, tau = wrangle.overlap(traj)
        ac_array[i, :] = ac[1:]
    return np.mean(ac_array, axis=0)

def autocorrelations_range_temp():
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    group = 'grouped'
    # iT = 27
    T_s = 22
    T_e = 40

    # overview of a few temps
    with h5py.File(file, 'r') as fin:
        parameters = fin[group]['sweep-alphas'][T_s:T_e]
        trajectory_ds = fin[group]['sweep-trajectories']
        nMods, nParameters, nReps, B, N = trajectory_ds.shape
        # let's just pick some...
        trajectories = trajectory_ds[0, T_s:T_e, :, :, :]
    print(trajectories.shape)
    nT, nR, B, N = trajectories.shape
    # maybe not plasma, maybe a diverging CM!
    # plasma, coolwarm
    cols = plt.get_cmap('plasma')(np.linspace(0, 1, nT))
    # acs = np.zeros((nT, B-1))
    fig, ax = plt.subplots()
    for iT in range(0, nT):
        ac = _calc_ac_over_repeats(trajectories[iT])
        print(parameters[iT], ac.shape)
        ax.plot(ac, label=f'{parameters[iT]:.2f}', marker=',', c=cols[iT])
    ax.set(xscale='log', ylabel=r'$C(\Delta t)$', xlabel=r'$\Delta t$', ylim=[-0.01, None])
    ax.legend(title=r'$T_f$', framealpha=1, fontsize=9, ncol=3, loc='upper left')
    plt.show()

def autocorrelations_hcp():
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    group = 'grouped'
    with h5py.File(file, 'r') as fin:
        print(fin.keys())
        print(fin['individuals'].keys())
        print(fin['individuals']['configurations'])
        trajectories = fin['individuals']['configurations'][:]

    ac = _calc_ac_over_repeats(trajectories)
    print(ac.shape)
    fig, ax = plt.subplots()
    ax.plot(ac, marker=',', ls='-', c='k')
    ax.set(xscale='log', ylabel=r'$C(\Delta t)$', xlabel=r'$\Delta t$', ylim=[-0.01, None])
   
    plt.show()

# CONCLUSION NO GRIFFITHS IN THE UNTHREHSOLDED MODEL!
# sweep_plot()
# autocorrelations_set_temp()
# autocorrelations_range_temp()
# autocorrelations_hcp()  # of course it's got the wierd shift, I have these plots already!!