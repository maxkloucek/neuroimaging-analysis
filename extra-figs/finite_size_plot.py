import numpy as np
import h5py
import matplotlib.pyplot as plt


def m_trajectory(repeat_trajectories):
    nR, B, N = repeat_trajectories.shape
    ms = np.zeros(nR)
    chis = np.zeros(nR)
    for iR in range(0, nR):
        traj = repeat_trajectories[iR, :, :]
        m_traj = np.abs(np.sum(traj, axis=1))
        # print(traj.shape, m_traj.shape)
        ms[iR] = np.mean(m_traj / N)
        chis[iR] = np.var(m_traj) / N
    return np.mean(ms), np.mean(chis)


file = './2DIsing.hdf5'
plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')


group_base = 'finite_size_scaling_L'
Ls = [4, 8, 16, 32]
linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, sharex=True)
ax[0, 0].axvline(2.2691852808, c='k', marker=',')
ax[1, 0].axvline(2.2691852808, c='k', marker=',')
for L, ls in zip(Ls, linestyles):
    group = group_base + str(L)
    with h5py.File(file, 'r') as fin:
        print(fin.keys())
        print(fin[group].keys())
        # loading temp sweep data
        temps = fin[group + '/sweep-temps'][()]
        sweep_trajectories = fin[group + '/sweep-trajectories'][()]

    ms = np.zeros(temps.size)
    chis = np.zeros(temps.size)
    for i in range(0, temps.size):
        m, chi = m_trajectory(sweep_trajectories[0, i, :, :, :])
        ms[i] = m
        chis[i] = chi
    chis = (chis/temps) * 25

    ax[0, 0].plot(temps, ms, marker=',', ls=ls, label=L**2)
    ax[1, 0].plot(temps, chis, marker=',', ls=ls)

ax[0, 0].legend()
ax[0, 0].set(ylabel=r'$|m|$')
ax[1, 0].set(ylabel=r'$\chi$', xlabel=r'$T$', xlim=[1.0, 5.0], ylim=[1, None])

plt.tight_layout()
plt.show()
