import h5py
import numpy as np
import matplotlib.pyplot as plt


def magnetisation(trajectory):
    m_traj = np.mean(trajectory, axis=1)
    m = np.mean(m_traj)
    print(m_traj.shape)
    return m


plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
# # file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
# file = './th_sweep.hdf5'
# with h5py.File(file, 'r') as fin:
#     # print(fin.keys())
#     print(fin['test2_sym'].keys())
#     # print(fin['grouped']['sweepTH_symmetric'].keys())
#     # let's take the lowest temp and plot the end of the trajecotry
#     # for the limits delta0 and delta-1
#     group = fin['test2_sym']
#     # group = fin['grouped']['sweepTH_symmetric']
#     print(group['sweep-alphas'].shape)
#     print(group['sweep-trajectories'].shape)
#     # delta_start = group['sweep-alphas'][0]
#     traj_start = group['sweep-trajectories'][0, 0, -1, -1000:, :]
#     # delta_end = group['sweep-alphas'][-1]
#     traj_end = group['sweep-trajectories'][13, 0, -1, -1000:, :]

# # print(group['trajectories'].shape)
# # print(delta_start, delta_end)
# m_start = magnetisation(traj_start)
# m_end = magnetisation(traj_end)
# # ohhh should I have calculated the absolute value before...
# #  yes I think averageing is destroying my m!
# print(m_start, m_end)
# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=False)
# ax[0, 0].matshow(traj_start.T, cmap='Greys')
# ax[1, 0].matshow(traj_end.T, cmap='Greys')
# plt.show()

file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
with h5py.File(file, 'r') as fin:
    print(fin.keys())
    print(fin['grouped'].keys())
    print(fin['grouped']['configurations'].shape)
    data = fin['grouped']['configurations'][0, :, :]
Cij = np.corrcoef(data.T)
np.fill_diagonal(Cij, 0)
fig, ax = plt.subplots()
im = ax.matshow(Cij, cmap='cividis')
# ax.set(xlabel=r'$j$', ylabel=r'$i$')
fig.colorbar(im, label=r'$R_{ij}$')
N = 360
nticks = 3
ax.xaxis.tick_bottom()
ax.xaxis.set_major_locator(plt.FixedLocator(np.linspace(0, N, nticks)))
ax.yaxis.set_major_locator(plt.FixedLocator(np.linspace(0, N, nticks)))
ax.set(
    xlim=[ax.get_xticks()[0], ax.get_xticks()[-1]],
    ylim=[ax.get_yticks()[-1], ax.get_yticks()[0]],
    xlabel='j',
    ylabel='i'
    )
plt.show()
