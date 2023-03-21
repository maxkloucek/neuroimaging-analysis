import numpy as np
import matplotlib.pyplot as plt
from figures import io
from figures import wrangle

plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
iD = 55
time_series = io.get_raw_data(iD)
z_series, spin_series = wrangle.binarize(time_series)

i_choice = 0
t_lim = 51
z = z_series[0:t_lim, i_choice]
s = spin_series[0: t_lim, i_choice]
# print(z_series.shape)
# exit()
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, sharex=False)
ax[0, 0].plot(z, marker=',', label=r'signal')
# xs = np.arange(0, 10)
z_y = z
z_x = np.arange(0, z.size)
# y_spins = spin_series[0:10, i_choice]
s_y = s
s_x = np.zeros_like(s)
q = ax[0, 0].quiver(z_x, z_y, s_x, s_y, pivot='mid', zorder=50)
# ax[0, 0].quiverkey(
#         q, X=0.3, Y=1.1, U=10,
#         label='spin', labelpos='E')
ax[0, 0].axhline(0, marker=',', ls='-', c='tab:orange', label='threshold')
ax[0, 0].legend()
ax[0, 0].set(ylabel=r'Activity ($i=0$)', xlabel='time')
plt.show()

binary_map = np.zeros((1, s.size))
for j in range(0, 1):
    binary_map[j, :] = s

fig, ax = plt.subplots()
ax.imshow(binary_map, cmap='Greys')
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
ax.set_xticks([], [])
ax.set_yticks([], [])
for i in range(0, t_lim):
    ax.axvline(i-0.5, ls='-', marker=',', c='grey', lw=1, zorder=500)
ax.set(xlabel=r'$\{ \boldsymbol{s}(t)\}_{t=1}^{B}$', ylabel=r'$s_{i}$')
plt.show()
# lets try plotting a binarized map?
