import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plots
from sklearn.metrics import r2_score

from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d

from pyplm.utilities import tools
import subplots
# from pyplm.pipelines import data_pipeline
# plm_pipeline = data_pipeline(file, group)
# plm_pipeline.subsample(no_ss_points=100)
# ok ok what am I doing today...
# I want to take the plot of Btilde vs thingy and do something of meaning with it!

plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
file = '/Users/mk14423/Desktop/Data/0_thesis/SubSampleSK/datasets.hdf5'
groups = ['N50', 'N100', 'N200', 'N400', 'N800']
g_labels = ['N = 50', 'N = 100', 'N = 200', 'N = 400', 'N = 800']
emin = np.array([0.14320461, 0.20949111, 0.29832886, 0.44711299, 0.69887905])
Btilde_linextrap = np.array([0.50661207, 0.74111249, 1.055392, 1.58174263, 2.47241035])
T_trues = [1.1, 1.175, 1.1, 1.1, 1.25]
Ns = np.array([50, 100, 200, 400, 800])
Btilde_linextrap *= 1e3
group = 'N800'
save = False
FIGDIR = '/Users/mk14423/Dropbox/Apps/Overleaf/thesis/thesis/1-results-sk-plm/figures/varying-N'


# let's normalize the tempeatures!
def get_subsampling_data_frame(file, group, T_true):
    df = pd.read_hdf(file, group + '/subsampling')
    df['mean_J'] = df['mean_J'] * df['N']         # rescale by N
    df['std_J'] = df['std_J'] * (df['N'] ** 0.5)
    df['mu'] = df['mean_J'] / df['std_J']
    df['T'] = 1 / df['std_J']
    # NORMALIZING T
    df['T'] = df['T'] / T_true
    df = df.sort_values(by=['B', 'iD'])
    return df

''' ------------------------------------------------------------- '''
# subplots.error_test()
# fig, ax = plt.subplots()
# subplots.error_scaling_explanation(ax)
# plt.show()
''' ------------------------------------------------------------- '''
''' ------------------------------------------------------------- '''
# Invese B vs bias plots
# fig, ax = plt.subplots()
# fitting_params = np.zeros((len(groups), 3))
# for iG in range(0, len(groups)):
#     print(iG, groups[iG])
#     print(T_trues[iG])
#     df = get_subsampling_data_frame(file, groups[iG], T_trues[iG])
#     df['std_J'] = df['std_J'] * T_trues[iG]
#     rescale = 1
#     df['B'] = df['B'] * rescale
#     df = df[df['B'] > 3e3 * rescale]
#     df = df.groupby(['B'], as_index=True).mean()
#     df = df.reset_index()
#     xfit = 1/np.linspace(df['B'].min(), df['B'].max(), 100)
#     ax.plot(1 / df['B'], df['std_J'], ls='none', label=f'N={Ns[iG]}')
#     df = df[df['B'] > 8e3 * rescale]
#     subplots.add_fit_to_ax(
#         ax, 1 / df['B'], df['std_J'], tools.linear, xfit=xfit, show_error=False,
#         marker=',', c='k', ls='-')
# ax.axvline(1/(8e3 * rescale), marker=',', ls='--', c='k')
# ax.axvspan(xfit[-1],1/(8e3 * rescale), fc='grey', alpha=0.3)
# ax.set(xlabel=r'$1/B (\times 10^4)$', ylabel=r'$\sigma^{*} / \sigma^{0}$')
# ax.set(
#     # xlim=[1, 3.2],
#     # ylim=[1, 2]
# )
# plt.legend(loc='upper left')
# plt.show()
''' ------------------------------------------------------------- '''
''' ------------------------------------------------------------- '''
# # Invese graident plot -> left this unfinished!
# fig, ax = plt.subplots()
# fitting_params = np.zeros((len(groups), 3))
# for iG in range(0, len(groups)):
#     print(iG, groups[iG])
#     print(T_trues[iG])
#     df = get_subsampling_data_frame(file, groups[iG], T_trues[iG])
#     # df['std_J'] = df['std_J'] * T_trues[iG]
#     rescale = 1e-4
#     # rescale = 1
#     df['B'] = df['B'] * rescale
#     df = df[df['B'] > 3e3 * rescale]
#     df = df.groupby(['B'], as_index=True).mean()
#     df = df.reset_index()
#     # ax.plot(1 / df['B'], df['std_J'], ls='none', label=f'N={Ns[iG]}')
#     # print(np.gradient(df['std_J']))
#     grad = np.gradient(df['std_J'])
#     # line, = ax.plot(1 / df['B'], grad, ls='none', alpha=0.1, zorder=1)
#     # c = line.get_color()
#     # grad = uniform_filter1d(grad, size=25)
#     # grad = np.gradient(grad)
#     grad = uniform_filter1d(grad, size=25)
#     # grad /= grad[-1]
#     ax.plot(1 / df['B'], grad, ls='-', marker=',', zorder=10, lw='2', label=f'N={Ns[iG]}')
#     # xfit = 1/np.linspace(df['B'].min(), df['B'].max(), 100)
#     # df = df[df['B'] > 8e3 * rescale]
#     # subplots.add_fit_to_ax(
#     #     ax, 1 / df['B'], df['std_J'], tools.linear, xfit=xfit, show_error=False,
#     #     marker=',', c='k', ls='-')
# # ax.axvline(1/(8e3 * rescale), marker=',', ls='--', c='k')
# # ax.axvspan(xfit[-1],1/(8e3 * rescale), fc='grey', alpha=0.3)
# ax.set(
#     xlabel=r'$1/B [\times 10^4]$',
#     ylabel=r'$\partial \sigma ^{*} / \partial (1/B)$')
# ax.set(
# #     xlim=[1, 3.2],
#     ylim=[-0.00137, 0]
#     # ylim=[-0.0002, 0.0002]
#     )
# plt.legend(loc='best')
# plt.show()
''' ------------------------------------------------------------- '''

fig, ax = plt.subplots()
fitting_params = np.zeros((len(groups), 3))
for iG in range(0, len(groups)):
    print(iG, groups[iG])
    df = get_subsampling_data_frame(file, groups[iG], T_trues[iG])
    popts = plots.subsampling_plot(ax, df, 5e3, iG, g_labels[iG])
    # ax.axhline(popts[0], ls='-', c='k', marker=',')
    fitting_params[iG, :] = popts
ax.set(ylim=[0.5, 1.0], xlabel=r'$B$', ylabel=r'$T^{*} / T^{0}$')
# ax.axhline(1, ls='-', c='k', marker=',')
print(fitting_params)
plt.legend()
if save is True:
    plt.savefig(os.path.join(FIGDIR, 'N-saturation-curves.png'))
plt.show()
''' ------------------------------------------------------------- '''


# I might want to use this for my other thingy as well?!?
plots.N_emin_Btilde(Ns, emin, fitting_params[:, 1], save=False)

# plt.close()

# OK LET'S RUN THE SUBSAMPLING ON MY GOOD BIG DATASET!! 


# # wtf how does this make any sense?!? they arent linear anymore it's all lies!
# fig, ax = plt.subplots()
# Ns = np.array([50, 100, 200, 400, 800])
# x = emin
# x *= np.sqrt(Ns)
# y = fitting_params[:, 1]
# x = np.log10(x)
# y = np.log10(y)
# popt, _ = curve_fit(tools.linear, x, y)
# yfit = tools.linear(x, *popt)

# x = 10 ** x
# y = 10 ** y
# yfit = 10 ** yfit
# # fitting in log space to give better weight to things!
# ax.plot(x, y, c='#4053d3', ls='none', label='data')
# ax.plot(x, yfit, c='k', marker=',', label=f'linear-fit r2={r2_score(y, yfit):.3f}')
# ax.set(xlabel=r'$ \varepsilon $', ylabel=r'$\tilde{B}$')
# plt.legend()
# plt.show()