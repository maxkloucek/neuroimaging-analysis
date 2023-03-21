import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import r2_score
from pyplm.utilities import tools
from pyplm import utilities
import seaborn as sns

from scipy.ndimage import gaussian_filter1d

import jackknife_helpers as helpers


def plot_histograms(ax, models, bins=None, colors='default', labels=None):
    if colors == 'default':
        colors = ['k']
        [
            colors.append(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']
        ]
    if labels == None:
        labels = ['true', 'inferred', 'cor-SC', 'cor-JK']
    if np.all(bins)==None:
        bins = np.linspace(models.min(), models.max(), 50)

    for i, mod in enumerate(models):
        params = tools.triu_flat(mod, k=1)
        n, x = np.histogram(params, bins, density=True)
        x = x[:-1]
        n = gaussian_filter1d(n, sigma=2)
        ax.plot(
            x, n,
            marker=',', ls='-', lw=2,
            c=colors[i],
            label=labels[i], zorder=10)
        # sns.kdeplot(params, marker=',', label=labels[i], ax=ax, c=colors[i], lw=2, ls='--')


def plot_correlations(ax, models, k=1, a = 0.25):

    ax.plot(tools.triu_flat(models[0], k=k), tools.triu_flat(models[0], k=k), marker=',', c='k')
    ax.plot(tools.triu_flat(models[0], k=k), tools.triu_flat(models[1], k=k), ls='none', alpha=a)
    ax.plot(tools.triu_flat(models[0], k=k), tools.triu_flat(models[2], k=k), ls='none', alpha=a)
    ax.plot(tools.triu_flat(models[0], k=k), tools.triu_flat(models[3], k=k), ls='none', alpha=a)
    # ax.set(ylim=[-2.75, 2.75])

    print(r2_score(tools.triu_flat(models[0], k=k), tools.triu_flat(models[1], k=k)))
    print(r2_score(tools.triu_flat(models[0], k=k), tools.triu_flat(models[2], k=k)))
    print(r2_score(tools.triu_flat(models[0], k=k), tools.triu_flat(models[3], k=k)))

plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
FIGDIR = '/Users/mk14423/Documents/tempfigs'
save = False
grps = ['SK_N24_T1.5', 'SK_N48_T1.5']
new_file_grps=[
    # 'SK_N50', 'SK_N100', 'SK_N50_B1500_T1.2',
    'SK_N50_B750_T2.0', 'SK_N50_B1500_T2.0', 'SK_N50_B3000_T2.0']


'''
# multiple models!
fig, ax = plt.subplots(nrows=2, ncols=2)
colors = ['k']
[colors.append(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
labels = ['true', 'inferred', 'cor-SC', 'cor-JK']

models, B = helpers.get_models(group=new_file_grps[3])
plot_histograms(ax[0, 0], models, colors=colors, labels=labels)
plot_correlations(ax[0, 1], models)

models, B = helpers.get_models(group=new_file_grps[4])
plot_histograms(ax[1, 0], models, colors=colors, labels=labels)
plot_correlations(ax[1, 1], models)
ax.legend()
ax[0, 0].legend()
if save is True:
    plt.savefig(os.path.join(FIGDIR, 'sigma_vs_B.png'))
plt.show()
'''




# example: (low B=750)
models, B = helpers.get_models(group=new_file_grps[0])
labels = ['true', 'PLM', 'cor-SC', 'cor-JK']
print(B)
on = True
if on == True:
    # distributions
    fig, ax = plt.subplots()
    plot_histograms(ax, models)
    ax.legend()
    ax.set(xlabel=r'$J_{ij}$', ylabel=r'$PDF$')
    plt.show()

on = False
if on == True:
    # imshows
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.ravel()
    for iA in range(0, ax.size):
        im = ax[iA].matshow(models[iA])
        # fig.colorbar(im, ax=ax)
        ax[iA].set(title=labels[iA])
    plt.show()

on = False
if on == True:
    # difference models and distributions
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.ravel()
    # vmin = models[0].min()
    # vmax = models[0].max()
    vmin = -0.05
    vmax = +0.05
    # let's do an difference distribution?
    diffs = []
    print('-----')
    for iA in range(0, 3):
        diff = models[iA + 1] - models[0]
        im = ax[iA].matshow(diff, vmin=vmin, vmax=vmax)
        ax[iA].set(title=labels[iA + 1])
        diffs.append(diff)
        print(np.mean(diff), np.std(diff))
    print('-----')
    fig.colorbar(im, ax=ax)
    diffs = np.array(diffs)
    cs = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_histograms(ax=ax[-1], models=diffs, colors=cs)
    plt.show()

on = False
if on == True:
    # let's compare how the sc and jk corrections change the model..
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    vmin = 1.07353515625 - (1.07353515625 * 0.5)
    vmax = 1.07353515625 + (1.07353515625 * 0.5)
    # let's set some vmax and vmin stuff to see if they're really similar
    # or whatever... so far I'm a little concerned that something isn't right..
    # matrix comparision.
    print(np.mean(models[1]/models[2]))
    print(np.mean(models[1]/models[3]))
    ax[0].matshow(models[1]/models[3], vmin=vmin, vmax=vmax)
    im = ax[1].matshow(models[1]/models[2], vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[1])
    plt.show()

on = True
if on == True:
    fig, ax = plt.subplots()
    iMods = [2]
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # cols 
    for i in iMods:
        # plt.matshow(models[3])
        # plt.show()
        models, B = helpers.get_models(group=new_file_grps[i])
        models = np.abs(models)
        input_params = tools.triu_flat(models[0], k=1)
        plm_params = tools.triu_flat(models[1], k=1)
        sc_params = tools.triu_flat(models[2], k=1)
        jk_params = tools.triu_flat(models[3], k=1)

        jk_percentage_cor = (jk_params - plm_params) / plm_params
        sc_percentage_cor = (sc_params - plm_params) / plm_params
        # print(np.mean(selfcon_percentage_cor))
        print(1 / (np.mean(sc_percentage_cor) + 1))
        line, = ax.plot(
            (input_params), jk_percentage_cor,
            ls='none', zorder=2, marker='o', c=cols[2],
            # label=f'JK, B={B}'
            label='Jackknife'
            )
        # B={B}: 
        line, = ax.plot(
            (input_params), sc_percentage_cor,
            ls='-', marker=',', c=line.get_color(), zorder=10, lw=2,
            # label=f'SC, B={B}',
            label='Self-consistency'
            )
        
        # ax.axhline(selfcon_percentage_cor, ls='-', marker=',', c=line.get_color(), lw=2, zorder=10)

        # let's do the absolute correction!
    ax.set(
        xlabel=r'$| J_{ij} ^{0} |$',
        ylabel=r'$\Delta | J_{ij}|$',
        # ylim=[-0.15, +0.05]
        # ylim=[-0.075, 0.025]
        ylim=[-0.15, 0.15]
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    iMods = [0, 1, 2]
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ci = 0
    bins_both = [
        np.linspace(-0.15, 0, 200),
        np.linspace(-0.15, 0, 200),
        np.linspace(-0.075, 0.025, 200),
    ]
    for i in iMods:
        models, B = helpers.get_models(group=new_file_grps[i])
        models = np.abs(models)

        sc_percentage_cor = (models[2] - models[1]) / models[1]
        jk_percentage_cor = (models[3] - models[1]) / models[1]

        plot_histograms(
            ax, models=np.array([jk_percentage_cor]),
            bins=bins_both[ci],
            labels=[f'B={B}'], colors=[None],
            )
        ax.axvline(np.mean(sc_percentage_cor), ls='-', marker=',', c=cols[ci], lw=2)
        ci+=1
    ax.legend()
    ax.set(xlabel=r'$\Delta |J_{ij}|$', ylabel=r'$PDF$')
    plt.show()
