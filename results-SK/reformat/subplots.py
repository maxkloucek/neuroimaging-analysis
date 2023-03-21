import glob
import os
import re
import warnings

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from inference import tools
from inference.tools import triu_flat
# from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
# from matplotlib.axes import _subplots
from matplotlib.axes import Axes

from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from inference.tools import plot_add_SK_analytical_boundaries


COLORS_CAT = [
    '#4053d3', '#ddb310', '#b51d14', '#00beff', '#fb49b0', '#00b25d', '#cacaca']


def phase_diagram(
        fig, ax,
        parameters, observables,
        observableName,
        contour=True,
        cbar=True,
        setmax=None,
        xlabel=None,
        ylabel=None):

    data = observables[observableName]
    if observableName == 'tau':
        data[data <= 1] = 1
    if setmax is None:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        norm = Normalize(vmin, vmax)
    else:
        vmin = np.nanmin(data)
        vmax = vmin * setmax
        norm = Normalize(vmin, vmax)
    # norm = LogNorm(data.min(), data.max())

    mesh = ax.pcolormesh(
        parameters['J'], parameters['T'], data,
        shading='auto',
        # cmap='cividis',
        cmap='Purples',
        norm=norm, zorder=0
        )
    # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_locator(plt.FixedLocator([0, 1, 2]))
    ax.yaxis.set_major_locator(plt.FixedLocator(np.linspace(0.5, 2, 4)))
    if xlabel is not None:
        # ax.set_xlabel(xlabel, fontsize=10)  # r'$\mu / \sigma$'
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        # ax.set_ylabel(ylabel, fontsize=10)  # r'$T / \sigma$'
        ax.set_ylabel(ylabel)
    # ax.set(xlabel=r'$\mu$')
    # ax.set(ylabel=r'$T$')
    if contour is True:
        vmax = vmin * 1.2
        data[data >= vmax] = vmax
        xx_smooth = parameters['J']
        yy_smooth = parameters['T']
        z_smooth = gaussian_filter(data, sigma=1)
        cs = ax.contour(
            xx_smooth, yy_smooth, z_smooth, levels=3,
            colors='w',
            # colors='#cacaca',
            )
        # ax.clabel(cs, inline=True, fontsize=10)
    plot_add_SK_analytical_boundaries(
        ax, np.min(parameters['J']), np.max(parameters['J']),
        np.min(parameters['T']), np.max(parameters['T']),
    )
    if cbar is True:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        return cbar
    else:
        return None


def error_bars(
        ax, x, obs_mean, obs_err, observableName,
        setmax=None, normbyMin=False, **pltkwrds):

    if normbyMin is True:
        obs_mean[observableName] = (
            obs_mean[observableName] / np.nanmin(obs_mean[observableName]))
        obs_err[observableName] = (
            obs_err[observableName] / np.nanmin(obs_mean[observableName]))
        # Bmin = 1016.4848550789357, so extra rescaling!
        # obs_mean[observableName] *= 1016.4848
        # obs_err[observableName] *= 1016.4848
    mean = obs_mean[observableName]
    err = obs_err[observableName] / np.sqrt(21)

    if setmax is not None:
        vmin = np.nanmin(mean) * 0.95
        vmax = vmin * setmax
        # x = x[mean <= vmax]
        # err = err[mean <= vmax]
        # mean = mean[mean <= vmax]
    else:
        vmin = np.nanmin(mean) * 0.95
        vmax = np.nanmax(mean)

    if observableName == 'chiSG':
        ax.set(ylim=[vmin, vmax], yscale='linear')
    if observableName == 'e':
        ax.set(ylim=[vmin, vmax], yscale='linear')
    if observableName == 'tau':
        mean[mean <= 1] = 1
        ax.set(ylim=[vmin, vmax], yscale='linear')

    line = ax.errorbar(
        x=x,
        y=mean,
        yerr=err,
        **pltkwrds
        )
    return line


def JR_T0vsTinfr(fig, ax, parameters, full_obs, Bs, observableName):
    full_data = full_obs[observableName]
    full_data = 1 / (full_data * 200 ** 0.5)
    # convert to T:
    # full_data = full_data
    # full_data = full_data[3:, :, :]
    nB, nT, nR, = full_data.shape
    # cm = plt.cm.get_cmap('cividis')
    # cm = plt.cm.get_cmap('YlOrRd')
    cm = plt.cm.get_cmap('summer')
    # cm = plt.cm.get_cmap('plasma')
    cols = cm(np.linspace(0, 1, nB))

    for iB in range(0, nB):
        lbl = '{:d}'.format(int(Bs[iB] / 1000))
        data = full_data[iB]
        Ts = np.nanmean(data, axis=0)
        Ts_err = np.nanstd(data, axis=0) / np.sqrt(nR)
        y = Ts
        # y = np.abs(Ts - parameters['T'][iB, :]) / parameters['T'][iB, :]
        # # y = y / y.min()
        ax.errorbar(
            x=parameters['T'][iB, :],
            # y=(Ts - parameters['T'][iB, :]) / parameters['T'][iB, :],
            marker='.',
            y=y,
            yerr=Ts_err,
            c=cols[iB],
            label=lbl)
    ax.plot(parameters['T'][iB, :], parameters['T'][iB, :], c='k', marker=',')
    # ax.set(xlabel=r'$T^{0} / \sigma^{0}$', ylabel=r'$T^{*} / \sigma^{0}$')
    ax.set(xlabel=r'$T^{0}$', ylabel=r'$T^{*}$')
    ax.set(ylim=[None, 2], xlim=[0.5, 2])

    T0 = [1.2506293810412095]
    T1e3 = [.5691474769252466]
    T2e3 = [0.876363657067979]
    T1e4 = [1.1675695662497456]
    T1e5 = [1.234653597037612]
    Ts = [T1e3, T2e3, T1e4, T1e5]
    ax.legend(loc='lower right', ncol=3, fontsize=8.5)
    for iT, T in enumerate(Ts):
        ax.plot(T0, T, zorder=50, marker='X', markersize=9, c=COLORS_CAT[iT])


def JR_Tsaturation(
        fig, ax, params, full_obs, Bs, Bcut=2, Tcut=0, Bmax=None, Tmax=None):
    T_theory = params['T'][0, :]
    T_curves = []
    T_curves_std = []

    for i, B in enumerate(Bs):
        obs = full_obs[i]
        Ts_inf = 1 / (obs['infrSig'] * 200 ** 0.5)
        T_curves.append(np.mean(Ts_inf, axis=0))
        T_curves_std.append(np.std(Ts_inf, axis=0))

    T_curves = np.array(T_curves)
    T_curves_std = np.array(T_curves_std)

    Bs = Bs[Bcut:Bmax]
    T_curves = T_curves[Bcut:Bmax, Tcut:Tmax]
    T_curves_std = T_curves_std[Bcut:Bmax, Tcut:Tmax]
    T_theory = T_theory[Tcut:Tmax]
    print(Bs)
    nB, nT = T_curves.shape
    cm = plt.cm.get_cmap('cividis')
    cm = plt.cm.get_cmap('YlGnBu')
    cm = plt.cm.get_cmap('winter')
    cTs = cm(np.linspace(0, 1, nT))
    popts = []
    r2s = []
    for iT in range(0, nT):
        lbl = '{:.2f}'.format(T_theory[iT])
        y = T_curves[:, iT]
        yerr = T_curves_std[:, iT]
        ax.errorbar(
            x=Bs,
            y=y,
            yerr=yerr,
            ls='none',
            c=cTs[iT],
            label=lbl
            )
        # func, popt = tools.curve_fit1D(Bs, y, tools.arctan)
        # func, popt = tools.curve_fit1D(Bs, y, tools.tanh)
        func = tools.arctan
        # ------- #
        popt, _ = curve_fit(tools.arctan, Bs, y)
        # ------- #
        # A = (2 * T_theory[iT]) / np.pi
        # popt, _ = curve_fit(lambda x, B: func(x, A, B), Bs, y)
        # popt = np.array([A, popt[0]])
        # ------- #
        # I'M FUCKING UP MY OTHER GRAPHS BY CHANGING THIS!!
        xfit = np.linspace(Bs.min(), Bs.max(), 100)
        # xfit = np.linspace(1e3, 5e4, 100)
        yfit = func(xfit, *popt)
        # plt.plot(Bs, y)
        ax.plot(xfit, yfit, ls='--', marker=',')
        # plt.show()
        r2 = r2_score(y, func(Bs, *popt))
        ax.plot(
            xfit,
            yfit,
            marker=',',
            c=cTs[iT],
            # label=lbl
            )
        popts.append(popt)
        r2s.append(r2)
    popts = np.array(popts)
    ax.set(xlabel=r'$B$', ylabel=r'$T^{*}$')
    # ax.legend(
    #     title=r'$T^{0}$',
    #     loc='best',
    #     ncol=2,
    #     # bbox_to_anchor=(1.1, 1),
    #     prop={'size': 6})

    return T_theory, popts, np.array(r2s)


def JR_Tsaturation_normalised(ax, params, full_obs, Bs, Bcut=2, Tcut=0, Tmax=None):
    T_theory = params['T'][0, :]
    T_curves = []
    T_curves_std = []

    for i, B in enumerate(Bs):
        obs = full_obs[i]
        Ts_inf = 1 / (obs['infrSig'] * 200 ** 0.5)
        Ts_inf = Ts_inf / T_theory
        T_curves.append(np.mean(Ts_inf, axis=0))
        T_curves_std.append(np.std(Ts_inf, axis=0))

    T_curves = np.array(T_curves)
    T_curves_std = np.array(T_curves_std)

    Bs = Bs[Bcut:]
    T_curves = T_curves[Bcut:, Tcut:Tmax]
    T_curves_std = T_curves_std[Bcut:, Tcut:Tmax]
    T_theory = T_theory[Tcut:Tmax]

    print('----')
    # sample_curve_i = [6, 12, -1]
    sample_curve_i = [6, 11, -1]
    ax.set(ylim=[0.75, 1])
    # sample_curve_i = [0, 1, 2, 3]
    # ax.set(ylim=[0, None])

    T_curves = np.array([T_curves[:, i] for i in sample_curve_i]).T
    T_curves_std = np.array([T_curves_std[:, i] for i in sample_curve_i]).T
    T_theory = np.array([T_theory[i] for i in sample_curve_i]).T
    print(T_curves.shape, T_curves_std.shape, T_theory.shape)
    print('----')

    nB, nT = T_curves.shape
    cm = plt.cm.get_cmap('cividis')
    cTs = cm(np.linspace(0, 1, nT))
    popts = []
    r2s = []
    # print()
    lines = [None for _ in range(0, int(2 * nT))]
    for iT in range(0, nT):
        lbl = 'T={:.2f}'.format(T_theory[iT])
        y = T_curves[:, iT]
        yerr = T_curves_std[:, iT]
        lines[iT] = ax.errorbar(
            x=Bs,
            y=y,
            yerr=yerr,
            ls='none',
            c=cTs[iT],
            label=lbl
            )

        # ------- #
        func = tools.arctan
        popt, _ = curve_fit(tools.arctan, Bs, y)
        # ------- #
        # # A = (2 * T_theory[iT]) / np.pi
        # A = (2) / np.pi
        # popt, _ = curve_fit(lambda x, B: func(x, A, B), Bs, y)
        # popt = np.array([A, popt[0]])
        # ------- #

        xfit = np.linspace(Bs.min(), Bs.max(), 100)
        yfit = func(xfit, *popt)
        r2 = r2_score(y, func(Bs, *popt))
        lines[iT+3], = ax.plot(
            xfit,
            yfit,
            marker=',',
            c=cTs[iT],
            label='Arctan fit; ' + 'R2=' + f'{r2:.3f}'
            )
        popts.append(popt)
        r2s.append(r2)
    print(lines)
    popts = np.array(popts)
    ax.set(xlabel=r'$B$', ylabel=r'$T^{*} / T^{0}$')
    ax.legend(handles=lines, ncol=2, loc='lower right')
    # lines = [lines[0], lines[1], lines[2], lines[3]]
    # ax.legend(handles=lines, ncol=1, loc='upper left')
    return 1 / popts[:, 1], T_theory, cTs


def discrete_cbar_label(fig, ax, line_colors, line_labels):
    # cmap = (mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
    #         .with_extremes(over='0.25', under='0.75'))
    cmap = mpl.colors.ListedColormap(list(line_colors))

    # bounds = [1, 2, 4, 7, 8]
    print(line_labels.size)
    # bounds = np.arange(0, line_labels.size) + 1
    bounds = line_labels
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        # boundaries=[0] + bounds + [13],  # Adding values for extensions.
        # extend='both',
        # ticks=bounds,
        ticks=line_labels,
        spacing='proportional',
        # orientation='horizontal',
        label='Discrete intervals, some other units',
    )


def JR_Tstauration_fitparamscaling(ax, Ts, popts, r2s, setmax=None):

    # ax.ylabel.fontsize = 10

    a_array = popts[:, 0]
    b_array = 1 / popts[:, 1]
    a_array = a_array * (np.pi / 2)  # T_B->inf
    r2cut = 0.90

    Ts = Ts[r2s >= r2cut]
    a_array = a_array[r2s >= r2cut]
    b_array = b_array[r2s >= r2cut]
    b_array = b_array / 1e3
    print('----')
    print(r2s)
    print(Ts)
    print('----')
    if setmax is not None:
        vmin = np.nanmin(b_array) * 0.95
        vmax = vmin * setmax
        # x = x[mean <= vmax]
        # err = err[mean <= vmax]
        # mean = mean[mean <= vmax]
    else:
        vmin = np.nanmin(b_array) * 0.95
        vmax = np.nanmax(b_array)

    line, = ax.plot(Ts, b_array, label=r'$\tilde{B}$', c=COLORS_CAT[4])
    ax.set(
        xlabel=r'$T$',
        ylabel=r'$\tilde{B}$', ylim=[vmin, vmax])
    return line


def distribution(ax, parameters, nbins, invert=False, **pltargs):
    n, bin_edges = np.histogram(parameters, bins=nbins, density=True)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    bin_centers = bin_centers[n > 0]
    n = n[n > 0]
    # decide on this later...
    n = n * (bin_edges[1] - bin_edges[0])
    ax.plot(
        bin_centers, n,
        # marker=',',
        **pltargs,
    )
    print(f'prob_sum = {np.sum(n):.3}')
    if invert is True:
        ax.invert_xaxis()

    return n, bin_centers


def loglog2side(axLeft, axRight, parameters, nbins, **pltargs):
    # n, x = distribution(ax, parameters, nbins, **pltargs)
    n_params = parameters[parameters < 0]
    p_params = parameters[parameters >= 0]
    distribution(axLeft, -n_params, nbins, invert=True, **pltargs)
    distribution(axRight, p_params, nbins, **pltargs)
    pass


def C2T_npzplot(
        ax, obsFiles,
        npzheaders=['plm', 'cor'],
        **pltargs):
    # T_files = [files_B1e3[0], files_B2e3[0], files_B1e4[0]]
    # C_files = [files_B1e3[1], files_B2e3[1], files_B1e4[1]]
    # titles = [r'$B=1 \times 10^{3}$', r'$B=2 \times 10^{3}$', r'$B=1 \times 10^{4}$']
    # xlabels = [None, r'$T^{0}$']
    # ylabels = [r'$T^{*}$', r'$\mathcal{C}^{2}$']
    # fig, ax = plt.subplots(figsize=(6, 5))

    C2Ts = []
    for file in obsFiles:
        C2T = file['tru'][0, :]
        C2Ts.append(C2T)
    C2Ts = np.array(C2Ts)
    print(C2Ts.shape)
    C2T_mean = np.mean(C2Ts, axis=0)
    C2T_sem = np.std(C2Ts, axis=0) # / np.sqrt(3)
    print(C2T_mean.shape, C2T_sem.shape)

    # cols = plt.cm.cividis(np.linspace(0, 1, 3))
    cols = [COLORS_CAT[0], COLORS_CAT[1]]
    linestyles = ['-', '-', '-']
    markers = ['o', 's', 'd']
    # lbls = [
    #         r'$B=1 \times 10^{3}$',
    #         r'$B=2 \times 10^{3}$',
    #         r'$B=1 \times 10^{4}$'
    #     ]
    lbls = ['PLM', 'cor-SC']
    if len(obsFiles) == 1:
        markers = ['d']
    handles = []

    l1 = ax.errorbar(
        x=obsFiles[0]['true_temps'],
        y=C2T_mean,
        yerr=C2T_sem,
        color='k',
        marker=',',
        label='True')
    handles.append(l1)
    for iB, file in enumerate(obsFiles):
        for inpz, _ in enumerate(npzheaders):
            l2 = ax.errorbar(
                x=file['true_temps'],
                y=file[npzheaders[inpz]][0, :],
                yerr=file[npzheaders[inpz]][1, :],
                color=cols[inpz],
                marker=markers[iB],
                ls=linestyles[iB],
                # label=lbls[iB],
                label=lbls[inpz])
            handles.append(l2)

        # l3 = ax.errorbar(
        #     x=file['true_temps'],
        #     y=file[npzheaders[1]][0, :],
        #     yerr=file[npzheaders[1]][1, :],
        #     color=cols[1],
        #     marker=markers[iB],
        #     ls=linestyles[iB],
        #     # label=r'$C^2$-correction'
        #     label='correction'
        #     )
    # print(handles)
    # plt.show()
    # handles = [l1, l2, l3]
    return handles


def npz_funcofT(ax, file_name, header_labels, cols, labels=None):
    file = np.load(file_name)
    print(file.files)
    x = file['true_temps']
    iH = 0
    lines = []
    for header in header_labels:
        print(header, file[header].shape)
        if header == 'tru':
            args = {'marker': ',', 'c': 'k', 'label': 'true'}
        else:
            args = {'marker': 'd', 'c': cols[iH]}
            if labels is not None:
                args['label'] = labels[iH]
            else:
                args['label'] = None
            iH += 1
        # print(args)
        line = ax.errorbar(
            x=x, y=file[header][0, :], yerr=file[header][1, :], **args)
        lines.append(line)
    return lines


''' old crap'''
# should specify that these have to be strings
def parameter_correlations(
        ax_corr, Nstring, T, h, J, j):
    # ax_corr.set_title('T={}, J={}'.format(T, J))
    ax_corr.set(xlabel=r'$P_{ij}^{0}$', ylabel=r'$P_{ij}^{*}$')
    ax_dist = ax_corr.inset_axes([0, 0.6, 0.4, 0.4])
    ax_dist.tick_params(
        axis='both', which='both', bottom=False, labelbottom=False,
        left=False, labelleft=False)

    # ax_dist.set_title('T={}, J={}'.format(T, J))
    # ax_corr = ax_dist.inset_axes([0, 0.6, 0.4, 0.4])
    # ax_corr.tick_params(
    #     axis='both', which='both', bottom=False, labelbottom=False,
    #     left=False, labelleft=False)
    sym_limit = 0.5
    run_dirs = sorted(glob.glob(Nstring))
    statepoint_string = 'T_{}-h_{}-J_{}-Jstd_{}.hdf5'.format(
        T, h, J, j)

    for counter, run_dir in enumerate(run_dirs):
        infile = os.path.join(run_dir, statepoint_string)
        with h5py.File(infile, 'r') as f:
            true_model = f['InputModel'][()]
            inferred_model = f['InferredModel'][()]
            true_params = triu_flat(true_model, k=0)
            infr_params = triu_flat(inferred_model, k=0)
        if counter == 0:
            truth = true_params
            output = infr_params
        else:
            truth = np.append(truth, true_params)
            output = np.append(output, infr_params)
    # delta_mu_sigma(truth, output)
    # distro
    bins = np.linspace(-sym_limit, sym_limit, 200)
    # bins = np.linspace(output.min(), output.max(), 1000)
    ax_dist.hist(truth, bins=bins, density=True, alpha=0.8, label='true')
    ax_dist.hist(output, bins=bins, density=True, alpha=0.8, label='inf')
    # ax_CDF = ax_dist.twinx()
    # ax_CDF.hist(
    #     truth, bins=bins, density=True, cumulative=True,
    #     histtype='step', color='tab:cyan')
    # ax_CDF.hist(
    #     output, bins=bins, density=True, cumulative=True,
    #     histtype='step', color='tab:olive')

    # residuals (I don't think this communicates any new information!)
    # I need to bin this somehow...
    # so I want to plot (y;error) = f(x;truth_size)
    # sort_indicies = np.argsort(truth)
    # sorted_truth = truth[sort_indicies]
    # sorted_output = output[sort_indicies]
    # sorted_difference = sorted_output - sorted_truth
    # # sorted_difference = abs(sorted_difference)
    # ax_dist.plot(
    # sorted_truth, sorted_difference, ls='none', mew=0, alpha=0.1)
    # # ax_dist.plot(sorted_truth, ls='none', mew=0, alpha=0.1)
    # # ax_dist.plot(sorted_output, ls='none', mew=0, alpha=0.1)
    # # ax_dist.set_ylim(truth.min(), truth.max())

    # correlations
    # ax_corr = ax_dist.inset_axes([0, 0.6, 0.4, 0.4])
    # xs = np.linspace(-sym_limit, sym_limit, 100)
    # ax_corr.set_ylim(-sym_limit, sym_limit)
    # ax_corr.set_xlim(-sym_limit, sym_limit)
    xs = np.linspace(truth.min(), truth.max(), 100)
    # ax_corr.set_ylim(truth.min(), truth.max())
    ax_corr.set_xlim(truth.min(), truth.max())
    ax_corr.plot(truth, output, ls='none', marker='.', alpha=0.1, rasterized=True)
    # hmm this might be abetter way of doing it... not for now anyway!
    # ax_corr.hexbin(truth, output, gridsize=50)
    ax_corr.plot(xs, xs, marker=',', ls='-', c='k')
    return true_model
    # pca_angle0 = pca_components(ax_corr, truth, output)
    # print(pca_angle0)


def connect(
        fig, ax_source, ax_inset, xy_source,
        inset_limLow, inset_limHigh, **pltkwrds):
    # ax_source.plot([xy_source[0]], [xy_source[1]], "o", c='tab:blue')
    con = ConnectionPatch(
        xyA=xy_source, coordsA=ax_source.transData,
        xyB=inset_limLow, coordsB=ax_inset.transAxes,
        arrowstyle="-",
        linewidth=1.5,
        **pltkwrds
        # zorder=1
    )
    fig.add_artist(con)
    con = ConnectionPatch(
        xyA=xy_source, coordsA=ax_source.transData,
        xyB=inset_limHigh, coordsB=ax_inset.transAxes,
        arrowstyle="-",
        linewidth=1.5,
        **pltkwrds
    )
    fig.add_artist(con)

'''THESE ARE WITH THE JREPEATS LOADED'''


def repJ_obscut(
        fig, ax_obs,
        params, full_obs, Bs,
        obsKwrds=['chiSG', 'tau'],
        errKwrds=['e', 'infrSig'],
        lbls=[
            r'$\chi _{SG}$', r'$\tau$',
            r'$\epsilon _{\gamma}$',
            r'$\epsilon _{\theta}$',
            r'$\epsilon _{\sigma}$'
            # r'$\Delta \sigma / \sigma ^{0}$'
            ]
        ):

    cols = plt.cm.get_cmap('cividis')(
        np.linspace(0, 1, len(obsKwrds + errKwrds)))
    ax_err = ax_obs.twinx()
    i = -1
    # i = 9  # is B1e4
    B = Bs[i]
    obs = full_obs[i]
    obs['infrSig'] = (obs['infrSig'] - obs['trueSig']) / obs['trueSig']
    obs['tau'][obs['tau'] <= 1] = 1
    # CONFIDENCE_LEVEL = 1 - 0.997
    # gauss_test = np.mean(obs['normTest'], axis=0)
    # gauss_test[gauss_test < CONFIDENCE_LEVEL] = 0
    # gauss_test[gauss_test >= CONFIDENCE_LEVEL] = 1
    gauss_test = np.ones_like(np.mean(obs['normTest'], axis=0))

    x = params['T'][0, :]
    # ax_obs.plot(x, gauss_test * 10, color='k')
    allKwrds = obs.dtype.names
    nJs, nTs = obs.shape
    means = np.zeros(nJs, dtype=obs.dtype)
    stds = np.zeros(nJs, dtype=obs.dtype)

    for kwrd in allKwrds:
        means[kwrd] = np.mean(obs[kwrd], axis=0)
        stds[kwrd] = np.std(obs[kwrd], axis=0)

    mins = []
    for kwrd in errKwrds:
        err_min = np.nanmin(means[kwrd])
        mins.append(err_min)
        means[kwrd] /= err_min
        stds[kwrd] /= err_min
    i = 0
    j = 0
    handles = []
    for kwrd in obsKwrds:
        xplot = x[gauss_test == 1]
        yplot = means[kwrd][gauss_test == 1]
        yerr = stds[kwrd][gauss_test == 1]
        eb = ax_obs.errorbar(
            x=xplot, y=yplot, yerr=yerr, c=cols[j], label=lbls[j])
        handles.append(eb)
        j += 1
    for kwrd in errKwrds:
        xplot = x[gauss_test == 1]
        yplot = means[kwrd][gauss_test == 1]
        yerr = stds[kwrd][gauss_test == 1]
        eb = ax_err.errorbar(
            x=xplot, y=yplot, yerr=yerr, c=cols[j],
            label='{0}: {1:.3f}'.format(lbls[j], mins[i]))
        ymin = np.nanmin(means[kwrd]) * 0.8
        ymax = 3 * ymin
        ax_err.set_ylim(ymin, ymax)
        handles.append(eb)
        i += 1
        j += 1
    ax_obs.legend(
        handles=handles,
        title=r'$\mathcal{O} / \mathcal{E}: \mathcal{E} ^{min}$',
        prop={'size': 10})
    ax_obs.set_ylabel(r'$\mathcal{O}$')
    ax_obs.set(yscale='log', title='B={}'.format(B))
    ax_err.set_ylabel(r'$\mathcal{E} / \mathcal{E} ^{min}$')


def repJ_Bscaling(
        fig, ax,
        params, full_obs, Bs,
        kwrd='pca',
        ):

    x = params['T'][0, :]
    cols = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(Bs)))
    Bs = Bs[2:]
    cols = cols[2:]
    full_obs = full_obs[2:, :, :]
    mins = []
    handles = []
    for i, B in enumerate(Bs):
        lbl = tools.sci_notation(B, decimal_digits=0)

        obs = full_obs[i]
        obs['infrSig'] = (obs['infrSig'] - obs['trueSig']) / obs['trueSig']
        obs['tau'][obs['tau'] <= 1] = 1
        CONFIDENCE_LEVEL = 1 - 0.997
        gauss_test = np.mean(obs['normTest'], axis=0)
        gauss_test[gauss_test < CONFIDENCE_LEVEL] = 0
        gauss_test[gauss_test >= CONFIDENCE_LEVEL] = 1
        nJs, nTs = obs.shape
        means = np.zeros(nJs, dtype=obs.dtype)
        stds = np.zeros(nJs, dtype=obs.dtype)

        means[kwrd] = np.mean(obs[kwrd], axis=0)
        stds[kwrd] = np.std(obs[kwrd], axis=0)

        err_min = np.nanmin(means[kwrd])
        # means[kwrd] /= err_min
        # stds[kwrd] /= err_min

        xplot = x[gauss_test == 1]
        yplot = means[kwrd][gauss_test == 1]
        yerr = stds[kwrd][gauss_test == 1]
        eb = ax.errorbar(
            x=xplot, y=yplot,
            yerr=yerr,
            c=cols[i],
            label=lbl)
        mins.append(err_min)
        handles.append(eb)

    ymin = 0.1
    ymax = 10
    ax.set_ylim(ymin, ymax)
    ax.legend(
        handles=handles,
        title='B',
        ncol=2,
        prop={'size': 8})
    # ax.set_ylabel(r'$\epsilon _{\theta} / \epsilon _{\theta}^{min}$')
    ax.set_ylabel(r'$\epsilon _{\theta}$')
    ax.set_xlabel(r'T')
    return Bs, mins
    # ax_obs.set_ylabel(r'$\mathcal{O}$')
    # ax_obs.set(yscale='log', title='B={}'.format(B))
    # ax_err.set_ylabel(r'$\mathcal{E} / \mathcal{E} ^{min}$')


def JR_Bscaling_Terr(
        fig, ax,
        params, full_obs, Bs,
        ):
    # x = params['T'][0, :]
    cols = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(Bs)))
    # Bs = Bs[2:]
    # cols = cols[2:]
    # full_obs = full_obs[2:, :, :]
    mins = []
    handles = []
    T_theory = params['T'][0, :]
    for i, B in enumerate(Bs):
        lbl = tools.sci_notation(B, decimal_digits=0)

        obs = full_obs[i]
        Ts_inf = 1 / (obs['infrSig'] * 200 ** 0.5)
        Terr = (Ts_inf - T_theory) / T_theory
        # print(Ts_inf.shape)
        # obs['infrSig'] = (obs['infrSig'] - obs['trueSig']) / obs['trueSig']
        obs['tau'][obs['tau'] <= 1] = 1
        CONFIDENCE_LEVEL = 1 - 0.997
        gauss_test = np.mean(obs['normTest'], axis=0)
        gauss_test[gauss_test < CONFIDENCE_LEVEL] = 0
        gauss_test[gauss_test >= CONFIDENCE_LEVEL] = 1
        nJs, nTs = obs.shape

        Terr_mean = np.mean(Terr, axis=0) * 100
        Terr_stds = np.std(Terr, axis=0) * 100
        Terr_max = abs(np.nanmax(Terr_mean))
        # print(Terr_max)
        # Terr_mean /= Terr_max
        # Terr_stds /= Terr_max

        xplot = T_theory[gauss_test == 1]
        yplot = -Terr_mean[gauss_test == 1]
        yerr = Terr_stds[gauss_test == 1]
        eb = ax.errorbar(
            x=xplot, y=yplot,
            yerr=yerr,
            c=cols[i],
            label=lbl)
        handles.append(eb)

    ymin = 1
    ymax = 100
    ax.set_ylim(ymin, ymax)
    ax.legend(
        handles=handles,
        title='B',
        ncol=2,
        prop={'size': 8})
    # ax.set_ylabel(r'$\epsilon _{\theta} / \epsilon _{\theta}^{min}$')
    ax.set_ylabel(r'$- \epsilon _{T} (\%)$')
    ax.set_xlabel(r'T')


def mkfigure(**subplotkwargs):
    fig, ax = plt.subplots(**subplotkwargs)
    labels = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 't', 'u', 'v'
        ]
    labels = [letter + ')' for letter in labels]
    labels = ['(' + letter for letter in labels]
    if isinstance(ax, Axes) is not True:
        ax_ravel = ax.ravel()
        for iax in range(0, ax_ravel.size):
            ax_ravel[iax].text(
                0.0, 1.0, labels[iax], transform=ax_ravel[iax].transAxes,
                # fontsize='medium', fontfamily='serif',
                horizontalalignment='left',
                verticalalignment='top',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=0))
    return fig, ax
