import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
from sklearn.metrics import r2_score

from pyplm.utilities.tools import p1, p2, p3, p_power, sqrt_x
from pyplm.plotting import mkfigure

plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
CATCOLS = plt.rcParams['axes.prop_cycle'].by_key()['color']

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


def convergence_check(x, y, nWindow):
    # print(x.shape)
    # print(y.shape)
    series_graidents = np.zeros((4, x.size))
    # print(series_graidents)
    series_graidents[0, :] = x
    series_graidents[1, :] = y
    # print(x.shape, np.diff(x).shape, y.shape)
    # exit()
    # print(x)
    # I think this makes more sense..?
    # print(x)
    # print(np.diff(x))
    
    dx = np.diff(x)
    dx = np.hstack((dx, dx[-1]))
  
    # dy = np.diff(y)
    # dy = np.hstack((dy, dy[-1]))
    # dxdy = dy/dx
    # dx2dy2 = dxdy / dx
    dydx = np.gradient(y, x) # / dx
    # dydx = uniform_filter1d(dydx, nWindow, mode='reflect')
    d2yd2x = np.gradient(dydx, x) # / dx
    # d2yd2x = uniform_filter1d(d2yd2x, 10, mode='reflect')
    # dydx = np.gradient(y) / dx
    # d2yd2x = np.gradient(dydx) # / dx
    series_graidents[2, :] = dydx
    series_graidents[3, :] = d2yd2x
    # series_graidents = series_graidents[:, nWindow: -nWindow]
    return series_graidents


def convergence_check2(x, y, polyorder):
    xfit = np.linspace(x.min(), x.max(), 5000)
    if polyorder == 1:
        popt, _ = curve_fit(p1, x, y)
        yfit = p1(xfit, *popt)
    elif polyorder == 2:
        popt_linear, _ = curve_fit(p1, x, y)
        linear_r2 = r2_score(y, p1(x, *popt_linear))
        popt, _ = curve_fit(p2, x, y)
        quadratic_r2 = r2_score(y, p2(x, *popt))
        # this is a silly quantity.
        print((quadratic_r2 - linear_r2) / quadratic_r2)
        yfit = p2(xfit, *popt)
    elif polyorder == 3:
        popt, _ = curve_fit(p3, x, y)
        yfit = p3(xfit, *popt)
    # print(popt)


    # polyfit = np.polynomial.Polynomial.fit(x, y, polyorder)
    # print(polyfit.coef)
    series_graidents = np.zeros((4, xfit.size))

    dydx = np.gradient(yfit, xfit)
    # dydx = dydx / b1
    # dydx = uniform_filter1d(dydx, 10, mode='reflect')
    d2yd2x = np.gradient(dydx, xfit) # / dx
    # this is too much....
    # d2yd2x = d2yd2x / (b1)
    # print(d2yd2x)
    # this seems wrong, surely these should go to 0..?
    series_graidents[0, :] = xfit
    series_graidents[1, :] = yfit
    series_graidents[2, :] = dydx
    series_graidents[3, :] = d2yd2x
    return series_graidents, popt

def graidents():
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    file = '/Users/mk14423/Desktop/Data/0_thesis/SubSampleSK/datasets.hdf5'
    groups = ['N50', 'N100', 'N200', 'N400', 'N800']
    g_labels = ['N = 50', 'N = 100', 'N = 200', 'N = 400', 'N = 800']
    emin = np.array([0.14320461, 0.20949111, 0.29832886, 0.44711299, 0.69887905])
    Btilde_linextrap = np.array([0.50661207, 0.74111249, 1.055392, 1.58174263, 2.47241035])
    T_trues = [1.1, 1.175, 1.1, 1.1, 1.25]
    Ns = np.array([50, 100, 200, 400, 800])
    # Btilde_linextrap *= 1e3
    # group = 'N800'
    # save = False
    figw, figh = plt.rcParams.get('figure.figsize')
    # print(figw, figh)
    fig, ax = mkfigure(nrows=1, ncols=1, sharex=True, figsize=(figw, figh * 1))
    # iG = 4
    # it works better in B than 1/B, whcich is annoying...
    iGs = [0, 1, 2, 3, 4]
    # iGs = [4]
    for iG in iGs:
        # print(iG)
        df = get_subsampling_data_frame(file, groups[iG], T_trues[iG])
        # print(df)
        # df['std_J'] = df['std_J'] * T_trues[iG]
        rescale = 1
        df['B'] = df['B'] * rescale
        df = df[df['B'] > 3e3 * rescale]
        df = df.groupby(['B'], as_index=True).mean()
        df = df.reset_index()
        nWindow = 10
        
        # df['std_J'] = uniform_filter1d(df['std_J'], nWindow, mode='reflect')
        xs = np.flip(1 / df['B'].to_numpy())
        ys = np.flip(df['std_J'].to_numpy()) # .to_numpy()

        # -- trying to "standardize" the variables -- #
        # ys = (ys - np.mean(ys))/np.std(ys)


        # what if I rescale to begin with.... here; report everything in units of b1!?
        # -- -- #
        # cuttoff = 1/(8e3 * rescale)
        # hmmm but why....
        # cuttoff = 1/(8e3 * rescale)
        # cut_poly = np.polynomial.Polynomial.fit(
        #     xs[xs < cuttoff],
        #     ys[xs < cuttoff],
        #     1)
        # b0, b1 = cut_poly.coef
        # print(cuttoff, b0, b1)
        # popt, _ = curve_fit(p1, xs[xs < cuttoff], ys[xs < cuttoff])
        # b0, b1 = popt
        # print(cuttoff, b0, b1)
        # let's see if the same thing happens with scipy.optimize?
        # ys = (ys - b0)
        # ys = ys / b1
        # -- -- #
        # s_gs = convergence_check(xs, ys)
        # idk if I can trust polyfit anymore now that I've seen this bullshit!
        s_gs, popt = convergence_check2(xs, ys, 2)
        # s_gs[1, :] = s_gs[1, :] / popt[1]
        # s_gs[2, :] = s_gs[2, :] / popt[1]
        s_gs[3, :] = s_gs[3, :] / (popt[1] ** 2)
        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax[0, 0].plot(s_gs[0, :], s_gs[1, :], marker=',', c=cols[iG])
        # ax[1, 0].plot(s_gs[0, :], s_gs[2, :], marker=',', c=cols[iG])
        # ax[2, 0].plot(s_gs[0, :], s_gs[3, :], marker=',', c=cols[iG])

        # print(' ------ ')
        # print(np.abs(popt / popt[1]))
        # print(popt)
        # print(iG, np.abs((popt[1] ** 2) / popt[2]))
        # print(popt[1]/ b1, popt[2]/ b1)
        # print(' ------ ')
        s_gs = convergence_check(xs, ys, nWindow)
        # s_gs[1, :] = s_gs[1, :]
        # s_gs[2, :] = s_gs[2, :] / popt[1]
        line, = ax[0, 0].plot(s_gs[0, :], s_gs[1, :], label=f'N={Ns[iG]}', marker='o', ls='none')
        # ax[1, 0].plot(s_gs[0, :], s_gs[2, :], label=f'N={Ns[iG]}', c=line.get_color(), marker='o', ls='none')
        # ax[2, 0].plot(s_gs[0, :], s_gs[3, :], label=f'N={Ns[iG]}', c=line.get_color(), marker='o', ls='none')

    legend = ax[0, 0].legend(loc='upper left', fontsize='8')
    legend.get_frame().set_alpha(None)

    ax[0, 0].set(ylabel=r'$y^{*} = \sigma^{*} N^{1/2}$', xlabel=r'$B^{-1}$')
    # ax[1, 0].set(ylabel=r'$\partial y / \left( \partial B^{-1} \right)$') # [ \times 10^{4}]
    # ax[2, 0].set(ylabel=r'$\partial ^2 y / \left( \partial ^2 B^{-1} \right)$', xlabel=r'$B^{-1}$')

    # ax[1, 0].set(ylim=[0.5, 1.5])
    ax[0, 0].set(
        xlim=[xs.min(), xs.max()],
        # ylim=[-10, 10]
        )

    # ax[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[0, 0].xaxis.set_major_locator(plt.MaxNLocator(5))

    # ax[0, 0].axvline(1/(8e3 * rescale), marker=',', ls='--', c='k')
    # ax[1, 0].axvline(1/(8e3 * rescale), marker=',', ls='--', c='k')
    # ax[2, 0].axvline(1/(8e3 * rescale), marker=',', ls='--', c='k')

    # ylabel=r'$\partial \sigma ^{*} / \partial (1/B)$')
    # ax[0, 1].set(ylim=[-0.05, 0])
    # ax[1, 0].set(ylim=[-0.005, 0.005])
    plt.show()

# this will all get renamed and everything.

def sliding_coalesc(x, y, ax, colour, method=False):
    nWindows = 10
    x_new, samples_new = make_windows(x, y, nWindows, ax=ax, method=method)
    CL = 0.05
    xs = []
    samples_again = []
    # print(x_new.shape)
    # compare first two, then add them together and compare the next one!
    accepted_sample = samples_new[0, :]
    # print(accepted_sample.shape)
    # print(accepted_sample)
    running_xs = [] # x_new[0]
    running_means = []
    running_stds = []
    accepted_indicies = [0]

    for i in range(1, x_new.size):
        # print(accepted_sample.shape)
        _, pvalue = ks_2samp(accepted_sample, samples_new[i], alternative='two-sided')
        # _, pvalue = ks_2samp(accepted_sample, samples_new[i], alternative='greater')
        # print(f'i={i}, p={pvalue:.3f} -> {pvalue >= CL}')
        if pvalue >= CL:
            # print(f'Accept')
            accepted_sample = np.hstack((accepted_sample, samples_new[i]))
            accepted_indicies.append(i)
            running_x = (x_new[0] + x_new[i]) / 2
            # accepted_xs = [x_new[ix] for ix in  accepted_indicies]
            # print(accepted_indicies, accepted_xs)
            # running_x = np.sum(accepted_xs) / len(accepted_indicies)
            running_xs.append(running_x)
            running_means.append(np.mean(accepted_sample))
            running_stds.append(np.std(accepted_sample, ddof=1))
        else:
            # print('REJECTED!')
            break
    i_cuttoff = accepted_sample.size - 1

    x_cuttoff = x[i_cuttoff]
    # print(i_cuttoff, x_cuttoff)
    if method == True:
        # ax.errorbar(
        #         x=x_new,
        #         y=np.mean(samples_new, axis=1),
        #         marker='o',
        #         ls='none',
        #         yerr=np.std(samples_new,axis=1),
        #         c='k')
        ax.axvline(x_cuttoff, c='k', marker=',', ls='--')
        ax.errorbar(
            x=running_xs[-1],
            y=running_means[-1],
            yerr=running_stds[-1],
            c='k', marker='^',
            markersize=5,
            elinewidth=2,
            zorder=100
        )
    return x_cuttoff


def make_windows(x, y, nWindows, ax, method):
    x_new = []
    window_length = int(y.size / nWindows)
    # print(nWindows, window_length)
    # window_length = 10
    # nWindows = int(dydx.size / window_length)
    samples = []
    for i in range(0, nWindows):
        w_start = i * window_length
        w_end = (i + 1) * window_length
        w_middle = int((w_start + w_end) / 2)
        data = y[w_start:w_end]
        # print(mean, std)
        x_new.append(x[w_middle])
        samples.append(data)
        if method == True:
            line, = ax.plot(x[w_start:w_end], data, ls='none')
            c = line.get_color()
            ax.axvspan(x[w_start], x[w_end-1], fc=c, alpha=0.5)
    # samples = np.array(samples)
    return np.array(x_new), np.array(samples)

def coalescing_example():
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    file = '/Users/mk14423/Desktop/Data/0_thesis/SubSampleSK/datasets.hdf5'
    groups = ['N50', 'N100', 'N200', 'N400', 'N800']
    g_labels = ['N=50', 'N=100', 'N=200', 'N=400', 'N=800']
    T_trues = [1.1, 1.175, 1.1, 1.1, 1.25]
    figw, figh = plt.rcParams.get('figure.figsize')
    # fig, ax = mkfigure(
    #     nrows=2, ncols=2,
    #     sharex=True,
    #     # figsize=(figw, figh * 1.4)
    #     )
    fig, ax = mkfigure(
        nrows=1, ncols=1,
        # sharex=True,
        # figsize=(figw, figh * 1.4)
        )
    ax = ax.ravel()
    # iG = 4
    iGs = [0, 1, 2, 3, 4]
    cuts = []
    # iGs = [3]
    iP = 0
    for iG in iGs:
        # print(iG)
        df = get_subsampling_data_frame(file, groups[iG], T_trues[iG])
        # print(df)
        df['std_J'] = df['std_J'] * T_trues[iG]
        rescale = 1
        df['B'] = df['B'] * rescale
        df = df[df['B'] > 3e3 * rescale]
        df = df.groupby(['B'], as_index=True).mean()
        df = df.reset_index()

        # df['std_J'] = uniform_filter1d(df['std_J'], nWindow, mode='reflect')
        xs = np.flip(1 / df['B'].to_numpy())
        ys = np.flip(df['std_J'].to_numpy()) # .to_numpy()
        # ax[0, 0].plot(xs, ys)
        dydx = np.gradient(ys, xs)
        x_cut = sliding_coalesc(xs, dydx, ax[iP], CATCOLS[iG])
        ax[0].plot(
            xs[xs <= x_cut], ys[xs <= x_cut],
            alpha=1, ls='none', c=CATCOLS[iG], marker='o', zorder=50,
            label=g_labels[iG])
        ax[0].plot(
            xs[xs > x_cut], ys[xs > x_cut],
            alpha=0.25, ls='none', c=CATCOLS[iG], marker='o', zorder=50)

        xfit = np.linspace(0, xs.max(), 1000)
        popt, _ = curve_fit(p1, xs[xs <= x_cut], ys[xs <= x_cut])
        yfit = p1(xfit, *popt)
        ax[0].plot(xfit, yfit, c='k', marker=',', zorder=1)

        # print(T_trues[iG], 1 / popt[0], 1 / np.mean(ys[xs <= x_cut]))
        # print(popt[1]) #  np.mean(dydx[xs <= x_cut])
        print(x_cut, popt)
        # ax[iP].plot(
        #     xs[xs <= x_cut], dydx[xs <= x_cut], alpha=1, ls='none', c=CATCOLS[iG], marker='o', label=g_labels[iG])
        # ax[iP].plot(
        #     xs[xs > x_cut], dydx[xs > x_cut], alpha=0.25, ls='none', c=CATCOLS[iG], marker='o')
        # cuts.append(x_cut)
        # ax[iP].set(
        #     xlim=[xs.min(), xs.max()],
        #     # ylabel=r'$\partial y / \partial \left( B^{-1} \right)$'
        # )
        # ax[iP].yaxis.set_major_locator(plt.MaxNLocator(3))
        # ax[iP].legend(loc='lower right', fontsize=9)
        # ax[iP].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # iP += 1

    # ax[0, 0].set(ylabel=r'$y = \sigma^{*} N^{1/2}$')
    # ax[1, 0].set(ylabel=r'$\partial y / \partial \left( B^{-1} \right)$', xlabel=r'$B^{-1}$')
    # # ax[2, 0].set(ylabel=r'$\partial ^2 y / \partial ^2 B^{-1}$', )
    # ax[1, 0]
    # ax[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax[0].legend()
    ax[0].set(
        xlabel=r'$B^{-1}$',
        ylabel=r'$\sigma^{*} / \sigma^{0}$',
        xlim=[xs.min(), xs.max()],
        # xlim=[xfit.min(), xfit.max()],
        ylim=[None, 2]
        )
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(4))
    # fig.supylabel(r'$\partial y / \partial \left( B^{-1} \right)$')
    # fig.supxlabel(r'$B^{-1}$')
    plt.show()


def coalescing_method():
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
    file = '/Users/mk14423/Desktop/Data/0_thesis/SubSampleSK/datasets.hdf5'
    groups = ['N50', 'N100', 'N200', 'N400', 'N800']
    g_labels = ['N=50', 'N=100', 'N=200', 'N=400', 'N=800']
    T_trues = [1.1, 1.175, 1.1, 1.1, 1.25]
    figw, figh = plt.rcParams.get('figure.figsize')
    # fig, ax = mkfigure(
    #     nrows=2, ncols=2,
    #     sharex=True,
    #     # figsize=(figw, figh * 1.4)
    #     )
    fig, ax = mkfigure(
        nrows=2, ncols=1,
        sharex=True,
        # figsize=(figw, figh * 1.4)
        )
    ax = ax.ravel()
    # iG = 4
    iGs = [0, 1, 2, 3, 4]
    cuts = []
    iGs = [2,3]
    iP = 0
    for iG in iGs:
        # print(iG)
        df = get_subsampling_data_frame(file, groups[iG], T_trues[iG])
        # print(df)
        df['std_J'] = df['std_J'] * T_trues[iG]
        rescale = 1
        df['B'] = df['B'] * rescale
        df = df[df['B'] > 3e3 * rescale]
        df = df.groupby(['B'], as_index=True).mean()
        df = df.reset_index()

        # df['std_J'] = uniform_filter1d(df['std_J'], nWindow, mode='reflect')
        xs = np.flip(1 / df['B'].to_numpy())
        ys = np.flip(df['std_J'].to_numpy()) # .to_numpy()
        # ax[0, 0].plot(xs, ys)
        dydx = np.gradient(ys, xs)
        x_cut = sliding_coalesc(xs, dydx, ax[iP], CATCOLS[iG], method=True)
        ax[iP].set(
        ylabel=r'$\partial y / \partial \left( B^{-1} \right)$',
        xlim=[xs.min(), xs.max()],
        # ylim=[None, 2]
        )
        ax[iP].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[iP].xaxis.set_major_locator(plt.MaxNLocator(4))
        iP += 1
    ax[-1].set(xlabel=r'$B^{-1}$')
    plt.show()

def b1_Btilde_N_helper(Ns, bias_measure, fitfunc, ax, fitls='-', **pltargs):
    line, = ax.plot(
        Ns, bias_measure,
        **pltargs)
    popt, _ = curve_fit(fitfunc, Ns, bias_measure)
    r2 = r2_score(bias_measure, fitfunc(Ns, *popt))
    print(popt, r2)
    xs = np.linspace(Ns.min(), Ns.max(), 1000)
    ax.plot(xs, fitfunc(xs, *popt), ls=fitls, marker=',', c=line.get_color())
    return popt, r2

def b1_Btilde_N():
    # this is normalised data!
    Btilde_results_5e3 = np.array(
        [
        [1.00600985e+00, 2.72014640e+02, 9.88233835e-01],
        [9.94183234e-01, 5.19063249e+02, 9.61574352e-01],
        [1.00267001e+00, 1.02930216e+03, 9.37379556e-01],
        [9.97783478e-01, 2.04440470e+03, 8.69604374e-01],
        [1.02747743e+00, 4.28098167e+03, 7.60336351e-01],
        ]
    )
    Btilde_results_8e3 = np.array(
        [
        [1.00335165e+00, 2.36632494e+02, 9.88233835e-01],
        [9.98044149e-01, 5.73135231e+02, 9.61574352e-01],
        [9.99475985e-01, 9.85167260e+02, 9.37379556e-01],
        [9.98207849e-01, 2.05093924e+03, 8.69604374e-01],
        [1.00340312e+00, 3.99806103e+03, 7.60336351e-01],
        ]
    )
    # Btildes = Btilde_results_8e3[:, 1]
    b1s = np.array([184.914, 377.723, 816.309, 1786.117, 4059.442])
    Bts = Btilde_results_5e3[:, 1]
    emin = np.array([0.14320461, 0.20949111, 0.29832886, 0.44711299, 0.69887905])
    Ns = np.array([50, 100, 200, 400, 800])
    fig, ax = mkfigure(nrows=1, ncols=1)
    # whoops, had this the wrong way round, but whatever..
    print('---------')
    func = p1
    _, r2_p1Bt = b1_Btilde_N_helper(
        Ns, Bts, func, ax[0, 0],
        c=CATCOLS[0], ls='none', label=r'$\tilde{B}$ fitting cut-off: $B = 5 \times 10^3$'
        )
    _, r2_p1b1 = b1_Btilde_N_helper(
        Ns, b1s, func, ax[0, 0],
        c=CATCOLS[1], ls='none', label=r'$b_1$ variable fitting cut-off'
        )
    print(r2_p1b1, r2_p1Bt)
    func = p_power
    _, r2_ppBt = b1_Btilde_N_helper(
        Ns, Bts, func, ax[0, 0], fitls='--',
        c=CATCOLS[0], ls='none'
        # , label=r'$\tilde{B}$ fitting cut-off: $B = 5 \times 10^3$'
        )
    _, r2_ppb1 = b1_Btilde_N_helper(
        Ns, b1s, func, ax[0, 0], fitls='--',
        c=CATCOLS[1], ls='none'
        # , label=r'$b_1$ variable fitting cut-off'
        )
    print(r2_ppb1, r2_ppBt)
    # improvements
    imp_b1 = (r2_ppb1 - r2_p1b1) / r2_ppb1
    imp_Bt = (r2_ppBt - r2_p1Bt) / r2_ppBt
    print(imp_b1, imp_Bt)
    print('---------')

    ax[0, 0].set(xlabel=r'$N$', ylabel='Bias measure')
    ax[0, 0].legend()

    axin = ax[0,0].inset_axes([0.6, 0.15, 0.3, 0.3])
    func = sqrt_x
    _, r2_emin = b1_Btilde_N_helper(
        Ns, emin, func, axin, fitls='--',
        c=CATCOLS[0], ls='none'
        # , label=r'$\tilde{B}$ fitting cut-off: $B = 5 \times 10^3$'
        )
    axin.set(xlabel=r'$N$', ylabel=r'$\varepsilon _{min}$')

    plt.show()

def b1_vs_emin_N():
    b1s = np.array([184.914, 377.723, 816.309, 1786.117, 4059.442])
    emins = np.array([0.14320461, 0.20949111, 0.29832886, 0.44711299, 0.69887905])
    Ns = np.array([50, 100, 200, 400, 800])
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisaside.mplstyle')
    fig, ax = plt.subplots()
    x = b1s
    y = emins * (Ns ** 0.5)
    ax.plot(x, y, ls='none')
    popt, _ = curve_fit(p1, x, y)
    r2 = r2_score(y, p1(x, *popt))
    print(popt, r2)
    ax.plot(
        x, p1(x, *popt), ls='--', c='k', marker=',',
        label=r'$R^2 = 0.999$')
    ax.set(xlabel=r'$b_{1}$', ylabel=r'$\varepsilon _{min} N^{1/2}$')
    ax.legend(loc='upper left')
    # plt.savefig('/Users/mk14423/Documents/tempfigs/analytical-bias-emin-vs-b1.png')
    plt.show()

# this is for a fixed B!
def analytical_error(E, N, mu, T, sigma):
    # E = 1/BD*0.5
    # err = E / (N**0.5)
    # err = err / T
    # factor = np.sqrt(((mu ** 2) / N) + (sigma ** 2))
    # err = err * factor
    # WHOOPS! NEED TO TRIPPLE CHECK THIS!
    # THIS SHOULD MAYBE BE DIFFEENT NOW!
    # err = E / (N**0.5)
    err = E * T * (N**0.5)
    factor = np.sqrt(((mu ** 2) / N) + (sigma ** 2))
    err = err / factor
    return err

def analytical_Kfactor(B, N, mu, T, sigma):
    # E = 1/BD*0.5
    # err = E / (N**0.5)
    # err = err / T
    # factor = np.sqrt(((mu ** 2) / N) + (sigma ** 2))
    # err = err * factor
    # WHOOPS! NEED TO TRIPPLE CHECK THIS!
    # THIS SHOULD MAYBE BE DIFFEENT NOW!
    # err = E / (N**0.5)
    K = T / (B * (N ** 0.5))
    factor = ((mu ** 2) / N) + (sigma ** 2)
    K = K / (factor ** 0.5)
    # err = E * T * (N**0.5)
    # factor = np.sqrt(((mu ** 2) / N) + (sigma ** 2))
    # err = err / factor
    return K


from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import LogNorm

from inference.scripts.paper2022May import load
from scipy.ndimage import gaussian_filter

def analytical_analysis():
    runsN50 = [
        'B1e4-Nscaling/N50_1', 'B1e4-Nscaling/N50_2', 'B1e4-Nscaling/N50_3',
        'B1e4-Nscaling/N50_4', 'B1e4-Nscaling/N50_5', 'B1e4-Nscaling/N50_6',
        ]
    runsN100 = [
        'B1e4-Nscaling/N100_1', 'B1e4-Nscaling/N100_2', 'B1e4-Nscaling/N100_3',
        'B1e4-Nscaling/N100_4', 'B1e4-Nscaling/N100_5', 'B1e4-Nscaling/N100_6',
        ]
    runsN800 = [
        'B1e4-Nscaling/N800_1',
        ]
    N = 50
    params, obs, = load.load_PD_fixedB(runsN50, '/Users/mk14423/Desktop/PaperData')
    # params, obs, = load.load_N200_Bfixed_obs('/Users/mk14423/Desktop/PaperData')
    print(params.shape, obs.shape)
    X = params['J']
    Y = params['T']
    error = obs['e']
    # error = gaussian_filter(error, sigma=0.5)
    # error[error > 1.5 * error.min()] = 1.5 * error.min()
    # Z = analytical_error(E=1, N=N, mu=X, T=Y, sigma=1)
    # bias_surface = error * Z
    K = analytical_Kfactor(B=10 ** 4, N=N, mu=X, T=Y, sigma=1)
    # I dont really understand why this doesnt show what I want it to :(
    # bias_surface = error / K
    bias_surface = error / K
    maxfactor = 10

    fig = plt.figure()
    gs = fig.add_gridspec(2, 4)
    ax = np.zeros((2, 2))
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2:])
    ax2 = fig.add_subplot(gs[1:, 0:])
    # ax[0]
    ax = [ax0, ax1, ax2]
    
    im = ax[0].pcolor(
        X, Y, error,
        # norm=LogNorm(vmin=error.min(), vmax=error.max())
        vmin=0.95 * error.min(),
        vmax=maxfactor * error.min()
    )
    im = ax[1].pcolor(
        X, Y, bias_surface,
        # norm=LogNorm(vmin=bias_surface.min(), vmax=bias_surface.max())
        vmin=0.95 * bias_surface.min(),
        vmax=maxfactor * bias_surface.min()
    )
    # fig.colorbar(im, ax=ax[1], label=r'$E$', use_gridspec=True)

    ax[0].set(xlabel=r'$\mu$', ylabel=r'$T$', title=r'$\varepsilon$')
    ax[1].set(xlabel=r'$\mu$', ylabel=r'$T$', title=r'$\varepsilon/K$')
    # ax[1].set(xlabel=r'$\mu$', ylabel=r'$T$')

    ax2 = ax[2].twinx()
    cut = 8
    print(X[0:cut, 0])
    e_collapsed = np.mean(error[0:cut, 0:21], axis=0)
    b_collapsed = np.mean(bias_surface[0:cut, 0:21], axis=0)
    print(b_collapsed.shape)
    print(Y[0, 0:21])
    ax[2].plot(Y[0, 0:21], e_collapsed, c=CATCOLS[0])
    ax2.plot(Y[0, 0:21], b_collapsed, c=CATCOLS[1])
    ax[2].set(
        xlabel=r'$T$',
        ylabel=r'$\varepsilon$',
        ylim=[0.95 * e_collapsed.min(), maxfactor * e_collapsed.min()])
    ax2.set(
        ylabel=r'$Bias=\epsilon / K$',
        ylim=[0.95 * b_collapsed.min(), maxfactor * b_collapsed.min()])
    # ax.set(yscale='log')
    # ax2.set(yscale='log')
    ax[2].yaxis.label.set_color(CATCOLS[0])
    ax2.yaxis.label.set_color(CATCOLS[1])
    plt.show()

    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisaside.mplstyle')
    fig, ax = plt.subplots()
    Ts = np.linspace(0.5, 2, 100)
    Ks = analytical_error(E=1, N=N, mu=0, T=Ts, sigma=1)
    # whyyyyyy is it all 1 Lol. FFS!!
    ax.plot(Ts, 1/Ks, marker=',')
    ax.set(xlabel=r'$T$', ylabel=r'$1/K$')
    # plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisaside.mplstyle')
    # plt.savefig('/Users/mk14423/Documents/tempfigs/analytical-bias-asideT.png')
    plt.show()


def analytical_analysis_includingTau():
    runsN50 = [
        'B1e4-Nscaling/N50_1', 'B1e4-Nscaling/N50_2', 'B1e4-Nscaling/N50_3',
        'B1e4-Nscaling/N50_4', 'B1e4-Nscaling/N50_5', 'B1e4-Nscaling/N50_6',
        ]
    runsN100 = [
        'B1e4-Nscaling/N100_1', 'B1e4-Nscaling/N100_2', 'B1e4-Nscaling/N100_3',
        'B1e4-Nscaling/N100_4', 'B1e4-Nscaling/N100_5', 'B1e4-Nscaling/N100_6',
        ]
    runsN800 = [
        'B1e4-Nscaling/N800_1',
        ]
    N = 200
    cut = 8

    # params, obs, = load.load_PD_fixedB(runsN800, '/Users/mk14423/Desktop/PaperData')
    params, obs, = load.load_N200_Bfixed_obs('/Users/mk14423/Desktop/PaperData')
    print(params.shape, obs.shape)
    X = params['J']
    Y = params['T']
    error = obs['e']
    tau = obs['tau']
    tau[tau < 1] = 1
    B = 1e4 / tau
    plt.plot(Y[0, 0:21], np.mean(tau[0:cut, 0:21], axis=0))
    plt.show()

    # FUCK THIS SHIT FOR NOW!
    # error = gaussian_filter(error, sigma=0.5)
    # error[error > 1.5 * error.min()] = 1.5 * error.min()
    Z = analytical_error(E=1, N=N, mu=X, T=Y, sigma=1)
    # Z = Z
    # plt.pcolor(X, Y, Z)
    # plt.show()
    bias_surface = error / Z


    fig = plt.figure()
    gs = fig.add_gridspec(2, 4)
    ax = np.zeros((2, 2))
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2:])
    ax2 = fig.add_subplot(gs[1:, 0:])
    # ax[0]
    ax = [ax0, ax1, ax2]
    
    im = ax[0].pcolor(
        X, Y, error,
        # norm=LogNorm(vmin=error.min(), vmax=error.max())
        vmin=0.95 * error.min(),
        vmax=maxfactor * error.min()
    )
    im = ax[1].pcolor(
        X, Y, bias_surface,
        # norm=LogNorm(vmin=bias_surface.min(), vmax=bias_surface.max())
        vmin=0.95 * bias_surface.min(),
        vmax=maxfactor * bias_surface.min()
    )
    # fig.colorbar(im, ax=ax[1], label=r'$E$', use_gridspec=True)

    ax[0].set(xlabel=r'$\mu$', ylabel=r'$T$', title=r'$\varepsilon$')
    ax[1].set(xlabel=r'$\mu$', ylabel=r'$T$', title=r'$\varepsilon/K$')
    # ax[1].set(xlabel=r'$\mu$', ylabel=r'$T$')

    ax2 = ax[2].twinx()
    print(X[0:cut, 0])
    e_collapsed = np.mean(error[0:cut, 0:21], axis=0)
    b_collapsed = np.mean(bias_surface[0:cut, 0:21], axis=0)
    print(b_collapsed.shape)
    print(Y[0, 0:21])
    ax[2].plot(Y[0, 0:21], e_collapsed, c=CATCOLS[0])
    ax2.plot(Y[0, 0:21], b_collapsed, c=CATCOLS[1])
    ax[2].set(
        xlabel=r'$T$',
        ylabel=r'$\varepsilon$',
        ylim=[0.95 * e_collapsed.min(), maxfactor * e_collapsed.min()]
        )
    ax2.set(
        ylabel=r'$Bias=\epsilon / K$',
        ylim=[0.95 * b_collapsed.min(), maxfactor * b_collapsed.min()]
        )
    # ax.set(yscale='log')
    # ax2.set(yscale='log')
    ax[2].yaxis.label.set_color(CATCOLS[0])
    ax2.yaxis.label.set_color(CATCOLS[1])
    plt.show()



# this also has some random stuff about the error!!
graidents()
# coalescing_example()
# coalescing_method()
# b1_Btilde_N()
# b1_vs_emin_N()
# analytical_analysis()
# analytical_analysis_includingTau()
# plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
