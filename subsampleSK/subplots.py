import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from pyplm.utilities import tools
from helpers import add_fit_to_ax

def saturation_split_iD(ax, df, observable, run_iD, **pltkwargs):
    subset_df = df.loc[df['iD'] == run_iD]
    x = subset_df['B']
    y = subset_df[observable]
    ax.plot(x, y, **pltkwargs)


def saturation_fit(ax, df, observable, threshold_B, **pltkwargs):
    df = df.groupby(['B'], as_index=True).mean()
    df = df.reset_index()
    ax.plot(df['B'], df[observable], c='k', marker='d', ls='none', alpha=1)
    df = df.loc[df['B'] >= threshold_B]

    x = df['B']
    y = df[observable]
    func = tools.arctan
    popt, _ = curve_fit(tools.arctan, x, y, p0 = [0.1, 0.1])
    xfit = np.linspace(0, x.max(), 1000)
    yfit = func(xfit, *popt)
    ax.plot(xfit, yfit, **pltkwargs)
    popt[0] = (popt[0] * np.pi) / 2
    popt[1] = 1 / popt[1]
    popt = np.append(popt, np.nanmax(y))
    return popt

def N_error(ax):
    x = np.array([50, 100, 200, 400, 800])
    y = np.array([0.14320461, 0.20949111, 0.29832886, 0.44711299, 0.69887905])
    mus = np.array([0.5, 0.4, 0.4, 0.6, 0.2])
    sigmas = np.array([1, 1, 1, 1, 1])
    Ts = np.array([1.1, 1.175, 1.1, 1.1, 1.25])

    # so if I plot simga mu n against thingy it should be a straight line going
    # throguh the origin right?

    # lbl = (r'$\varepsilon _{\mathrm{min}}(N) = A N ^\gamma$')
    
    
    # lbl = (r'$\varepsilon _{\mathrm{min}}(N) = A N ^{1/2}$')
    ax.plot(x, y, ls='none', color='#4053d3')

    xs = np.linspace(0, 1000, 1000)
    fc = 'grey'
    lbl = (r'$\varepsilon (N) = A N ^\gamma$')
    add_fit_to_ax(
        ax, x, y, tools.pure_power, xfit=xs, err_col=fc, show_error=False,
        marker=',', c='k', ls='-', label=lbl)
    fc = '#00beff'
    lbl = (r'$\varepsilon (N) = A N ^{1/2} + c$')
    print('-0-')
    add_fit_to_ax(
        ax, x, y, tools.sqrt_x, xfit=xs, err_col=fc, show_error=False,
        marker=',', c='r', ls='-', label=lbl)
    print('-0-')
    # fc = '#fb49b0'

    ax.set(
        # ylim=[y.min() * 0.97, y.max() * 1.03],
        ylim=[0, None],
        xlim=[0, 805],
        xlabel=r'$N$',
        ylabel=r'$\varepsilon$')
    ax.legend()


def error_scaling_explanation(ax):
    # xs = np.array([50, 100, 200, 400, 800])
    Ns = np.array([50, 100, 200, 400, 800])
    errs = np.array([0.14320461, 0.20949111, 0.29832886, 0.44711299, 0.69887905])
    mus = np.array([0.5, 0.4, 0.4, 0.6, 0.2])
    sigmas = np.array([1, 1, 1, 1, 1])
    Ts = np.array([1.1, 1.175, 1.1, 1.1, 1.25])

    xs = (Ns ** 0.5)
    xs = xs * (Ts ** -1)
    xs_uncorrected = np.copy(xs)
    sqrt_term = np.sqrt( ((mus ** 2) / Ns) + (sigmas ** 2))
    xs = xs * sqrt_term
    ys = errs

    # explain this somehow! So acutally the error is a pretty
    # bad measure of the phase diagram thingy. Really
    # I should do my B subsmapling to figure it out i think!
    # this comes from numerics....
    # I should plot my error surface for a fixed thing!
    
    # lbl = (r'$\varepsilon _{\mathrm{min}}(N) = A N ^{1/2}$')
    ax.plot(xs, ys, ls='none', color='#4053d3')
    ax.plot(xs_uncorrected, ys, ls='none', color='#b51d14')

    # xfits = np.linspace(0, 1000, 1000)
    xfits = np.linspace(xs.min(), xs.max(), 1000)
    fc = 'grey'
    lbl = (r'$\varepsilon (N) = A N ^\gamma$')
    add_fit_to_ax(
        ax, xs, ys, tools.linear, xfit=xfits, err_col=fc, show_error=True,
        marker=',', c='k', ls='-', label=lbl)
    # fc = '#00beff'
    # lbl = (r'$\varepsilon (N) = A N ^{1/2} + c$')
    # print('-0-')
    # add_fit_to_ax(
    #     ax, x, y, tools.sqrt_x, xfit=xs, err_col=fc, show_error=False,
    #     marker=',', c='r', ls='-', label=lbl)
    # print('-0-')
    # # fc = '#fb49b0'

    ax.set(
        # ylim=[y.min() * 0.97, y.max() * 1.03],
        # ylim=[0, None],
        # xlim=[0, 805],
        xlabel=r'$N$',
        ylabel=r'$\varepsilon$')
    ax.legend()

def error_test():
    E = 1
    N = 200
    # Ts = np.linspace(0.3, 2, 100)
    T = 1
    mu = 0.1
    sigma = 1
    # err = [err_analytical(E, N, T, mu, sigma) for T in Ts]
    # plt.plot(Ts, err)
    Ns = np.linspace(10, 200, 20)
    plt.plot(Ns, err_analytical(E, Ns, T, mu, sigma))
    plt.show()
    pass

def err_analytical(E, N, T, mu, sigma):
    err = E * (N ** -0.5) * (T ** -1)
    sqrt_term = np.sqrt( ((mu ** 2) / N) + (sigma ** 2))
    err = err * sqrt_term
    return err

def plot_with_fit(ax, x, y, fitfunc, color, label):
    xrange = abs(x.max() - x.min())
    x_lim_offset = xrange * 0.02
    xlim_min = x.min() - x_lim_offset
    xlim_max = x.max() + x_lim_offset

    yrange = abs(y.max() - y.min())
    y_lim_offset = yrange * 0.02
    ylim_min = y.min() - y_lim_offset
    ylim_max = y.max() + y_lim_offset


    xfit = np.linspace(xlim_min, xlim_max, 200)
    # x = np.log10(x)
    # y = np.log10(y)
    # xfit = np.log10(xfit)
    # popt, _ = curve_fit(fitfunc, x, y)
    # yfit = fitfunc(xfit, *popt)
    # score = r2_score(y, fitfunc(x, *popt))
    # score = 1 - ((1-score) * ((y.size - 1) / (y.size - popt.size - 1)))
    # print(popt, score)
    # x = 10 ** x
    # y = 10 ** y
    # xfit = 10 ** xfit
    # yfit = 10 ** yfit
    # ax.plot(xfit, yfit, c='k', marker=',', ls='-')
    add_fit_to_ax(
        ax, x, y, fitfunc, xfit=xfit, err_col=color, show_error=False,
        marker=',', c='k', ls='-')
    ax.set(xlim=[xlim_min, xlim_max], ylim=[ylim_min, ylim_max])
    line, = ax.plot(x, y, c=color, ls='none', label=label, zorder=50)
    return line
