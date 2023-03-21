import numpy as np

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def add_fit_to_ax(ax, xraw, yraw, func, xfit=None, show_error=True, err_col='grey', **pltargs):
    # func = tools.sqrt_x
    # xs = np.linspace(x.min(), x.max(), 100)
    if np.all(xfit == None):
        xfit = xraw

    popt, pcov = curve_fit(func, xraw, yraw)
    perr = np.sqrt(np.diag(pcov))
    popt_min = popt - perr
    popt_max = popt + perr
    score = r2_score(yraw, func(xraw, *popt))
    score = 1 - ((1-score) * ((yraw.size - 1) / (yraw.size - popt.size - 1)))
    print('params:', popt)
    print('errors:', perr)
    print(f'adj-r2 = {score:.3f}')

    yfit = func(xfit, *popt)
    yfit_min = func(xfit, *popt_min)
    yfit_max = func(xfit, *popt_max)
    if show_error == True:
        ax.plot(xfit, yfit_min, marker=',', c=err_col, ls='--', zorder=1)
        ax.plot(xfit, yfit_max, marker=',', c=err_col, ls='--', zorder=1)
        ax.fill_between(xfit, yfit_min, yfit_max, color=err_col, alpha=0.4, zorder=1)
    line, = ax.plot(
        xfit, yfit, **pltargs, zorder=0)
    return line