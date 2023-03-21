import numpy as np
from pyplm.utilities import tools
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import linregress


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

import matplotlib.pyplot as plt
def get_tail_PowLaws(x, pdf):
    # filter only x > 0
    # I dont do the whole literning out the zeros from the pdf...
    # but why should I after I do the filter level thing...?
    # log data
    logx = np.log10(x)
    logpdf = np.log10(pdf)
    # print(logpdf)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(logx, logpdf)
    # plt.show()

    number_of_points = logx.size
    min_points = 20
    fitting_array = []

    for start in range(0, number_of_points - min_points):
        stop = -1
        logx_cut = logx[start:stop]
        logpdf_cut = logpdf[start:stop]
        res = linregress(logx_cut, logpdf_cut)
        fitting_array.append(np.array([res.rvalue ** 2, res.slope, res.intercept]))
    fitting_array = np.array(fitting_array)
    return fitting_array
# heres my fit with the big thingy wheres it goe?
