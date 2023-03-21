import numpy as np
from scipy.optimize import curve_fit
from pyplm.utilities import tools

def subsampling_plot(ax, df, fitting_th, iG, label):
    COLORS_CAT = [
        '#4053d3', '#ddb310', '#b51d14',
        '#00beff', '#fb49b0', '#00b25d', '#cacaca'
    ]
    iDs = df['iD'].unique()
    for iD in iDs:
        # subplots.saturation_split_iD(ax, df, 'T', iD, ls='none', alpha=0.1, c=COLORS_CAT[iD])
        saturation_split_iD(ax, df, 'T', iD, ls='none', alpha=0.25, c='grey')
    popts = saturation_fit(
        ax, df, 'T', fitting_th,
        c=COLORS_CAT[iG], ls='-', marker=',', lw=2, label=label)
    return popts


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


# this is meant to just plot, nothing more nothing less
def saturation(ax, df, xkey, ykey, **mpl_args):
    x = df[xkey]
    y = df[ykey]
    ax.plot(x, y, **mpl_args)