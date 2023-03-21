import numpy as np
import matplotlib.pyplot as plt
from pyplm.utilities import tools
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# this uses a dataframe as input
def df_plot(ax, xkey, ykey, df_mean, df_std=None, **pltkwargs):
    x = df_mean[xkey]
    y = df_mean[ykey]
    if np.all(df_std == None):
        yerr = None
    else:
        yerr=df_std[ykey]
    # let's do a fill betweem
    # with color somehting...?
    line, = ax.plot(df_mean[xkey], df_mean[ykey], **pltkwargs)
    if np.all(df_std) != None:
        ec = line.get_color()
        ax.fill_between(
            x=df_mean[xkey],
            y1=df_mean[ykey] - df_std[ykey],
            y2=df_mean[ykey] + df_std[ykey],
            color=ec,
            alpha=0.5
            )
    # ax.errorbar(
    #     x=df_mean[xkey],
    #     y=df_mean[ykey],
    #     yerr=yerr,
    #     **pltkwargs)

# this uses numpy arrays as input... not super helpful to have both!
# def add_fit(ax, xraw, yraw, func, xfit=None, show_error=True, err_col='grey', **pltargs):
def df_add_fit(ax, df, xkey, ykey, fitfunc, xfit=None, show_error=True, err_col='grey', **pltargs):
    # func = tools.sqrt_x
    # xs = np.linspace(x.min(), x.max(), 100)
    xraw = df[xkey]
    yraw = df[ykey]

    if np.all(xfit == None):
        xfit = xraw

    popt, pcov = curve_fit(fitfunc, xraw, yraw)
    perr = np.sqrt(np.diag(pcov))
    popt_min = popt - perr
    popt_max = popt + perr
    score = r2_score(yraw, fitfunc(xraw, *popt))
    score = 1 - ((1-score) * ((yraw.size - 1) / (yraw.size - popt.size - 1)))
    print('params:', popt)
    print('errors:', perr)
    print('ybest:', yraw.iloc[[-1]])
    print(f'adj-r2 = {score:.3f}')
    yfit = fitfunc(xfit, *popt)
    yfit_min = fitfunc(xfit, *popt_min)
    yfit_max = fitfunc(xfit, *popt_max)
    if show_error == True:
        ax.plot(xfit, yfit_min, marker=',', c=err_col, ls='--', zorder=1)
        ax.plot(xfit, yfit_max, marker=',', c=err_col, ls='--', zorder=1)
        ax.fill_between(xfit, yfit_min, yfit_max, color=err_col, alpha=0.4, zorder=1)
    line, = ax.plot(
        xfit, yfit, **pltargs)
    return line, popt, perr

def modshow(ax, model, cmap='cividis', nticks=3, show_cbar=True):
    N, _ = model.shape
    # nice I like this, gives me the option to make it look nice :)!
    mat = ax.matshow(model, cmap=cmap)  #, vmin=model.min(), vmax=0.2
    ax.xaxis.tick_bottom()
    
    ax.xaxis.set_major_locator(plt.FixedLocator(np.linspace(0, N, nticks)))
    ax.yaxis.set_major_locator(plt.FixedLocator(np.linspace(0, N, nticks)))
    ax.set(
        xlim=[ax.get_xticks()[0], ax.get_xticks()[-1]],
        ylim=[ax.get_yticks()[-1], ax.get_yticks()[0]],
        xlabel='j',
        ylabel='i'
        )
    if show_cbar == True:
        plt.colorbar(mat)

def histogram(ax, data, filter_level=1, **histargs):
    # this I'm just not sure on, don't know whether to fill or not...
    # oooh cause I've got autobins that's why it's different
    # somehwat concerning that this has such a big effect....
    pdf, bin_edges = np.histogram(data, **histargs)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    # pdf = pdf * np.diff(bin_edges)
    # so why does this now no longer work to fit?
    # print(filter_level)
    print('----')
    print(pdf.shape)
    if filter_level is not None:
        for level in range(0, filter_level):
            mins = np.min(pdf)
            # print(mins, n[n == mins].size)
            bin_centers = bin_centers[pdf > mins]
            pdf = pdf[pdf > mins]
            print(pdf.shape)

    # print(bin_centers)
    # cdf = pdf * np.diff(bin_edges)
    # print(np.sum(pdf * np.diff(bin_edges)))
    # fig2, ax2 = plt.subplots(nrows=2, ncols=1)
    # ax2 = ax2.ravel()
    # shall I plot smooved version?
    # cdf_plot = np.convolve(cdf, np.ones(5)/5, mode='same')
    # ax2[0].plot(bin_centers, pdf_plot)
    # ax2[1].plot(bin_centers, cdf_plot)
    # plt.show()
    # smoothed_pdf = np.convolve(pdf, np.ones(10)/10, mode='same')
    # ax.plot(bin_centers, smoothed_pdf, marker=',',c=color)
    ax.plot(
        bin_centers, pdf, ls='none',
        # ls='none', 
    )
    # ok ok so let's just do it, plot the points and the smoothed line..?
    # maybe a bit unclear?
    # where the hell is my new version of mkfig that always amkes it a 2d array
    # I want to find this, anyway!
    # filter out the zeros!
    # should i really be doing this?
    # bin_length = bin_edges[1] - bin_edges[0]
    # n = n * bin_length
    # mins = np.min(n)
    # bin_centers = bin_centers[n > mins]
    # n = n[n > mins]
    # cdf = np.cumsum(n)
    # ax.plot(bin_centers, n)
    return bin_centers, pdf