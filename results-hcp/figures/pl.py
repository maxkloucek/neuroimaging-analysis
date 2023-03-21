import numpy as np
import powerlaw
import matplotlib.pyplot as plt

from pyplm.utilities import tools

from mpmath import gammainc

def truncated_powerlaw(x, alpha, Lambda, xmin):
    # print(Lambda*xmin)
    # print(1-alpha)
    # print(gammainc(1-alpha, Lambda*xmin))
    # C = ( Lambda**(1+alpha) /
    #             float(gammainc(1+alpha, Lambda*xmin)))
    C = ( Lambda**(1-alpha) /
                float(gammainc(1-alpha, Lambda*xmin)))
    # C = xmin
    # I don't get this, but whatever...
    return C * (x**(alpha) * np.exp(Lambda * x))


def powerlaw_fit_method2(model):
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax2 = ax2.ravel()

    model_LL = model[0:180, 0:180]
    model_RR = model[180:, 180:]

    p_LL = tools.triu_flat(model_LL)
    p_RR = tools.triu_flat(model_RR)

    p_LL = p_LL[p_LL > 0]
    p_RR = p_RR[p_RR > 0]

    fit_LL = powerlaw.Fit(p_LL, sigma_threshold=0.1)
    fit_RR = powerlaw.Fit(p_RR, sigma_threshold=0.1)

    print(f'LL: xmin = {fit_LL.power_law.xmin:.3f} alpha  = {fit_LL.power_law.alpha:.3f}')
    print(f'RR: xmin = {fit_RR.power_law.xmin:.3f} alpha  = {fit_RR.power_law.alpha:.3f}')
    # sigma_threshold
    print('-------')
    print(fit_LL.power_law.sigma / fit_LL.power_law.alpha)
    print(fit_RR.power_law.sigma / fit_RR.power_law.alpha)
    print('-------')
    print(fit_LL.power_law.sigma, fit_RR.power_law.sigma)
    ax2[0].plot(fit_LL.xmins, fit_LL.Ds, marker=',', label='Left')
    ax2[0].plot(fit_RR.xmins, fit_RR.Ds, marker=',', label='Right')
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax2[0].axvline(fit_LL.power_law.xmin, c=default_colors[0], marker=',', ls='--')
    ax2[0].axvline(fit_RR.power_law.xmin, c=default_colors[1], marker=',', ls='--')

    ax2[1].plot(fit_LL.xmins, 100 * (fit_LL.sigmas / fit_LL.alphas), marker=',')
    ax2[1].plot(fit_RR.xmins, 100 * (fit_RR.sigmas / fit_RR.alphas), marker=',')
    # ax2[1].plot(fit_LL.xmins, fit_LL.sigmas, marker=',')
    # ax2[1].plot(fit_RR.xmins, fit_RR.sigmas, marker=',')
    # ax2[1].plot(fit_LL.xmins, fit_LL.alphas, marker=',')
    # ax2[1].plot(fit_RR.xmins, fit_RR.sigmas, marker=',')
    ax2[0].set(ylabel=r'$D$')
    # ax2[1].set(ylabel=r'Standard Error', xlabel=r'$x_{min}$')
    ax2[1].set(
            ylabel=r'$\gamma$ $\%$ error', xlabel=r'$x_{min}$',
            xlim=[0, 0.2], ylim=[0, 10])
    ax2[1].xaxis.set_major_locator(plt.MaxNLocator(4))
    ax2[0].legend()
    

    plt.show()
    exit()
    return fit_LL.power_law.xmin, fit_RR.power_law.xmin

def add_powerlaw_fit(ax, data, xmin, plot=True):
    # fit = powerlaw.Fit(data, xmax=0.1) # -> result: power_law: 0.005017020688106992 1.8649063756555213
    # fit = powerlaw.Fit(data, xmin=0.005017020688106992, xmax=0.1)
    # fit = powerlaw.Fit(data, xmin=0.006777) # first minimum R-R!
    # fit = powerlaw.Fit(data, xmin=xmin)

    fit = powerlaw.Fit(data) # CHANGED!

    # powerlaw.plot_pdf(data, ax=ax, marker=',', c='#4053d3')
    # print('power_law:', fit.power_law.xmin, fit.power_law.alpha)
    # print('truncated_power_law:', fit.truncated_power_law.alpha, fit.truncated_power_law.Lambda)
    # fit.power_law.plot_pdf(ax=ax, ls='-', marker=',', c='r', label='power-law method 2', lw=2)

    # CHANGED!
    if plot == True:
        fit.power_law.plot_pdf(
            ax=ax, ls='--', marker=',', c='k',
            label=r'$\gamma = $' + f'{fit.power_law.alpha:.3f}',
            lw=1.5)
    return fit.power_law.alpha


def MLE_continuous_powerlaw(ax, data):
    # fit = powerlaw.Fit(data)
    # fit = powerlaw.Fit(data, xmin=0.001)
    # fit = powerlaw.Fit(data, xmax=0.08)

    # this is for truncated power law! # i can try it again with xmin=(0.002, 0.008)
    fit = powerlaw.Fit(data, xmin=0.0033009008398471765, xmin_distribution = 'truncated_power_law')
    # this shomehow killed my computer...
    # fit = powerlaw.Fit(data, xmin=(0.002, 0.008), xmin_distribution = 'truncated_power_law')
    # print(fit.power_law.xmin, fit.power_law.alpha)
    # print('-Comparing Distributions-')
    print('pl vs expo:', fit.distribution_compare('power_law', 'exponential'))
    print('pl vs truncpl:', fit.distribution_compare('power_law', 'truncated_power_law'))
    print('truncpl vs expo:', fit.distribution_compare('truncated_power_law', 'exponential'))
    
    powerlaw.plot_pdf(data, ax=ax)
    # xs = np.linspace(data.min(), data.max())
    # --- without xmax ---
    # first peak: 0.0055 1.82 (lower error)
    # second peak: 0.09792419199781272 4.964750163704389 (lower minimum)
    # alpha = 1.82
    # xmin =  0.0055
    # alpha = fit.power_law.alpha
    # xmin =  fit.truncated_power_law.xmin
    # fitted_pl = ((alpha - 1) / (xmin) ) * (( (xs) / (xmin)) ** -alpha)
    # print(fit.power_law.xmin)
    print('power_law:', fit.power_law.xmin, fit.power_law.alpha)
    print('truncated_power_law:', fit.truncated_power_law.alpha, fit.truncated_power_law.Lambda)
    # fitted_trunc_pl = truncated_powerlaw(
    #     xs, -fit.truncated_power_law.alpha, -fit.truncated_power_law.Lambda, xmin)
    # ax.plot(xs, fitted_pl, c='y', marker=',')
    # ax.plot(xs, fitted_trunc_pl, c='y', marker=',')
    
    # maybe fit a trunkated power law...?

    # fit.lognormal.plot_pdf(ax=ax, ls='--', marker=',', c='b')
    # fit.exponential.plot_pdf(ax=ax, ls='-', marker=',', c='r')
    # fit.stretched_exponential.plot_pdf(ax=ax, ls='--', marker=',', c='k')
    # fit.power_law.plot_pdf(ax=ax, ls='-', marker=',', c='k')
    fit.truncated_power_law.plot_pdf(ax=ax, ls='--', marker=',', c='k')
    ax.set(
        xlabel=r'$J_{ij}$', ylabel=r'$PDF$',
        xlim=[1e-3, 1e0],
        ylim=[3 * 1e-3, 300]
        )
    plt.show()
    # fig, ax  = plt.subplots(nrows=3, ncols=1, sharex=True)
    # ax = ax.ravel()
    # ax[0].plot(fit.xmins, fit.Ds, marker=',', label=r'$D$')
    # ax[1].plot(fit.xmins, fit.alphas, marker=',', label=r'$\alpha$')
    # ax[2].plot(fit.xmins, fit.sigmas, marker=',', label=r'$\sigma / \alpha$')

    # ax[0].set(ylabel=r'$D$')
    # ax[1].set(ylabel=r'$\alpha$')
    # ax[2].set(xlabel=r'$x_{min}$', ylabel=r'$Error$')
    
    # plt.show()