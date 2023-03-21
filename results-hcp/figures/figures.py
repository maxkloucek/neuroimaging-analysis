import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from figures.analysis import calc_C2
from scipy.ndimage import gaussian_filter1d
from . import axes
from . import io
from . import pl
from . import wrangle
from . import generic_axis_elements

from pyplm.utilities import tools
import hcp_helpers as helpers

CAT_COLS = [
        '#4053d3', '#ddb310', '#b51d14',
        '#00beff', '#fb49b0', '#00b25d', '#cacaca'
    ]

def mkfigure(**subplotkwargs):
    fig, ax = plt.subplots(squeeze=False, **subplotkwargs)
    labels = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 't', 'u', 'v'
        ]
    labels = [letter + ')' for letter in labels]
    labels = ['(' + letter for letter in labels]

    if ax.size != 1:
        ax_ravel = ax.ravel()
        for iax in range(0, ax_ravel.size):
            ax_ravel[iax].text(
                0.0, 1.0, labels[iax], transform=ax_ravel[iax].transAxes,
                # fontsize='medium', fontfamily='serif',
                horizontalalignment='left',
                verticalalignment='top',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=0))
    return fig, ax

# This is where I build completed figures
# A figure is a collection of axes that I format in some way
# surely it makes more sense to have it all here
# much less complicated
# I can have a little file that does each preparation and everything
# this makes more sense :)!
plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
HCP_FIGDIR = '/Users/mk14423/Dropbox/Apps/Overleaf/thesis/thesis/3-results-hcp/figures'

def bias_invB(invB, a, b, c):
    bias = (a * invB) + ((b ** 2) * (invB ** 2)) + c
    return bias

def bias_linB(B, a, c):
    bias = (a / B) + c
    return bias

def hcp_subsampling(
    file='/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
    group='grouped',
    true_temp=None,
    xkey = 'invB',
    ykey = 'std_J',
    threshold_B = 760400 * 0.5,
    label=None,
    save=False,
    ):  
        # zoom region function!
        df = io.get_subsampling_data_frame(file, group, true_temp)
        print(df)
        func = tools.linear
        # func = bias_linB
        fig, ax = plt.subplots()
        popt, perr = axes.plot_saturation(
            ax, df, xkey, ykey,
            run_iD=0, rescale_factor=1e4, threshold_B = threshold_B, fit_func=func)

        if ykey == 'std_J':
            ax_inset = ax.inset_axes([0.55, 0.12, 0.4, 0.4])
            ax_inset.set(xlim=[0, 0.1], ylim=[0.326, 0.338])
            ax.indicate_inset_zoom(ax_inset, edgecolor="black")
            axes.plot_saturation(
                ax_inset, df, xkey, ykey,
                run_iD=0, rescale_factor=1e4, threshold_B = threshold_B, fit_func=func)
        
        # do I want to return the thingy, I think I do!
        if ykey == 'T':
            ax_inset = ax.inset_axes([0.12, 0.06, 0.4, 0.4])
            ax_inset.set(xlim=[0.01, 0.1], ylim=[2.98, 3.058])
            ax.indicate_inset_zoom(ax_inset, edgecolor="black")
            axes.plot_saturation(
                ax_inset, df, xkey, ykey,
                run_iD=0, rescale_factor=1e4, threshold_B = threshold_B, fit_func=func)
        
        # ax.set(xlabel=r'$1/B(\times 10^{-4})$', ylabel=r'$T = 1 / \left( \sigma N^{1/2} \right)$')
        # ax.set(ylim=[0.9, 3])
        ax.set(xlabel=r'$1/B(\times 10^{-4})$', ylabel=r'$\sigma^{*} N^{1/2}$')
        if save is True:
            plt.savefig(os.path.join(HCP_FIGDIR, 'invB-std_J.png'))
        plt.show()

def hcp_subsampling_sig_and_T(
    file='/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
    group='grouped',
    true_temp=None,
    threshold_B = 760400 * 0.13,
    label=None,
    save=False,
    ):  
        # zoom region function!
        df = io.get_subsampling_data_frame(file, group, true_temp)
        print(df)
        func = tools.linear
        # func = bias_linB
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax = ax.ravel()
        popt, perr = axes.plot_saturation(
            ax[0], df, 'invB', 'std_J',
            run_iD=0, rescale_factor=1e4, threshold_B = threshold_B, fit_func=func)
        popt, perr = axes.plot_saturation(
            ax[1], df, 'invB', 'T',
            run_iD=0, rescale_factor=1e4, threshold_B = threshold_B, fit_func=func)

    
        ax_inset = ax[0].inset_axes([0.7, 0.15, 0.25, 0.5])
        ax_inset.set(xlim=[0, 0.1], ylim=[0.326, 0.338])
        ax[0].indicate_inset_zoom(ax_inset, edgecolor="black")
        axes.plot_saturation(
            ax_inset, df, 'invB', 'std_J',
            run_iD=0, rescale_factor=1e4, threshold_B=threshold_B, fit_func=func)
        
        ax_inset = ax[1].inset_axes([0.7, 0.4, 0.25, 0.5])
        ax_inset.set(xlim=[0, 0.1], ylim=[2.96, 3.071])
        ax[1].indicate_inset_zoom(ax_inset, edgecolor="black")
        axes.plot_saturation(
            ax_inset, df, 'invB', 'T',
            run_iD=0, rescale_factor=1e4, threshold_B=threshold_B, fit_func=func)
        
        # ax.set(ylim=[0.9, 3])
        ax[0].set(
            ylabel=r'$\sigma^{*} N^{1/2}$')
        ax[1].set(
            xlabel=r'$1/B_{ss}(\times 10^{-4})$',
            ylabel=r'$T = 1 / \left( \sigma^{*} N^{1/2} \right)$',
            xlim=[0, 2.7],
            ylim=[1.5, None])
        if save is True:
            plt.savefig(os.path.join(HCP_FIGDIR, 'invB-std_J-T.png'))
        plt.show()

def hcp_modshow(
    file='/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
    group='grouped',
    iD=0,
    save=False,
    ):
        model = io.get_model(file=file, group=group, iD=iD)
        fig, ax = plt.subplots()
        axes.plot_model(ax, model)
        if save is True:
            plt.savefig(os.path.join(HCP_FIGDIR, 'model-grouped.png'))
        plt.show()

def hcp_dissipation(save=False):
    CAT_COLS = [
            '#4053d3', '#ddb310', '#b51d14',
            '#00beff', '#fb49b0', '#00b25d', '#cacaca'
        ]
    # plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisaside.mplstyle')
    # let's plot %tiles
    # to understand the meaning of the bias better
    # need to mutiply these by 1e4 to get right
    popt = np.array([0.10822103, 0.32592218])
    popt[0] *= 1e4
    # how to understand the meaning of b1
    # T_true = 1 / sigma_true
    # T_infr = 1 / sigma_infr
    # y = np.abs(T_infr - T_true) / T_true
    Bs = np.logspace(3, 7, 100)
    sigma_infr = bias_linB(Bs, *popt)
    sigma_true = popt[1]
    error_levels = np.array([1e-1, 1e-2, 1e-3, 1e-4])

    x = Bs
    y = (sigma_infr - sigma_true) / sigma_true

    
    fig, ax = plt.subplots()
    for er in error_levels:
        ax.axhline(er, c='k', marker=',')
        txt = f'{er * 100:}%'
        ax.text(Bs[70], er * 1.05, txt)
    # this doesn't quite make sense... what I do want to measure?
    lbl = r'hcp: $b_{1}$' + f'= {int(popt[0])}'
    # turn this on again to add hcp b1!
    # ax.plot(x, y, marker=',', ls='--', label=lbl, c=CAT_COLS[-1])

    popts = [
        np.array([0.01, 0.32592218]),
        np.array([0.1, 0.32592218]),
        np.array([1, 0.32592218]),
        np.array([10, 0.32592218]),
    ]
    i = 0
    for popt in popts:
        popt[0] *= 1e4
        sigma_infr = bias_linB(Bs, *popt)
        y = (sigma_infr - popt[1]) / popt[1]
        lbl = r'ref: $b_{1}$' + f'= {int(popt[0])}'
        ls = '-'
        if i == 1:
            ls = '--'
        ax.plot(x, y, marker=',', label=lbl, c=CAT_COLS[i], ls=ls)
        i += 1
    ax.set(
        xlim=[x.min(),x.max()],
        xscale='log',
        yscale='log',
        xlabel=r'$B$',
        ylabel=r'$\varepsilon _{\sigma}$'
    )
    # ax.axhline(0.0042, c='k', marker=',', ls='--')
    # ax.text(Bs[70], 0.0042 * 1.05, f'{0.0042 * 100:}%')

    ax.legend(fontsize=10)
    if save is True:
            plt.savefig(os.path.join(HCP_FIGDIR, 'b1-explanation.png'))
    plt.show()

# I want the error as a function of b1 for fixe B
def error_unfinished(save=False):
    B = 1e5  # let's keep this fixed
    c = 1
    b1 = np.logspace(-4, +4, 100)
    # popt = np.array([0.10822103, 0.32592218])
    # popt[0] *= 1e4
    # how to understand the meaning of b1
    # T_true = 1 / sigma_true
    # T_infr = 1 / sigma_infr
    # y = np.abs(T_infr - T_true) / T_true
    # Bs = np.logspace(3, 7, 100)
    sigma_infr = bias_linB(B, b1, c)
    y = (sigma_infr - c) / c
    plt.plot(b1, y)
    plt.show()


def tail():
    file='/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5'
    group='grouped'
    iD=0
    model = io.get_model(file=file, group=group, iD=iD)

    # -- method2 -- #
    xmin_LL, xmin_RR = pl.powerlaw_fit_method2(model)
    parameters_LL = tools.triu_flat(model[0:180, 0:180])
    parameters_RR = tools.triu_flat(model[180:, 180:]) 
    # model = model[0:180, 0:180]

    parameters_LL = parameters_LL[parameters_LL > 0]
    parameters_RR = parameters_RR[parameters_RR > 0]
    # model = model[0:180, 0:180]
    fig, ax = mkfigure(nrows=2, ncols=1, sharex=True)
    print(ax.shape)
    # I need to start using the make
    # let's do both thingies on the same plot!
    axes.distribution_tail_fit(ax[0, 0], parameters_LL)
    pl.add_powerlaw_fit(ax[0, 0], parameters_LL, xmin_LL)

    axes.distribution_tail_fit(ax[1, 0], parameters_RR)
    pl.add_powerlaw_fit(ax[1, 0], parameters_RR, xmin_RR)
    # generic_axis_elements.histogram(ax, parameters, bins='auto', density=True)
    # powerlaw.plot_pdf(parameters, ax=ax)
    
    # ax.axvline(0.005, c='k', marker=',')
    ax[0, 0].set(
        xlim=[5e-3, 0.4],
        ylim=[3e-2, 100],
        ylabel=r'$PDF$ - Left'
        )
    ax[1, 0].set(
        xlim=[5e-3, 0.4],
        ylim=[3e-2, 100],
        xlabel=r'$J_{ij}$',
        ylabel=r'$PDF$ - Right'
        )
    ax[0, 0].legend()
    plt.show()

# plot hi vs sum sigma Jij!
    

def hcp_mod_tail_method1(
    file='/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
    group='grouped',
    iD=0,
    save=False):
        model = io.get_model(file=file, group=group, iD=iD)
        fig, ax = plt.subplots()
        axes.single_model_tail(ax, model, showMethod=True)
        plt.show()

def power_law_construction(
    file='/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
    group='grouped',
    iD=0,
    save=False):
        model = io.get_model(file=file, group=group, iD=iD)
        model = model[0:180, 0:180]
        # model = model[180:, 180:]
        parameters = tools.triu_flat(model)
        parameters = parameters[parameters > 0]
        print(parameters.shape)
        fig, ax = plt.subplots()
        # bins=np.logspace(-3, 0, 100)
        # bins='auto'
        # bins=int(parameters.size/100)
        bins='auto'
        pdf, x = np.histogram(parameters, bins=bins, density=True)
        ax.plot(x[:-1], pdf, c='grey', ls='none')
        # add on the fitting as well...?
        # makes a complicated graph really!
        print(parameters.shape)
        pl.MLE_continuous_powerlaw(ax, parameters)
        # have these on the same axis to really get to grips with it...
        # plt.matshow(model)
        plt.show()


# ---- pre-processing --- #
def time_stuff(
    # file='/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
    # group='individuals',
    # iD=0,
    save=False
    ):  
        # time_series = io.get_raw_data(iD=0)
        # z_trajectory, spin_trajectory = wrangle.binarize(time_series)
        # spin_trajectory2 = wrangle.binarize2(time_series)
        # print(z_trajectory.shape, spin_trajectory.shape)
        # i = 44
        # single_spin_time_series = time_series[:, i]
        # # label = ? get label might be nice here!
        # # print(time_series.shape)
        # # let's just pretend I didn't z-score it!
        # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        # ax = ax.ravel()
        # ax[0].plot(time_series[0:200, i], marker=',')
        # ax[1].plot(z_trajectory[0:200, i], marker=',')
        # ax[2].plot(spin_trajectory[0:200, i], marker=',', alpha=0.5)
        # ax[2].plot(spin_trajectory2[0:200, i], marker=',', alpha=0.5)
        # plt.show()
        # ok this clearly means something isn't matched up properly...?
        # # let's plot the distributions as well..?
        # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        # ax = ax.ravel()
        # ax[0].hist(time_series.ravel(), bins='auto')
        # ax[1].hist(z_trajectory.ravel(), bins='auto')
        # ax[2].hist(spin_trajectory.ravel(), bins='auto')
        # plt.show()
        # let's get the mean out of this stuff!

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        iD_max = 10 # max is 161
        # why does it have this periodicity..?
        iDs = np.arange(0, iD_max)
        raw_ac = np.zeros((iD_max, 4724))
        z_ac = np.zeros((iD_max, 4724))
        spin_ac = np.zeros((iD_max, 4724))
        for iD in iDs:
            print(iD)
            time_series = io.get_raw_data(iD)
            z_series, spin_series = wrangle.binarize(time_series)
            # spin_series = wrangle.binarize2(time_series)
            print(time_series.shape, z_series.shape, spin_series.shape)

            raw_autocorr, tau = wrangle.overlap(time_series)
            raw_ac[iD, :] = raw_autocorr
            z_autocorr, tau = wrangle.overlap(z_series)
            z_ac[iD, :] = z_autocorr
            spin_autocorr, tau = wrangle.overlap(spin_series)
            spin_ac[iD, :] = spin_autocorr

        subset = 1000
        raw_ac = raw_ac[:, 0:subset]
        z_ac = z_ac[:, 0:subset]
        spin_ac = spin_ac[:, 0:subset]

        print(spin_ac.shape)
        raw_ac_mean = np.mean(raw_ac, axis=0)
        raw_ac_stds = np.std(raw_ac, axis=0)

        z_ac_mean = np.mean(z_ac, axis=0)
        z_ac_stds = np.std(z_ac, axis=0)

        spin_ac_mean = np.mean(spin_ac, axis=0)
        spin_ac_stds = np.std(spin_ac, axis=0)
        print(spin_ac_stds.shape)
        xs = np.arange(0, raw_ac_mean.size)
    
        # raw_ac_mean = np.convolve(raw_ac_mean, np.ones(2)/2, mode='same')
        # z_ac_mean = np.convolve(z_ac_mean, np.ones(2)/2, mode='same')
        # spin_ac_mean = np.convolve(spin_ac_mean, np.ones(2)/2, mode='same')
        ax.errorbar(
            x=xs, y=raw_ac_mean,
            # yerr=raw_ac_stds,
            marker=',', label='raw data')
        ax.errorbar(
            x = xs, y=z_ac_mean,
            # yerr=z_ac_stds,
            marker=',', label='z scored')
        ax.errorbar(
            x = xs, y=spin_ac_mean,
            # yerr=spin_ac_stds,
            marker=',', label='binarised (spins)')
        ax.set(
            xlabel=r'$\Delta t$',
            ylabel=r'$C_{t} \left( \Delta t \right)$',
            ylim=[-0.06, 0.06],)
        ax.legend()
        if save is True:
            plt.savefig(os.path.join(HCP_FIGDIR, 'autocorrelationx.png'))
        plt.show()
        # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        # autocorr, tau = wrangle.overlap(time_series)
        # print(tau)
        # ax.plot(autocorr[0:200], label='raw')
        # autocorr, tau = wrangle.overlap(z_trajectory)
        # print(tau)
        # ax.plot(autocorr[0:200], label='z')
        # autocorr, tau = wrangle.overlap(spin_trajectory)
        # print(tau)
        # ax.plot(autocorr[0:200], label='config')
        # ax.legend()
        # plt.show()
        # then let's do the time series!

# --- trajecotry figures --- #
from . import analysis

def sweep_plot(save=False):
    print('hi!')
    # file = '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5'
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    # file = '/Volumes/IT047719/HCP_analyised/HCP_rsfmri_added_data.hdf5'
    group = 'grouped'
    trajecotries = analysis.SweepAnalysis(file, group, ['m', 'q', 'C2'])
    # trajecotries.calculate_observables(0)
    obs = trajecotries.load_analysed_data()
    # fig, ax = plt.subplots()
    fig, ax = mkfigure(nrows=2, ncols=1, sharex=True)
    print(ax.shape)
    axes.sweep_plot(ax, obs)
    ax[0, 0].legend()
    # ax[1, 0].legend()
    # we find the same thing, so that's good!
    ax[0, 0].set(xlim=[None, 1.5])
    ax[0, 0].set(ylim=[-0.1, None], ylabel=r'$m$, $q$')
    ax[1, 0].set(ylim=[0.1, 60], xlabel=r'$T_{f}$', ylabel=r'$C^2$')
    ax[0, 0].axvline(1, marker=',', c='k')
    ax[1, 0].axvline(1, marker=',', c='k')
    if save is True:
        plt.savefig(os.path.join(HCP_FIGDIR, 'hcp-sweep.png'))
    plt.show()


def threshold_test(save=False):
    print('hi!')
    # file = '/Volumes/IT047719/HCP_analyised/HCP_rsfmri_added_data2.hdf5'
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    # file = '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5'
    # group = 'grouped'
    # sweep_group = 'sweepTH_symmetric'
    # trajecotries = analysis.SweepAnalysis2(file, group, sweep_group, ['m', 'q', 'C2'])
    # # trajecotries.calculate_observables(ds_label='th-sweep')
    # obs = trajecotries.load_analysed_data(ds_label='th-sweep')

    
    # fig, ax = plt.subplots()
    fig, ax = mkfigure(nrows=2, ncols=1, sharex=True)
    print(ax.shape)

    group = 'grouped'
    sweep_group = 'sweepTH_symmetric'
    trajecotries = analysis.SweepAnalysis2(file, group, sweep_group, ['m', 'q', 'C2'])
    # trajecotries.calculate_observables()
    obs = trajecotries.load_analysed_data()

    axes.sweep_plot(ax, obs, param_key='param')
    ax[0, 0].legend()
    # ax[1, 0].legend()
    # we find the same thing, so that's good!
    # ax[0, 0].set(xlim=[None, 1.5])

    # group = 'grouped'
    # sweep_group = 'sweepTH_positive'
    # trajecotries = analysis.SweepAnalysis2(file, group, sweep_group, ['m', 'q', 'C2'])
    # # trajecotries.calculate_observables()
    # obs = trajecotries.load_analysed_data()
    # # I want to look at some trajectories to understand this better...!

    # axes.sweep_plot(ax, obs, param_key='param')
    ax[0, 0].legend()
    # ax[1, 0].legend()
    # we find the same thing, so that's good!
    # ax[0, 0].set(xlim=[None, 1.5])

    ax[0, 0].set(
        ylabel=r'$m$, $q$',
        xscale='log'
        )
    ax[1, 0].set(
        xlabel=r'$\alpha$', ylabel=r'$C^2$',
        xscale='log'
        )
    # ax[0, 0].axvline(1, marker=',', c='k')
    # ax[1, 0].axvline(1, marker=',', c='k')
    # not sure I even need to plot m and q!
    if save is True:
        plt.savefig(os.path.join(HCP_FIGDIR, 'hcp-sweep.png'))
    plt.show()


def threshold_analysis(save=False):
    # file = '/Volumes/IT047719/HCP_analyised/HCP_rsfmri_added_data2.hdf5'
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    # file = '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5'
    group = 'grouped'
    sweep_group = 'sweepTH_symmetric'
    sweep_analysis = analysis.SweepAnalysis2(file, group, sweep_group, ['m', 'q', 'C2'])
    # trajecotries.calculate_observables(ds_label='th-sweep')

    # this is the simplest underlying strucutre before it breaks... maybe not that helpful?

    # th_mod, th_val = sweep_analysis.load_threshold_model(30)
    # fig, ax = plt.subplots()
    # mat = ax.matshow(th_mod)
    # fig.colorbar(mat)
    # plt.show()
    # distrios for the 3 points that I hihglight on the method :)!
    fig, ax = plt.subplots()
    iMs = [30, 35, 41]
    # I don't know how to compare these histograms properly...
    # this desont really make sense...
    # oh cause my bins keep changing
    # or maybe not...? surely at higher N the bins should be unaffected!
    bins = np.linspace(-0.06212955659652472, 0.1, 800)
    for i in range(0, len(iMs)):
        th_mod, th_val = sweep_analysis.load_threshold_model(iMs[i])
        # th_mod = th_mod[0:180, 0:180]
        # print((0.1-th_mod.min()) / th_val)
        # print(th_mod.min())
        # bins = np.linspace(th_mod.min(), 0.1, nbins[i])
        params = tools.triu_flat(th_mod)
        params_all = params.size
        params = params[params != 0]
        params_neq0 = params.size
        cp = params_neq0/params_all * 100
        print(th_val, params_neq0/params_all * 100)
        # this doesn;t really seem to have worked, but I really don't
        # know why...?
        # something is wrong for sure...
        pdf, x = np.histogram(params, bins=bins, density=False)
        x = x[:-1] + (0.5 * np.diff(x))
        # x = x[:-1]
        lbl = r'$\delta=$ ' + f'{th_val:.3f}, {cp:.0f}% retained'
        ax.plot(x, pdf, marker=',', label=lbl)
        # ax.axvspan(-th_val, +th_val, alpha=0.5, color='orange')
        # ax.axvline(-th_val, marker=',', c='k')
        # ax.axvline(+th_val, marker=',', c='k')
    ax.set(xlabel=r'$J_{ij}$', ylabel=r'$Count$')
    plt.legend()
    plt.show()
    
    workout_best_th = True
    if workout_best_th == True:
        fig, ax = mkfigure()

        param_key = 'param'

        trajectory_df = sweep_analysis.load_analysed_data(ds_label='th-sweep')
        df_means = trajectory_df.groupby([param_key], as_index=True).mean()
        df_stds = trajectory_df.groupby([param_key], as_index=True).std(ddof=0)
        df_stds.fillna(0)
        df_means = df_means.reset_index()
        df_stds = df_stds.reset_index()
        generic_axis_elements.df_plot(
            ax[0, 0], param_key, 'C2',
            df_means, df_stds, marker=',', label=r'$C^{2}$', c='#b51d14')
        
        model_df = sweep_analysis.load_threshold_models_obs(ds_label='th-sweep')
        print(model_df)
        # I think that the threhold value is something
        ax2 = ax[0, 0].twinx()
        # ax2.plot(model_df[param_key], model_df['k-mean'], marker='.', label='k-mean')
        # ax2.fill_between(
        #     model_df[param_key],
        #     model_df['k-mean'] - model_df['k-std'],
        #     model_df['k-mean'] + model_df['k-std'], alpha=0.5
        #     )
        ax2.plot(model_df[param_key], model_df['sG-nN'] / 360, marker=',', label='Nodes')
        ax2.plot(model_df[param_key], model_df['sG-nE'] / 64620, marker=',', label='Edges')
        # ax2.fill_between(
        #     model_df[param_key],
        #     model_df['sG-k-mean'] - model_df['sG-k-std'],
        #     model_df['sG-k-mean'] + model_df['sG-k-std'], alpha=0.5
        #     )
        # generic_axis_elements.df_plot(ax2, param_key, 'k-mean', model_df, marker='.', label='mean')
        # generic_axis_elements.df_plot(ax2, param_key, 'k-mean', model_df, marker='.')
        # generic_axis_elements.df_plot(ax2, param_key, 'k-max', model_df, marker='.', label='max')
        ax2.legend(loc='lower left')
        ax[0, 0].set(
            xlabel=r'$\delta$', ylabel=r'$C^2$',
            xscale='log'
            )
        ax2.set(
            ylabel=r'Subgraph percentage $(\%)$'
        )
        ax[0, 0].axvline(model_df[param_key].iloc[30], marker=',', c='k')
        ax[0, 0].axvline(model_df[param_key].iloc[35], marker=',', c='k')
        ax[0, 0].axvline(model_df[param_key].iloc[41], marker=',', c='k')

        if save is True:
            plt.savefig(os.path.join(HCP_FIGDIR, 'hcp-sweep.png'))
        plt.show()

        print(df_means)

        # fig, ax = plt.subplots()
        # model_df['sG-nN'] = model_df['sG-nN'] / 360.0
        # df_means['C2'] = (df_means['C2'] - df_means['C2'].iloc[-1]) 
        # df_means['C2'] = df_means['C2'] / df_means['C2'].iloc[0]
        # print(df_means)
        # ax.plot(model_df[param_key], np.abs(df_means['C2'] - model_df['sG-nN']))
        # ax.set(xscale='log')
        # ax.axvline(model_df[param_key].iloc[30], marker=',', c='k')
        # ax.axvline(model_df[param_key].iloc[35], marker=',', c='k')
        # ax.axvline(model_df[param_key].iloc[41], marker=',', c='k')
        # plt.show()


# either way I'm learning some cool stuff here :)!
def threshold_analysis2(save=False):
    # work on this later! For now try get the time sereis!
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    group = 'grouped'
    # soemthings happening already - check this later!
    sweep_group = 'sweepTH_symmetric'
    # sweep_group = 'sweepTH_positive'
    sweep_analysis = analysis.SweepAnalysis2(file, group, sweep_group, ['m', 'q', 'C2'])

    # with h5py.File(file, 'r') as f:
    #     g = f[group]
    #     data_traj = g['configurations'][0, :, :]
    #     print(data_traj.shape)
    # C2_data = calc_C2(data_traj)
    # print(C2_data)

    fig, ax = plt.subplots()
    iMs = [30, 35, 41]
    bins = np.linspace(-0.06212955659652472, 0.1, 800)
    for i in range(0, len(iMs)):
        th_mod, th_val = sweep_analysis.load_threshold_model(iMs[i])
        # this can still change to to use load_models!
        # deltas, sym_mod, _, pos_mod, _ = helpers.get_example_threshold_trajecotries(
        # iMs[i])
        # th_mod = pos_mod
        # th_val = deltas[iMs[i]]
        # not sure wjhat I'm doing today..
        params = tools.triu_flat(th_mod)
        params_all = params.size
        params = params[params != 0]
        params_neq0 = params.size
        cp = params_neq0/params_all * 100
        print(th_val, params_neq0/params_all * 100)

        pdf, x = np.histogram(params, bins=bins, density=False)
        x = x[:-1] + (0.5 * np.diff(x))
        # x = x[:-1]
        lbl = r'$\delta=$ ' + f'{th_val:.3f}, {cp:.0f}% retained'
        ax.plot(x, pdf, marker=',', label=lbl)
        # ax.axvspan(-th_val, +th_val, alpha=0.5, color='orange')
        # ax.axvline(-th_val, marker=',', c='k')
        # ax.axvline(+th_val, marker=',', c='k')
    ax.set(xlabel=r'$J_{ij}$', ylabel=r'$Count$')
    plt.legend()
    plt.show()

    workout_best_th = True
    if workout_best_th == True:
        fig, ax = mkfigure()

        param_key = 'param'

        trajectory_df = sweep_analysis.load_analysed_data()
        df_means = trajectory_df.groupby([param_key], as_index=True).mean()
        df_stds = trajectory_df.groupby([param_key], as_index=True).std(ddof=0)
        df_stds.fillna(0)
        df_means = df_means.reset_index()
        df_stds = df_stds.reset_index()
        generic_axis_elements.df_plot(
            ax[0, 0], param_key, 'C2',
            df_means, df_stds, marker=',', label=r'$C^{2}$', c='#b51d14')
        
        model_df = sweep_analysis.load_threshold_models_obs()
        # print(model_df)
        # this won't work because it's ass!
        # I think that the threhold value is something
        ax2 = ax[0, 0].twinx()
        # ax2.plot(model_df[param_key], model_df['k-mean'], marker='.', label='k-mean')
        # ax2.fill_between(
        #     model_df[param_key],
        #     model_df['k-mean'] - model_df['k-std'],
        #     model_df['k-mean'] + model_df['k-std'], alpha=0.5
        #     )
        ax2.plot(model_df[param_key], model_df['sG-nN'] / 360, marker=',', label='Nodes')
        ax2.plot(model_df[param_key], model_df['sG-nE'] / 64620, marker=',', label='Edges')
        # ax2.fill_between(
        #     model_df[param_key],
        #     model_df['sG-k-mean'] - model_df['sG-k-std'],
        #     model_df['sG-k-mean'] + model_df['sG-k-std'], alpha=0.5
        #     )
        # generic_axis_elements.df_plot(ax2, param_key, 'k-mean', model_df, marker='.', label='mean')
        # generic_axis_elements.df_plot(ax2, param_key, 'k-mean', model_df, marker='.')
        # generic_axis_elements.df_plot(ax2, param_key, 'k-max', model_df, marker='.', label='max')
        ax2.legend(loc='lower left')
        ax[0, 0].set(
            xlabel=r'$\delta$', ylabel=r'$C^2$',
            xscale='log'
            )
        ax2.set(
            ylabel=r'Subgraph percentage $(\%)$'
        )
        ax[0, 0].axvline(model_df[param_key].iloc[30], marker=',', c='k')
        ax[0, 0].axvline(model_df[param_key].iloc[35], marker=',', c='k')
        ax[0, 0].axvline(model_df[param_key].iloc[41], marker=',', c='k')

        if save is True:
            plt.savefig(os.path.join(HCP_FIGDIR, 'hcp-sweep.png'))
        plt.show()

        # print(df_means)

        # fig, ax = plt.subplots()
        # model_df['sG-nN'] = model_df['sG-nN'] / 360.0
        # df_means['C2'] = (df_means['C2'] - df_means['C2'].iloc[-1]) 
        # df_means['C2'] = df_means['C2'] / df_means['C2'].iloc[0]
        # print(df_means)
        # ax.plot(model_df[param_key], np.abs(df_means['C2'] - model_df['sG-nN']))
        # ax.set(xscale='log')
        # ax.axvline(model_df[param_key].iloc[30], marker=',', c='k')
        # ax.axvline(model_df[param_key].iloc[35], marker=',', c='k')
        # ax.axvline(model_df[param_key].iloc[41], marker=',', c='k')
        # plt.show()



def comparing_threshold_model_obs(save=False):
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    group = 'grouped'
    labels = ['symmetric', 'positive']
    param_key = 'param'
    sweep_groups = ['sweepTH_symmetric', 'sweepTH_positive']
    mqC2_dfs = []
    network_dfs = []
    for sg in sweep_groups:
        sweep_analysis = analysis.SweepAnalysis2(file, group, sg, ['m', 'q', 'C2'])
        mqC2_df = sweep_analysis.load_analysed_data()
        mqC2_df = mqC2_df.groupby([param_key], as_index=True).mean()
        mqC2_df = mqC2_df.reset_index() # this has m, q, C2
        network_df = sweep_analysis.load_threshold_models_obs()
        # mqC2_df['param'] = np.arange(0, 50)
        # network_df['param'] = np.arange(0, 50)
        mqC2_dfs.append(mqC2_df)
        network_dfs.append(network_df)

    figsize = plt.rcParams['figure.figsize']  # w, h
    figsize[1] = figsize[1] * 1.5
    print(figsize)
    fig, ax = mkfigure(nrows=3, ncols=1, sharex=True, figsize=figsize)
    
    linestyles = ['-', '--']
    for i in range(0, 2):
        ax[1, 0].plot(
            mqC2_dfs[i]['param'], mqC2_dfs[i]['C2'],
            ls=linestyles[i], label=labels[i], marker='.')

        # ax[2, 0].plot(
        #     mqC2_dfs[i]['param'], mqC2_dfs[i]['m'],
        #     ls=linestyles[i], label=labels[i], marker='.')
        ax[0, 0].plot(
            mqC2_dfs[i]['param'], mqC2_dfs[i]['q'],
            ls=linestyles[i], label=labels[i], marker='.')
        # ax[1, 0].plot(
        #     network_dfs[i]['param'], network_dfs[i]['sG-nN'] / 360,
        #     # c=CAT_COLS[0],
        #     ls=linestyles[i], label=labels[i], marker='.')
        ax[2, 0].plot(
            network_dfs[i]['param'], 100 * (network_dfs[i]['sG-nE'] / 64620),
            # c=CAT_COLS[0],
            ls=linestyles[i], label=labels[i], marker='.')

    ax[1, 0].axhline(mqC2_dfs[0]['C2'].iloc[0], c='k', marker=',', ls=':')
    # ax[0, 0].axhline(
    #         mqC2_dfs[0]['C2'].iloc[0] - (mqC2_dfs[0]['C2'].iloc[0] * 0.05),
    #         c='k', marker=',', ls=':')
    for a in ax.ravel():
        print('------------------')
        print(network_dfs[i][param_key].iloc[30])
        print(network_dfs[i][param_key].iloc[35])
        print(network_dfs[i][param_key].iloc[41])
        print('------------------')
        a.axvline(network_dfs[i][param_key].iloc[30], marker=',', c='k', ls='-')
        # a.axvline(network_dfs[i][param_key].iloc[35], marker=',', c='k', ls=':')
        a.axvline(network_dfs[i][param_key].iloc[41], marker=',', c='k', ls='-')
        a.axvspan(
            network_dfs[i][param_key].iloc[41],
            network_dfs[i][param_key].iloc[-1], fc='Grey', alpha=0.5)
        a.set(xscale='log')
    
    ax[0, 0].legend(loc='lower left')

    ax[0, 0].set(ylabel=r'$q$')
    ax[1, 0].set(ylabel=r'$C^2$')
    # ax[1, 0].set(ylabel=r'$\%$ kept nodes')
    ax[-1, 0].set(xlabel=r'$\delta$', ylabel=r'$\%$ kept edges')
    ax[-1, 0].set(xlim=[None, network_dfs[i][param_key].iloc[-1]])
    # ax.set(
    #     xlabel=r'$\delta$', ylabel=r'$obs$',
    #     xscale='log')
    plt.show()



def comparing_threshold_model_trajectories():
    # ----- #
    # let's compare what the tranjectories look like...?
    # it could be completely different models though...
    i_start = 0
    i_5percent = 30
    i_positive_peak = 38
    i_last_full = 41

    _, sym_mod, sym_traj, pos_mod, pos_traj = helpers.get_example_threshold_trajecotries(
            0, i_rep=5)

    N, _ = sym_mod.shape
    nrows = 3
    fw, fh = plt.rcParams['figure.figsize']
    A = fw/fh
    length = int(N * (A * nrows))
    print(length,pos_traj.shape)
    fig, ax = plt.subplots(nrows=nrows, ncols=1, sharex=True, sharey=True)
    ax = ax.ravel()
    start = 3000
    for i in range(0, nrows):
        print(start, start + length)
        # plot_traj = sym_traj[start:start + length, :]
        plot_traj = pos_traj[start:start + length, :]
        ax[i].matshow(plot_traj.T, interpolation='none')
        start = start + length
    
    nticks = 5
    ax[i].xaxis.tick_bottom()
    ax[0].tick_params(axis='y',  which='both', left=False, right=False, labelleft=False)
    ax[2].tick_params(axis='y',  which='both', left=False, right=False, labelleft=False)
    ax[1].yaxis.set_major_locator(plt.FixedLocator([0, 180, 359]))
    ax[-1].xaxis.set_major_locator(plt.MaxNLocator(3))
 
    # fig.suptitle('Positive Threshold Peak')
    # fig.supxlabel(r'$t \rightarrow$')
    # fig.supylabel(r'$\leftarrow i$')
    ax[-1].set_xlabel(r'$t \rightarrow$')
    ax[1].set_ylabel(r'$\leftarrow i$')

    plt.show()
    
    # fig, ax = plt.subplots()


    # fig, ax = plt.subplots(nrows= 1, ncols=2, squeeze=False)
    # cmap = 'PRGn'
    # for ia, th_choice in enumerate([i_start]):
    #     _, sym_mod, sym_traj, pos_mod, pos_traj = helpers.get_example_threshold_trajecotries(
    #         th_choice, i_rep=5)
    #     N, _ = sym_mod.shape

    #     sym_traj = sym_traj[500:1500, :]
    #     pos_traj = pos_traj[500:1500, :]

    #     sym_dot_matrix = helpers.configuration_similarity_matrix(sym_traj)
    #     pos_dot_matrix = helpers.configuration_similarity_matrix(pos_traj)
    #     ax[ia, 0].matshow(sym_dot_matrix, cmap=cmap, vmin=-0.2, vmax=0.2)
    #     ax[ia, 1].matshow(pos_dot_matrix, cmap=cmap, vmin=-0.2, vmax=0.2)
    # plt.show()
    # this is crap!
    # aha, this is ferromagnetic; I need to see the thingy!!!
    # plot qs and ms as well!!
    # np.fill_diagonal(dot_trajectories, 0)
    # fig, ax = plt.subplots()
    # im = ax.matshow(
    #     dot_trajectories,
    #     cmap=cmap,
    #     vmin=-0.2, vmax=0.2
    #     )
    # fig.colorbar(mappable=im, ax=ax)
    # ax.xaxis.tick_bottom()
    # ax.set(xlabel=r'$t$', ylabel=r'$t$')
    # plt.show()

    exit()


    start = 5000
    roi_lim = 360
    length = int(roi_lim)

    cmap_mod = 'cividis'
    cmap_traj = 'viridis'

    sym_mod = sym_mod[0:roi_lim, 0:roi_lim]
    pos_mod = pos_mod[0:roi_lim, 0:roi_lim]


    sym_traj = sym_traj.T
    pos_traj = pos_traj.T
   
    sym_traj = sym_traj[0:roi_lim, start: start + length]
    pos_traj = pos_traj[0:roi_lim, start: start + length]


    print(sym_traj.shape)

    vmin = sym_mod.min()
    vmax = sym_mod.max()
    print(vmin, vmax)
    fig, ax = mkfigure(nrows=2, ncols=2, sharey=True)
    ax[0, 0].matshow(sym_mod, cmap=cmap_mod, vmin=vmin, vmax=vmax)
    ax[0, 1].matshow(sym_traj, cmap=cmap_traj, vmin=vmin, vmax=vmax)

    ax[1, 0].matshow(pos_mod, cmap=cmap_mod)
    ax[1, 1].matshow(pos_traj, cmap=cmap_traj)

    for a in ax.ravel():
        # a.xaxis.set_major_locator(plt.MaxNLocator(3))
        # a.yaxis.set_major_locator(plt.MaxNLocator(3))
        a.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labeltop=False, labelbottom=False, labelleft=False)
    plt.show()



# either way I'm learning some cool stuff here :)!
def threshold_distributions(save=False):
    # work on this later! For now try get the time sereis!
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    group = 'grouped'
    # soemthings happening already - check this later!
    # sweep_group = 'sweepTH_symmetric'
    sweep_group = 'sweepTH_positive'
    sweep_analysis = analysis.SweepAnalysis2(file, group, sweep_group, ['m', 'q', 'C2'])

    # with h5py.File(file, 'r') as f:
    #     g = f[group]
    #     data_traj = g['configurations'][0, :, :]
    #     print(data_traj.shape)
    # C2_data = calc_C2(data_traj)
    # print(C2_data)

    

    fig, ax = plt.subplots()
    iMs = [30, 38, 41] # this is the spread cool!
    # oh cause I set the bin lim to 0.1 what if I set it higher?
    bins = np.linspace(-0.06212955659652472, 0.1, 800)
    delta_labels = [r'$\delta_{1}$', r'$\delta_{2}$', r'$\delta_{3}$']
    for i in range(0, len(iMs)):
        th_mod, th_val = sweep_analysis.load_threshold_model(iMs[i])
        # this can still change to to use load_models!
        # deltas, sym_mod, _, pos_mod, _ = helpers.get_example_threshold_trajecotries(
        # iMs[i])
        # th_mod = pos_mod
        # th_val = deltas[iMs[i]]
        # not sure wjhat I'm doing today..
        params = tools.triu_flat(th_mod)
        params_all = params.size
        params = params[params != 0]
        params_neq0 = params.size
        cp = params_neq0/params_all * 100
        print(th_val, params_neq0/params_all * 100)

        pdf, x = np.histogram(params, bins=bins, density=False)
        x = x[:-1] + (0.5 * np.diff(x))
        # pdf = gaussian_filter1d(pdf, sigma=1) # NO!
        # smooth this?
        # x = x[:-1]
        # lbl = r'$\delta=$ ' + f'{th_val:.3f}, {cp:.0f}% retained'
        lbl = delta_labels[i] + f', {cp:.0f}% retained'
        ax.plot(x, pdf, marker=',', label=lbl)
        # ax.axvspan(-th_val, +th_val, alpha=0.5, color='orange')
        # ax.axvline(-th_val, marker=',', c='k')
        # ax.axvline(+th_val, marker=',', c='k')
    ax.set(xlabel=r'$J_{ij}$', ylabel=r'$Count$')
    xmin_tail_th = 0.0045 
    ax.axvline(xmin_tail_th, ls='--', marker=',', c='k', label=r'$x_{min}$ tail')
    if sweep_group == 'sweepTH_positive':
        ax.set(xlim=[0.001, None])
    plt.legend()
    plt.show()
