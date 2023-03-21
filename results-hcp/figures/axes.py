import numpy as np
import matplotlib.pyplot as plt
from . import generic_axis_elements
from . import subsample_wrangle
from pyplm.utilities import tools

CAT_COLS = [
        '#4053d3', '#ddb310', '#b51d14',
        '#00beff', '#fb49b0', '#00b25d', '#cacaca'
    ]

def plot_saturation(ax, df, xkey, ykey, run_iD, rescale_factor, threshold_B, fit_func):

    df = df.loc[df['iD'] == run_iD]
    df['invB'] = df['invB'] * rescale_factor
    df_means = df.groupby(['B'], as_index=True).mean()
    df_stds = df.groupby(['B'], as_index=True).std(ddof=0)
    df_stds.fillna(0)
    df_means = df_means.reset_index()
    df_stds = df_stds.reset_index()
    # I can pass two different dictionaries to mkae it work :)! i.e. one for df_plot args and
    # one for add fit plt args!
    generic_axis_elements.df_plot(
        ax, xkey, ykey,
        df_means,
        # df_stds,
        ls='-', alpha=1, c=CAT_COLS[3], marker='o', markeredgewidth=0.3,)

    # threshold_B = 760400 * 0.5
    df_means = df_means.loc[df_means['B'] >= threshold_B]
    # print(df_means)

    xfit = np.linspace(0, df[xkey].max(), 1000)
    # xfit = np.linspace(1000, df[xkey].max(), 1000)
    # xfit = np.linspace(100, 1e6, 1000)
    # xfit = np.linspace(df_means[xkey].min(), df_means[xkey].max(), 1000)
    line, popt, perr, = generic_axis_elements.df_add_fit(
        ax, df_means, xkey, ykey, fit_func, xfit=xfit, show_error=False, marker=',', c='k', ls='--', zorder=50)
    return popt, perr

def plot_model(ax, model):
    generic_axis_elements.modshow(ax, model)


def distribution_tail_fit(ax, data, **pltargs):
    histargs = {
        # 'bins': int(data.size/100),
        'bins': 'auto',
        'density': True}
    x, pdf = generic_axis_elements.histogram(
        ax, data, filter_level=0, **histargs)
    mins = np.min(pdf)
    x = x[pdf > mins]
    pdf = pdf[pdf > mins]
    # if I don't do this then it's a mess
    pdf = pdf[x > 0]
    x = x[x > 0]
    r2mc = subsample_wrangle.get_tail_PowLaws(x, pdf)
    iBestFit = np.argmax(r2mc[:, 0])
    iCutOff = len(x[x<=0]) + iBestFit
    print(x[iCutOff], r2mc[iBestFit])

    xfit = np.linspace(x[x>0].min(), x[x>0].max(), r2mc[:, 0].size)
    yfit = 10**((r2mc[iBestFit, 1] * np.log10(xfit)) + r2mc[iBestFit, 2])
    lbl = r' $\gamma = $' + f'{r2mc[iBestFit, 1]:.2f}'
    # ax.plot(x, pdf, color=colors[iM], **pltargs)
    ax.plot(xfit, yfit, marker=',',  ls='-', label='power-law method 1', c='k', lw=2)


def single_model_tail(ax, model, showMethod=False, **pltargs):
    N, _ = model.shape
    modL = model[0:180, 0:180]
    modR = model[180:, 180:]
    modLR = model[180:, 0:180]
    np.fill_diagonal(modLR, 0)
    # mods = [modL, modR, modLR, model]
    mods = [modL, modR]
    # mods = [modLR]
    colors = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600']
    # colors = [colors[0], colors[2], colors[-1]]
    lbls = ['L-L', 'L-R', 'L-R', 'tot']
    results_dictionary = {}
    for iM, mod in enumerate(mods):
        ax.axvline(0, c='k', marker=',')
        # ax.axhline(0.5, c='k', marker=',')
        Js = tools.triu_flat(mod)
        # maybe we should do js > 0 only?
        Js = Js[Js > 0]
        bins = 'auto'
        # bins = int(Js.size/100)
        histargs = {'bins': bins, 'density': True}
        x, pdf = generic_axis_elements.histogram(
            ax, Js, filter_level=0, **histargs)

        mins = np.min(pdf)
        x = x[pdf > mins]
        pdf = pdf[pdf > mins]
        # if I don't do this then it's a mess
        pdf = pdf[x > 0]
        x = x[x > 0]
        r2mc = subsample_wrangle.get_tail_PowLaws(x, pdf)
        iBestFit = np.argmax(r2mc[:, 0])

        iCutOff = len(x[x<=0]) + iBestFit
        # return the cuttoff value and some other stuff?
        print(f'--- {lbls[iM]} tail features ---')
        # print(f'i_cutoff = {iCutOff:d}, J_cutoff = {x[iCutOff]:.3}, P(Jij <= J_cuttoff) = {cdf[iCutOff]:.3}')
        print(f'i_cutoff = {iCutOff:d}, J_cutoff = {x[iCutOff]:.3}')
        print(f'--------------------------------')
        results_dictionary[lbls[iM]] = [x[iCutOff], r2mc[iBestFit]]
        print([x[iCutOff], r2mc[iBestFit]])

        xfit = np.linspace(x[x>0].min(), x[x>0].max(), r2mc[:, 0].size)
        yfit = 10**((r2mc[iBestFit, 1] * np.log10(xfit)) + r2mc[iBestFit, 2])

        lbls[iM] = lbls[iM] + r' $\gamma = $' + f'{r2mc[iBestFit, 1]:.2f}'
        # ax.plot(x, pdf, color=colors[iM], **pltargs)
        ax.plot(xfit, yfit, marker=',', color=colors[iM], ls='--', label=lbls[iM])
        ax.set(
            ylabel=r'$PDF$',
            xlim=[1e-3, 1e0],
            # ylim=[0.99 * pdf.min(), 1.01 * pdf.max()]
            ylim=[0.02, 57]
            )
    ax.set(xscale='log', yscale='log')
    ax.legend()
    plt.show()
    # this currently only does right I think
    if showMethod is True:
        print(showMethod)
        print(x.shape, r2mc.shape, len(r2mc))
        # figM, axM = plt.subplots()
        fig = plt.figure()
        spec = fig.add_gridspec(ncols=3, nrows=2)
        axP = fig.add_subplot(spec[0, 0])
        axT = fig.add_subplot(spec[1, 0])
        axM = fig.add_subplot(spec[0:, 1:])

        axP.get_shared_x_axes().join(axP, axT)
        axP.set_xticklabels([])

        axM.plot(x, pdf, color=colors[iM], label=lbls[iM], **pltargs)
        cols = plt.cm.plasma(np.linspace(0, 1, len(r2mc)))
        xfit = np.linspace(x[x>0].min(), x[x>0].max(), r2mc[:, 0].size)
        for i in range(0, len(r2mc)):
            yfit = 10**((r2mc[i, 1] * np.log10(xfit)) + r2mc[i, 2])
            # axM.plot(xfit, yfit, marker=',', color=cols[i], ls='--', alpha=0.3)
            axM.plot(xfit, yfit, marker=',', color='grey', ls='--', alpha=0.3)

        R2_threshold_value = 0.9
        mask = r2mc[:, 0] > R2_threshold_value
        masked_fitting_array = r2mc[mask, :]

        for fit_params in masked_fitting_array:
            yfit = 10**((fit_params[1] * np.log10(xfit)) + fit_params[2])
            axM.plot(xfit, yfit, marker=',', c='#ddb310', alpha=0.5)
        
        yfit = 10**((r2mc[iBestFit, 1] * np.log10(xfit)) + r2mc[iBestFit, 2])
        axM.plot(xfit, yfit, marker=',', c='k')
        axM.set(
            xlabel=r'$J_{ij}$', ylabel=r'$P(J_{ij})$',
            xscale='log', yscale='log', ylim=[0.99 * pdf.min(), 1.01 * pdf.max()])
        # plt.savefig('./prelim-results/tail-method-fits.png')

        # axP = axP.ravel()
        # let's call it gamma!
        ylabels = [r'$R^{2}$', r'$\gamma$', r'$c$']
        axP.plot(x[x > 0][:-20], r2mc[:, 0], c='grey', marker=',')
        axP.plot(x[x > 0][:-20][mask], r2mc[mask, 0], c='#ddb310', marker=',')
        axP.plot(x[x > 0][:-20][iBestFit], r2mc[iBestFit, 0], c='k')

        axT.plot(x[x > 0][:-20], r2mc[:, 1], c='grey', marker=',')
        axT.plot(x[x > 0][:-20][mask], r2mc[mask, 1], c='#ddb310', marker=',')
        axT.plot(x[x > 0][:-20][iBestFit], r2mc[iBestFit, 1], c='k')
        axP.set(ylabel=ylabels[0], ylim=[0.7, .95])
        axT.set(ylabel=ylabels[1], xlabel=r'$x_{min}$', xlim=[None, 0.1], ylim=[-2.5, -1.5])
        plt.show()

        
        # figP, axP = plt.subplots(3, 1, sharex=True)
        # axP = axP.ravel()
        # ylabels = [r'$R^{2}$', r'$\gamma$', r'$c$']
        # for ia in range(0, axP.size):
        #     # axP[ia].plot(logx[:-10][mask], r2mc[:, ia])
        #     axP[ia].plot(x[x > 0][:-20], r2mc[:, ia], c='grey', marker=',')
        #     axP[ia].plot(x[x > 0][:-20][mask], r2mc[mask, ia], c='#ddb310', marker=',')
        #     axP[ia].plot(x[x > 0][:-20][iBestFit], r2mc[iBestFit, ia], c='k')
        #     axP[ia].set(ylabel=ylabels[ia])
        # axP[0].set(ylim=[0, 1])
        # # axP[0].axhline(R2_threshold_value, marker=',', c='k')
        # axP[-1].set(
        #     # xlabel=r'$\it{fitting-threshold}$ $(J_{ij})$',
        #     xlabel=r'$x_{min}$',
        #     # xscale='log'
        #     )
        # plt.show()
        # plt.savefig('./prelim-results/tail-method-params.png')
    return results_dictionary

def sweep_plot(ax, df, param_key='alpha'):
    print(df)
    # print('----')
    df_means = df.groupby([param_key], as_index=True).mean()
    df_stds = df.groupby([param_key], as_index=True).std(ddof=0)
    df_stds.fillna(0)
    df_means = df_means.reset_index()
    df_stds = df_stds.reset_index()
    generic_axis_elements.df_plot(
        ax[0, 0], param_key, 'm', df_means, df_stds, marker='.', label=r'$m$')
    generic_axis_elements.df_plot(
        ax[0, 0], param_key, 'q', df_means, df_stds, marker='.', label=r'$q$')
    generic_axis_elements.df_plot(
        ax[1, 0], param_key, 'C2', df_means, df_stds, marker='.', label=r'$C^{2}$', c='#b51d14')