import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score

from pyplm.pipelines import model_pipeline
from pyplm.utilities import tools
# from pyplm.analyse.models import centered_histogram
import afunctions as afunc

def single_model_matrix_fullprocess(iMod, mod_selector, threshold_dictionary=None):
    file =  mod_selector['file']
    group = mod_selector['group']
    dataset = mod_selector['dataset']
    pname = mod_selector['pname']
    modpipe = model_pipeline(file, group, dataset)
    print(modpipe.datasets.shape)

    model = modpipe.datasets[iMod]
    modL = model[0:180, 0:180]
    modR = model[180:, 180:]
    modLR = model[180:, 0:180]
    copy_modL = np.copy(modL)
    copy_modR = np.copy(modR)
    afunc.LLLR_distribution_comparison(modL, modLR)

    afunc.LR_intercoupling_summary(modLR)
    mods = [modL, modR]
    lbls = [group + '-left', group + '-right']

    fig, ax = plt.subplots()
    matrix = ax.matshow(model, cmap='cividis')
    ax.axvline(180, marker=',', c='k')
    ax.axhline(180, marker=',', c='k')
    ax.set(xlabel=r'$i$', ylabel=r'$j$')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_label_position("top")
    cbar = fig.colorbar(matrix, ax=ax)
    cbar.set_label(r'$\theta _{ij}$')
    # plt.savefig('./prelim-results/grp-matrix-overview.png')
    plt.show()

    paramsL = tools.triu_flat(modL, k=0)
    paramsR = tools.triu_flat(modR, k=0)
    afunc.LLRR_parameters(paramsL, paramsR)

    if threshold_dictionary != None:
        fig, ax = plt.subplots(2, 2)
    else:
        fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    for iM in range(0, len(mods)):
        m = ax[iM].matshow(mods[iM], cmap='cividis')
        cbar = fig.colorbar(m, ax=ax[iM])
        cbar.set_label(r'$\theta _{ij}$')
    if threshold_dictionary != None:
        for iM in range(0, len(mods)):
            print(lbls[iM])
            TH = threshold_dictionary[lbls[iM]][0]
            # TH = 0
            # afunc.threshold_model(mods[iM], TH)
            mods[iM] = afunc.threshold_model(mods[iM], TH)
            mod_plot = np.copy(mods[iM])
            mod_plot[mod_plot == 0] = np.NaN
            m = ax[iM + 2].matshow(mod_plot, cmap='cividis')
            cbar = fig.colorbar(m, ax=ax[iM+2])
            cbar.set_label(r'$\theta _{ij}$')
    for a in ax:
        a.xaxis.set_major_locator(plt.MaxNLocator(4))
        a.yaxis.set_major_locator(plt.MaxNLocator(4))
    # plt.savefig('./prelim-results/grp-matrix-tailTH.png')
    plt.show()
    categorical_model = afunc.show_categorical_asymmetry_matrix(mods[0], mods[1])
    cat_flat = tools.triu_flat(categorical_model)
    # --- hot spice --- #
    cpy_mods = [copy_modL, copy_modR]
    lbls = ['L-L', 'R-R']
    THs = [0.00606, 0.00617]
    N, _ = categorical_model.shape
    conditions = [0, 1, 2, 3]
    cond_lbls = ['L-L & R-R DC', 'L-L & R-R C', 'L-L C, R-R DC', 'L-L DC, R-R C', ]
    
    for i in range(0, len(cpy_mods)):
        fig, ax = plt.subplots()

        params = tools.triu_flat(cpy_mods[i])
        # bins = int(params.size/100)
        bins = np.linspace(params.min(), params.max(), int(params.size/100))
        histargs = {'bins': bins, 'density': False}
        x, pdf = afunc.centered_histogram(params, 0, **histargs)
        # bin_length = x[1] - x[0]
        # pdf = pdf * bin_length
        mins = np.min(pdf)
        x = x[pdf > mins]
        pdf = pdf[pdf > mins]

        ax.plot(x, pdf, label=lbls[i])
        ax.axvline(THs[i], c='k', marker=',')
        nParams = int(params.size)
        print(f'--- nParams = {nParams} ---')

        paramCount = 0
        for condtion in conditions:
            condition_params = params[cat_flat == condtion]
            x, pdf = afunc.centered_histogram(condition_params, 0, **histargs)
            # bin_length = x[1] - x[0]
            # pdf = pdf * bin_length
            mins = np.min(pdf)
            x = x[pdf > mins]
            pdf = pdf[pdf > mins]
            paramCount += int(condition_params.size)
            print(
                f'{cond_lbls[condtion]}; nP={int(condition_params.size)},' +
                f'parameter_count = {paramCount}, P(condition) = {condition_params.size/nParams:.3}')
            ax.plot(x, pdf, label=cond_lbls[condtion])
        ax.set(xlabel=r'$J_{ij}$', ylabel=r'$N(J_{ij})$', yscale='log')
        plt.legend()
        # figname = './prelim-results/grp-tailTH-category-histogram' + lbls[i] + '.png'
        # plt.savefig(figname)
        plt.show()


def single_model_LR_shared_subnet(iMod, mod_selector, threshold_dictionary):
    file =  mod_selector['file']
    group = mod_selector['group']
    dataset = mod_selector['dataset']
    pname = mod_selector['pname']
    modpipe = model_pipeline(file, group, dataset)
    print(modpipe.datasets.shape)

    model = modpipe.datasets[iMod]
    modL = model[0:180, 0:180]
    modR = model[180:, 180:]
    # modLR = model[180:, 0:180]
    # copy_modL = np.copy(modL)
    # copy_modR = np.copy(modR)
    # might have to redo this elsewhere as well!!!! BUG ALERT!

    THL = threshold_dictionary[group + '-left'][0]
    THR = threshold_dictionary[group + '-right'][0]
    print(THL, THR)
    modL = afunc.threshold_model(modL, THL)
    modR = afunc.threshold_model(modR, THR)
    categorical_model = afunc.show_categorical_asymmetry_matrix(modL, modR)
    modL[categorical_model != 1] = 0
    modR[categorical_model != 1] = 0

    # check that I have chosen overlapping model
    # modL[modL != 0] = 1
    # modR[modR != 0] = 1
    # print(np.allclose(modL, modR))

    # paramsL = tools.triu_flat(modL, k=0)
    # paramsR = tools.triu_flat(modR, k=0)
    # afunc.LLRR_parameters(paramsL[paramsR != 0], paramsR[paramsR != 0])

    # input is shared subnetwork
    x = afunc.subnet_vary_threshold(modL, modR)

def single_model_histogram(ax, iMod, mod_selector, **pltargs):
    file =  mod_selector['file']
    group = mod_selector['group']
    dataset = mod_selector['dataset']
    pname = mod_selector['pname']
    print(pltargs)
    modpipe = model_pipeline(file, group, dataset)
    print(modpipe.datasets.shape)
    model = modpipe.datasets[iMod]
    N, _ = model.shape
    modL = model[0:180, 0:180]
    modR = model[180:, 180:]
    modLR = model[180:, 0:180]

    mods = [modL, modR, modLR]
    mods = [modL, modR]
    colors = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600']
    colors = [colors[0], colors[-1]]
    lbls = [group + '-left', group + '-right']
    for iM, mod in enumerate(mods):
        ax.axvline(0, c='k', marker=',')
        # ax.axhline(0.5, c='k', marker=',')

        Js = tools.triu_flat(mod)
        histargs = {'bins': int(Js.size/100), 'density': True}
        x, pdf = afunc.centered_histogram(Js, 0, **histargs)
        # pdf = np.convolve(pdf, np.ones(5)/5, mode='same')
        bin_length = x[1] - x[0]
        pdf = pdf * bin_length

        # filter out the zeros!
        mins = np.min(pdf)
        x = x[pdf > mins]
        pdf = pdf[pdf > mins]

        cdf = np.cumsum(pdf)
        Jpeak = x[pdf == pdf.max()]
        N = 180
        mean = np.mean(Js) * (N)
        std = np.std(Js) * (N ** 0.5)
        print(f'----{lbls[iM]}----')
        print(f'mean = {mean:.2f},\nstd = {std:.2f}')
        print(f'Jpeak is at {Jpeak}')
        print('-------------------')
        
        ax.plot(x, pdf, color=colors[iM], label=lbls[iM], **pltargs)
        
        # ax.plot(x, cdf, color=colors[iM], label=lbls[iM], **pltargs)

        x0_approx = tools.find_nearest(x, 0)
        cdf0 = cdf[x == x0_approx]
        print(x0_approx, cdf0[0])


def single_model_tail(ax, iMod, mod_selector, showMethod=False, **pltargs):
    file =  mod_selector['file']
    group = mod_selector['group']
    dataset = mod_selector['dataset']
    pname = mod_selector['pname']
    print(pltargs)
    modpipe = model_pipeline(file, group, dataset)
    print(modpipe.datasets.shape)
    model = modpipe.datasets[iMod]
    N, _ = model.shape
    modL = model[0:180, 0:180]
    modR = model[180:, 180:]
    modLR = model[180:, 0:180]

    mods = [modL, modR, modLR]
    mods = [modL, modR]
    colors = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600']
    colors = [colors[0], colors[-1]]
    lbls = [group + '-left', group + '-right']
    results_dictionary = {}
    for iM, mod in enumerate(mods):
        ax.axvline(0, c='k', marker=',')
        # ax.axhline(0.5, c='k', marker=',')

        Js = tools.triu_flat(mod)
        histargs = {'bins': int(Js.size/100), 'density': True}
        x, pdf = afunc.centered_histogram(Js, 0, **histargs)
        # pdf = np.convolve(pdf, np.ones(5)/5, mode='same')
        bin_length = x[1] - x[0]
        pdf = pdf * bin_length
        
        # filter out the zeros!
        mins = np.min(pdf)
        x = x[pdf > mins]
        pdf = pdf[pdf > mins]
        cdf = np.cumsum(pdf)

        r2mc = afunc.get_tail_PowLaws(x, pdf)
        iBestFit = np.argmax(r2mc[:, 0])

        iCutOff = len(x[x<=0]) + iBestFit
        # return the cuttoff value and some other stuff?
        print(f'--- {lbls[iM]} tail features ---')
        print(f'i_cutoff = {iCutOff:d}, J_cutoff = {x[iCutOff]:.3}, P(Jij <= J_cuttoff) = {cdf[iCutOff]:.3}')
        print(f'--------------------------------')
        results_dictionary[lbls[iM]] = [x[iCutOff], r2mc[iBestFit]]

        xfit = np.linspace(x[x>0].min(), x[x>0].max(), r2mc[:, 0].size)
        yfit = 10**((r2mc[iBestFit, 1] * np.log10(xfit)) + r2mc[iBestFit, 2])

        lbls[iM] = lbls[iM] + r' $\gamma = $' + f'{r2mc[iBestFit, 1]:.2f}'
        ax.plot(x, pdf, color=colors[iM], label=lbls[iM], **pltargs)
        ax.plot(xfit, yfit, marker=',', color=colors[iM], ls='--')
        ax.set(ylim=[0.99 * pdf.min(), 1.01 * pdf.max()])

    # this currently only does right I think
    if showMethod is True:
        print(showMethod)
        print(r2mc.shape, len(r2mc))
        figM, axM = plt.subplots()
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
            axM.plot(xfit, yfit, marker=',', c='r', alpha=0.5)
        
        yfit = 10**((r2mc[iBestFit, 1] * np.log10(xfit)) + r2mc[iBestFit, 2])
        axM.plot(xfit, yfit, marker=',', c='k')
        axM.set(
            xlabel=r'$J_{ij}$', ylabel=r'$P(J_{ij})$',
            xscale='log', yscale='log', ylim=[0.99 * pdf.min(), 1.01 * pdf.max()])
        # plt.savefig('./prelim-results/tail-method-fits.png')

        
        figP, axP = plt.subplots(3, 1, sharex=True)
        axP = axP.ravel()
        ylabels = [r'$R^{2}$', r'$\gamma$', r'$c$']
        for ia in range(0, axP.size):
            # axP[ia].plot(logx[:-10][mask], r2mc[:, ia])
            axP[ia].plot(x[x > 0][:-20], r2mc[:, ia], c='grey')
            axP[ia].plot(x[x > 0][:-20][mask], r2mc[mask, ia], c='r')
            axP[ia].plot(x[x > 0][:-20][iBestFit], r2mc[iBestFit, ia], c='k')
            axP[ia].set(ylabel=ylabels[ia])
        axP[0].set(ylim=[0, 1])
        # axP[0].axhline(R2_threshold_value, marker=',', c='k')
        axP[-1].set(xlabel=r'$\it{fitting-threshold}$ $(J_{ij})$', xscale='log')
        # plt.savefig('./prelim-results/tail-method-params.png')
    return results_dictionary