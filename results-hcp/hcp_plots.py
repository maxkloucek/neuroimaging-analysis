import numpy as np
import matplotlib.pyplot as plt
import h5py
import hcp_helpers as helpers
from pyplm.utilities import tools
from figures import analysis
from figures import pl
from pyplm.plotting import mkfigure

from figures import analysis

def coupling_distribution():
    plm_model = helpers.load_grouped_model()
    N, _ = plm_model.shape
    params_plm = tools.triu_flat(plm_model)

    bins = np.linspace(params_plm.min(), params_plm.max(), 200)
    T = 1/(np.std(params_plm) * (N**0.5))
    print(T)

    fig, ax = plt.subplots()
    lbl = r'hcp, $T^{*}$' + f'={T:.3f}'
    helpers.distribution(ax, params_plm, bins=bins, marker=',', lw=2, label=lbl)
    ax.set(
        xlabel=r'$J_{ij}$',
        ylabel=r'$PDF$',
        ylim=[5e-4, None], yscale='log')
    ax.legend()
    plt.show()


def distribution_comparison():
    noMM_model, _ = helpers.load_noMM_model()
    N, _ = noMM_model.shape
    noMM_params = tools.triu_flat(noMM_model)
    noMM_T = 1/(np.std(noMM_params) * (N**0.5))
    print(N, noMM_T)

    hcp_model = helpers.load_grouped_model()
    N, _ = hcp_model.shape
    hcp_params = tools.triu_flat(hcp_model)
    hcp_T = 1/(np.std(hcp_params) * (N**0.5))
    print(N, hcp_T)

    print(np.min(noMM_params), np.min(hcp_params))
    print(np.max(noMM_params), np.max(hcp_params))
    bins = np.linspace(-0.215, 0.6423, 200)

    'noMM'
    fig, ax = plt.subplots()
    lbl = r'no-MM, $T^{*}$' + f'={noMM_T:.3f}'
    helpers.distribution(ax, noMM_params, bins=bins, marker=',', lw=2, label=lbl)
    lbl = r'hcp, $T^{*}$' + f'={hcp_T:.3f}'
    helpers.distribution(ax, hcp_params, bins=bins, marker=',', lw=2, label=lbl)
    ax.set(
        xlabel=r'$J_{ij}$',
        ylabel=r'$PDF$',
        ylim=[5e-4, None], yscale='log')
    ax.legend()
    plt.show()


def hvsJ():
    hcp_model = helpers.load_grouped_model()
    hcp_model = np.abs(hcp_model)
    hs = np.diag(hcp_model)
    sumJs = np.sum(hcp_model, axis=0)
    fig, ax = plt.subplots()
    ax.plot(hs, sumJs, ls='none')
    ax.set(xlabel=r'$|h_{i}|$', ylabel=r'$\Sigma_{i}$ $|J_{ij}|$')
    plt.show()


def example_trajectories():
    # lol these are just completley wrong...
    i_sg = 15
    i_crit = 31 # 26
    i_plm = 38 # 
    i_cor = 39 #
    # interesting that cirticality is agian at 0.8
    # what dose that tell us -> both models are shifted a constant amount
    # away from Tc? somewhat wierd right..?


    # selections = [i_sg, i_crit, i_plm, i_cor, -1]
    selections = [i_sg, i_crit, i_plm, -1]
    labels=[
        r'SG, $T_{f}=0.4$',
        r'Critical, $T_{f}=0.8$',
        r'PLM, $T_{f}=1.0$',
        r'Para, $T_{f}=2.5$']
    # selections = [0, 10, -1]
    print(labels)
    start = 8200
    roi_lim = 180
    # length = int(roi_lim * 1.8)
    length = int(roi_lim * 2.5)
    # hmmmm not sure what to do about this...

    fig = plt.figure()
    spec = fig.add_gridspec(3, 2, wspace=0.0, hspace=0.0)
    ax0 = fig.add_subplot(spec[0, :])
    ax1 = fig.add_subplot(spec[1, 0])
    ax2 = fig.add_subplot(spec[1, 1])
    ax3 = fig.add_subplot(spec[2, 0])
    ax4 = fig.add_subplot(spec[2, 1])
    ax = [ax0, ax1, ax2, ax3, ax4]
    # fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    # ax = ax.ravel()
    # plt.show()
    fpath ='/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    with h5py.File(fpath, 'r') as fin:
        input_trajectory = fin['grouped']['configurations'][0, start:start+(2*length), 0:roi_lim]
        print(input_trajectory.shape)
    # input_trajectory = input_trajectory[]
    # exit()
    ax[0].matshow(input_trajectory.T)
    ax[0].set(title='Brain Signal')
    # ax.plot(correlations_input, correlations_input, ls='-', c='k', marker=',', zorder=10)
    for iS in range(0, len(selections)):
        # fig, ax = plt.subplots()
        mc_trajectory = helpers.get_example_trajecotry(selections[iS])
        
        mc_trajectory = mc_trajectory[start:start+length, 0:roi_lim]
        print(iS, mc_trajectory.shape)
        ax[iS+1].matshow(mc_trajectory.T)
        ax[iS+1].set(title=labels[iS])
    for a in ax:
        # a.xaxis.set_major_locator(plt.MaxNLocator(3))
        # a.yaxis.set_major_locator(plt.MaxNLocator(3))
        a.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labeltop=False, labelbottom=False, labelleft=False)
    # ax[iA].yaxis.set_major_locator(plt.MaxNLocator(4))
    ax[-1].xaxis.tick_bottom()
    ax[-2].xaxis.tick_bottom()
    fig.supxlabel(r'time, $t \rightarrow$')
    fig.supylabel(r'$\leftarrow i$')
    plt.show()

# from tslearn.metrics import dtw
# from tslearn.clustering import TimeSeriesKMeans
# from tslearn.utils import to_time_series_dataset


def identifying_states():
    # lol these are just completley wrong...
    i_sg = 15
    i_crit = 29 # 31 # 26
    i_plm = 38 # 
    # interesting that cirticality is agian at 0.8
    # what dose that tell us -> both models are shifted a constant amount
    # away from Tc? somewhat wierd right..?

    # selections = [i_sg, i_crit, i_plm, i_cor, -1]
    selections = [i_sg, i_crit, i_plm, -1]
    labels=[
        r'SG, $T_{f}=0.4$',
        r'Critical, $T_{f}=0.8$',
        r'PLM, $T_{f}=1.0$',
        r'Para, $T_{f}=2.5$']
    # selections = [0, 10, -1]
    print(labels)
    start = 8200
    roi_lim = 360
    # length = int(roi_lim * 1.8)
    length = int(roi_lim * 2.5)
    # hmmmm not sure what to do about this...
    # let's try just with one simple trajectory!

    fpath ='/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    with h5py.File(fpath, 'r') as fin:
        print(fin['grouped'].keys())
        model = fin['grouped']['inferredModels'][0, :, :]
        N, _ = model.shape
        input_trajectory = fin['grouped']['configurations'][0, start:start+(length), 0:roi_lim]
    
    # ----- matrix  ----- #
    # helpers.configuration_similarity_matrix(input_trajectory)
    # # i.e. do I have "anti-states"
    # i = i_plm # i_sg = 15, i_crit = 31, i_plm = 38
    # mc_trajectory, _ = helpers.get_example_trajecotry(i)
    # mc_trajectory = mc_trajectory[start:start+length, 0:roi_lim]
    # helpers.configuration_similarity_matrix(mc_trajectory)
    # ----- running average ----- #
    mc_trajectory, _ = helpers.get_example_trajecotry(i_plm)
    # qs_plm = helpers.configuration_similarity_running_avrg(mc_trajectory)
    # print(np.mean(qs_plm), np.var(qs_plm))
    qs_plm = helpers.configuration_similarity_local_dot(mc_trajectory)
    print(np.mean(qs_plm), np.var(qs_plm))

    mc_trajectory, _ = helpers.get_example_trajecotry(i_crit)
    # qs_crit = helpers.configuration_similarity_running_avrg(mc_trajectory)
    # print(np.mean(qs_crit), np.var(qs_crit))
    qs_crit = helpers.configuration_similarity_local_dot(mc_trajectory)
    print(np.mean(qs_crit), np.var(qs_crit))

    mc_trajectory, _ = helpers.get_example_trajecotry(i_sg)
    # qs_sg = helpers.configuration_similarity_running_avrg(mc_trajectory)
    # print(np.mean(qs_sg), np.var(qs_sg))
    qs_sg = helpers.configuration_similarity_local_dot(mc_trajectory)
    print(np.mean(qs_sg), np.var(qs_sg))

    fig, ax = plt.subplots()
    ax.plot(qs_plm, marker=',', label='plm')
    ax.plot(qs_crit, marker=',', label='crit')
    ax.plot(qs_sg, marker=',', label='sg')
    ax.legend()
    plt.show()

    # what if I tried a "local dot product...?"
    temps = []
    chis = []
    for i in range(0, 100):
        mc_trajectory, temp = helpers.get_example_trajecotry(i)
        # qs = helpers.configuration_similarity_running_avrg(mc_trajectory)
        qs = helpers.configuration_similarity_local_dot(mc_trajectory)
        chi_q = np.var(qs)
        temps.append(temp)
        chis.append(chi_q)
    temps = np.array(temps)
    chis = np.array(chis)
    fig, ax = plt.subplots()
    ax.plot(temps, chis, marker=',')
    ax.set(xlabel=r'$T$', ylabel=r'$\chi$')
    ax.axvline(1, marker=',', ls='--', c='k')
    ax2 = ax.twinx()

    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    trajecotries = analysis.SweepAnalysis(file, 'grouped', ['m', 'q', 'C2'])
    obs = trajecotries.load_analysed_data()
    df_means = obs.groupby(['alpha'], as_index=True).mean()
    df_stds = obs.groupby(['alpha'], as_index=True).std(ddof=0)
    df_stds.fillna(0)
    df_means = df_means.reset_index()
    df_stds = df_stds.reset_index()
    ax2.plot(df_means['alpha'], df_means['C2'], c='r', marker=',')
    # changing window lenght causes agreement...
    plt.show()

def avalanche_distribution():
    i_sg = 15
    i_crit = 31
    i_plm = 38
    # let's try a few low temperature ones...
    # selections = [10, 15, 20, 25, 30, i_plm]
    # selections = [10, 15, 20, 31, 38]
    # selections = [25, 26, 27, 28, 29, 30, 31, 32, 33]
    selections = np.arange(10, 42, 2)
    # selections = [20, 25, 31, 38]

    bins = np.linspace(0, 1e4, 3000)
    fig, ax = plt.subplots()

    fpath ='/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    with h5py.File(fpath, 'r') as fin:
        input_trajectory = fin['grouped']['configurations'][0, 0:int(5e4), :]
        print(input_trajectory.shape)
    avs = analysis.calc_avalanche(input_trajectory)
    # gamma = pl.add_powerlaw_fit(ax, avs, None, plot=True)
    # lbl = r'$\gamma =$' + f' {gamma:.2f}'
    _, _, line = helpers.distribution(
            ax, avs, bins=bins, marker=',', lw=1.5, c='k', ls='-')

    lines = []
    cols = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(selections)))
    iC = 0
    nReps = 12
    for i in selections:
        for i_rep in range(0, nReps):
            mc_trajectory, temp = helpers.get_example_trajecotry(i)
            avs = analysis.calc_avalanche(mc_trajectory)
            if i_rep == 0:
                avalanches = np.copy(avs)
            else: 
                avalanches = np.hstack((avalanches, avs))
            # print(avalanches.shape)
        if i < 0:
            gamma = pl.add_powerlaw_fit(ax, avalanches, None, plot=True)
            lbl = r'$T_{f} =$ ' + f' {temp:.2f}, ' + r'$\gamma =$' + f' {gamma:.2f}'
        else:
            gamma = pl.add_powerlaw_fit(ax, avalanches, None, plot=False)
            # lbl = r'$T_{f} =$ ' + f' {temp:.2f}'
            lbl = f'{temp:.2f}'
            print(temp, gamma)

        # lbl = f'{i}, T={temp:.2f}'
        if i == 38:
            ls=':'
        elif i == 39:
            ls=':'
        elif i == 31:
            ls= '--'
        elif i == 30:
            ls= '--'
        else:
            ls='-'
        
        _, _, line = helpers.distribution(
            ax, avalanches, bins=bins, marker=',', lw=1.5, label=lbl, c=cols[iC], ls=ls)
        lines.append(line)
        iC +=1
    ax.set(xscale='log', yscale='log', ylim=[1e-7, None])
    ax.set(xlabel='Event duration', ylabel=r'$PDF$')
    ax.legend(handles=lines, title=r'$T_{f}$', fancybox=True, ncol=2, fontsize=8)
    plt.show()

    # these take long at the moment!
    nTs = 42
    nReps = 12
    gammas = np.zeros(nTs)
    temps = np.zeros(nTs)
    fig, ax = plt.subplots()
    for i in range(0, nTs):
        for i_rep in range(0, nReps):
            mc_trajectory, temp = helpers.get_example_trajecotry(i)
            avs = analysis.calc_avalanche(mc_trajectory)
            if i_rep == 0:
                avalanches = np.copy(avs)
            else:
                avalanches = np.hstack((avalanches, avs))
        # mc_trajectory, temp = helpers.get_example_trajecotry(i)
        # avalanches = analysis.calc_avalanche(mc_trajectory)
        gamma = pl.add_powerlaw_fit(ax, avalanches, None)
        gammas[i] = gamma
        temps[i] = temp
    plt.close()
    fig, ax = plt.subplots()
    ax.plot(temps, gammas)
    ax.set(xlabel=r'$T_{f}$', ylabel=r'$\gamma$')
    print(gammas[i])
    plt.show()
