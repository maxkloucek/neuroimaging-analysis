import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from sklearn.metrics import r2_score
from pyplm.utilities import tools
import nxfuncs


def centered_histogram(data, filter_level, **histargs):
    n, bin_edges = np.histogram(data, **histargs)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    # print(filter_level)
    if filter_level is not None:
        for level in range(0, filter_level):
            mins = np.min(n)
            # print(mins, n[n == mins].size)
            bin_centers = bin_centers[n > mins]
            n = n[n > mins]

            # print(mins, bin_centers.shape, n.shape)
    return bin_centers, n


def get_tail_PowLaws(x, pdf):
    # filter only x > 0
    pdf = pdf[x > 0]
    x = x[x > 0]
    # log data
    logx = np.log10(x)
    logpdf = np.log10(pdf)

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

def threshold_model(mod, TH, print_sumstats=True):
    model = np.copy(mod)
    parameters = tools.triu_flat(model, k=0)

    mask = parameters >= TH
    nParameters = parameters.size
    nParameters_selected = np.sum(mask)
    if print_sumstats is True:
        print('------')
        print(f'modShape = {model.shape}, TH = {TH:.3}')
        print(
            f'N>TH = {nParameters_selected},N = {nParameters}, N>TH/N = {nParameters_selected / nParameters:.3}')
        print('------')
    # model[model < TH] = np.NaN
    model[model < TH] = 0
    return model


def show_categorical_asymmetry_matrix(modL, modR):
    # I AM CHANING THIS TO COPY. THIS WILL BE A SOURCE OF BUGS!
    categorical_model = get_categorical_asymmetry_matrix(modL, modR)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('PuOr', np.max(categorical_model) - np.min(categorical_model) + 1)
    # cmap = plt.get_cmap('tab10')[0: np.max(categorical_model) - np.min(categorical_model) + 1]
    # print(cmap)
    mat = ax.matshow(categorical_model, cmap=cmap, vmin=np.min(categorical_model) - 0.5, 
                      vmax=np.max(categorical_model) + 0.5)
    cbar = fig.colorbar(mat, ticks=np.arange(np.min(categorical_model), np.max(categorical_model) + 1))
    cbar.set_label('Category')
    ax.xaxis.set_label_position("top")
    ax.set(xlabel=r'$i$', ylabel=r'j')
    lbls = ['0 : both-disconnected', '1 : both-connected', '2: left-connected right-disconnected', 'right-connected left-disconnected']
    print(lbls)
    # plt.savefig('./prelim-results/grp-matrix-tailTH-categorical.png')
    # plt.show()
    plt.close() # CHANGE ONCE FINISHED MESSING AROUND!
    return categorical_model

def get_categorical_asymmetry_matrix(modelL, modelR):
    modL = np.copy(modelL)
    modR = np.copy(modelR)
    modL[modL != 0] = 1
    modR[modR != 0] = 1

    categorical_model = np.empty(shape=modL.shape, dtype=int)
    categorical_model[np.logical_and(modL == 0, modR == 0)] = 0
    categorical_model[np.logical_and(modL == 1, modR == 1)] = 1
    categorical_model[np.logical_and(modL == 1, modR == 0)] = 2
    categorical_model[np.logical_and(modL == 0, modR == 1)] = 3
    return categorical_model

# I should change this function to take parameters, not models!
# this will change my other use of it too!!!!
def LLRR_parameters(paramsL, paramsR):
    # paramsL = tools.triu_flat(modL, k=0)
    # paramsR = tools.triu_flat(modR, k=0)

    fig, ax = plt.subplots()
    score = r2_score(paramsL, paramsR)
    from matplotlib import colors
    # ax.plot(paramsL, paramsR, ls='none', alpha=0.1, label=r'$R^{2} =$' + f'{score:.3f}')
    axmax = np.max([paramsL.max(), paramsR.max()])
    axmin = np.min([paramsL.min(), paramsR.min()])

    h = ax.hist2d(paramsL, paramsR, bins=100, norm=colors.LogNorm(), cmap='cividis')
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label(r'$N(\theta _{ij})$')

    ax.plot(np.linspace(axmin, axmax, 10), np.linspace(axmin, axmax, 10), c='k', marker=',')
    ax.text(
        0.05, 0.85,
        r'$R^{2} =$' + f'{score:.3f}', 
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),
        transform=ax.transAxes)
    # plt.legend()
    ax.set(
        xlabel=r'$\theta _{ij} ^{L}$', ylabel=r'$\theta _{ij} ^{R}$',
        xlim=[axmin, axmax], ylim=[axmin, axmax])
    # plt.savefig('./prelim-results/grp-LR-parameter-correlation.png')
    plt.show()

def LR_intercoupling_summary(modLR):
    fig, ax = plt.subplots()
    # modLR = modLR * (180 ** 0.5)
    m = ax.matshow(modLR, cmap='cividis')
    ax.set(xlabel=r'$i^{L}$', ylabel=r'$j^{R}$')
    ax.xaxis.set_label_position("top")
    cbar = fig.colorbar(m, ax=ax)
    cbar.set_label(r'$\theta _{ij}$')

    plt.show()
    diag = np.diag(modLR)
    off_diag = tools.triu_flat(modLR)

    mean_d = np.mean(diag)
    std_d = np.std(diag)
    mean_od = np.mean(off_diag)
    std_od = np.std(off_diag)
    print('------------')
    print(f'Diagonal: mean={mean_d:.2} sd={std_d:.2}')
    print(f'Off-Diagonal: mean={mean_od:.2} sd={std_od:.2}')
    print('------------')

    fig, ax = plt.subplots()
    ax.hist(diag, bins=80, density=True, label=r'$i=j$',)
    ax.hist(off_diag, bins=80, density=True, label=r'$i \neq j$',)
    ax.set(xlabel=r'$\theta_{i^{L} j^{R}}$', ylabel='PDF')
    plt.legend()
    # plt.savefig('./prelim-results/grp-LR-histogram.png')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(diag)
    ax.set(xlabel=r'$i^{L}$', ylabel=r'$\theta_{i^{L} j^{R}}$')
    # plt.savefig('./prelim-results/grp-LR-diagonal-params.png')
    plt.show()


def LLLR_distribution_comparison(modL, modLR):
    paramsL = tools.triu_flat(modL, k=1)
    paramsLR = tools.triu_flat(modLR, k=1)

    fig, ax = plt.subplots()
    diagLR = np.diag(modLR)
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.hist(diagLR, bins=80, density=True, label=r'$L-R: i=j$', facecolor=cols[0])
    ax.hist(paramsL, bins=80, density=True, label=r'$L-L: i \neq j$', facecolor=cols[2])
    ax.hist(paramsLR, bins=80, density=True, label=r'$L-R: i \neq j$', facecolor=cols[1])

    ax.set(xlabel=r'$J_{ij}$', ylabel='PDF', yscale='log')
    GRP_tail_cutoffL = 0.00606
    ax.axvline(GRP_tail_cutoffL, c='k', marker=',')
    plt.legend()
    # plt.savefig('./prelim-results/grp-LL-LR-histogram.png')
    plt.show()


def subnet_vary_threshold(modL, modR):
    nComps=3
    min_param = np.min([modL[modL != 0].min(), modR[modR != 0].min()])
    max_param = np.max([modL[modL != 0].max(), modR[modR != 0].max()])
    print(min_param, max_param)

    logrange = np.logspace(np.log10(min_param), np.log10(max_param), 3)
    # components = np.zeros_like(logrange)
    # nNodes = np.zeros_like(logrange)
    # nEdges = np.zeros_like(logrange)
    # put the networkx stuff in it's own function?
    nNodes = np.zeros((logrange.size, nComps))
    nEdges = np.zeros((logrange.size, nComps))
    nCommunities = np.zeros((logrange.size, nComps))
    CommMax = np.zeros((logrange.size, nComps))
    CommMean = np.zeros((logrange.size, nComps))

    for i,th in enumerate(logrange):
        shared_model = np.copy((modL + modR) / 2)
        shared_model = threshold_model(shared_model, th, print_sumstats=False)
        # shared_model[shared_model != 0] = 1 # changes from weighted to unweighted adjacency matrix

        # print(np.count_nonzero(tools.triu_flat(shared_model)))

        # this applys TH to each model, and then picks only those that remain overlapping
        # modL_th = threshold_model(modL, th, print_sumstats=False)
        # modR_th = threshold_model(modR, th, print_sumstats=False)
        # categorical_model = get_categorical_asymmetry_matrix(modL_th, modR_th)
        # adjShared = np.copy((modL_th + modR_th) / 2)
        # adjShared[categorical_model != 1] = 0
        # print(np.count_nonzero(tools.triu_flat(adjShared)))
        # fig, ax = plt.subplots()
        # ax.matshow(shared_model)
        # plt.show()
        nodes, edges, no_communities, comm_maxSize, comm_meanSize = nxfuncs.connected_component_analysis(shared_model, nComps)
        nNodes[i, :] = nodes
        nEdges[i, :] = edges
        nCommunities[i, :] = no_communities
        CommMax[i, :] = comm_maxSize
        CommMean[i, :] = comm_meanSize

    exit()
    fig, ax = plt.subplots()
    for i in range(0, nComps):
        ax.plot(logrange, nNodes[:, i], label=f'{i}')
    ax.plot(logrange, np.nansum(nNodes,axis=1),c='k', marker=',')
    ax.set(xlabel='threshold', ylabel='No. nodes')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    for i in range(0, nComps):
        ax.plot(logrange, nEdges[:, i], label=f'{i}')
    ax.set(xlabel='threshold', ylabel='No. edges')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    for i in range(0, nComps):
        ax.plot(logrange, nCommunities[:, i], label=f'communties {i}')
    for i in range(0, nComps):
        ax.plot(logrange, nNodes[:, i], label=f'nodes {i}')
    ax.set(xlabel='threshold', ylabel='No. communities')
    ax.plot(logrange, np.nansum(nNodes,axis=1),c='k', marker=',')
    plt.legend()
    plt.legend()
    plt.show()

    # this needs to be normalised by no nodes I think!
    fig, ax = plt.subplots()
    # for i in range(0, nComps):
    #     ax.plot(logrange, CommMax[:, i] / nNodes[:, i], label=f'Max: {i}')
    for i in range(0, nComps):
        ax.plot(logrange, CommMean[:, i] / nNodes[:, i], label=f'Mean: {i}')
    ax.set(xlabel='threshold', ylabel='Community size / no. nodes')
    plt.legend()
    plt.show()

    # I can do some funky combination measure, e.g. no communties per node?
    fig, ax = plt.subplots()
    for i in range(0, nComps):
        ax.plot(logrange, nCommunities[:, i] / nNodes[:, i], label=f'{i}')
    ax.set(xlabel='threshold', ylabel='No. communities / No. Nodes')
    plt.legend()
    plt.show()
    # fig, ax = plt.subplots()
    # cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # ax.plot(logrange, nEdges)
    # ax.set(xlabel='threshold', ylabel=r'$N_{J}$', xscale='log', yscale='log')
    # ax2 = ax.twinx()
    # ax2.plot(logrange, nNodes, c=cols[1])
    # ax2.set(xlabel='threshold', ylabel=r'$N$', xscale='log', yscale='log')
    # plt.show()
    # return logrange, nCouplings, NetSize
    return 0
