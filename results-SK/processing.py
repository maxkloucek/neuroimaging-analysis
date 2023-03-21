import os
import glob
import numpy as np
import pandas as pd
import observables as obs
import h5py
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

from pyplm.plotting import mkfigure

from pyplm.utilities import tools
# import argparse

def get_directories():
    root='/Users/mk14423/Desktop/PaperData'
    runsN50 = [
            'B1e4-Nscaling/N50_1',
            'B1e4-Nscaling/N50_2',
            'B1e4-Nscaling/N50_3',
            'B1e4-Nscaling/N50_4',
            'B1e4-Nscaling/N50_5',
            'B1e4-Nscaling/N50_6',
        ]
    runsN100 = [
            'B1e4-Nscaling/N100_1',
            'B1e4-Nscaling/N100_2',
            'B1e4-Nscaling/N100_3',
            'B1e4-Nscaling/N100_4',
            'B1e4-Nscaling/N100_5',
            'B1e4-Nscaling/N100_6',
        ]
    runsN200 = [
            # 'B1e4-Nscaling/N200_1',
            'B1e4-Nscaling/N200_2',
            'B1e4-Nscaling/N200_3',
        ]
    runsN400 = [
            'B1e4-Nscaling/N400_1',
            'B1e4-Nscaling/N400_2',
            'B1e4-Nscaling/N400_3',
        ]
    runsN800 = [
            'B1e4-Nscaling/N800_1',
        ]
   
    runsNx = [runsN50, runsN100, runsN200, runsN400, runsN800]
    for i in range(0, len(runsNx)):
        runsNx[i] = [os.path.join(root, run) for run in runsNx[i]]
    return runsNx

# this is going to be a long long list of observations...
# let's get n, mu ....
# input...
def calculate_observations():
    Nx_dirs = get_directories()
    # Nx_dirs = Nx_dirs[0:2]
    # headers = ['iD', 'N', 'mu', 'T', 'm', 'q', 'C2', 'err']
    headers = ['iD', 'N', 'mu', 'T', 'tau', 'err']

    f = open('summarised_results_tau.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(headers)
    # check if exists else write over this.
    
    for dirs in Nx_dirs:
        for iD in range(0, len(dirs)):
            print(iD, dirs[iD])
            files = glob.glob(dirs[iD] + '/*.hdf5')
            print(len(files))
            for file in tqdm(files):
                # print(file)
                with h5py.File(file, 'r') as fin:
                    N = fin['configurations'].attrs['N']
                    mu = fin['configurations'].attrs['J']
                    T = fin['configurations'].attrs['T']

                    # print(
                    #     int(fin['configurations'].attrs['eq_cycles'] / 1e4),
                    #     int(fin['configurations'].attrs['prod_cycles'] /1e4),
                    #     int(fin['configurations'].attrs['cycle_dumpfreq'] / 1e4)
                    #     )
                    # I NEED TO MAKE SURE TO DISCARD THE TRAJECOTRY!
                    discard = int(
                        fin['configurations'].attrs['eq_cycles'] /
                        fin['configurations'].attrs['cycle_dumpfreq']
                    )
                    true_mod = fin['InputModel'][()]
                    infr_mod = fin['InferredModel'][()]
                    trajectory = fin['configurations'][discard:, :]
                    
                    m = obs.calc_m(trajectory)
                    q = obs.calc_q(trajectory)
                    C2 = obs.calc_C2(trajectory)
                    tau = obs.calc_tau(trajectory)
                    error, _, _ = obs.recon_error_nguyen(true_mod, infr_mod)

                    # row = [iD, N, mu, T, m, q, C2, error]
                    row = [iD, N, mu, T, tau, error]
                    writer.writerow(row)
    f.close()

# I can append to the dataframe, sick!
# ok this is going to be a slight mess, but I guess we just make
# liest and append to them...
# calculate m, q, chi2 and error...?
# yep that's all I want for now I think!
# I have these functions somehwere
# print(fin['configurations'].attrs['T'])
# print(fin['configurations'].attrs['h'])
# print(fin['configurations'].attrs['J'])
# print(fin['configurations'].attrs['Jstd'])

def make_da_plot(df_all, obs1, obs2):
    fig_w, fig_h = plt.rcParams['figure.figsize']
    fig_h = 1.5 * fig_h
    fig, ax = mkfigure(nrows=5, ncols=1, sharex=True, figsize=(fig_w, fig_h))
    # ax_ravel = ax.ravel()
    ax = ax.ravel()
    Ns = [50, 100, 200, 400, 800]
    # fig, ax = plt.subplots()
    # hmmm, doesn't look great!
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for iN, N in enumerate(Ns):
        df = df_all[df_all['N'] == N]
        print(df)
        df = df[df['mu'] < 1.0]
        df_mean = df.groupby(['T'], as_index=True).mean()
        df_mean = df_mean.reset_index()
        df_std = df.groupby(['T'], as_index=True).std()
        df_std = df_std.reset_index()
        # df_mean[df_mean[obs1] < 1] = 1
        df_mean[obs1][df_mean[obs1] < 1] = 1
        print(df_mean)
        ax1 = ax[iN]
        ax2 = ax1.twinx()
        
        ax1.text(0.5, 0.75, f'N={N}', transform=ax1.transAxes,
                # fontsize='medium', fontfamily='serif',
                horizontalalignment='center',
                verticalalignment='top',
                # bbox=dict(facecolor='1.0', edgecolor='k', pad=0.2)
                )
        # really these should be two different functions by now...

        # ax1.plot(df_mean['T'], df_mean[obs1], marker='o', label=N, c=cols[0])
        # ax1.errorbar(x=df_mean['T'], y=df_mean[obs1], yerr=df_std[obs1], marker='o', label=N, c=cols[0])
        if obs1 == 'C2':
            ax1.errorbar(x=df_mean['T'], y=df_mean[obs1], yerr=df_std[obs1], marker='o', label=N, c=cols[0])
            ax2.plot(df_mean['T'], df_mean[obs2], marker='o', label=N, c=cols[2])
            df_std = df_std[df_mean[obs2] < df_mean[obs2].min() * 1.5]
            df_mean = df_mean[df_mean[obs2] < df_mean[obs2].min() * 1.5]
            ax2.errorbar(x=df_mean['T'], y=df_mean[obs2], yerr=df_std[obs2], marker='o', label=N, c=cols[2])
        elif obs1 == 'tau':
            ax1.plot(df_mean['T'], df_mean[obs1], marker='o', label=N, c=cols[0])
            df_std_tf = df_std[df_mean[obs1] < 3]
            df_mean_tf = df_mean[df_mean[obs1] < 3]
            ax1.errorbar(x=df_mean_tf['T'], y=df_mean_tf[obs1], yerr=df_std_tf[obs1], marker='o', label=N, c=cols[0])
            ax2.plot(df_mean['T'], df_mean[obs2], marker='o', label=N, c=cols[2])
            df_std = df_std[df_mean[obs2] < df_mean[obs2].min() * 1.5]
            df_mean = df_mean[df_mean[obs2] < df_mean[obs2].min() * 1.5]
            ax2.errorbar(x=df_mean['T'], y=df_mean[obs2], yerr=df_std[obs2], marker='o', label=N, c=cols[2])
        # ax2.plot(df_mean['T'], df_mean[obs2], marker='o', label=N, c=cols[2])
        # df_std = df_std[df_mean[obs2] < df_mean[obs2].min() * 1.5]
        # df_mean = df_mean[df_mean[obs2] < df_mean[obs2].min() * 1.5]
    
        # ax2.errorbar(x=df_mean['T'], y=df_mean[obs2], yerr=df_std[obs2], marker='o', label=N, c=cols[2])

        # ax1.plot(df['T'], df[obs1], ls='none', alpha=0.1, c=cols[0], markeredgecolor='none')
        # ax2.plot(df['T'], df[obs2], ls='none', alpha=0.1, c=cols[2], markeredgecolor='none')
        if iN == 2:
            if obs1 == 'C2':
                ax1.set(ylabel=r'$C^2$')
            elif obs1 == 'tau':
                ax1.set(ylabel=r'$\tau$')
            ax2.set(ylabel=r'$\varepsilon$')

        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(2))

        ax2.set(ylim=[df_mean[obs2].min() * 0.98, df_mean[obs2].min() * 1.5])
        if obs1 == 'tau':
            ax1.set(ylim=[0.98, 3])
        if obs1 == 'C2':
            ax1.set(ylim=[0, None])
        ax1.yaxis.label.set_color(cols[0])
        ax2.yaxis.label.set_color(cols[2])
    ax1.set(xlabel=r'$T$')
    plt.show()


def min_maxplot(df_allN):
    plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisaside.mplstyle')
    Ns = [50, 100, 200, 400, 800]
    T_diffs = []
    T_peak_C2 = []
    T_min_err = []
    for N in Ns:
        df = df_allN[df_allN['N'] == N]
        df = df[df['mu'] < 1.0]
        df_mean = df.groupby(['T'], as_index=True).mean()
        df_mean = df_mean.reset_index()
        # idmax = df_mean['C2'].idxmax()
        # print(df['C2'] == df_mean['C2'].max())
        # print(df_mean['C2'].max())
        maxC2_df = df_mean[df_mean['C2'] == df_mean['C2'].max()]
        minErr_df = df_mean[df_mean['err'] == df_mean['err'].min()]
        T1 = maxC2_df['T'].iloc[0]
        T2 = minErr_df['T'].iloc[0]
        # print(N, np.abs(T2-T1))
        T_diffs.append(np.abs(T2-T1))
        T_peak_C2.append(T1)
        T_min_err.append(T2)
    # I maybe don't want this to be on here...
    # Ns, delTs = min_maxplot(df_all)
    fig, ax = mkfigure()
    ax[0, 0].plot(Ns, T_peak_C2, label=r'$T_{C^2-max}$')
    ax[0, 0].plot(Ns, T_min_err, label=r'$T_{\varepsilon-min}$')
    # ax[0, 0].plot(Ns, T_diffs)
    ax[0, 0].set(xlabel=r'$N$', ylabel=r'$T$')
    ax[0, 0].legend()
    plt.savefig('/Users/mk14423/Documents/tempfigs/pd-avrg-C2max-emin.png')
    plt.show()
    # return np.array(Ns), np.array(T_diffs)

plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')

# calculate_observations()

# df = pd.read_csv('summarised_results.csv')
# make_da_plot(df, 'C2', 'err')
# # min_maxplot(df)

# df = pd.read_csv('summarised_results_tau.csv')
# make_da_plot(df, 'tau', 'err')


# Ok! Let's ge the plots from the paper data into nicer formats
# over here!