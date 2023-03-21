import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LogNorm
from scipy.interpolate import Rbf

from pyplm.pipelines import data_pipeline
from pyplm.utilities.hdf5io import write_models_to_hdf5, write_configurations_to_hdf5
from pyplm.utilities.metadataio import get_metadata_df
from figures import analysis
plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')


def _get_models():
    # how do I set this up... I really don't know!
    file = '/Users/mk14423/Desktop/PaperData/HCP_data_analysed/HCP_rsfmri_added_data3.hdf5'
    group = 'grouped'
    # labels = ['symmetric', 'positive']
    param_key = 'param'
    sweep_groups = ['sweepTH_symmetric', 'sweepTH_positive']
    # print(sweep_groups[0])
    sweep_analysis = analysis.SweepAnalysis2(file, group, sweep_groups[0], ['m', 'q', 'C2'])
    i_range = np.arange(27, 41)
    print(len(i_range))
    th_mods = np.zeros((len(i_range), 360, 360))
    th_vals = np.zeros(len(i_range))
    for i, i_mod in enumerate(i_range):
        th_mod, th_val = sweep_analysis.load_threshold_model(i_mod)
        # print(th_mod.shape)
        # th_mod_flat = th_mod[th_mod != 0]
        # print(i, th_val, th_mod_flat.min(), th_mod_flat.max())
        print(th_mod.shape)
        th_mods[i, :, :] = th_mod
        th_vals[i] = th_val
    # print(th_vals)
    th_vals = np.array(th_vals, dtype=str)
    # print(th_vals)
    return th_mods, th_vals


def write_th_mods_to_hdf5(file, group):
    mods, ths = _get_models()
    print(mods.shape, ths.shape)
    for mod in mods:
        print(mod.min(), mod.max())

    write_models_to_hdf5(file, group, 'inferredModels', mods, ths)


def do_sweeps():
    file = './th_sweep.hdf5'
    group = 'test2_sym'
    # just do 2ce as many temperature points..?
    write_th_mods_to_hdf5(file, group)

    pipeline = data_pipeline(file, group)
    alphas = np.linspace(0.5, 2, 10)
    print(alphas)
    # exit()
    pipeline.ficticiousT_sweep(
        alphas, 1000, 6,
    )

def _make_pd_dataframe(name, group):
    file = './th_sweep.hdf5'
    # group = 'test'
    # somehow collapse this onto a phase diagram, let's start with m and q.
    with h5py.File(file, 'r') as fin:
        g = fin[group]
        print(g.keys())
        deltas = get_metadata_df(g, 'inferredModels')
        deltas = deltas.to_numpy()[:, 0]
        temps = g['sweep-alphas'][()]
        trajectories = g['sweep-trajectories'][()]
        print(trajectories.shape)
        print(temps)
        print(deltas)
    nDelta, nT, nR, B, N = trajectories.shape
    d = np.zeros((nDelta * nT * nR, 6))
    print(d.shape)
    i = 0
    for iDelta in range(0, nDelta):
        for iT in range(0, nT):
            for iR in range(0, nR):
                delta = deltas[iDelta]
                T = temps[iT]
                d[i, 0] = delta
                d[i, 1] = T
                d[i, 2] = iR

                data = trajectories[iDelta, iT, iR]
                d[i, 3] = np.abs(analysis.calc_m(data))
                d[i, 4] = analysis.calc_q(data)
                d[i, 5] = analysis.calc_C2(data)
                # print(data.shape)
                i+=1
    df = pd.DataFrame(d, columns=['deltas', 'temps', 'reps', 'm', 'q', 'C2'])
    df.to_hdf(file, name)

def lines3d(df, xkey='temps', ykey='deltas', zkey='q'):
    xs = df[xkey].unique()
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    cols = plt.get_cmap('cividis')(np.linspace(0, 1, len(xs)))
    # print(xs)
    # print(cols)
    # I think I want to connect the Ts at differnet deltas!!
    # so gotta switch this for it to make sense :)!
    ax.plot([xs.min(), xs.max()], [1, 1], [0, 0], c='k', marker=',', ls='--')
    x1 = []
    y1 = []
    z1 = []
    for ix, x in enumerate(xs):
        df_fixed_delta = df[df[xkey] == x]
        x = df_fixed_delta[xkey].to_numpy()
        y = df_fixed_delta[ykey].to_numpy()
        z = df_fixed_delta[zkey].to_numpy()
        # z = gaussian_filter1d(z, sigma=1)

        ax.plot(x, y, z, c=cols[ix], marker=',', lw='2', zorder=10)
        # ax.plot([x[3]], [y[3]], [z[3]], c=cols[ix], marker='o', lw='5', zorder=10)
        x1.append(x[3])
        y1.append(y[3])
        z1.append(z[3])
        ax.add_collection3d(
            plt.fill_between(y,0,z, color=cols[ix], alpha=0.5, zorder=2),
            zs=x[0], zdir='x')
    y1 = np.array(y1)
    print(y1)
    y1 = np.ones_like(x1) * df[ykey].max()
    ax.plot(x1, y1, z1, c='grey', marker=',', lw='2', zorder=11)
    ax.add_collection3d(
            plt.fill_between(x1,0,z1, color='grey', alpha=0.5),
            zs=df[ykey].max(), zdir='y')

    # ax.plot(x1, y1, z1, zdir='y', color='grey')
    # offset=df[ykey].max()
    # ax.view_init(elev=28., azim=-30)
    # ax.view_init(elev=56., azim=-20)
    ax.view_init(elev=36., azim=-57)
    # ax.set(xlabel=xkey, ylabel=ykey, zlabel=zkey)
    ax.set(xlabel=r'$\delta$', ylabel=r'$T$', zlabel=r'$m$')
    ax.set(
        xlim=[df[xkey].min(), df[xkey].max()],
        ylim=[df[ykey].min(), df[ykey].max()],
        # zlim=[df[zkey].min(), df[zkey].max()],
        )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    xx, zz = np.meshgrid(xlim, zlim)
    yy = np.ones_like(xx)
    # ax.plot_surface(xx, yy, zz, alpha=0.8, color='grey', zorder=1)
    plt.show()

# ok this is kind of interesting...?
# I've got the best way to visualise it now :)!
def surface3d(df, obs='q'):
    deltas = df['deltas'].unique()
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    cols = plt.get_cmap('cividis')(np.linspace(0, 1, len(deltas)))
    # print(cols)
    print(df)
    x = df['deltas'].to_numpy()
    y = df['temps'].to_numpy()
    z = df[obs].to_numpy()
    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
    plt.show()

# let's do the same but for the symmetric ths!
def quick_pd():
    file = './th_sweep.hdf5'
    # df = pd.read_hdf(file, 'analysisDF')

    # _make_pd_dataframe('test2_symobs', 'test2_sym')
    # df = pd.read_hdf(file, 'test2_symobs')

    # _make_pd_dataframe('test2obs', 'test2')
    df = pd.read_hdf(file, 'test2obs')

    df = df.groupby(['deltas', 'temps'], as_index=True).mean().reset_index()
    # print(df)
    print(df[df['temps'] == 0.5])
    lines3d(df, 'deltas', 'temps', 'm')
    # surface3d(df, 'C2')
    # imshow(df)
    # matshow(df, 'C2')
    # god showing this data is an abolute nightmare...
    # I'm ready to give up on this now!! Fuck..
    # check what firth C2 is would be quite a good thing to do!
    # the clearest graph is indeed the 3D one! how wierd!

# write_th_mods_to_hdf5()
# do_sweeps()
quick_pd()



# # ---- not that helpful below ---- #
# def imshow(df):
#     x = df['deltas']
#     y = df['temps']
#     z = df['q']
#     # xi, yi = np.meshgrid(xi, yi)
#     # Set up a regular grid of interpolation points
#     spacing = 10
#     xi, yi = np.linspace(x.min(), x.max(), spacing), np.linspace(y.min(), 
#                         y.max(), spacing)
#     XI, YI = np.meshgrid(xi, yi)
#     # Interpolate
#     rbf = Rbf(x, y, z, function='linear')
#     ZI = rbf(XI, YI)
#     #plot
#     fig, ax = plt.subplots()
#     sc = ax.imshow(
#         ZI, vmin=z.min(), vmax=z.max(), origin='lower',
#         aspect="auto",
#         extent=[x.min(), x.max(), y.min(), 
#                 y.max()],
#                 # cmap="GnBu",
#                 # norm=colors.LogNorm(vmin=ZI.min(), vmax=ZI.max())
#                 )
#     fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.01)
#     ax.set(xscale='log')
#     plt.show()


# def matshow(df, vals='q'):
#     piv = df.pivot(index='temps', columns='deltas', values=vals)
#     print(piv)
#     x = piv.columns.to_numpy()
#     y =  piv.index.to_numpy()
#     print(x)
#     print(y)
#     z = piv.to_numpy()

#     fig, ax = plt.subplots()
#     c = ax.imshow(
#         z,
#         # cmap='RdBu',
#         # vmin=z.min(), vmax=z.max(),
#         extent=[x.min(), x.max(), y.min(), y.max()],
#         norm=LogNorm(vmin=z.min(), vmax=z.max()),
#         interpolation='nearest', origin='lower', aspect='auto')
#     # ax.set_title('image (nearest, aspect="auto")')
#     fig.colorbar(c, ax=ax)
#     # ax.set(xscale='log')
#     plt.show()