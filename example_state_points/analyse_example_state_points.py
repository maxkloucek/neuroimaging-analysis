import numpy as np
import matplotlib.pyplot as plt
import h5py
from pyplm.utilities import tools

import analyse_esp_figs as figs
import analyse_esp_aside as aside

file = '/Users/mk14423/Desktop/Data/0_thesis/ExampleInferenceOutputs/datasets.hdf5'
# group = 'SK_N24'
# group = 'SK_N72'
group = 'SK_N120'

with h5py.File(file, 'r') as fin:
    dataset_names = fin[group].keys()
    # print(dataset_names)
    print(fin[group]['inputModels_metadata'][()])
    configs = fin[group]['configurations'][()]
    true_mods = fin[group]['inputModels'][()]
    infr_mods = fin[group]['inferredModels'][()]

phases = ['P', 'F', 'SG', 'P-SG']
plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
# for config in configs:
#     print(config.shape)
#     m_i = np.mean(config, axis=0)
#     m = np.mean(m_i)
#     print(m_i.shape, m)
figs.autocorrs(configs, phases)
exit()
print('----')
# figs.error_sum_vs_i(phases, true_mods, infr_mods, save=False)
# figs.models_and_correlations(phases, true_mods, infr_mods, save=False)
figs.models_correlations_distributions(phases, true_mods, infr_mods, save=False)
# figs.models_error_distributions(phases, true_mods, infr_mods, save=False)
plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisaside.mplstyle')
# aside.correlation_ferro(phases, true_mods, infr_mods, save=True)

# for iP in range(0, len(phases)):
#     fig, ax = plmplt.mkfigure(nrows=2, ncols=2)
#     # ax[0, 0].matshow(true_mods[iP])
#     # # ax[0, 1].matshow(infr_mods[iP], vmin=true_min, vmax=true_max)
#     # ax[0, 1].matshow(infr_mods[iP] - true_mods[iP], vmin=true_min, vmax=true_max)

#     # plmplt.histogram(ax[1, 0], true_params, 25, c='k')
#     # # plmplt.histogram(ax[1, 0], infr_params, 25, c=plmplt.category_col(0))
#     # plmplt.histogram(ax[1, 0], error_params, 25, c=plmplt.category_col(0))

#     # ax[1, 1].plot(true_params, infr_params, c=plmplt.category_col(0), ls='none', alpha=0.5)
#     # ax[1, 1].plot(true_params, true_params, c='k', marker=',')
    
#     # ax[0, 0].set(ylabel=f'Phase: {phases[iP]}')

#     # ax[ia, 0].xaxis.set_ticks_position('bottom')
#     # ax[ia, 1].xaxis.set_ticks_position('bottom')
#     # ax[ia, 2].xaxis.set_ticks_position('bottom')
#     plt.show()


exit()
for iP in range(0, len(phases)):
    fig, ax = plmplt.mkfigure(nrows=2, ncols=2)
    # ax = np.reshape(ax, (1, 4))
    true_params = tools.triu_flat(true_mods[iP])
    infr_params = tools.triu_flat(infr_mods[iP])
    true_max = np.nanmax(true_mods[iP])
    true_min = np.nanmin(true_mods[iP])

    ax[0, 0].matshow(true_mods[iP])
    ax[0, 1].matshow(infr_mods[iP], vmin=true_min, vmax=true_max)

    plmplt.histogram(ax[1, 0], true_params, 25, c='k')
    plmplt.histogram(ax[1, 0], infr_params, 25, c=plmplt.category_col(0))
    ax[1, 1].plot(true_params, infr_params, c=plmplt.category_col(0), ls='none', alpha=0.5)
    ax[1, 1].plot(true_params, true_params, c='k', marker=',')
    # ax[ia, 2].matshow(infr_mods[iP]-true_mods[iP], vmin=true_min, vmax=true_max)
    ax[0, 0].set(ylabel=f'Phase: {phases[iP]}')

    # ax[ia, 0].xaxis.set_ticks_position('bottom')
    # ax[ia, 1].xaxis.set_ticks_position('bottom')
    # ax[ia, 2].xaxis.set_ticks_position('bottom')
    plt.show()
