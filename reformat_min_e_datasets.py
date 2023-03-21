import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from pyplm.pipelines import data_pipeline

plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
# emin value mu T
# 50 [0.14320461 0.5        1.1       ]
# 100 [0.20949111 0.4        1.175     ]
# 200 [0.29832886 0.4        1.1       ]
# 400 [0.44711299 0.6        1.1       ]
# 800 [0.69887905 0.2        1.25      ]

data_root_path = '/Users/mk14423/Desktop/PaperData/B1e4-Nscaling/'
# N50_emin_states = ['N50_' + str(i) + '/' for i in range(1, 7)]
# N50_emin_states = [sp + 'T_1.100-h_0-J_.500-Jstd_1.hdf5' for sp in N50_emin_states]
# N50_emin_states = [os.path.join(data_root_path, sp) for sp in N50_emin_states]
# print(N50_emin_states)
# # 'T_.500-h_0-J_1.100-Jstd_1.hdf5'
# # N50_1/

def get_emin_files(root, N_label, state_point, rep_labels):
    emin_files = [N_label + str(i) + '/' for i in rep_labels]
    emin_files = [repeat + state_point for repeat in emin_files]
    emin_files = [os.path.join(root, repeat) for repeat in emin_files]
    return np.array(emin_files)

N50_files = get_emin_files(data_root_path, 'N50_', 'T_1.100-h_0-J_.500-Jstd_1.hdf5', [1,2,3,4,5,6])
N100_files = get_emin_files(data_root_path, 'N100_', 'T_1.175-h_0-J_.400-Jstd_1.hdf5', [1,2,3,4,5,6])

N200_files = get_emin_files(data_root_path, 'N200_', 'T_1.100-h_0-J_.400-Jstd_1.hdf5', [2, 3])
N400_files = get_emin_files(data_root_path, 'N400_', 'T_1.100-h_0-J_.600-Jstd_1.hdf5', [1,2,3])

N800_files = get_emin_files(data_root_path, 'N800_', 'T_1.250-h_0-J_.200-Jstd_1.hdf5', [1])

list_of_Nfiles = [N50_files, N100_files, N200_files, N400_files, N800_files]
Ns = [50, 100, 200, 400, 800]
file = '/Users/mk14423/Desktop/Data/0_thesis/SubSampleSK/datasets.hdf5'
for iN, Nx_files in enumerate(list_of_Nfiles):
    group = 'N' + str(Ns[iN])
    print(group)
    nRepeats = len(Nx_files)
    repeat_configs = np.zeros((nRepeats, 10000, Ns[iN]))
    for i in range(0, nRepeats):
        with h5py.File(Nx_files[i], 'r') as fin:
            print(Nx_files[i], fin['configurations'].shape)
            production_configs = fin['configurations'][1000:, :]
            # print(production_configs.shape)
        repeat_configs[i, :, :] = production_configs
    
    # plm_pipeline = data_pipeline(file, group)
    # plm_pipeline.write_data(repeat_configs, np.array(Nx_files, dtype=str))
