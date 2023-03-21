import os
import numpy as np
import matplotlib.pyplot as plt
from pyplm.pipelines import data_pipeline
import json
import h5py
import glob


def get_sample_trajectories(dataset):
    files = glob.glob(os.path.join(dataset, '*'))
    files = np.array(files)
    config_datasets = []
    for iF in (range(0, len(files))):
        time_series = np.loadtxt(files[iF], delimiter=',')
        spin_trajectory = np.copy(time_series)
        spin_trajectory[spin_trajectory <= 0] = -1
        spin_trajectory[spin_trajectory > 0] = 1
        config_datasets.append(spin_trajectory)
        # print(files[iF], spin_trajectory.shape)
    return files, np.array(config_datasets)


plt.style.use('/Users/mk14423/Dropbox/mpl-styles/paper-1col.mplstyle')
file = '/Users/mk14423/Desktop/Data/0_thesis/ExampleInferenceOutputs/datasets.hdf5'
# file = '/Users/mk14423/Desktop/PLM_DifferentModels/test.hdf'
# group = 'SK_N25_T1.5'
group = 'SK_N120'
N = 120
Ts = [1.5, 0.6, 0.35, 0.5]
mus = [0.8, 2.0, 0.0, 0.3]
# then I zip :)!
mod_choices = ['SK' for _ in Ts]
mod_args = [
    {'N': N, 'T': T, 'h': 0, 'jmean': mu, 'jstd': 1} for T,mu in zip(Ts, mus)]
sim_args = [{'B_eq': 1e5, 'B_sample': 1e4, 'nChains': 6} for _ in Ts]


plm_pipeline = data_pipeline(file, group)
# plm_pipeline.generate_model(mod_choices, mod_args)
# plm_pipeline.simulate_data(sim_args, n_jobs=6)
# plm_pipeline.infer(Parallel_verbosity=5)
# plm_pipeline.correct_jacknife()
plm_pipeline.correct_C2()

# plm_pipeline.ficticiousT_sweep(
#     np.linspace(0.1, 4, 200), 1e4, 6)



# file = 'testdata/test1.hdf5'
# group = 'simtest'
# Ts = [0.5, 1, 2.25, 5]
# mod_choices = ['2D_ISING_PBC' for _ in Ts]
# mod_args = [{'L': 7, 'T': T, 'h': 0, 'jval': 1} for T in Ts]
# sim_args = [{'B_eq': 1e3, 'B_sample': 5e3, 'nChains': 6} for _ in Ts]
# # print(mod_args)
# print(len(mod_args), len(sim_args))
# plm_pipeline = pipeline(file, group)
# plm_pipeline.generate_model(mod_choices, mod_args)
# plm_pipeline.simulate_data(sim_args, n_jobs=6)
# # plm_pipeline.write_data(data, labels)
# plm_pipeline.infer(Parallel_verbosity=5)


# file = 'testdata/test1.hdf5'
# group = 'htest_lowcoupling'
# hs = np.linspace(-4, +4, 100)
# mod_choices = ['2D_ISING_PBC' for _ in hs]
# mod_args = [{'L': 7, 'T': 1, 'h': h, 'jval': 0.1} for h in hs]
# sim_args = [{'B_eq': 1e3, 'B_sample': 5e3, 'nChains': 6} for _ in hs]
# # print(mod_args)
# print(len(mod_args), len(sim_args))
# plm_pipeline = pipeline(file, group)
# plm_pipeline.generate_model(mod_choices, mod_args)
# plm_pipeline.simulate_data(sim_args, n_jobs=6)
# # plm_pipeline.write_data(data, labels)
# plm_pipeline.infer(Parallel_verbosity=5)