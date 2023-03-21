import numpy as np
from pyplm.pipelines import data_pipeline

file = './2DIsing.hdf5'

# Ls = [4, 8, 16]
Ls = [32]

# sim_args = [{'B_eq': 2e3, 'B_sample': 1e3, 'nChains': 6} for _ in Ts]
# exit()
for L in Ls:
    group = 'finite_size_scaling_L' + str(L)
    print(group)
    mod_choices = ['2D_ISING_PBC' for _ in Ls]
    mod_args = [{'L': L, 'T': 1, 'h': 0, 'jval': 1}]
    plm_pipeline = data_pipeline(file, group)
    plm_pipeline.generate_model(mod_choices, mod_args)
    plm_pipeline.ficticiousT_sweep(
        np.linspace(1, 5, 50), 5e3, 6, mod_name='inputModels')
