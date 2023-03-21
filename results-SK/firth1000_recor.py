import numpy as np
import h5py

from pyplm.utilities.hdf5io import write_models_to_hdf5, write_configurations_to_hdf5
from pyplm.utilities.metadataio import get_metadata_df
from pyplm.pipelines import data_pipeline


def _add_to_pipe():
    # root = '/Users/mk14423/Desktop/Data/N200_J0.1_optimizeT/B1e3_1/'
    file = './B1000_firth.hdf5'
    with h5py.File(file, 'r') as f:
        g = f['infMod_is_firth']
        print(g.keys())
        # md = get_metadata_df(g, 'inferredModels')
        
        mods = g['inferredModels'][()]
        md = g['configurations_metadata'].asstr()[()]
        configs = g['configurations'][()]
        print(configs.shape, mods.shape, md.shape)
    # group = 'infMod_is_firth_thenC2'
    # write_models_to_hdf5(
    #     file, 'infMod_is_firth_thenC2', 'inferredModels',
    #     mods, md)
    # write_configurations_to_hdf5(
    #     file, 'infMod_is_firth_thenC2',
    #     configs, md)


def run_firthC2():
    file = './B1000_firth.hdf5'
    group = 'infMod_is_firth_thenC2'
    # group = 'small_test'
    # this is the frith run, then I need to make some C2s!!!
    with h5py.File(file, 'r') as f:
        print(group)
        print(f[group].keys())
        print(f[group]['configurations'].shape)
    pipeline = data_pipeline(file, group)
    # pipeline.correct_firth(mod_name='inferredModels')
    pipeline.correct_C2()
    # so then I want to look at firth C2 and also
    # something else.. ok this should be fine...
    # pipeline.ficticiousT_sweep(np.array([1]), 10000, 6)

# _add_to_pipe()
run_firthC2()
