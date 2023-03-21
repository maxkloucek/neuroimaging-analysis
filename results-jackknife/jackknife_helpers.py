import numpy as np
import h5py

from pyplm import utilities

# yikes it probably does matter... oh whatever!
def get_models(
        file='/Users/mk14423/Desktop/Data/0_thesis/jackknife/jackknife_dataset.hdf5',
        group='SK_N100'):

    with h5py.File(file, 'r') as fin:
        # print(fin[group].keys())
        true_mod = fin[group]['inputModels'][0]
        infr_mod = fin[group]['inferredModels'][0]
        C2cr_mod = fin[group]['correctedModels'][0]

        jack_mods = fin[group]['correctedModels_jacknife'][0]
        sampling_md = utilities.get_metadata_df(fin[group], 'configurations')
        B = sampling_md['B_sample'][0] * sampling_md['nChains'][0]

    jack_mean = np.mean(jack_mods, axis=0)
    jn_bias = (B - 1) * (jack_mean - infr_mod)
    jkcr_mod = infr_mod - jn_bias
    # maybe I don't even have enough samples, either way this doesn't
    # seem to have worked...?
    # lol I was returning the wrong thing!
    return np.array([true_mod, infr_mod, C2cr_mod, jkcr_mod]), B