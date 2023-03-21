import numpy as np
import h5py
file_all = '/Users/mk14423/Desktop/data/HCP_data_Ryota/HCP_AllRESTfMRI_360ROIs.hdf5'
with h5py.File(file_all, 'r') as fin:
    print(fin.keys())
    group_append = fin['appended']
    print(group_append.keys())
    infrMod = group_append['InferredModel'][()]
    infrMods = np.zeros((1, 360, 360))
    infrMods[:, :, :] = infrMod
print(infrMods.shape)

# md = ['transferred from diff file']
# md = np.array(md, dtype=str)
# from pyplm.utilities import write_models_to_hdf5
# write_models_to_hdf5(
#     '/Users/mk14423/Desktop/data/HCP_data/HCP_rsfmri.hdf5',
#     'grouped',
#     'inferredModels',
#     infrMods,
#     md
# )