import os
import pandas as pd
import numpy as np
import h5py


def get_subsampling_data_frame(file, group, T_true=None):
    df = pd.read_hdf(file, group + '/subsampling')
    df['mean_J'] = df['mean_J'] * df['N']         # rescale by N
    df['std_J'] = df['std_J'] * (df['N'] ** 0.5)
    df['mu'] = df['mean_J'] / df['std_J']
    df['T'] = 1 / df['std_J']
    df['invB'] = 1 / df['B']
    print(T_true)
    if T_true != None:
        df['T'] = df['T'] / T_true
    df = df.sort_values(by=['B', 'iD'])
    return df

def get_model(file, group, iD):
    with h5py.File(file, 'r') as fin:
        print(iD)
        model = fin[group]['inferredModels'][iD, :, :]
        print(model.shape)
    return model

def get_raw_data(iD):
    raw_data_dir = '/Users/mk14423/Desktop/PaperData/HCP_data_Ryota/HCP_AllRESTfMRI_360ROIs'
    files = [
        f for f in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, f))]
    files = [os.path.join(raw_data_dir, f) for f in files]
    files = np.array(files)
    file = files[iD]
    
    time_series = np.loadtxt(file, delimiter=',')
    print(file, time_series.shape)
    return time_series

# this is making me somehwat worried...
# somehow somethings not lining up, but what is that something?
# the structure looks right, so I'm going to ignore that this might be a problem for now!
# I think if I just binarised my data it might look the same anyway...?