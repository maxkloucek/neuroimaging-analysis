import pandas as pd


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