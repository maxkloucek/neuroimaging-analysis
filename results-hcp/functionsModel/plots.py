import numpy as np
import matplotlib.pyplot as plt
import h5py

from pyplm.utilities import tools
x = 1
# my linting is fucking up big time for some reason??
def plot_model(file, group, iD):
    with h5py.File(file, 'r') as fin:
        print(fin.keys())
        print(fin[group].keys())
        inf_mod = fin[group]['inferredModels'][iD, :, :]  # type: ignore
    N, _ = inf_mod.shape  # type: ignore
    params = tools.triu_flat(inf_mod, k=1)
    T = 1 / (np.std(params) * (N **0.5))
    print(T)
    plt.matshow(inf_mod)
    plt.show()
