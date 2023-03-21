# trying to make the most basic working version!
import numpy as np
import matplotlib.pyplot as plt
from pyplm.plotting import mkfigure
import helpers
# from tqdm import 
from time import perf_counter
plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')


def test_1():
    in_mod, out_mod, dataset = helpers.load_trajectory(choice='P')
    nSamples, nSpins = dataset.shape
    # dataset = dataset[::50, :]
    print('Dataset shape:', dataset.shape)


    # dataset = dataset.astype(int)
    # cij = np.cov(dataset.T)

    # let's work out a minimum working example..?
    plm_model = helpers.plm(dataset)
    # plm_model = out_mod
    # dataset = dataset[:, 20:20]
    fith_model = helpers.plm_firth(dataset)
    # we can just load plm, no need to do it again! I trust this stuff now!
    print('-----')
    print('TRU', in_mod[0, 0:5])
    print('PLM', plm_model[0, 0:5])
    print('FIR', fith_model[0, 0:5])
    T_tru = helpers._calc_temp(in_mod)
    T_plm = helpers._calc_temp(plm_model)
    T_fir = helpers._calc_temp(fith_model)
    print(T_tru)
    print(T_plm)
    print(T_fir)

    pre_edit_coefs = [-0.00451931, -0.02433114, 0.04946996, -0.0487061, 0.02857161]
    print(np.allclose(fith_model[0, 0:5], pre_edit_coefs))

    # fig, ax = mkfigure(nrows=3, ncols=1, sharex=True, sharey=True)
    # vmax = in_mod.max()
    # vmin = in_mod.min()
    # ax = ax.ravel()
    # ax[0].matshow(in_mod, vmin=vmin, vmax=vmax)
    # ax[1].matshow(plm_model, vmin=vmin, vmax=vmax)
    # ax[2].matshow(fith_model, vmin=vmin, vmax=vmax)
    # ax[1].set(ylabel=r'$i$')
    # ax[2].set(xlabel=r'$j$')
    # plt.show()

test_1()