# this is the main testing script!!
import h5py
import numpy as np
import matplotlib.pyplot as plt
from firthlogist import FirthLogisticRegression, load_sex2
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from pyplm.utilities.tools import triu_flat
from pyplm.plotting import mkfigure

def plm(trajectory):
    print('--- PLM ---')
    nSamples, nSpins = trajectory.shape
    infr_model = np.zeros((nSpins, nSpins))
    # nSpins = 1
    for row_index in tqdm(range(0, nSpins)):
        # print(row_index)
        X = np.delete(trajectory, row_index, 1)
        y = trajectory[:, row_index]  # target
        log_reg = LogisticRegression(
            penalty='none',
            solver='lbfgs',
            max_iter=200)

        log_reg.fit(X, y)
        # factor of 2 from equations! wieghts size = N-1
        weights = log_reg.coef_[0] / 2
        bias = log_reg.intercept_[0] / 2

        left_weights = weights[0:row_index]
        right_weights = weights[row_index:]

        infr_model[row_index, 0:row_index] = left_weights
        infr_model[row_index, row_index+1:] = right_weights
        infr_model[row_index, row_index] = bias
        # row_parameters[0:row_index] = left_weights
        # row_parameters[row_index+1:] = right_weights
        # row_parameters[row_index] = bias


    # model_inf = np.array(model_inf)
    infr_model = (infr_model + infr_model.T) * 0.5
    # print(model_inf.shape)
    return infr_model

def plm_firth(trajectory):
    print('--- Firth ---')
    nSamples, nSpins = trajectory.shape
    infr_model = np.zeros((nSpins, nSpins))
    nSpins = 1
    for row_index in tqdm(range(0, nSpins)):
        # print(row_index)
        X = np.delete(trajectory, row_index, 1)
        y = trajectory[:, row_index]  # target

        fl = FirthLogisticRegression(skip_pvals=True, skip_ci=True, max_iter=100)
        fl.fit(X, y)
        # log_reg = LogisticRegression(
        #     penalty='none',
        #     solver='lbfgs',
        #     max_iter=200)

        # log_reg.fit(X, y)
        # factor of 2 from equations! wieghts size = N-1

        weights = fl.coef_ / 2
        bias = fl.intercept_ / 2

        left_weights = weights[0:row_index]
        right_weights = weights[row_index:]

        infr_model[row_index, 0:row_index] = left_weights
        infr_model[row_index, row_index+1:] = right_weights
        infr_model[row_index, row_index] = bias
        # row_parameters[0:row_index] = left_weights
        # row_parameters[row_index+1:] = right_weights
        # row_parameters[row_index] = bias


    # model_inf = np.array(model_inf)
    infr_model = (infr_model + infr_model.T) * 0.5
    # print(model_inf.shape)
    return infr_model

def temp(model):
    N, _ = model.shape
    Js = triu_flat(model)
    T = 1 / (np.std(Js) * (N ** 0.5))
    return T

# these are the extreme examples
def load_trajectory(choice='SG'):
    if choice == 'SG':
        file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_.500-h_0-J_.100-Jstd_1.hdf5'

        # file = '/Users/mk14423/Desktop/Data/Nconst/N50_1/T_.500-h_0-J_.100-Jstd_1.hdf5'
    elif choice == 'F':
        # file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_.500-h_0-J_2.000-Jstd_1.hdf5'

        file = '/Users/mk14423/Desktop/Data/Nconst/N50_1/T_.500-h_0-J_2.000-Jstd_1.hdf5'
    elif choice == 'FC':
        file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_1.025-h_0-J_1.500-Jstd_1.hdf5'

        # file = '/Users/mk14423/Desktop/Data/Nconst/N50_1/T_1.025-h_0-J_1.500-Jstd_1.hdf5'
    elif choice == 'P':
        # file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_1.025-h_0-J_.500-Jstd_1.hdf5'
        # file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_2.000-h_0-J_.500-Jstd_1.hdf5'

        file = '/Users/mk14423/Desktop/Data/Nconst/N50_1/T_2.000-h_0-J_.500-Jstd_1.hdf5'
    with h5py.File(file, 'r') as f:
        print(f.keys())
        input_model = f['InputModel'][()]
        output_model = f['InferredModel'][()]
        dataset = f['configurations'][1000:, :200]
    return input_model, output_model, dataset


plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
in_mod, out_mod, dataset = load_trajectory(choice='SG')
nSamples, nSpins = dataset.shape
# dataset = dataset.astype(int)
cij = np.cov(dataset.T)

# plm_model = plm(dataset)
plm_model = out_mod
# dataset = dataset[:, 20:20]
fith_model = plm_firth(dataset)
# we can just load plm, no need to do it again! I trust this stuff now!
print('-----')
print('TRU', in_mod[0, 0:5])
print('PLM', plm_model[0, 0:5])
print('FIR', fith_model[0, 0:5])
T_tru = temp(in_mod)
T_plm = temp(plm_model)
T_fir = temp(fith_model)
print(T_tru)
print(T_plm)
print(T_fir)

fig, ax = mkfigure(nrows=3, ncols=1, sharex=True, sharey=True)
vmax = in_mod.max()
vmin = in_mod.min()
ax = ax.ravel()
ax[0].matshow(in_mod, vmin=vmin, vmax=vmax)
ax[1].matshow(plm_model, vmin=vmin, vmax=vmax)
ax[2].matshow(fith_model, vmin=vmin, vmax=vmax)
ax[1].set(ylabel=r'$i$')
ax[2].set(xlabel=r'$j$')
plt.show()
