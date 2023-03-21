import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from firthlogreg import FirthLogisticRegression

# from tqdm import tqdm
from time import perf_counter
from pyplm.utilities.tools import triu_flat



def plm(trajectory):
    print('--- PLM ---')
    nSamples, nSpins = trajectory.shape
    infr_model = np.zeros((nSpins, nSpins))
    # nSpins = 1
    t0 = perf_counter()
    for row_index in (range(0, nSpins)):
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
    t1 = perf_counter()
    print(f'Time taken PLM: {t1-t0:.3f}')
    infr_model = (infr_model + infr_model.T) * 0.5
    # print(model_inf.shape)
    return infr_model
    

def plm_firth(trajectory):
    # this is running the plm, I want to profile the firth!
    print('--- Firth ---')
    nSamples, nSpins = trajectory.shape
    infr_model = np.zeros((nSpins, nSpins))
    nSpins = 10
    t0 = perf_counter()
    for row_index in (range(0, nSpins)):
        print(row_index)
        X = np.delete(trajectory, row_index, 1)
        y = trajectory[:, row_index]  # target

        fl = FirthLogisticRegression(skip_pvals=True, skip_ci=True, max_iter=100)
        fl.fit(X, y)

        weights = fl.coef_ / 2
        bias = fl.intercept_ / 2

        left_weights = weights[0:row_index]
        right_weights = weights[row_index:]

        infr_model[row_index, 0:row_index] = left_weights
        infr_model[row_index, row_index+1:] = right_weights
        infr_model[row_index, row_index] = bias
    t1 = perf_counter()
    print(f'Time taken firth: {t1-t0:.3f}')
    infr_model = (infr_model + infr_model.T) * 0.5
    return infr_model

def _calc_temp(model):
    N, _ = model.shape
    Js = triu_flat(model)
    T = 1 / (np.std(Js) * (N ** 0.5))
    return T


# these are the extreme examples, loading 50 or N200. just to profile!
def load_trajectory(choice='SG'):
    if choice == 'SG':
        # file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_.500-h_0-J_.100-Jstd_1.hdf5'
        file = '/Users/mk14423/Desktop/Data/Nconst/N50_1/T_.500-h_0-J_.100-Jstd_1.hdf5'
    elif choice == 'F':
        # file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_.500-h_0-J_2.000-Jstd_1.hdf5'
        file = '/Users/mk14423/Desktop/Data/Nconst/N50_1/T_.500-h_0-J_2.000-Jstd_1.hdf5'
    elif choice == 'FC':
        # file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_1.025-h_0-J_1.500-Jstd_1.hdf5'
        file = '/Users/mk14423/Desktop/Data/Nconst/N50_1/T_1.025-h_0-J_1.500-Jstd_1.hdf5'
    elif choice == 'P':
        file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_2.000-h_0-J_.500-Jstd_1.hdf5'
        # file = '/Users/mk14423/Desktop/Data/Nconst/N50_1/T_2.000-h_0-J_.500-Jstd_1.hdf5'
    with h5py.File(file, 'r') as f:
        print(f.keys())
        input_model = f['InputModel'][()]
        output_model = f['InferredModel'][()]
        dataset = f['configurations'][1000:, :200]
    return input_model, output_model, dataset

def load_const_mu_models():
    root = '/Users/mk14423/Desktop/Data/N200_J0.1_Bscaling/B1e4_1/'
    files = [
        'T_0.500-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_0.575-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_0.650-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_0.725-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_0.800-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_0.875-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_0.950-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.025-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.100-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.175-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.250-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.325-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.400-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.475-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.550-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.625-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.700-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.775-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.850-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_1.925-h_0.000-J_0.000-Jstd_1.000.hdf5',
        'T_2.000-h_0.000-J_0.000-Jstd_1.000.hdf5'
        ]
    per_temp_in_models = []
    per_temp_traj_configs = []
    temps = np.linspace(0.5, 2, 21)
    # print(temps)
    # let's write some temperatures to the md?
    for file in files:
        path = root + file
        print(file)
        with h5py.File(path, 'r') as f:
            # print(f.keys())
            in_mod = f['InputModel'][()]
            # configs = f['configurations'][1000:, :]
            configs = f['configurations'][1000::10, :]
            # print(configs.shape)
            # print(in_mod.shape, configs.shape)
        per_temp_in_models.append(in_mod)
        per_temp_traj_configs.append(configs)
    per_temp_in_models = np.array(per_temp_in_models)
    per_temp_traj_configs = np.array(per_temp_traj_configs)
    # print(per_temp_in_models.shape, per_temp_traj_configs.shape)
    return temps, per_temp_in_models, per_temp_traj_configs











