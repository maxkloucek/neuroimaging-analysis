import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.linear_model import LogisticRegression
import sep_help as sep


plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')

# these are the extreme examples
def load_trajectory(choice='SG'):
    if choice == 'SG':
        file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_.500-h_0-J_.100-Jstd_1.hdf5'
    elif choice == 'F':
        file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_.500-h_0-J_2.000-Jstd_1.hdf5'
    elif choice == 'FC':
        file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_1.025-h_0-J_1.500-Jstd_1.hdf5'
    elif choice == 'P':
        # file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_1.025-h_0-J_.500-Jstd_1.hdf5'
        file = '/Users/mk14423/Desktop/Data/Nconst/N200_2/T_2.000-h_0-J_.500-Jstd_1.hdf5'
    with h5py.File(file, 'r') as f:
        print(f.keys())
        input_model = f['InputModel'][()]
        output_model = f['InferredModel'][()]
        dataset = f['configurations'][()]
    return input_model, output_model, dataset

def analyse():
    # not sure how exactly to catgorize these...
    in_mod, out_mod, dataset = load_trajectory(choice='F')
    nSamples, nSpins = dataset.shape
    dataset = dataset.astype(int)
    fig, ax = plt.subplots()
    ax.matshow(dataset.T[0:200, 0:600])
    plt.show()
   
    row_index = 0
    col_index = 58
    X = np.delete(dataset, row_index, 1)
    y = dataset[:, row_index]  # target

    # plt.plot(out_mod[58, :])
    # plt.show()
    
    # mi = np.mean(dataset, axis=0)
    # m = np.mean(mi)
    # why no change?
    print('m is  ', np.mean(dataset))
    print('y mean = ', np.mean(y))

   
    # defo don't sum them is what I've learned!
    for j in range(0, 59):
        # j = col_index
        x = X[:, j]
        mm = 0
        mp = 0
        pm = 0
        pp = 0
        # print('x mean = ', np.mean(x))

        for yi, xi in zip(y, x):
            if xi == -1 and yi == -1:
                mm += 1
            elif xi == -1 and yi == 1:
                mp += 1
            elif xi == 1 and yi == -1:
                pm += 1
            elif xi == 1 and yi == 1:
                pp += 1
        score_array = np.zeros((2,2))
        score_array[0, 0] = pp
        score_array[0, 1] = mp
        score_array[1, 0] = pm
        score_array[1, 1] = mm
        np.set_printoptions(suppress=True)
        

        norm = nSamples * (nSpins - 1)
        # score_array = score_array / norm
        print(score_array)


def firth_analysis():
    in_mod, out_mod, dataset = load_trajectory(choice='P')
    nSamples, nSpins = dataset.shape
    dataset = dataset.astype(int)
    # fig, ax = plt.subplots()
    # ax.matshow(dataset.T[0:200, 0:600])
    # plt.show()
   
    row_index = 0
    col_index = 58
    X = np.delete(dataset, row_index, 1)
    y = dataset[:, row_index]  # target
    x = X[:, 0:2]
    print(X.shape, x.shape, y.shape)
    log_reg = LogisticRegression(
        penalty='none',
        # C=10,
        # random_state=0,
        solver='lbfgs',
        max_iter=200
    )
    log_reg.fit(x, y)
    logreg_weights = np.array(
        [log_reg.coef_[0, 0], log_reg.coef_[0, 1], log_reg.intercept_[0]])
    print(logreg_weights)
    # firth_weights = sep.firth_logit(x,y, num_iter=250)
    # print(firth_weights.shape)
    # print(firth_weights)
    # cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # fig, ax = plt.subplots()
    # for i in range(0, 2):
    #     ax.plot(firth_weights[:, i], c=cols[i], ls='-')
    #     ax.axhline(logreg_weights[i], c=cols[i], ls='--', marker=',')
    # plt.show()
    # print(y_pred)
    pass

# analyse()
firth_analysis()

# hmmmmmm this parameter should be fine...?
# log_reg = LogisticRegression(
#     penalty='none',
#     # C=10,
#     # random_state=0,
#     solver='lbfgs',
#     max_iter=200)
# log_reg.fit(X, y)
# weights = log_reg.coef_[0] / 2
# bias = log_reg.intercept_[0] / 2
# print(weights[col_index])
# print(in_mod[col_index, row_index], out_mod[col_index, row_index])
# let's get m of the dataset...