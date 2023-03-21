from pyplm.pipelines import data_pipeline
from pyplm.utilities import tools
from scipy.optimize import curve_fit

file = '/Users/mk14423/Desktop/Data/0_thesis/SubSampleSK/datasets.hdf5'
group = 'N800'

# plm_pipeline = data_pipeline(file, group)
# plm_pipeline.subsample(no_ss_points=100)

# check no l1 regularisation is on at the moment!
# that would be a mare! also use the lgbfs minimizer.

import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np

plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')
# wtf am I esitmating here... this seems wrong; maybe I just calculate my thingy wrong?!?
# T should really be 1.1 not 0.3!?!?
# wow this works nicely :)!
# np.std(infr_params) * (N**0.5)
df = pd.read_hdf(file, group + '/subsampling')

# rescale by N
df['mean_J'] = df['mean_J'] * df['N']
df['std_J'] = df['std_J'] * (df['N'] ** 0.5)

# calculate mu and T
# mu_infr = np.mean(infr_params) * N
# sigma_infr = np.std(infr_params) * (N**0.5)       
# mu_infr = mu_infr / sigma_infr
df['mu'] = df['mean_J'] / df['std_J']
df['T'] = 1 / df['std_J']

df = df.sort_values(by=['B', 'iD'])
print(df)
# for iD_label in range(0, 6):
#     identifier = 'mu'
#     subset_df = df.loc[df['iD'] == iD_label]
#     x = subset_df['B']
#     y = subset_df[identifier]
#     func = tools.arctan
#     subset_df = subset_df.loc[subset_df['B'] > 2500]
#     popt, _ = curve_fit(tools.arctan, subset_df['B'], subset_df[identifier], p0 = [0.1, 0.1])
#     # the popt here don't make any sense... the value of A should clearly not be negative?!?
#     xfit = np.linspace(x.min(), x.max(), 200)
#     yfit = func(xfit, *popt)
#     # hmmmm this doesn't make massive sense!
#     plt.plot(x, y, ls='none', alpha=1)
#     plt.plot(xfit, yfit, ls='--', marker=',', c='k', lw=2)
#     popt[0] = (popt[0] * np.pi) / 2
#     popt[1] = 1 / popt[1]
#     print(iD_label, popt)
# plt.show()

# yay made it work :)!
# I can also estimate mu in the same way now that I understand it a bit better!!
# 

# I should get the infred temp as well!
# and see the error like that?

identifier = 'T'
x = df['B']
y = df[identifier]
df = df.loc[df['B'] >= 2500]
func = tools.arctan
popt, _ = curve_fit(tools.arctan, df['B'], df[identifier], p0 = [0.1, 0.1])
xfit = np.linspace(x.min(), x.max(), 200)
yfit = func(xfit, *popt)
# hmmmm this doesn't make massive sense!
plt.plot(x, y, ls='none', alpha=0.5)
plt.plot(xfit, yfit, ls='--', marker=',', c='k', lw=2)
popt[0] = (popt[0] * np.pi) / 2
popt[1] = 1 / popt[1]
print(popt)
plt.show()