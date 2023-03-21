import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subsamplingFunctions as sF
from subsamplingFunctions import io as ss_io
from subsamplingFunctions import plots as ss_plots
import functionsModel.plots as modplots

import figures as figs


# plt.style.use('/Users/mk14423/Dropbox/mpl-styles/thesisbody.mplstyle')

file = '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5'
groups = ['grouped']
# T_trues = [3.06087293]
T_trues = [None]
g_labels = ['HCP dataset']
figs.hcp_subsampling(file, groups[0], true_temp=T_trues[0], label=g_labels)

# file = '/Users/mk14423/Desktop/Data/0_thesis/SubSampleSK/datasets.hdf5'
# groups = ['N50', 'N100', 'N200', 'N400', 'N800']
# T_trues = [None]
# figs.hcp_subsampling(file, groups[0], true_temp=T_trues[0], label=g_labels, threshold_B=3e3)
# I want a little more control over the axies and stuff
# figs.hcp_Tscaling(file, groups[0], true_temp=T_trues[0], label=g_labels)

# ss_df = ss_io.get_subsampling_data_frame(file, groups[0], T_trues[0])

# ss_plots.model(file, groups[0], iD=0)
# construct figures... I think I want a better structure!
# fig, ax = plt.subplots()
# fitting_th = 1e4
# popts = ss_plots.subsampling_plot(ax, ss_df, fitting_th, 0, g_labels[0])
# ax.set(
#     # ylim=[0.5, 1.0],
#     xlabel=r'$B$', ylabel=r'$T^{*} / T^{0}$')
# # ax.axhline(1, ls='-', c='k', marker=',')
# print(popts)
# plt.legend()
# # if save is True:
# #     plt.savefig(os.path.join(FIGDIR, 'N-saturation-curves.png'))
# plt.show()

# I need to compare this with what I was finding before!!
# because 3 seems larger than what it was, maybe I did something wrong
# this is when I find out it would have been good to save the models...

# it's annoying that these don't talk to eachother