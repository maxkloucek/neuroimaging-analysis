import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np

from pyplm.pipelines import data_pipeline
from pyplm.utilities import tools
from scipy.optimize import curve_fit

file = '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5'
group = 'grouped'

plm_pipeline = data_pipeline(file, group)
# plm_pipeline.subsample(no_ss_points=100)
plm_pipeline.threshold_sweep(1e3, 6)
