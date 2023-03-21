import os
import numpy as np
import matplotlib.pyplot as plt

# import inference.analysis.new as analysis
# from statsmodels.stats.weightstats import DescrStatsW
from inference import tools

import inference.scripts.postobs as post

import load
import plots
import niceplots as nplots
import subplots as splots
# import plots

SIGMA = 0.5
JPICK = 1
PLOTS = [
    # '3PANEL',
    # 'POWERLAW',
    # 'ESCALING',
]
# plt.style.use('seaborn-colorblind')
# plots.figOverview(save=False)
# plots.figSaturation(save=False)
# plots.figSaturation_2pannel(save=False)
# plots.figTcurves_and_distro(save=True)
# plots.figDistribution(save=False)
# plots.figDistribution_single(save=False)
# plots.figCorrectionTC2(TorC2='C2', save=False)
# plots.figCorrectionTC2(TorC2='T', save=False)
# plots.figKajimuraDemonstration(save=False)


# nice plots correctly sized
# nplots.PhaseDiagramOverview(save=False) # NOT USED
nplots.PhaseDiagramOverview_squareTau(save=False) # USED
# nplots.PhaseDiagramOverview_allN(save=False) # USED, MISSING EMINVSB, CODE FOR THAT IS ELSEWHERE!
# nplots.BtildeNscaling_convtoB(save=False) # RELATION INVALID!

# nplots.FixedN_observablesCut(save=False) # USED
# nplots.InferredT_and_distribution(save=False) # USED -> is now inset instead of multifig

# nplots.TSaturation_and_fit_inset(save=False) # USED -> REFORMED INTO TWO FIGRUES
# nplots.TSaturation_vary_includedB(save=False) # NOT SURE WHAT THIS SHOWS
# nplots.TSaturation_rescale_test(save=False)   # NOT SURE WHAT THIS SHOWS

# nplots.cor2_infr_temp(save=False) # USED, SHOWS STATURATION CORRECTION!

# nplots.Correction_T_C2(save=False) # USED -> REFORMATED AND RECOLOURED 
# ^ I use this twice, and chagned it to produce the differnet figures
# Not going to be easy to edit if I need to remake these figures.


# nplots.C2_B_fakedata(save=True) # USED

# didn't use any of this! All moved to it's onw nice new place!
# nplots.KajimuraSubSampling(save=False)
# nplots.Kajimura_model_similarity(save=False)
# nplots.KajimuraSweep(save=False)
# nplots.KajimuraShowModel(save=False)
# nplots.KajimuraShowCij(save=False)

# nplots.rework_1overB(save=False)
# nplots.rework_var_and_B(save=False)