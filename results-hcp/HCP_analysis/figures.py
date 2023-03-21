import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

from pyplm.pipelines import model_pipeline
from pyplm.utilities import tools
from pyplm.analyse.models import centered_histogram

import fighelpers

def GroupVsIndividual_distro():
    indi_selector = {
        'file': '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
        'group': 'individuals',
        'dataset': 'inferredModels',
        'pname': 'J'
    }
    grp_selector = {
        'file': '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
        'group': 'grouped',
        'dataset': 'inferredModels',
        'pname': 'J'
    }
    iMod = 0

    fig, ax = plt.subplots()
    fighelpers.single_model_histogram(ax, iMod, indi_selector, marker='o')
    fighelpers.single_model_histogram(ax, iMod, grp_selector, marker='s')
    ax.set(
        xlabel=r'$J_{ij}$', ylabel=r'$P(J_{ij})$',
        # xlabel=r'$J_{ij}$', ylabel=r'$P(J < J_{ij})$',
        # xscale='log', yscale='log',
        # xlim=[1e-3, 1e0],
    )
    ax.legend()
    plt.savefig('./prelim-results/grp-indi-pdf.png')
    plt.show()

def GroupVsIndividual_distro_tail():
    indi_selector = {
        'file': '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
        'group': 'individuals',
        'dataset': 'inferredModels',
        'pname': 'J'
    }
    grp_selector = {
        'file': '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
        'group': 'grouped',
        'dataset': 'inferredModels',
        'pname': 'J'
    }
    fig, ax = plt.subplots()
    fighelpers.single_model_tail(ax, 0, indi_selector, marker='o')
    # fighelpers.single_model_tail(ax, 54, indi_selector, marker='*')
    fighelpers.single_model_tail(ax, 0, grp_selector, showMethod=False, marker='s')
    # use this in a new plot! print(res_dict)
    ax.set(
        xlabel=r'$J_{ij}$', ylabel=r'$P(J_{ij})$',
        # xlabel=r'$J_{ij}$', ylabel=r'$P(J < J_{ij})$',
        xscale='log', yscale='log',
        xlim=[1e-3, 1e0],
    )
    ax.legend()
    # plt.savefig('./prelim-results/grp-indi-pdf-tail.png')
    plt.show()


def GroupModMatrix():
    grp_selector = {
        'file': '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
        'group': 'grouped',
        'dataset': 'inferredModels',
        'pname': 'J'
    }
    fig, ax = plt.subplots()
    results = fighelpers.single_model_tail(ax, 0, grp_selector, showMethod=False, marker='s')
    plt.close()

    fighelpers.single_model_matrix_fullprocess(0, grp_selector, threshold_dictionary=results)


def GroupModMatrixSharedSubNet():
    grp_selector = {
        'file': '/Users/mk14423/Desktop/PaperData/HCP_rsfmri_added_data.hdf5',
        'group': 'grouped',
        'dataset': 'inferredModels',
        'pname': 'J'
    }
    fig, ax = plt.subplots()
    results = fighelpers.single_model_tail(ax, 0, grp_selector, showMethod=False, marker='s')
    plt.close()
    fighelpers.single_model_LR_shared_subnet(0, grp_selector, threshold_dictionary=results)