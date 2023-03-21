import numpy as np
import pandas as pd
import os
import h5py
import networkx as nx
from tqdm import tqdm

import matplotlib.pyplot as plt

import hcp_helpers as helpers
from scipy.ndimage import label
from tqdm import tqdm

# have them all dump to a hdf5 file, thats a dataframe..?

class SweepAnalysis:
    def __init__(self, file, group, observables=['m', 'q', 'C2']):
        print(file, group)
        print(observables)
        self.file = file
        self.group = group

        with h5py.File(file, 'r') as fin:
            alphas = fin[group]['sweep-alphas'][()]
            trajectories = fin[group]['sweep-trajectories']
            nModels, nAlphas, nReps, B, N = trajectories.shape
        # print(alphas.shape, alphas.size)
    
        self.alphas = alphas
        self.obs = np.array(observables)
        self.nReps = nReps


        # then for each alpha load and clculate...
    def setup_output(self, outfile, ds_label):
        # outfile = os.path.join(outdir, 'analysed_data.hdf5')
        print(outfile)
        # print(self.alphas.size, self.obs.size)
        # here I need to know how to phrase this!
        if os.path.exists(outfile) is True:
            fopen_kwrds = {
                'mode': 'a',
            }
        else:
            fopen_kwrds = {
                'mode': 'w',
                'track_order': True,
                'fs_strategy': 'fsm',
                'fs_persist': True
            }
        print(fopen_kwrds)
        with h5py.File(outfile, **fopen_kwrds) as fout:
            g = fout.require_group(self.group)
            d = g.require_dataset(
                ds_label,
                shape=(self.alphas.size * self.nReps, self.obs.size + 2),
                dtype=float,
                compression="gzip")
            # let's fill the thing yiwth the alphas as well!
            # d[:, 0] = self.alphas
            # print(group['sweep-observables'].shape)
            obs_labels = np.append(['alpha', 'iD'], self.obs)
            print(obs_labels, obs_labels.dtype)
            # no I think I should have it go, 0, 1, 2, 3...
            for iObs, obs_label in enumerate(obs_labels):
                d.attrs[str(iObs)] = str(obs_label)
            for key, value in d.attrs.items():
                print(key, value)


    def calculate_observables(self, iModel, ds_label='sweep-observables'):
        outdir = os.path.dirname(self.file)
        outfile = os.path.join(outdir, 'analysed_data.hdf5')
        file_exists = os.path.exists(outfile)
        print(file_exists)
        if file_exists:
            print(f'Whatch out, file {outfile} already exists.')
            print('Running this method will overwride your calculated data.')
            return

        self.setup_output(outfile, ds_label)
        for iAlpha in tqdm(range(0, self.alphas.size)):
            # print(iAlpha)
            trajectories = helper_load_repeats(self.file, self.group, iModel, iAlpha)
            with h5py.File(outfile, 'a') as fout:
                # print(fout[self.group][ds_label][:, 0])
                d = fout[self.group][ds_label]
                for iRep in range(0, self.nReps):
                    iVar = (iAlpha * self.nReps) + iRep
                    # print(iAlpha, iRep, iVar)
                    d[iVar, 0] = self.alphas[iAlpha]
                    d[iVar, 1] = iRep
                    d[iVar, 2] = calc_m(trajectories[iRep])
                    d[iVar, 3] = calc_q(trajectories[iRep])
                    d[iVar, 4] = calc_C2(trajectories[iRep])

    def load_analysed_data(self, ds_label='sweep-observables'):
        outdir = os.path.dirname(self.file)
        outfile = os.path.join(outdir, 'analysed_data.hdf5')
        with h5py.File(outfile, 'r') as fin:
            d = fin[self.group][ds_label]
            observation_table = d[()]
            keys = list(d.attrs.keys())
            values = [d.attrs[key] for key in keys]
        dataframe = pd.DataFrame(observation_table, columns=values)
        return dataframe


def helper_load_repeats(file, group, iModel, iAlpha):
    with h5py.File(file, 'r') as fin:
        trajectories = fin[group]['sweep-trajectories'][iModel, iAlpha, :, :, :]
    return trajectories    

def calc_m(trajectory):
    si_avrg = np.mean(trajectory, axis=0)
    return np.mean(si_avrg)

def calc_q(trajectory):
    si_avrg = np.mean(trajectory, axis=0)
    si_avrg_sqr = si_avrg ** 2
    return np.mean(si_avrg_sqr)

def calc_C2(trajectory):
    cij = np.cov(trajectory.T)
    N, _ = cij.shape
    C = np.sum(cij ** 2)
    C = C / N
    return C

# @njit
def calculate_E(configuration, parameter_matrix):
    ss_matrix = np.outer(configuration, configuration)
    ss_matrix = ss_matrix * 0.5
    np.fill_diagonal(ss_matrix, configuration)
    E = -np.sum(ss_matrix * parameter_matrix)
    return E

def calc_avalanche(trajectory):
    B, N = trajectory.shape
    # print(B, N)
    # avalanches = np.array([])
    # print(avalanches, avalanches.shape)
    print('Calculating avalanche distribution...')
    for i_spin in tqdm(range(0, N)):
        # print(i_spin)
        spin_traj = trajectory[:, i_spin]
        spin_traj[spin_traj == -1] = 0
        labeled_array, num_features = label(spin_traj)
        # print(num_features, np.max(labeled_array))
        # print(labeled_array == num_features)
        # so we wnat to go for in range(0, num_features + 1)
        # print(np.unique(labeled_array))
        # for i in range(0, 10):
        #     print(spin_traj[i], labeled_array[i])
        # print(num_features)
        # print(labeled_array.shape, num_features)
        # print(spin_traj.shape)
        avalanche_durations = np.zeros(num_features + 1) # includes zero!
        for iF in range(0, num_features + 1):
            # print(iF) # shape should be plus !
            feature_duration = labeled_array[labeled_array == iF].size
            avalanche_durations[iF] = feature_duration
        avalanche_durations = avalanche_durations[1:]
        if i_spin == 0:
            avalanches = np.copy(avalanche_durations)
        else:
            avalanches = np.hstack((avalanches, avalanche_durations))
        # print(avalanches.shape)
        # avalanches = np.hstack((avalanches, avalanche_durations)) if avalanches.size else avalanche_durations
        # print(num_features, avalanche_durations.shape)
        # print(avalanche_durations.min(), avalanche_durations.max())
    # print('------')
    return avalanches



class SweepAnalysis2:
    def __init__(self, file, group, sweep_group, observables=['m', 'q', 'C2']):
        print(file, group, sweep_group)
        print(observables)
        self.file = file
        self.group = group
        self.sweep_group = sweep_group

        with h5py.File(file, 'r') as fin:
            parameters = fin[group][sweep_group]['parameters'][()]
            trajectories = fin[group][sweep_group]['trajectories']
            nParameters, nReps, B, N = trajectories.shape
        # print(alphas.shape, alphas.size)
    
        self.parameters = parameters
        self.obs = np.array(observables)
        self.nReps = nReps

    def setup_output(self, outfile, ds_label):
        # outfile = os.path.join(outdir, 'analysed_data.hdf5')
        print(outfile)
        # print(self.alphas.size, self.obs.size)
        # here I need to know how to phrase this!
        if os.path.exists(outfile) is True:
            fopen_kwrds = {
                'mode': 'a',
            }
        else:
            fopen_kwrds = {
                'mode': 'w',
                'track_order': True,
                'fs_strategy': 'fsm',
                'fs_persist': True
            }
        print(fopen_kwrds)
        with h5py.File(outfile, **fopen_kwrds) as fout:
            g = fout.require_group(self.group)
            d = g.require_dataset(
                ds_label,
                shape=(self.parameters.size * self.nReps, self.obs.size + 2),
                dtype=float,
                compression="gzip")

            obs_labels = np.append(['param', 'iD'], self.obs)
            print(obs_labels, obs_labels.dtype)
            # no I think I should have it go, 0, 1, 2, 3...
            for iObs, obs_label in enumerate(obs_labels):
                d.attrs[str(iObs)] = str(obs_label)
            for key, value in d.attrs.items():
                print(key, value)


    def calculate_observables(self, ds_label=None):
        if ds_label == None:
            ds_label = self.sweep_group
            print('Label in analysed_data.hdf5 is ', ds_label)
        outdir = os.path.dirname(self.file)
        outfile = os.path.join(outdir, 'analysed_data.hdf5')
        file_exists = os.path.exists(outfile)
        print(file_exists)
        if file_exists:
            print(f'Whatch out, file {outfile} already exists.')
            print('Running this method will overwride your calculated data.')
            cont = input("Do you want to continue? (y/n):")
            if cont == 'y':
                pass
            if cont == 'n':
                exit()
        self.setup_output(outfile, ds_label)
        for iP in tqdm(range(0, self.parameters.size)):
            with h5py.File(self.file, 'r') as fin:
                trajectories = fin[self.group][self.sweep_group]['trajectories'][iP, :, :, :]
            with h5py.File(outfile, 'a') as fout:
                d = fout[self.group][ds_label]
                for iRep in range(0, self.nReps):
                    iVar = (iP * self.nReps) + iRep
                    d[iVar, 0] = self.parameters[iP]
                    d[iVar, 1] = iRep
                    d[iVar, 2] = calc_m(trajectories[iRep])
                    d[iVar, 3] = calc_q(trajectories[iRep])
                    d[iVar, 4] = calc_C2(trajectories[iRep])

    def load_analysed_data(self, ds_label=None):
        if ds_label == None:
            ds_label = self.sweep_group

        outdir = os.path.dirname(self.file)
        outfile = os.path.join(outdir, 'analysed_data.hdf5')
        with h5py.File(outfile, 'r') as fin:
            d = fin[self.group][ds_label]
            observation_table = d[()]
            keys = list(d.attrs.keys())
            values = [d.attrs[key] for key in keys]
        dataframe = pd.DataFrame(observation_table, columns=values)
        return dataframe

    def load_threshold_models_obs(self, ds_label=None):
        if ds_label == None:
            ds_label = self.sweep_group
        print(ds_label)

        with h5py.File(self.file, 'r') as fin:
            model = fin[self.group]['inferredModels'][0, :, :]
            thresholds = fin[self.group][self.sweep_group]['parameters'][()]
        print(model.shape)
        print(thresholds.shape)
        N, _ = model.shape
        nModels, = thresholds.shape
        models = np.zeros((nModels, N, N))
        degree_mean = np.zeros((nModels))
        degree_std = np.zeros((nModels))
        degree_max = np.zeros((nModels))

        sG_kmean = np.zeros((nModels))
        sG_kstd = np.zeros((nModels))
        sG_nNodes = np.zeros((nModels))
        sG_nEdges = np.zeros((nModels))
        # SG_nNodes[iSubGraph] = subgraph.number_of_nodes()
        # nEdges[iSubGraph] = subgraph.number_of_edges()

        for iM, _ in enumerate(models):
            mod = np.copy(model)
            count = np.sum(np.abs(mod) <= thresholds[iM])
            # print(iM, count, mod.size)
            np.fill_diagonal(mod, 0)
            # mod[np.abs(mod) <= thresholds[iM]] = 0
            # mod[mod != 0] = 1
            # models[iM] = mod

            if ds_label == 'sweepTH_symmetric':
                mod[np.abs(mod) <= thresholds[iM]] = 0
            elif ds_label == 'sweepTH_positive':
                mod[mod <= thresholds[iM]] = 0
            else:
                print(ds_label, ' is not a valid ds_label!')
                exit()
            mod[mod != 0] = 1
            models[iM] = mod


            G = nx.from_numpy_matrix(mod)
            degree_sequence = list((d for n, d in G.degree()))
            # print(degree_sequence)
            degree_mean[iM] = np.mean(degree_sequence)
            degree_std[iM] = np.std(degree_sequence)
            degree_max[iM] = np.max(degree_sequence)

            connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
            biggest_component = connected_components[0]
            sG = G.subgraph(biggest_component)
            degree_sequence = list((d for n, d in sG.degree()))
            sG_kmean[iM] = np.mean(degree_sequence)
            sG_kstd[iM] = np.std(degree_sequence)

            sG_nNodes[iM] = sG.number_of_nodes()
            sG_nEdges[iM] = sG.number_of_edges()
            # nSubgraphs = len(connected_components)
            # print(nSubgraphs)
            
            # print(sG)
            # let's measure some things!
            # subgraphs = [G.subgraph(c) for c in connected_components]

        degree_df = pd.DataFrame(
            {
                'param': thresholds,
                'k-mean': degree_mean,
                'k-std': degree_std,
                'k-max': degree_max,
                'sG-k-mean': sG_kmean,
                'sG-k-std': sG_kstd,
                'sG-nN': sG_nNodes,
                'sG-nE': sG_nEdges
            }
        )
        return degree_df

    def load_threshold_model(self, iM):
        with h5py.File(self.file, 'r') as fin:
            model = fin[self.group]['inferredModels'][0, :, :]
            threshold = fin[self.group][self.sweep_group]['parameters'][iM]
        # print(model.shape, threshold)
        if self.sweep_group == 'sweepTH_symmetric':
            model[np.abs(model) <= threshold] = 0
        elif self.sweep_group == 'sweepTH_positive':
            model[model <= threshold] = 0
        # model[np.abs(model) <= threshold] = 0
        return model, threshold
        