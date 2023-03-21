import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import nxwrappers as nxw


def get_ROInames():
    name_file = './ROIname.txt'
    ROInames = np.genfromtxt(name_file, dtype=str, delimiter=',')
    ROInames = np.char.strip(ROInames, "'")
    ROInames = np.char.replace(ROInames, ' ', '')
    return ROInames


# this will output a ragged list of list, not sure I love this!
def connected_component_analysis(model, nComponents=3):
    # this should happen for the 3 biggest connected components right
    # so let's get community outputs!
    ROInames = get_ROInames()
    node_mapping = {i: ROInames[i] for i in range(0, len(ROInames))}
    # print(node_mapping)
    G = nx.from_numpy_matrix(model)
    G = nx.relabel_nodes(G, node_mapping)

    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    connected_components = connected_components[:nComponents]
    nSubgraphs = len(connected_components)
    subgraphs = [G.subgraph(c) for c in connected_components]

    nNodes = np.empty(nComponents)
    nNodes[:] = np.nan

    nEdges = np.empty(nComponents)
    nEdges[:] = np.nan

    nCommunities = np.empty(nComponents)
    nCommunities[:] = np.nan

    CommMax = np.empty(nComponents)
    CommMax[:] = np.nan

    CommMean = np.empty(nComponents)
    CommMean[:] = np.nan


    # print('----')
    # print(nSubgraphs)
    print('--loop start--')
    for iSubGraph in range(0, nSubgraphs):
        subgraph = subgraphs[iSubGraph]
        # print(iSubGraph, len(list(subgraph.nodes)))
        nNodes[iSubGraph] = subgraph.number_of_nodes()
        nEdges[iSubGraph] = subgraph.number_of_edges()

        # nxw.plot_communities(subgraph) # ok this works yay!
        communities = nxw.get_communities(subgraph)
        comm_sizes = [len(c) for c in communities]
        nCommunities[iSubGraph] = len(communities)
        CommMax[iSubGraph] = comm_sizes[0]
        CommMean[iSubGraph] = np.mean(comm_sizes)
        # I could either do degree for communties or for network as a whole?
        nxw.get_degrees(subgraph)
        # print(comm_sizes[0], np.mean(comm_sizes))
        # closeness = nx.closeness_centrality(graph)
        # closeness = dict(sorted(closeness.items(), key=lambda item: item[1]))
        # plt.bar(closeness.keys(), closeness.values())
        # plt.show()
        # print(closeness)
    # print(nNodes)
    # print(nEdges)
    # print(nNodes)
    # print(CommMax, CommMean)
    return nNodes, nEdges, nCommunities, CommMax, CommMean
