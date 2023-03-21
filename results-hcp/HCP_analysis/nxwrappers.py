import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_communities(G):
    communities = get_communities(G)
    nCommunities = len(communities)
    nCols = int(nCommunities ** 0.5 + 1)
    nRows = int(nCommunities / nCols + 1)
    # print(nCols, nRows)
    fig, ax = plt.subplots(ncols=nCols, nrows=nRows, figsize=(10, 10))
    ax=ax.ravel()
    cols = plt.cm.RdYlBu(np.linspace(0, 1, len(communities)))

    for iCom in range(0, len(communities)):
        community_graph = G.subgraph(communities[iCom])
        pos = nx.spring_layout(community_graph)
        # pos = nx.kamada_kawai_layout(community_graph)
        # pos = nx.circular_layout(G) # MAYVE
        # pos = nx.random_layout(community_graph) # NO as name suggests it's random ;)!
        # pos = nx.spectral_layout(community_graph) # NO
        nx.draw_networkx_nodes(community_graph, pos, ax=ax[iCom], node_size=300, node_color=[cols[iCom]])
        nx.draw_networkx_labels(community_graph, pos, ax=ax[iCom], font_size=10)
        nx.draw_networkx_edges(community_graph, pos, ax=ax[iCom], alpha=0.3)
    for a in ax:
        a.axis('off')
    plt.show()
    return communities

def get_communities(G):
    communities = nx.algorithms.community.label_propagation_communities(G)
    communities = list(communities)
    communities.sort(key=len, reverse=True)
    return communities

def get_degrees(G):
    # print(G)
    # print(G.degree(weight='weight'))
    degrees = list((d for name, d in G.degree(weight='weight')))
    print(degrees)
    fig, ax = plt.subplots()
    ax.hist(degrees)
    plt.show()