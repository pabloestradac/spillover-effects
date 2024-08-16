"""
Useful functions for spillover effects estimation
"""

__author__ = """ Pablo Estrada pabloestradace@gmail.com """

import numpy as np
import pandas as pd
from scipy import sparse as spr
from scipy.stats import hypergeom


def adjacency_matrix(edges, directed=True, nodes=None):
    """
    Adjacency matrix and nodelist from edge list dataframe

    Parameters
    ----------
    edges     : pandas.core.frame.DataFrame
                Dataframe with j+1 columns
                Column 1 is the source node
                Column 2,3,j+1 are the target nodes
    directed  : bool
                Whether the graph is directed or not
    nodes     : array
                n x 1 array of the nodes order
    """
    # Transform edges to two columns of source and target nodes
    data = edges.iloc[:, 0:2].dropna()
    data.columns = [0, 1]
    for j in range(2, edges.shape[1]):
        data_j = edges.iloc[:, [0, j]].dropna()
        data_j.columns = [0, 1]
        data = pd.concat([data, data_j], ignore_index=True)
    # Remove self-loops
    self_loops = data[0] == data[1]
    if self_loops.sum() > 0:
        data = data[~self_loops]
        print('Removed {} self-loops'.format(self_loops.sum()))
    # Check for repeated (i,j) and (j,i) edges when undirected
    if not directed:
        data = pd.DataFrame({tuple(sorted(i)): i for i in data.values}.values())
    # Get unique nodes and drop edges not in nodes
    if nodes is None:
        nodes = edges.stack().unique()
    else:
        data = data[data[0].isin(nodes) & data[1].isin(nodes)]
    n = len(nodes)
    # Create mapping of nodes to indices
    nodes_map = {nodes[i]: str(i) for i in range(n)}
    data = data.replace(nodes_map)
    rows = data[0].astype(int)
    cols = data[1].astype(int)
    ones = np.ones(len(rows), np.uint32)
    A = spr.coo_matrix((ones, (rows, cols)), shape=(n, n))
    if not directed:
        A = A + A.T
    return A, nodes


def spillover_treatment(treatment, A, interaction=False):
    """
    Treatment matrix for spillover effects estimation

    Parameters
    ----------
    treatment   : array
                  n x 1 array of treatment assignment
    A           : array
                  n x n adjacency matrix
    interaction : bool
                  Whether to include the interaction of direct and spillover treatments
    """
    if interaction:
        spillover = ((A @ treatment) > 0) * 1
        return np.vstack([(1-treatment) * (1-spillover), 
                          (1-treatment) * spillover, 
                          treatment     * (1-spillover), 
                          treatment     * spillover]).T
    else:
        spillover = ((A @ treatment) > 0) * 1
        return spillover


def spillover_pscore(A, treated, blocks=None, exposure='spillover'):
    """
    Compute the propensity score of having at least one friend treated

    Parameters
    ----------
    A         : array
                n x n adjacency matrix
    treated   : float
                Number or fraction of treated individuals in the block
    blocks    : pandas Series
                n x 1 array of block assignment
    exposure  : str
                Type of exposure mapping (spillover, direct, interaction)
    """
    n = A.shape[0]
    if spr.issparse(A):
        A = A.toarray().astype(int)
    if blocks is None:
        # Protocol: all units (students) are in the same block (school)
        degree = A @ np.ones(n)
        n_treated = round(treated * n) if treated < 1 else treated
        pscore_spillover = 1 - hypergeom(n, n_treated, degree).pmf(0)
        # pscore_spillover = 1 - binom(degree, treated).pmf(0)
        pscore_direct = n_treated / n
    else:
        # Protocol: propensity score by blocks, e.g., classrooms
        unique_blocks = blocks.unique()
        # Each row is a vector giving the number of friends of each unit (student) that are in block (classroom) k
        degree_by_block = np.vstack([A @ (blocks==k).values for k in unique_blocks])
        # K blocks (classrooms) of n_k units (students), n_treated in each block
        blocks_size = blocks.value_counts().loc[unique_blocks].values
        K = len(unique_blocks)
        p0_block = np.zeros((K, n))
        # Probability of having zero treated friends out of the n_k units in the k block
        for k in range(K):
            n_treated = round(treated * blocks_size[k]) if treated < 1 else treated
            p0_block[k, :] = hypergeom(blocks_size[k], n_treated, degree_by_block[k, :]).pmf(0)
        pscore_spillover = 1 - p0_block.prod(axis=0) # product across k classrooms
        pscore_direct = [n_treated / blocks.value_counts().loc[i] for i in blocks] if treated >= 1 else [treated] * n
    if exposure == 'spillover':
        return pscore_spillover
    elif exposure == 'direct':
        return pscore_direct
    elif exposure == 'interaction':
        return np.vstack([(1-pscore_direct) * (1-pscore_spillover),
                          (1-pscore_direct) * pscore_spillover,
                          pscore_direct     * (1-pscore_spillover),
                          pscore_direct     * pscore_spillover]).T
    else:
        raise ValueError('Exposure must be either "spillover", "direct", or "interaction"')


def kernel(A, bw=-1, K=1):
    """
    Kernel matrix for covariance estimation

    Parameters
    ----------
    A         : array
                n x n adjacency matrix
    bw        : int
                If negative, use optimal bandwidth
    K         : int
                K-neighborhood exposure
    """
    if spr.issparse(A):
        A_mat = A.toarray().astype(int)
    n = A.shape[0]
    # Calculate shortest path distance matrix
    dist_matrix = spr.csgraph.dijkstra(csgraph=A_mat, directed=False, unweighted=True)
    _, labels = spr.csgraph.connected_components(csgraph=A, directed=False, return_labels=True)
    unique, counts = np.unique(labels, return_counts=True)
    Gcc_label = unique[np.argmax(counts)]
    APL = dist_matrix[labels == Gcc_label, :][:, labels == Gcc_label].sum() / counts.max() / (counts.max() - 1)
    avg_deg = A.sum() / n
    if bw < 0:
        bw = round(APL/2) if APL < 2*np.log(n)/np.log(avg_deg) else round(APL**(1/3))
        bw = max(2*K, bw)
    # Calculate kernel matrix
    weights = (dist_matrix <= bw) * 1
    # Check for negative eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(weights)
    if eigenvalues[0] < 0:
        weights = eigenvectors @ np.diag(np.maximum(eigenvalues, 0)) @ eigenvectors.T
    return weights, bw


def load_data():
    """
    Load data for spillover effects estimation
    """
    path_data = 'https://raw.githubusercontent.com/pabloestradac/spillover-effects/main/data/'
    edges = pd.read_csv(path_data + 'edges.csv')
    data = pd.read_csv(path_data + 'data.csv')
    A, nodes = adjacency_matrix(edges, directed=True)
    data = data.set_index('node').loc[nodes].reset_index()
    data['pscore'] = spillover_pscore(A, data['D'].sum())
    data['exposure'] = spillover_treatment(data['D'], A)
    distances = kernel(A, 3)
    return data, distances