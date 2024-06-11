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
    # Get unique nodes
    nodes = edges.stack().unique() if nodes is None else nodes
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
        return np.vstack([1-spillover, spillover]).T


def spillover_pscore(A, n_treated, blocks=None, matrix=False):
    """
    Compute the propensity score of having at least one friend treated

    Parameters
    ----------
    A         : array
                n x n adjacency matrix
    n_treated : int
                Number of treated individuals in the block
    blocks    : array
                n x 1 array of block assignments, k unique blocks
    matrix    : bool
                Whether to return the matrix of propensity scores
    """
    n = A.shape[0]
    if blocks is None:
        # Protocol: all students are in the same block
        degree = A @ np.ones(n)
        pscore_spillover = 1 - hypergeom(n, n_treated, degree).pmf(0)
        pscore_direct = n_treated / n
        # pscore0_spillover = binom(degree, pscore0_direct).pmf(0)
    else:
        # Protocol: propensity score by blocks, e.g., classrooms
        unique_blocks = blocks.unique()
        # Each row is a vector giving the number of friends of each student that are in classroom k
        degree_by_block = np.vstack([A @ (blocks==k).values for k in unique_blocks])
        # 174 blocks (classrooms) of x students, 4 treated
        blocks_size = blocks.value_counts().loc[unique_blocks].values
        k = len(unique_blocks)
        p0_block = np.zeros((k, n))
        # Probability of having zero treated friends out of the x students in the k classroom
        for i in range(k):
            p0_block[i, :] = hypergeom(blocks_size[i], n_treated, degree_by_block[i, :]).pmf(0)
        pscore_spillover = 1 - p0_block.prod(axis=0)
        pscore_direct = [n_treated / blocks.value_counts().loc[i] for i in blocks]
    if matrix:
        return np.vstack([(1-pscore_direct) * (1-pscore_spillover), 
                          (1-pscore_direct) * pscore_spillover, 
                          pscore_direct     * (1-pscore_spillover), 
                          pscore_direct     * pscore_spillover]).T
    else:
        return np.vstack([1-pscore_spillover, pscore_spillover]).T


def kernel(A, bw):
    """
    Kernel matrix for covariance estimation

    Parameters
    ----------
    A         : array
                n x n adjacency matrix
    bw        : int
                Bandwidth to calculate kernel matrix
    """
    if spr.issparse(A):
        A = A.toarray()
    # Calculate shortest path distance matrix
    dist_matrix = spr.csgraph.dijkstra(csgraph=A, directed=False, unweighted=True)
    # Calculate kernel matrix
    weights = (dist_matrix <= bw) * 1
    # Check for negative eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(weights)
    if eigenvalues[0] < 0:
        weights = eigenvectors @ np.diag(np.maximum(eigenvalues, 0)) @ eigenvectors.T
    return weights