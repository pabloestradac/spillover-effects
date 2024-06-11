"""
WLS Estimation of Spillover Bounds
"""

__author__ = """ Pablo Estrada pabloestradace@gmail.com """

import numpy as np
from scipy import sparse as spr
from .utils import kernel


class Bounds():
    
        """
        Bounds using WLS estimation of spillover effects under a pre-specified exposure mapping.
    
        Parameters
        ----------
        y             : array
                        ns x 1 array of dependent variable
        selection     : array
                        n x 1 array of selection variable
        Z             : array
                        n x t array of exposure mapping
        pscore        : array
                        n x t array of propensity scores
        group_exposed : array
                        t x 1 boolean array indicating the group exposed (1) or control (0)
        A             : array
                        n x n adjacency matrix
        bw            : int
                        Bandwidth for kernel matrix
    
        Attributes
        ----------
        params        : array
                        (t+k) x 1 array of WLS coefficients
        vcov          : array
                        Variance covariance matrix
        """
    
        def __init__(self,
                    y,
                    selection,
                    Z,
                    pscore,
                    group_exposed,
                    A=None,
                    subsample=None,
                    bw=2):
            
            # Kernel matrix
            n = Z.shape[0]
            weights = kernel(A, bw) if A is not None else np.identity(n)
            # Filter by subsample of interest
            if subsample is not None:
                y = y[subsample[selection]]
                Z = Z[subsample]
                pscore = pscore[subsample]
                selection = selection[subsample]
                weights = weights[subsample,:][:,subsample]
            # Check for propensity score outside (0.01, 0.99)
            valid = (np.sum(Z*pscore, axis=1) > 0.01) & (np.sum(Z*pscore, axis=1) < 0.99)
            drop_obs = np.sum(~valid)
            if drop_obs > 0:
                print('Warning: {} observations have propensity scores outside (0.01, 0.99)'.format(drop_obs))
                y = y[valid[selection]]
                Z = Z[valid]
                n = Z.shape[0]
                pscore = pscore[valid]
                weights = weights[valid,:][:,valid]
                selection = selection[valid]
            # Calculate trimming probability
            p_hat, group_trimmed, alpha_hat = trim_prob(selection, Z, pscore, group_exposed)
            n_list = ns_for_variance(selection, Z, pscore, group_exposed, group_trimmed)
            # Calculate trimming indicators
            Zs = Z[selection]
            pscore_s = pscore[selection]
            weights = weights[selection,:][:,selection]
            trim_bounds, quantile_bounds = trim_quantile(y, Zs, group_exposed, p_hat, group_trimmed)
            beta_bounds, V_bounds = [], []
            for i in range(2):
                trim = trim_bounds[i]
                quantile = quantile_bounds[i]
                y_trim = y[trim >= 0]
                Z_trim = Zs[trim >= 0]
                pscore_trim = pscore_s[trim >= 0]
                weights_trim = weights[trim >= 0, :][:, trim >= 0]
                # Weight with propensity score
                W = np.diag(1 / np.sum(Z_trim*pscore_trim, axis=1))
                # Fit WLS
                XWXi = sinv(Z_trim.T @ W @ Z_trim)
                beta = XWXi @ Z_trim.T @ W @ y_trim
                # Variance
                e = np.diag(y_trim - Z_trim @ beta)
                V = XWXi @ Z_trim.T @ W @ e @ weights_trim @ e @ W @ Z_trim @ XWXi
                v_y = (quantile - beta[group_exposed==group_trimmed])**2 * (1-p_hat)/p_hat / n_list[3-group_trimmed]
                v_q = ((quantile - beta[group_exposed==group_trimmed])**2) * ((p_hat-alpha_hat)/alpha_hat/n_list[0] + (1-alpha_hat)/alpha_hat/n_list[1])
                V_unadj = V[group_exposed==group_trimmed, group_exposed==group_trimmed]
                V[group_exposed==group_trimmed, group_exposed==group_trimmed] = V_unadj + v_y + v_q
                beta_bounds.append(beta)
                V_bounds.append(V)

            self.params_lb = beta_bounds[0]
            self.params_ub = beta_bounds[1]
            self.vcov_lb = V_bounds[0]
            self.vcov_ub = V_bounds[1]
            self.n = n
            self.p_hat = p_hat
            self.group_trimmed = group_trimmed
            self.alpha_hat = alpha_hat


def sinv(A):
    """
    Find inverse of matrix A using numpy.linalg.solve
    Helpful for large matrices
    """
    if spr.issparse(A):
        n = A.shape[0]
        Ai = spr.linalg.spsolve(A.tocsc(), spr.identity(n, format='csc'))
    else:
        try:
            n = A.shape[0]
            Ai = np.linalg.solve(A, np.identity(n))
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                Ai = np.linalg.pinv(A)
            else:
                raise
    return Ai


def trim_prob(S, Z, p, group):
    """
    Calculate the trimming probability q and the trimmed-group indicator g

    Parameters
    ----------
    S : array 
        n x 1 boolean array indicating the selected sample
    Z : array
        n x t exposure mapping matrix
    p : array
        n x t propensity score matrix
    group : array
        t x1 boolean array indicating the group exposed (1) or control (0)
    """
    Zs = Z[S]
    ps = p[S]
    W = 1 / np.sum(Z*p, axis=1)
    Ws = 1 / np.sum(Zs*ps, axis=1)
    qT = np.sum(Ws[Zs[:, group==1].sum(axis=1) == 1]) / np.sum(W[Z[:, group==1].sum(axis=1) == 1])
    qC = np.sum(Ws[Zs[:, group==0].sum(axis=1) == 1]) / np.sum(W[Z[:, group==0].sum(axis=1) == 1])
    # qT = np.sum(Zs[:, group==1]/ps[:, group==1]) / np.sum(Z[:, group==1]/p[:, group==1])
    # qC = np.sum(Zs[:, group==0]/ps[:, group==0]) / np.sum(Z[:, group==0]/p[:, group==0])
    qnum = np.min([qT, qC])
    qden = np.max([qT, qC])
    q = qnum / qden
    g = 0 if qT < qC else 1
    return q, g, qnum


def ns_for_variance(S, Z, p, group, g):
    """
    Calculate the trimming probability q and the trimmed-group indicator g.

    Parameters
    ----------
    S : array 
        n x 1 boolean array indicating the selected sample
    Z : array
        n x t exposure mapping matrix
    p : array
        n x t propensity score matrix
    group : array
        t x1 boolean array indicating the group exposed (1) or control (0)
    """
    Zs = Z[S]
    ps = p[S]
    W = 1 / np.sum(Z*p, axis=1)
    Ws = 1 / np.sum(Zs*ps, axis=1)
    n1 = W[Z[:, group==1].sum(axis=1) == 1].sum() / W.sum() * Z.shape[0]
    n0 = W[Z[:, group==0].sum(axis=1) == 1].sum() / W.sum() * Z.shape[0]
    ns1 = Ws[Zs[:, group==g].sum(axis=1) == 1].sum() / Ws.sum() * Zs.shape[0]
    ns0 = Ws[Zs[:, group!=g].sum(axis=1) == 1].sum() / Ws.sum() * Zs.shape[0]
    return [n1, n0, ns1, ns0]


def trim_quantile(Y, Z, group, p_hat, g):
    """
    Calculate the trimming indicators [lb, ub] for lower and upper bound. 
    Assign 1 when the observation stays, -1 if the observation is trimmed.

    Parameters
    ----------
    Y : array
        Outcome vector.
    Z : array
        (n x 2) treatment matrix.
    group : array
        Boolean array indicating the group exposed.
    p_hat : float
        Trimming probability.
    g : int
        Trimmed-group indicator.
    """
    # Sort the outcome Y in ascending order only for the group g
    trim_ind = Z[:, group==g].sum(axis=1) == 1
    sortidx = np.argsort(Y[trim_ind])
    # Calculate the index at the threshold for trimming
    q_lb = int(np.floor((p_hat * sortidx.size))) if g == 1 else int(np.floor(((1-p_hat) * sortidx.size)))
    q_ub = int(np.floor(((1-p_hat) * sortidx.size))) if g == 1 else int(np.floor((p_hat * sortidx.size)))
    # Trimming indicators: 1 stays, -1 drops
    lb = np.full(Y[trim_ind].size, -1)
    ub = np.full(Y[trim_ind].size, -1)
    if g == 1:
        lb[sortidx[:q_lb]] = 1
        ub[sortidx[q_ub+1:]] = 1
    else:
        lb[sortidx[q_lb+1:]] = 1
        ub[sortidx[:q_ub]] = 1
    trim_lb, trim_ub = trim_ind.copy()*1, trim_ind.copy()*1
    trim_lb[trim_ind], trim_ub[trim_ind] = lb, ub
    y_q = np.quantile(Y[trim_ind], p_hat)
    y_1q = np.quantile(Y[trim_ind], 1-p_hat)
    return [trim_lb, trim_ub], [y_q, y_1q]
