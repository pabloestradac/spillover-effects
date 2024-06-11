"""
WLS Estimation of Spillover Effects
"""

__author__ = """ Pablo Estrada pabloestradace@gmail.com """

import numpy as np
import pandas as pd
from scipy import sparse as spr
from scipy.stats import norm


class WLS():
    
        """
        WLS estimation of spillover effects under a pre-specified exposure mapping.
    
        Parameters
        ----------
        y             : array
                        n x 1 array of dependent variable
        Z             : array
                        n x t array of exposure mapping
        pscore        : array
                        n x 1 array of propensity scores
        A             : array
                        n x n adjacency matrix
        X             : array
                        n x k array of covariates
        interaction   : bool
                        Whether to include the interaction of Z and X
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
                    name_y,
                    name_z,
                    name_pscore,
                    data,
                    kernel_weights=None,
                    name_x=None,
                    interaction=True,
                    subsample=None,
                    contrast='spillover'):

            # Outcome and treatment exposure
            y = data[name_y].values
            Z = data[name_z].values
            pscore = data[name_pscore].values
            # Standardize or create matrix X
            t = Z.shape[1]
            name_x = [name_x] if isinstance(name_x, str) else name_x
            X = data[name_x].values if name_x is not None else None
            if X is not None:
                X = (X - X.mean(axis=0)) # / X.std(axis=0)
                if interaction:
                    ZX = np.hstack([Z[:, i:i+1] * X for i in range(t)])
                    X = np.hstack((Z, ZX))
                else:
                    X = np.hstack((Z, X))
            else:
                X = Z.copy() if X is None else X
            # Kernel matrix
            n = Z.shape[0]
            weights = np.identity(n) if kernel_weights is None else kernel_weights
            # Filter by subsample of interest
            if subsample is not None:
                y = y[subsample]
                Z = Z[subsample]
                pscore = pscore[subsample]
                selection = selection[subsample]
                weights = weights[subsample,:][:,subsample]
                X = X[subsample] if X is not None else None
            # Check for propensity score outside (0.01, 0.99)
            valid = (np.sum(Z*pscore, axis=1) > 0.01) & (np.sum(Z*pscore, axis=1) < 0.99)
            drop_obs = np.sum(~valid)
            if drop_obs > 0:
                print('Warning: {} observations have propensity scores outside (0.01, 0.99)'.format(drop_obs))
                y = y[valid]
                Z = Z[valid]
                pscore = pscore[valid]
                weights = weights[valid,:][:,valid]
                X = X[valid] if X is not None else None
            # Weight with propensity score
            W = np.diag(1 / np.sum(Z*pscore, axis=1))
            # Fit WLS
            XWXi = sinv(X.T @ W @ X)
            beta = XWXi @ X.T @ W @ y
            # Variance
            e = np.diag(y - X @ beta)
            V = XWXi @ X.T @ W @ e @ weights @ e @ W @ X @ XWXi
            # Summary of results
            if t == 4:
                G = 1/2 * np.array([-1, -1, 1, 1]) if contrast is 'direct' else 1/2 * np.array([-1, 1, -1, 1])
            elif t == 2:
                G = np.array([-1, 1])
            else:
                raise ValueError('Contrast not available for T lenght = {}'.format(t))
            coef = np.insert(beta, 0, G @ beta[:t])
            se = np.insert(np.sqrt(V.diagonal()), 0, np.sqrt(G @ V[:t, :t] @ G.T))
            tval = coef / se
            pval = 2 * (1 - norm.cdf(np.abs(tval)))
            ci_low = coef - 1.96*se
            ci_up = coef + 1.96*se
            if name_x is None:
                name_vars = [contrast] + name_z
            else:
                if interaction:
                    name_vars = [contrast] + name_z + [zi + '*' + xi for zi in name_z for xi in name_x]
                else: 
                    name_vars = [contrast] + name_z + name_x
            df_results = pd.DataFrame({'coef': coef, 'se': se, 't-val': tval, 'p-val': pval,
                                       'ci-low': ci_low, 'ci-up': ci_up},
                                      index=name_vars)

            self.params = beta
            self.vcov = V
            self.summary = df_results



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