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
    name_y        : str
                    Name of the outcome variable
    name_z        : str or list
                    Name of the treatment exposure variable(s)
    name_pscore   : str or list
                    Name of the propensity score variable(s)
    dataframe     : DataFrame
                    Data containing the variables of interest
    kernel_weights: array
                    Kernel weights for the estimation
    name_x        : str or list
                    Name of covariate(s)
    interaction   : bool
                    Whether to include interaction terms between Z and X
    subsample     : array
                    Subsample of observations to consider
    contrast      : str
                    Type of contrast to estimate (direct or spillover)
    alpha         : float
                    Significance level
    warn          : bool
                    Whether to print warnings

    Attributes
    ----------
    params        : array
                    WLS coefficients
    vcov          : array
                    Variance covariance matrix
    summary       : DataFrame
                    Summary of WLS results
    """

    def __init__(self,
                name_y,
                name_z,
                name_pscore,
                dataframe,
                kernel_weights=None,
                name_x=None,
                interaction=True,
                subsample=None,
                contrast='spillover',
                alpha = 0.05, warn=True):

        # Kernel matrix
        data = dataframe.copy()
        n = data.shape[0]
        weights = np.identity(n) if kernel_weights is None else kernel_weights
        # Filter by subsample of interest and nonmissing values
        if subsample is not None:
            print('Warning: Filtering by subsample of {} observations'.format(subsample.sum())) if warn else None
            weights = weights[subsample,:][:,subsample]
            data = data[subsample].copy()
        name_x = [name_x] if isinstance(name_x, str) else name_x
        if isinstance(name_z, str):
            data[name_z+'0'] = 1 - data[name_z]
            data = data.rename(columns={name_z: name_z+'1'})
            name_z = [name_z+'0', name_z+'1']
        if name_x is not None:
            missing = data[[name_y] + name_z + name_x].isna().any(axis=1)
        else:
            missing = data[[name_y] + name_z].isna().any(axis=1)
        missy = data[name_y].isna().sum()
        if missing.sum() > 0: 
            print('Warning: {} observations have missing values ({} missing outcomes)'.format(missing.sum(), missy)) if warn else None
            weights = weights[~missing,:][:,~missing]
            data = data[~missing].copy()
        # Check for propensity score outside (0.01, 0.99)
        if isinstance(name_pscore, str):
            psvals = data[name_pscore].values
            data[name_pscore+'0'] = 1 - psvals
            data = data.rename(columns={name_pscore: name_pscore+'1'})
            name_pscore = [name_pscore+'0', name_pscore+'1']
        full_pscores = data[name_z].values * data[name_pscore].values
        valid = (np.sum(full_pscores, axis=1) > 0.01) & (np.sum(full_pscores, axis=1) < 0.99)
        if np.sum(~valid) > 0:
            print('Warning: {} observations have propensity scores outside (0.01, 0.99)'.format(np.sum(~valid))) if warn else None
            weights = weights[valid,:][:,valid]
            data = data[valid].copy()
        # Outcome and treatment exposure
        y = data[name_y].values
        Z = data[name_z].values
        pscore = data[name_pscore].values
        # Standardize or create matrix X
        t = Z.shape[1]
        X = data[name_x].values if name_x is not None else None
        if X is not None:
            X = (X - np.mean(X, axis=0))
            if interaction:
                ZX = np.hstack([Z[:, i:i+1] * X for i in range(t)])
                X = np.hstack((Z, ZX))
            else:
                X = np.hstack((Z, X))
        else:
            X = Z.copy()
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
            G = 1/2 * np.array([-1, -1, 1, 1]) if contrast == 'direct' else 1/2 * np.array([-1, 1, -1, 1])
        elif t == 2:
            G = np.array([-1, 1])
        else:
            raise ValueError('Contrast not available for T lenght = {}'.format(t))
        coef = np.insert(beta, 0, G @ beta[:t])
        se = np.insert(np.sqrt(V.diagonal()), 0, np.sqrt(G @ V[:t, :t] @ G.T))
        tval = coef / se
        pval = 2 * (1 - norm.cdf(np.abs(tval)))
        z_alpha = norm(0,1).ppf(1-alpha/2)
        ci_low = coef - z_alpha*se
        ci_up = coef + z_alpha*se
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
        self.X = X
        self.y = y
        self.Z = Z
        self.pscore = pscore
        self.W = W



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