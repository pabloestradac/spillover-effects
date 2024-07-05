"""
WLS Estimation of Spillover Bounds
"""

__author__ = """ Pablo Estrada pabloestradace@gmail.com """

import numpy as np
import pandas as pd
from scipy import sparse as spr
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, QuantileRegressor
from sklearn.utils.fixes import sp_version, parse_version


class BoundsML():
    """
    Bounds using WLS estimation of spillover effects under a pre-specified exposure mapping.

    Parameters
    ----------
    name_y        : str
                    Name of the outcome variable
    name_s        : str
                    Name of the selection variable
    name_z        : str or list
                    Name of the treatment exposure variable(s)
    name_pscore   : str
                    Name of the propensity score variable
    data          : DataFrame
                    Data containing the variables of interest
    kernel_weights: array
                    Kernel weights for the estimation
    name_x        : str or list
                    Name of covariate(s)
    interaction   : bool
                    Whether to include interaction terms between Z and X
    subsample     : boolean array
                    Subsample of observations to consider
    contrast      : str
                    Type of contrast to estimate (direct or spillover)

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
                name_s,
                name_z,
                name_pscore,
                name_x,
                data,
                kernel_weights=None,
                subsample=None,
                interaction='treatment',
                contrast='spillover',
                n_splits=5, n_cvs=5,
                lambdas_proba=np.geomspace(1e-4, 1e4, 10), 
                lambdas_quant=np.geomspace(1e-4, 1e4, 10), 
                semi_cf=True, method='parametric', verbose=False, seed=None):

        # Outcome and treatment exposure
        selection = data[name_s].astype('bool').values
        Y = data.loc[selection, name_y].values
        Z = data[name_z].values
        pscore = data[name_pscore].values
        # Create interated matrix X
        name_x = [name_x] if isinstance(name_x, str) else name_x
        X = data[name_x].values
        if interaction=='treatment':
            X = X - X.mean(axis=0)
            X = np.hstack([Z[:, 1:2], X, Z[:, 1:2] * X])
            name_all = [name_z[1]] + name_x + ['treatment*'+cols for cols in name_x]
        elif interaction=='all':
            k = X.shape[1]
            XX = np.hstack([X[:, i:i+1] * X[:, j:j+1] for i in range(k) for j in range(k) if i != j])
            X = X - X.mean(axis=0)
            XX = XX - XX.mean(axis=0)
            X = np.hstack([Z[:, 1:2], X, Z[:, 1:2] * X, Z[:, 1:2] * XX])
            name_xx = ['{}*{}'.format(name_x[i], name_x[j]) for i in range(k) for j in range(k) if i != j]
            name_all = [name_z[1]] + name_x + ['treatment*'+cols for cols in name_x] + ['treatment*'+cols for cols in name_xx]
        else:
            print('Warning: Interaction term not included')
            X = np.hstack([Z[:, 1:2], X])
            name_all = [name_z[1]] + name_x
        X = pd.DataFrame(X, columns=name_all)
        # Kernel matrix
        n = Z.shape[0]
        weights = np.identity(n) if kernel_weights is None else kernel_weights
        # Filter by subsample of interest
        if subsample is not None:
            y = y[subsample[selection]]
            Z = Z[subsample]
            pscore = pscore[subsample]
            selection = selection[subsample]
            weights = weights[subsample,:][:,subsample]
            X = X[subsample]
        # Check for propensity score outside (0.01, 0.99)
        valid = (np.sum(Z*pscore, axis=1) > 0.01) & (np.sum(Z*pscore, axis=1) < 0.99)
        drop_obs = np.sum(~valid)
        if drop_obs > 0:
            print('Warning: {} observations have propensity scores outside (0.01, 0.99)'.format(drop_obs))
            y = y[valid[selection]]
            Z = Z[valid]
            pscore = pscore[valid]
            selection = selection[valid]
            weights = weights[valid,:][:,valid]
            X = X[valid]
        # Calculate trimming probability
        s1, s0, vars_proba = first_stage_proba(data[name_s], X, name_z[1],
                                               n_splits, n_cvs, 1/lambdas_proba,
                                               semi_cf, method, verbose, seed)
        p0 = s0 / s1
        # Round p0 to be at most 0.99
        p0 = np.where((p0>0.99) & (p0<=1), 0.99, p0)
        p0 = np.where((p0>1) & (p0<(1/0.99)), 1/0.99, p0)
        # Round p0 to be at least 0.01
        p0 = np.where(p0<0.01, 0.01, p0)
        p0 = np.where(p0>100, 100, p0)
        # Group individuals
        ind_help = p0 <= 1
        ind_hurt = p0 > 1
        # Always-observed individuals
        p0s = p0[selection]
        ind_helps = ind_help[selection]
        ind_hurts = ind_hurt[selection]
        # If one group of individuals < 5%, then strict monotonicity
        thresh_max_ind = 0.95
        if ind_help.mean() >= thresh_max_ind:
            p0 = np.where(p0>0.999, 0.999, p0)
        if ind_hurt.mean() >= thresh_max_ind:
            p0 = np.where(p0<1.001, 1.001, p0)
        # Calculate quantile at the trimming probabilities
        ns = np.sum(selection)
        Zs = Z[selection]
        Xs = X.iloc[selection]
        pscore_s = pscore[selection]
        s0s = s0[selection]
        s1s = s1[selection]
        p0s_all = pd.Series(np.hstack([p0s[p0s<1], 1/p0s[p0s>1], 1-p0s[p0s<1], 1-1/p0s[p0s>1]])) * 100
        q_grid = np.sort((p0s_all.apply(np.floor) / 100).unique())
        q1, q0, vars_quant = first_stage_quant(data.loc[selection, name_y], Xs, name_z[1],
                                               n_splits, n_cvs, lambdas_quant,
                                               q_grid, semi_cf, method, verbose, seed)
        # Estimate outcome at the trimming probabilities
        y_p0, y_1p0 = np.full(ns, np.nan), np.full(ns, np.nan)
        # Impute quantile levels for p0
        if ind_helps.sum() > 0:
            # Individuals that are helped: p0 <= 1
            for q in range(q_grid.size):
                y_p0[ind_helps] = np.where(np.abs(p0s[ind_helps] - q_grid[q]) <= 0.01, q1[ind_helps, q], y_p0[ind_helps])
        if ind_hurt.sum() > 0:
            # Individuals that are hurted: p0 > 1
            for q in range(q_grid.size):
                y_p0[ind_hurts] = np.where(np.abs((1/p0s)[ind_hurts] - q_grid[q]) <= 0.01, q0[ind_hurts, q], y_p0[ind_hurts])
        # Impute quantile levels for 1-p0
        if ind_help.sum() > 0:
            # Individuals that are helped: p0 <= 1
            for q in range(q_grid.size):
                y_1p0[ind_helps] = np.where(np.abs(1-p0s[ind_helps] - q_grid[q]) <= 0.01, q1[ind_helps, q], y_1p0[ind_helps])
        if ind_hurt.sum() > 0:
            # Individuals that are hurted: p0 > 1
            for q in range(q_grid.size):
                y_1p0[ind_hurts] = np.where(np.abs(1-(1/p0s)[ind_hurts] - q_grid[q]) <= 0.01, q0[ind_hurts, q], y_1p0[ind_hurts])
        # Transform outcome to ensure orthogonality
        Y_u1 = (p0s <= 1) * (Y*(Y>=y_1p0) + y_1p0 * (((Y<=y_1p0) - (1-p0s)) - p0s*(1-s1s))) + (p0s > 1) * (Y + y_p0 * (1-s1s))
        Y_u0 = (p0s <= 1) * (Y - y_1p0 * (1-s0s)) + (p0s > 1) * (Y*(Y<=y_p0) + y_p0 * ((1/p0s)*(1-s0s) - ((Y<=y_p0) - (1/p0s))))
        Y_l1 = (p0s <= 1) * (Y*(Y<=y_p0) + y_p0 * (((Y<=y_p0) - p0s) - p0s*(1-s1s))) + (p0s > 1) * (Y + y_1p0 * (1-s1s))
        Y_l0 = (p0s <= 1) * (Y - y_p0 * (1-s0s)) + (p0s > 1) * (Y*(Y>=y_1p0) + y_1p0 * ((1/p0s)*(1-s0s) - ((Y<=y_1p0) - (1-1/p0s))))
        Y_ub = Zs[:, 1] * Y_u1 + Zs[:, 0] * Y_u0
        Y_lb = Zs[:, 1] * Y_l1 + Zs[:, 0] * Y_l0
        # Estimate bounds
        G = np.array([-1, 1])
        W = np.diag(1 / np.sum(Zs*pscore_s, axis=1))
        ZWZi = sinv(Zs.T @ W @ Zs)
        Yhat_lb = ZWZi @ Zs.T @ W @ (Y_lb)
        Yhat_ub = ZWZi @ Zs.T @ W @ (Y_ub)
        # Share of always-observed individuals
        AO_share = np.zeros(ns)
        if ind_helps.sum() > 0:
            AO_share[ind_helps] = (Zs[:, 0]*(1-s0s)/pscore_s[:, 0] + s0s)[ind_helps]
        if ind_hurts.sum() > 0:
            AO_share[ind_hurts] = (Zs[:, 1]*(1-s1s)/pscore_s[:, 1] + s1s)[ind_hurts]
        # AO_share = np.mean(np.vstack([s0s, s1s]).min(axis=0))
        coef = np.array([G @ Yhat_lb / np.mean(AO_share), G @ Yhat_ub / np.mean(AO_share)])
        coef = np.sort(coef)
        # Variance
        Q = np.array([[1, 0, -coef[0]/np.mean(AO_share)], [0, 1, -coef[1]/np.mean(AO_share)]])
        neyman = np.array([Y_l1 / (Zs[:, 1]/pscore_s[:, 1]).sum() - Y_l0 / (Zs[:, 0]/pscore_s[:, 0]).sum(),
                           Y_u1 / (Zs[:, 1]/pscore_s[:, 1]).sum() - Y_u0 / (Zs[:, 0]/pscore_s[:, 0]).sum(),
                           AO_share]).T
        V = Q @ np.cov(neyman, rowvar=False) @ Q.T
        se = np.sqrt(np.diag(V))
        ci_low = coef[0] - 1.645*se[0]
        ci_up = coef[1] + 1.645*se[1]
        df_results = pd.DataFrame({'lower-bound': coef[0], 'upper-bound': coef[1], 
                                    'ci-low': ci_low, 'ci-up': ci_up},
                                    index=[contrast])

        self.summary = df_results
        self.params_lb = Yhat_lb
        self.params_ub = Yhat_ub
        self.vcov = V
        self.vars_proba = vars_proba
        self.vars_quant = vars_quant
        self.always_observed = AO_share
        self.p0 = p0
        self.q_grid = q_grid


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


def first_stage_proba(Y, X, treatment, n_splits, n_cvs, lambdas, semi_cf=True, method='parametric', verbose=False, seed=42):
    """
    Calculate post-lasso selection probabilities

    Parameters
    ----------
    Y         : pandas Series
                Selection variable.
    X         : pandas DataFrame
                Covariates.
    treatment : str
                Treatment variable.
    """

    # Pipeline
    preprocessor = ColumnTransformer(transformers=[('treatment', 'passthrough', [treatment])],
                                     remainder=StandardScaler())
    logit = Pipeline(steps=[('prep', preprocessor), 
                            ('clf', LogisticRegression())])
    if method == 'lasso':
        logit_cv = Pipeline(steps=[('prep', preprocessor),
                                   ('clf', LogisticRegressionCV(penalty='l1', solver='liblinear', cv=n_cvs, Cs=lambdas))])
        if semi_cf:
            logit_cv.fit(X, Y)
            lambda_best = logit_cv['clf'].C_[0]
    model_vars = []
    s0, s1 = np.zeros(X.shape[0]), np.zeros(X.shape[0])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train, test in skf.split(X, Y):
        if method == 'parametric':
            selected_vars = X.columns.tolist()
            model_vars.append(selected_vars)
            logit.set_params(clf__penalty=None, clf__solver='newton-cg').fit(X.iloc[train][selected_vars], Y.iloc[train])
            if verbose:
                accuracy = logit.score(X.iloc[test][selected_vars], Y.iloc[test])
                print("Accuracy={:.2f}".format(accuracy))
        elif method == 'lasso':
            if semi_cf:
                logit.set_params(clf__penalty='l1', clf__solver='liblinear', clf__C=lambda_best).fit(X.iloc[train], Y.iloc[train])
            else:
                logit_cv.fit(X.iloc[train], Y.iloc[train])
                lambda_best = logit_cv['clf'].C_[0]
                logit.set_params(clf__penalty='l1', clf__solver='liblinear', clf__C=lambda_best).fit(X.iloc[train], Y.iloc[train])
            logit_coef = logit['clf'].coef_.ravel()
            selected_vars = logit.feature_names_in_[np.where(logit_coef != 0)[0]].tolist()
            model_vars.append(selected_vars)
            selected_vars = [treatment] + selected_vars if treatment not in selected_vars else selected_vars
            logit.set_params(clf__penalty=None, clf__solver='newton-cg').fit(X.iloc[train][selected_vars], Y.iloc[train])
            if verbose:
                accuracy = logit.score(X.iloc[test][selected_vars], Y.iloc[test])
                pct_nonzero = np.mean(logit_coef != 0)*100
                print("{} ({:.0f}%) selected variables; lambda={:.2f}; accuracy={:.2f}".format(len(selected_vars), 
                                                                                               pct_nonzero, 1/lambda_best, accuracy))
        else:
            raise ValueError("Method not recognized")
        # Create treatment and interaction variables
        X1, X0 = X.iloc[test][selected_vars].copy(), X.iloc[test][selected_vars].copy()
        X1[treatment] = 1
        X0[treatment] = 0
        # Predict selection
        s1[test] = logit.predict_proba(X1)[:, 1]
        s0[test] = logit.predict_proba(X0)[:, 1]

    return s1, s0, model_vars


def first_stage_quant(Y, X, treatment, n_splits, n_cvs, lambdas, q_grid, semi_cf=True, method='parametric', verbose=False, seed=42):
    """
    Calculate post-lasso quantiles

    Parameters
    ----------
    Y         : pandas Series
                Outcome variable.
    X         : pandas DataFrame
                Covariates.
    treatment : str
                Treatment variable.
    """

    # Pipeline
    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    lambdas = {'alpha': lambdas}
    preprocessor = ColumnTransformer(transformers=[('treatment', 'passthrough', [treatment])],
                                     remainder=StandardScaler())
    qr = Pipeline(steps=[('prep', preprocessor), 
                         ('reg', QuantileRegressor(solver=solver))])
    if method == 'lasso':
        qr_cv = Pipeline(steps=[('prep', preprocessor),
                                ('reg', GridSearchCV(QuantileRegressor(solver=solver), lambdas, cv=n_cvs))])
        if semi_cf:
            qr_cv.fit(X, Y)
            lambda_best = qr_cv['reg'].best_params_['alpha']
    model_vars = []
    q0, q1 = np.zeros((Y.shape[0], q_grid.size)), np.zeros((Y.shape[0], q_grid.size))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train, test in kf.split(X, Y):
        if not semi_cf:
            qr_cv.fit(X.iloc[train], Y.iloc[train])
            lambda_best = qr_cv['reg'].best_params_['alpha']

        for q in range(q_grid.size):
            if method == 'parametric':
                selected_vars = X.columns.tolist()
                model_vars.append(selected_vars)
                qr.set_params(reg__quantile=q_grid[q], reg__alpha=0).fit(X.iloc[train][selected_vars], Y.iloc[train])
                if verbose:
                    r2 = qr.score(X.iloc[test][selected_vars], Y.iloc[test])
                    if q_grid[q]*100 in [1, 10, 25, 50, 75, 90, 99]:
                        print("Q {:.2f}) R2 = {:.2f}".format(q_grid[q], r2))
            elif method == 'lasso':
                qr.set_params(reg__quantile=q_grid[q], reg__alpha=lambda_best).fit(X.iloc[train], Y.iloc[train])
                qr_coef = qr['reg'].coef_.ravel()
                selected_vars = qr.feature_names_in_[np.where(qr_coef != 0)[0]].tolist()
                model_vars.append(selected_vars)
                selected_vars = [treatment] + selected_vars if treatment not in selected_vars else selected_vars
                qr.set_params(reg__quantile=q_grid[q], reg__alpha=0).fit(X.iloc[train][selected_vars], Y.iloc[train])
                if verbose:
                    r2 = qr.score(X.iloc[test][selected_vars], Y.iloc[test])
                    pct_nonzero = np.mean(qr_coef != 0)*100
                    if q_grid[q]*100 in [1, 10, 25, 50, 75, 90, 99]:
                        print("Q {:.2f}) {} ({:.1f}%) selected variables; lambda = {}; R2 = {:.2f}".format(q_grid[q], len(selected_vars), 
                                                                                                           pct_nonzero, lambda_best, r2))
            else:
                raise ValueError("Method not recognized")
            # Create treatment variable
            X1, X0 = X.iloc[test][selected_vars].copy(), X.iloc[test][selected_vars].copy()
            X1[treatment] = 1
            X0[treatment] = 0
            # Predict outcome at the given quantile
            q1[test, q] = qr.predict(X1)
            q0[test, q] = qr.predict(X0)

    return np.sort(q1), np.sort(q0), model_vars
