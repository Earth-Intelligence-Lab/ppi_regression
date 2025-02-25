import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, RegressionResults
from statsmodels.stats.weightstats import _zconfint_generic
from statsmodels.stats.sandwich_covariance import cov_hc0, cov_hc1, cov_hac
import statsmodels
from tqdm import tqdm

def cross_cov(A, B):
    '''
    A is (p x n)
    B is (p x n)
    
    Output: (p x p) cross-covariance matrix
    '''
    n = A.shape[1]
    C = 1/(n-1) * (A-A.mean(axis=1)[:, np.newaxis]) @ (B-B.mean(axis=1)[:, np.newaxis]).T
    return C

def resample_datapoints(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled):
    n = len(X)
    N = len(Xhat_unlabeled)
    resampled_indices = np.random.choice(np.arange(0, n+N), size=n+N, replace=True)
    
    calibration_indices = resampled_indices[resampled_indices < n]
    X_trial = X[calibration_indices]
    Xhat_trial = Xhat[calibration_indices]
    Y_trial = Y[calibration_indices]
    Yhat_trial = Yhat[calibration_indices]
    
    preds_indices = resampled_indices[resampled_indices >= n] - n
    Xhat_unlabeled_trial = Xhat_unlabeled[preds_indices]
    Yhat_unlabeled_trial = Yhat_unlabeled[preds_indices]
    
    return X_trial, Xhat_trial, Xhat_unlabeled_trial, Y_trial, Yhat_trial, Yhat_unlabeled_trial

def ptd_bootstrap(algorithm, X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, B=2000, alpha=0.05, tuning_method='optimal_diagonal'):
    '''
    B: number of bootstrap steps
    tuning_method: method used to create the tuning matrix (None, 'optimal_diagonal', or 'optimal')
    '''
    p = X.shape[1]
    
    coeff_calibration_list = []
    coeff_preds_calibration_list = []
    coeff_preds_all_list = []
    
    # get bootstrap coefficient estimates
    for i in range(B):
        X_trial, Xhat_trial, Xhat_unlabeled_trial, Y_trial, Yhat_trial, Yhat_unlabeled_trial = resample_datapoints(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled)

        coeff_calibration = algorithm(X_trial, Y_trial)
        coeff_calibration_list.append(coeff_calibration)

        coeff_preds_calibration = algorithm(Xhat_trial, Yhat_trial)
        coeff_preds_calibration_list.append(coeff_preds_calibration)
        
        coeff_preds_all = algorithm(Xhat_unlabeled_trial, Yhat_unlabeled_trial)
        coeff_preds_all_list.append(coeff_preds_all)

    coeff_calibration_list = np.array(coeff_calibration_list)
    coeff_preds_calibration_list = np.array(coeff_preds_calibration_list)
    coeff_preds_all_list = np.array(coeff_preds_all_list)
    
    # compute tuning matrix
    #tuning_matrix = cross_cov(coeff_calibration_list.T, coeff_preds_calibration_list.T) @ np.linalg.inv(np.cov(coeff_preds_calibration_list.T))
    if tuning_method is None:
        tuning_matrix = np.identity(p)
    
    # compute confidence interval for regression coefficient
    outputs = []
    for i in range(B):
        coeff_calibration = coeff_calibration_list[i]
        coeff_preds_calibration = coeff_preds_calibration_list[i]
        coeff_preds_all = coeff_preds_all_list[i]
        output = tuning_matrix @ coeff_preds_all + (coeff_calibration - tuning_matrix @ coeff_preds_calibration)
        outputs.append(output)
        
    lo = np.percentile(outputs, 100*alpha/2, axis=0)
    hi = np.percentile(outputs, 100*(1-alpha/2), axis=0)
    
    pointestimate = (lo+hi)/2
    ci = (lo, hi)
    return tuning_matrix, pointestimate, ci

def algorithm_ols(X, Y):
    regression = OLS(Y, exog=X).fit()
    coeff = regression.params
    return coeff

def ptd_ols(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, B=2000, alpha=0.05, tuning_method='optimal_diagonal'):
    return ptd_bootstrap(algorithm_ols, X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, B=B, alpha=alpha, tuning_method=tuning_method)

def classical_ols_ci(X, Y, alpha=0.05, alternative="two-sided"):
    regression = OLS(Y, exog=X).fit()
    theta = regression.params
    se = regression.HC0_se
    ci = _zconfint_generic(theta, se, alpha, alternative)
    return (ci[0], ci[1])