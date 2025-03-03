import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import WLS, RegressionResults
from statsmodels.stats.weightstats import _zconfint_generic
from statsmodels.stats.sandwich_covariance import cov_hc0, cov_hc1, cov_hac
import statsmodels
from tqdm import tqdm

'''
HELPER FUNCTIONS
'''

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

'''
MAIN PTD BOOTSTRAP FUNCTION
'''

def ptd_bootstrap(algorithm, X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, w, w_unlabeled, B=2000, alpha=0.05, tuning_method='optimal_diagonal'):
    """
    Computes tuning matrix, point estimates, and confidence intervals for regression coefficients using the Predict-then-Debias bootstrap algorithm from Kluger et al. (2025), 'Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling,' <https://arxiv.org/abs/2501.18577>.
    
    Args:
        algorithm: python function that takes in (x, y) data and weights, and returns regression coefficients (e.g. linear regression or logistic regression function)
        X (ndarray): ground truth covariates in labeled data (dimensions n x p)
        Xhat (ndarray): predicted covariates in labeled data (dimensions n x p)
        Xhat_unlabeled (ndarray): predicted covariates in unlabeled data (dimensions N x p)
        Y (ndarray): ground truth response variable in labeled data (length n)
        Yhat (ndarray): predicted response variable in labeled data (length n)
        Yhat_unlabeled (ndarray): predicted response variable in unlabeled data (length N)
        w (ndarray): sample weights for the labeled dataset (length n)
        w_unlabeled (ndarray): sample weights for the unlabeled dataset (length N)
        B (int, optional): number of bootstrap steps
        alpha (float, optional): error level (must be in the range (0, 1)). The PTD confidence interval will target a coverage of 1 - alpha. 
        tuning_method (str, optional): method used to create the tuning matrix (None, "optimal_diagonal", or "optimal")
        
    Returns:
        ndarray: the tuning matrix (dimensions p x p) computed from the selected tuning method
        ndarray: PTD point estimate of the coefficients (length p)
        tuple: lower and upper bounds of PTD confidence intervals for the coefficients
    """
    p = X.shape[1]
    
    coeff_calibration_list = []
    coeff_preds_calibration_list = []
    coeff_preds_unlabeled_list = []
    
    # get bootstrap coefficient estimates
    for i in range(B):
        X_trial, Xhat_trial, Xhat_unlabeled_trial, Y_trial, Yhat_trial, Yhat_unlabeled_trial = resample_datapoints(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled)

        coeff_calibration = algorithm(X_trial, Y_trial, w)
        coeff_calibration_list.append(coeff_calibration)

        coeff_preds_calibration = algorithm(Xhat_trial, Yhat_trial, w)
        coeff_preds_calibration_list.append(coeff_preds_calibration)
        
        coeff_preds_unlabeled = algorithm(Xhat_unlabeled_trial, Yhat_unlabeled_trial, w_unlabeled)
        coeff_preds_unlabeled_list.append(coeff_preds_unlabeled)

    coeff_calibration_list = np.array(coeff_calibration_list)
    coeff_preds_calibration_list = np.array(coeff_preds_calibration_list)
    coeff_preds_unlabeled_list = np.array(coeff_preds_unlabeled_list)
    
    # compute tuning matrix
    if tuning_method is None:
        tuning_matrix = np.identity(p)
    else:
        cross_cov_calibration = cross_cov(coeff_calibration_list.T, coeff_preds_calibration_list.T)
        cov_preds_calibration = np.cov(coeff_preds_calibration_list.T)
        cov_preds_unlabeled = np.cov(coeff_preds_unlabeled_list.T)
        if tuning_method == "optimal":
            tuning_matrix = cross_cov_calibration @ np.linalg.inv(cov_preds_calibration + cov_preds_unlabeled)
        elif tuning_method == "optimal_diagonal":
            tuning_matrix = np.diag(np.diag(cross_cov_calibration)/(np.diag(cov_preds_calibration) + np.diag(cov_preds_unlabeled)))
    
    # compute confidence interval for regression coefficient
    pointestimates = []
    for i in range(B):
        coeff_calibration = coeff_calibration_list[i]
        coeff_preds_calibration = coeff_preds_calibration_list[i]
        coeff_preds_unlabeled = coeff_preds_unlabeled_list[i]
        pointestimate = tuning_matrix @ coeff_preds_unlabeled + (coeff_calibration - tuning_matrix @ coeff_preds_calibration)
        pointestimates.append(pointestimate)
        
    lo = np.percentile(pointestimates, 100*alpha/2, axis=0)
    hi = np.percentile(pointestimates, 100*(1-alpha/2), axis=0)
    
    ptd_pointestimate = (lo+hi)/2
    ptd_ci = (lo, hi)
    return tuning_matrix, ptd_pointestimate, ptd_ci

'''
LINEAR REGRESSION
'''

def algorithm_linear_regression(X, Y, w):
    if w is None:
        w=1
    regression = WLS(endog=Y, exog=X, weights=w).fit()
    coeff = regression.params
    return coeff

def ptd_linear_regression(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, w=None, w_unlabeled=None, B=2000, alpha=0.05, tuning_method='optimal_diagonal'):
    """
    Tuning matrix, point estimate, and confidence interval for linear regression coefficients using the Predict-then-Debias bootstrap algorithm. 
    """
    return ptd_bootstrap(algorithm_linear_regression, X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, w, w_unlabeled, B=B, alpha=alpha, tuning_method=tuning_method)

def classical_linear_regression_ci(X, Y, w=None, alpha=0.05):
    """
    Confidence interval for linear regression coefficients using the classical method.

    Args:
        X (ndarray): labeled covariates
        Y (ndarray): labeled responses
        w (ndarray, optional): sample weights for the labeled data set
        alpha (float, optional): error level (must be in the range (0, 1)). Confidence interval will target a coverage of 1 - alpha.

    Returns:
        tuple: lower and upper bounds of classical confidence intervals for the coefficients
    """
    if w is None:
        w = 1
    regression = WLS(endog=Y, exog=X, weights=w).fit()
    theta = regression.params
    se = regression.HC0_se
    ci = _zconfint_generic(theta, se, alpha, alternative="two-sided")
    return (ci[0], ci[1])