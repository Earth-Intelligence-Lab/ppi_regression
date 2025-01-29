import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, RegressionResults
from statsmodels.stats.weightstats import _zconfint_generic
from statsmodels.stats.sandwich_covariance import cov_hc0, cov_hc1, cov_hac
import statsmodels
from tqdm import tqdm

def ppi_pointestimate(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, tuning_matrix):
    coeff_calibration = OLS(Y, exog=statsmodels.tools.add_constant(X)).fit().params
    coeff_preds_calibration = OLS(Yhat, exog=statsmodels.tools.add_constant(Xhat)).fit().params
    coeff_preds_all = OLS(Yhat_unlabeled, exog=statsmodels.tools.add_constant(Xhat_unlabeled)).fit().params
    
    coeff_ppi = tuning_matrix @ coeff_preds_all + (coeff_calibration - tuning_matrix @ coeff_preds_calibration)
    
    return coeff_ppi

def ppi_calib_correction(X, Xhat, Y, Yhat, tuning_matrix):
    coeff_calibration = OLS(Y, exog=statsmodels.tools.add_constant(X)).fit().params
    coeff_preds_calibration = OLS(Yhat, exog=statsmodels.tools.add_constant(Xhat)).fit().params
    
    coeff_calib_correction = coeff_calibration - tuning_matrix @ coeff_preds_calibration
    return coeff_calib_correction

def cross_cov(A, B):
    '''
    A is (p x n)
    B is (p x n)
    
    Output: (p x p) cross-covariance matrix
    '''
    n = A.shape[1]
    C = 1/(n-1) * (A-A.mean(axis=1)[:, np.newaxis]) @ (B-B.mean(axis=1)[:, np.newaxis]).T
    return C

def resample_datapoints(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, calib_only=False):
    calibration_indices = np.random.choice(np.arange(0, len(X)), size=len(X), replace=True)

    X_trial = X[calibration_indices]
    Xhat_trial = Xhat[calibration_indices]
    Y_trial = Y[calibration_indices]
    Yhat_trial = Yhat[calibration_indices]
    
    if calib_only:
        return X_trial, Xhat_trial, Y_trial, Yhat_trial
    
    preds_indices = np.random.choice(np.arange(0, len(Xhat_unlabeled)), size=len(Xhat_unlabeled), replace=True)
    Xhat_unlabeled_trial = Xhat_unlabeled[preds_indices]
    Yhat_unlabeled_trial = Yhat_unlabeled[preds_indices]
    
    return X_trial, Xhat_trial, Xhat_unlabeled_trial, Y_trial, Yhat_trial, Yhat_unlabeled_trial

def bootstrap_ppi(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, tune=True, quick_convolve=False, trials=100):
    p = X.shape[1]
    if tune:
        # bootstrap to get tuning matrix
        coeff_calibration_list = []
        coeff_preds_calibration_list = []
        for trial in range(trials):
            X_trial, Xhat_trial, Xhat_unlabeled_trial, Y_trial, Yhat_trial, Yhat_unlabeled_trial = resample_datapoints(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled)

            coeff_calibration = OLS(Y_trial, exog=statsmodels.tools.add_constant(X_trial)).fit().params
            coeff_calibration_list.append(coeff_calibration)

            coeff_preds_calibration = OLS(Yhat_trial, exog=statsmodels.tools.add_constant(Xhat_trial)).fit().params
            coeff_preds_calibration_list.append(coeff_preds_calibration)

        coeff_calibration_list = np.array(coeff_calibration_list).T
        coeff_preds_calibration_list = np.array(coeff_preds_calibration_list).T

        tuning_matrix = cross_cov(coeff_calibration_list, coeff_preds_calibration_list) @ np.linalg.inv(np.cov(coeff_preds_calibration_list))
    else:
        tuning_matrix = np.identity(X.shape[1]+1)
    
    # bootstrap to get confidence interval for regression coefficient
    outputs = []
    regression_preds_all = OLS(Yhat_unlabeled, exog=statsmodels.tools.add_constant(Xhat_unlabeled)).fit()
    coeff_preds_all = regression_preds_all.params
    coeff_preds_cov = cov_hc0(regression_preds_all)
    coeff_preds_cov_cholesky = np.linalg.cholesky(coeff_preds_cov)
    for trial in range(trials):
        if quick_convolve:
            # speedup using CLT to estimate coefficient for unlabeled data
            X_trial, Xhat_trial, Y_trial, Yhat_trial = resample_datapoints(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, calib_only=True)
            coeff_calib_correction = ppi_calib_correction(X_trial, Xhat_trial, Y_trial, Yhat_trial, tuning_matrix)
            coeff_preds_all_noisy = coeff_preds_all + coeff_preds_cov_cholesky @ np.random.normal(size=(p+1))
            output = tuning_matrix @ coeff_preds_all_noisy + coeff_calib_correction
        else:
            X_trial, Xhat_trial, Xhat_unlabeled_trial, Y_trial, Yhat_trial, Yhat_unlabeled_trial = resample_datapoints(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled)
            output = ppi_pointestimate(X_trial, Xhat_trial, Xhat_unlabeled_trial, Y_trial, Yhat_trial, Yhat_unlabeled_trial, tuning_matrix)
        outputs.append(output)
        
    lo = np.percentile(outputs, 2.5, axis=0)
    hi = np.percentile(outputs, 97.5, axis=0)
    return (lo, hi)

def classical_ols_ci(X, Y, alpha, alternative="two-sided"):
    regression = OLS(Y, exog=statsmodels.tools.add_constant(X)).fit()
    theta = regression.params
    se = regression.HC0_se
    ci = _zconfint_generic(theta, se, alpha, alternative)
    return (ci[0], ci[1])