#!/usr/bin/env python
# coding: utf-8

# # Canada VAR Pipeline — Library Implementation
# Includes PyMC for Bayesian methods and Scikit-learn + Statsmodels for Frequentist.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import warnings
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from scipy import stats
import pymc as pm
import arviz as az
from statsmodels.datasets import get_rdataset
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

LAMBDA_RIDGE = 0.1
NORMAL_PRIOR_SCALE = 1.0
LASSO_LAMBDA = 1.0
T_TEST_DIFFS = 4
LAG_MIN = 1
LAG_MAX = 12

MCMC_TUNE = 500
MCMC_DRAWS = 500
MCMC_CHAINS = 2

N_BOOT = 50
BLOCK_SIZE = 4
CI_ALPHA = 0.05
SEED = 123
COL_NAMES = ['e', 'prod', 'rw', 'U']
OUTPUT_DIR = os.path.join(os.getcwd(), 'canada_var_results_library')
os.makedirs(OUTPUT_DIR, exist_ok=True)



# In[ ]:


local_csv = os.path.join(os.getcwd(), 'canada_data.csv')
if os.path.exists(local_csv):
    df_canada = pd.read_csv(local_csv, index_col=0)
else:
    
    canada_ds = get_rdataset('Canada', 'vars')
    df_canada = canada_ds.data
df_canada.columns = [c.strip() for c in df_canada.columns]
if 'e' not in df_canada.columns:
    df_canada.columns = COL_NAMES[:df_canada.shape[1]]

Y_levels = df_canada[COL_NAMES].values.astype(np.float64)
Tfull, d = Y_levels.shape
Y_diff = np.diff(Y_levels, axis=0)

holdout_idx = Y_diff.shape[0] - T_TEST_DIFFS
Ytrain_diff = Y_diff[:holdout_idx, :]
Ytest_diff = Y_diff[holdout_idx:, :]
Ytrain_levels = Y_levels[:holdout_idx + 1, :]
Ytest_levels = Y_levels[holdout_idx:, :]
actual_test_levels = Ytest_levels[1:, :]

def make_var_design_p(Y, p):
    T, d = Y.shape
    if T <= p: return None, None
    X = np.zeros((T - p, d * p))
    Y_out = Y[p:, :]
    for t in range(p, T):
        lags = []
        for lag_i in range(1, p + 1):
            lags.extend(Y[t - lag_i, :])
        X[t - p, :] = lags
    return X, Y_out



# In[ ]:


class VARModel(ABC):
    @property
    @abstractmethod
    def name(self): pass
    @property
    def is_bayesian(self): return False
    @abstractmethod
    def fit(self, X_train, Y_train, d, p_fit): pass
    @abstractmethod
    def get_coefficients(self): pass
    @abstractmethod
    def get_intervals(self, alpha=0.05): pass
    def predict(self, X_test):
        B = this.get_coefficients()
        return X_test @ B.T
    def get_posterior_samples(self): return None

class SklearnRidgeVAR(VARModel):
    def __init__(self, alpha=LAMBDA_RIDGE, n_boot=N_BOOT, block_size=BLOCK_SIZE, cl_alpha=CI_ALPHA):
        self.alpha = alpha
        self.n_boot = n_boot
        self.block_size = block_size
        self.cl_alpha = cl_alpha
        self._name = 'SklearnRidge'

    @property
    def name(self): return self._name

    def fit(self, X_train, Y_train, d, p_fit):
        self.model = MultiOutputRegressor(Ridge(alpha=self.alpha, fit_intercept=False))
        self.model.fit(X_train, Y_train)
        self.B_hat_ = np.vstack([est.coef_ for est in self.model.estimators_])
        
        n = X_train.shape[0]
        B_boot = np.zeros((self.n_boot, d, d * p_fit))
        for b in tqdm(range(self.n_boot), desc=f"Bootstrap {self._name}", leave=False):
            nblocks = int(np.ceil(n / self.block_size))
            starts = np.random.randint(0, max(1, n - self.block_size + 1), size=nblocks)
            idx = [i for s in starts for i in range(s, s + self.block_size)][:n]
            Xb, Yb = X_train[idx], Y_train[idx]
            boot_model = MultiOutputRegressor(Ridge(alpha=self.alpha, fit_intercept=False)).fit(Xb, Yb)
            B_boot[b] = np.vstack([est.coef_ for est in boot_model.estimators_])

        self.B_sd_ = np.std(B_boot, axis=0)
        z_val = stats.norm.ppf(1 - self.cl_alpha / 2)
        self.B_lower_ = self.B_hat_ - z_val * self.B_sd_
        self.B_upper_ = self.B_hat_ + z_val * self.B_sd_
        return self

    def get_coefficients(self): return self.B_hat_
    def get_intervals(self, alpha=0.05): return self.B_lower_, self.B_upper_

# Using LinearRegression on built design matrix is exactly OLS VAR but faster to code interface
class SklearnOLSVAR(VARModel):
    def __init__(self):
        self._name = 'SklearnOLS'
    @property
    def name(self): return self._name
    def fit(self, X_train, Y_train, d, p_fit):
        self.model = MultiOutputRegressor(LinearRegression(fit_intercept=False))
        self.model.fit(X_train, Y_train)
        self.B_hat_ = np.vstack([est.coef_ for est in self.model.estimators_])
        
        # Simple analytic CI approx for speed
        resid = Y_train - self.model.predict(X_train)
        sigma2 = np.var(resid, axis=0)
        XtX_inv = np.linalg.pinv(X_train.T @ X_train)
        diag_inv = np.diag(XtX_inv)
        se = np.sqrt(np.clip(np.outer(sigma2, diag_inv), 0, None))
        z_val = stats.norm.ppf(1 - CI_ALPHA / 2)
        self.B_lower_ = self.B_hat_ - z_val * se
        self.B_upper_ = self.B_hat_ + z_val * se
        return self
    def get_coefficients(self): return self.B_hat_
    def get_intervals(self, alpha=0.05): return self.B_lower_, self.B_upper_
    
class SklearnLassoVAR(VARModel):
    def __init__(self, alpha=LASSO_LAMBDA):
        self.alpha = alpha
        self._name = 'SklearnLasso'
    @property
    def name(self): return self._name
    def fit(self, X_train, Y_train, d, p_fit):
        self.model = MultiOutputRegressor(Lasso(alpha=self.alpha, fit_intercept=False))
        self.model.fit(X_train, Y_train)
        self.B_hat_ = np.vstack([est.coef_ for est in self.model.estimators_])
        
        # Simple analytic CI approx for speed
        resid = Y_train - self.model.predict(X_train)
        sigma2 = np.var(resid, axis=0)
        XtX_inv = np.linalg.pinv(X_train.T @ X_train)
        diag_inv = np.diag(XtX_inv)
        se = np.sqrt(np.clip(np.outer(sigma2, diag_inv), 0, None))
        z_val = stats.norm.ppf(1 - CI_ALPHA / 2)
        self.B_lower_ = self.B_hat_ - z_val * se
        self.B_upper_ = self.B_hat_ + z_val * se
        return self
    def get_coefficients(self): return self.B_hat_
    def get_intervals(self, alpha=0.05): return self.B_lower_, self.B_upper_



# In[ ]:


class PyMCBayesianVAR(VARModel):
    def __init__(self, prior_type, prior_scale=1.0, tune=MCMC_TUNE, draws=MCMC_DRAWS, chains=MCMC_CHAINS):
        self.prior_type = prior_type
        self.prior_scale = prior_scale
        self.tune = tune
        self.draws = draws
        self.chains = chains
        self._name = f'PyMC_{prior_type}'
    @property
    def name(self): return self._name
    @property
    def is_bayesian(self): return True

    def fit(self, X_train, Y_train, d, p_fit):
        q = X_train.shape[1]
        
        with pm.Model() as model:
            if self.prior_type == "Normal":
                B = pm.Normal("B", mu=0, sigma=self.prior_scale, shape=(d, q))
            elif self.prior_type == "Lasso":
                B = pm.Laplace("B", mu=0, b=1.0/self.prior_scale, shape=(d, q))
            elif self.prior_type == "Horseshoe":
                tau = pm.HalfCauchy("tau", beta=1, shape=(d, 1))
                lambd = pm.HalfCauchy("lambda", beta=1, shape=(d, q))
                B = pm.Normal("B", mu=0, sigma=tau * lambd, shape=(d, q))
            elif self.prior_type == "SpikeSlab":
                # Continuous relaxation of spike and slab using Mixture
                pi = pm.Beta("pi", 1, 1)
                sigma_slab = pm.HalfNormal("sigma_slab", sigma=1)
                
                # We use large sigma for slab, very small fixed for spike
                slab = pm.Normal.dist(mu=0, sigma=sigma_slab)
                spike = pm.Normal.dist(mu=0, sigma=1e-4)
                
                w = pm.math.stack([pi, 1-pi])
                components = [slab, spike]
                
                B = pm.Mixture("B", w=w, comp_dists=components, shape=(d, q))

            sigma = pm.HalfCauchy("sigma", beta=2.5, shape=d)
            mu = pm.math.dot(X_train, B.T)
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y_train)

            # Silence logging intentionally
            import logging
            logger = logging.getLogger("pymc")
            logger.setLevel(logging.ERROR)
            
            trace = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains, 
                              target_accept=0.85, random_seed=SEED, progressbar=True, compute_convergence_checks=False)

        self.B_samples_ = trace.posterior["B"].values.reshape(-1, d, q)
        self.B_hat_ = np.mean(self.B_samples_, axis=0)
        self.B_lower_ = np.percentile(self.B_samples_, 2.5, axis=0)
        self.B_upper_ = np.percentile(self.B_samples_, 97.5, axis=0)
        return self

    def get_coefficients(self): return self.B_hat_
    def get_intervals(self, alpha=0.05): return self.B_lower_, self.B_upper_
    def get_posterior_samples(self): return self.B_samples_

MODEL_REGISTRY = {
    'OLS': lambda: SklearnOLSVAR(),
    'Ridge': lambda: SklearnRidgeVAR(),
    'Lasso': lambda: SklearnLassoVAR(alpha=LASSO_LAMBDA),
    'Normal_PyMC': lambda: PyMCBayesianVAR('Normal', prior_scale=NORMAL_PRIOR_SCALE),
    'Lasso_PyMC': lambda: PyMCBayesianVAR('Lasso', prior_scale=LASSO_LAMBDA),
    'Horseshoe_PyMC': lambda: PyMCBayesianVAR('Horseshoe'),
    'SpikeSlab_PyMC': lambda: PyMCBayesianVAR('SpikeSlab'),
}



# In[ ]:


def run_rolling_forecast(B_hat, Ytrain_diff, Ytrain_levels, Ytest_levels, p):
    d = Ytrain_diff.shape[1]
    Ttest = Ytest_levels.shape[0] - 1
    preds = np.zeros((Ttest, d))
    diff_history = Ytrain_diff[-p:].copy()
    current_level = Ytrain_levels[-1:].copy()
    
    for i in tqdm(range(Ttest), desc="Rolling Forecast", leave=False):
        lag_vec = diff_history[::-1].flatten().reshape(1, -1)
        pred_diff = lag_vec @ B_hat.T
        pred_level = current_level + pred_diff
        preds[i] = pred_level
        actual_next = Ytest_levels[i + 1:i + 2]
        new_diff = actual_next - current_level
        if p > 1:
            diff_history = np.vstack([diff_history[1:], new_diff])
        else:
            diff_history = new_diff.copy()
        current_level = actual_next.copy()
    return preds

class MetricsEngine:
    @staticmethod
    def rmse(actual, predicted): return float(np.sqrt(np.mean((actual - predicted)**2)))
    @staticmethod
    def mape(actual, predicted, eps=1e-8):
        mask = np.abs(actual) > eps
        if not np.any(mask): return np.nan
        return float(100.0 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))
    @staticmethod
    def mae(actual, predicted): return float(np.mean(np.abs(actual - predicted)))
    @staticmethod
    def smape(actual, predicted, eps=1e-8):
        denom = np.abs(actual) + np.abs(predicted)
        mask = denom > eps
        if not np.any(mask): return np.nan
        return float(100.0 * np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denom[mask]))
    @staticmethod
    def directional_accuracy(actual, predicted, actual_prev):
        actual_diff = actual - actual_prev
        pred_diff = predicted - actual_prev
        return float(100.0 * np.mean(np.sign(actual_diff) == np.sign(pred_diff)))

    @staticmethod
    def aic_bic(actual_train, pred_train, d, p):
        T = actual_train.shape[0]
        eps = actual_train - pred_train
        Sigma = eps.T @ eps / T
        Sigma += np.eye(d) * 1e-8
        det_Sigma = np.linalg.det(Sigma)
        if det_Sigma <= 0: det_Sigma = 1e-8
        log_det = np.log(det_Sigma)
        num_params = d * d * p
        aic = T * log_det + 2 * num_params
        bic = T * log_det + np.log(T) * num_params
        return float(aic), float(bic)

    @staticmethod
    def compute_all_metrics(actual, predicted, actual_prev, col_names, aic=np.nan, bic=np.nan):
        results = []
        d = actual.shape[1]
        for j in range(d):
            a_j = actual[:, j]
            p_j = predicted[:, j]
            prev_j = actual_prev[:, j]
            results.append({
                'variable': col_names[j],
                'RMSE': MetricsEngine.rmse(a_j, p_j),
                'MAPE': MetricsEngine.mape(a_j, p_j),
                'MAE': MetricsEngine.mae(a_j, p_j),
                'SMAPE': MetricsEngine.smape(a_j, p_j),
                'DirAcc': MetricsEngine.directional_accuracy(a_j, p_j, prev_j)
            })
        
        a_f = actual.flatten()
        p_f = predicted.flatten()
        prev_f = actual_prev.flatten()
        results.append({
            'variable': 'All',
            'RMSE': MetricsEngine.rmse(a_f, p_f),
            'MAPE': MetricsEngine.mape(a_f, p_f),
            'MAE': MetricsEngine.mae(a_f, p_f),
            'SMAPE': MetricsEngine.smape(a_f, p_f),
            'DirAcc': MetricsEngine.directional_accuracy(a_f, p_f, prev_f),
            'AIC': aic,
            'BIC': bic
        })
        return results



# In[ ]:


np.random.seed(SEED)
all_results = []
all_predictions = {}
actual_prev_levels = Ytest_levels[:-1, :]

# Run tests
print(f"Running pass for lags {LAG_MIN} to {LAG_MAX}...")
for p_val in range(LAG_MIN, LAG_MAX + 1):
    X_train, Y_train = make_var_design_p(Ytrain_diff, p_val)
    if X_train is None: continue
    
    print(f"\n--- Lag {p_val} ---")
    for model_name, factory in MODEL_REGISTRY.items():
        model = factory()
        t0 = time.time()
        model.fit(X_train, Y_train, d, p_val)
        B_hat = model.get_coefficients()
        
        # Calculate training predictions for AIC/BIC
        pred_train = X_train @ B_hat.T
        aic, bic = MetricsEngine.aic_bic(Y_train, pred_train, d, p_val)
        
        preds = run_rolling_forecast(B_hat, Ytrain_diff, Ytrain_levels, Ytest_levels, p_val)
        all_predictions[(p_val, model_name)] = dict(preds=preds, B=B_hat)
        metrics = MetricsEngine.compute_all_metrics(actual_test_levels, preds, actual_prev_levels, COL_NAMES, aic, bic)
        for m in metrics:
            m['p'] = p_val
            m['Method'] = model_name
        all_results.extend(metrics)
        sys_metric = [m for m in metrics if m['variable']=='All'][0]
        print(f"  {model_name:15s} | RMSE: {sys_metric['RMSE']:.4f} | SMAPE: {sys_metric['SMAPE']:.2f}% | AIC: {sys_metric['AIC']:.1f} | BIC: {sys_metric['BIC']:.1f} | [{time.time()-t0:.1f}s]")

df_results = pd.DataFrame(all_results)
df_results.head(10)


