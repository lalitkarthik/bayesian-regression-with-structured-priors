import json
import os

def new_notebook():
    return {"cells": [], "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}

def add_md(nb, text):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(nb, code, num=None):
    lines = [f"{line}\n" for line in code.split('\n')]
    nb["cells"].append({"cell_type": "code", "execution_count": num, "metadata": {}, "outputs": [], "source": lines})

# 1. CANADA PIPELINE
nb_can = new_notebook()
add_md(nb_can, "# Canada VAR Pipeline — Library Implementation\nIncludes PyMC for Bayesian methods and Scikit-learn + Statsmodels for Frequentist.")
add_code(nb_can, """\


import os
os.environ["PYTENSOR_FLAGS"] = "device=cuda,floatX=float32"

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
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

LAMBDA_RIDGE = 0.1
NORMAL_PRIOR_SCALE = 1.0
LASSO_LAMBDA = 1.0
T_TEST_DIFFS = 4
LAG_MIN = 1
LAG_MAX = 12

MCMC_TUNE = 1000
MCMC_DRAWS = 1000
MCMC_CHAINS = 2

N_BOOT = 50
BLOCK_SIZE = 4
CI_ALPHA = 0.05
SEED = 123
COL_NAMES = ['e', 'prod', 'rw', 'U']
OUTPUT_DIR = os.path.join(os.getcwd(), 'canada_var_results_library')
os.makedirs(OUTPUT_DIR, exist_ok=True)
""")

add_code(nb_can, """\
local_csv = os.path.join(os.getcwd(), 'canada_data.csv')
if os.path.exists(local_csv):
    df_canada = pd.read_csv(local_csv, index_col=0)
else:
    from statsmodels.datasets import get_rdataset
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
""")

add_code(nb_can, """\
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
        B = self.get_coefficients()
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
""")

add_code(nb_can, """\
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
    'FreqLasso': lambda: SklearnLassoVAR(alpha=LASSO_LAMBDA),
    'Normal': lambda: PyMCBayesianVAR('Normal', prior_scale=NORMAL_PRIOR_SCALE),
    'Lasso': lambda: PyMCBayesianVAR('Lasso', prior_scale=LASSO_LAMBDA),
    'Horseshoe': lambda: PyMCBayesianVAR('Horseshoe'),
    'Spike and Slab': lambda: PyMCBayesianVAR('SpikeSlab'),
}
""")

add_code(nb_can, """\
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
""")

add_code(nb_can, """\
np.random.seed(SEED)
all_results = []
all_predictions = {}
actual_prev_levels = Ytest_levels[:-1, :]

# Run tests
print(f"Running pass for lags {LAG_MIN} to {LAG_MAX}...")
for p_val in range(LAG_MIN, LAG_MAX + 1):
    X_train, Y_train = make_var_design_p(Ytrain_diff, p_val)
    if X_train is None: continue
    
    print(f"\\n--- Lag {p_val} ---")
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

# --- Save full results ---
csv_path = os.path.join(OUTPUT_DIR, 'all_metrics.csv')
df_results.to_csv(csv_path, index=False)
print("Results saved to", csv_path)

# --- Best-lag summary by AIC and RMSE ---
df_all = df_results[df_results['variable'] == 'All'].copy()
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 120)
print()
print("=== Best Lag per Method (by AIC) ===")
best_aic = df_all.loc[df_all.groupby('Method')['AIC'].idxmin(), ['Method', 'p', 'AIC', 'BIC', 'RMSE']]
print(best_aic.reset_index(drop=True))
print()
print("=== Best Lag per Method (by RMSE) ===")
best_rmse = df_all.loc[df_all.groupby('Method')['RMSE'].idxmin(), ['Method', 'p', 'RMSE', 'SMAPE', 'AIC']]
print(best_rmse.reset_index(drop=True))

df_results.head(10)
""")

add_code(nb_can, """\
# ============================================================
# VISUALIZATION — Forecast RMSE heatmap across Lags x Methods
# ============================================================
import seaborn as sns

df_all = df_results[df_results['variable'] == 'All'].copy()
pivot_rmse = df_all.pivot_table(index='p', columns='Method', values='RMSE')

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.heatmap(pivot_rmse, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0],
            linewidths=0.5, cbar_kws={'label': 'RMSE'})
axes[0].set_title('Forecast RMSE by Lag and Method', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Method')
axes[0].set_ylabel('Lag (p)')

pivot_aic = df_all.pivot_table(index='p', columns='Method', values='AIC')
sns.heatmap(pivot_aic, annot=True, fmt='.0f', cmap='Blues_r', ax=axes[1],
            linewidths=0.5, cbar_kws={'label': 'AIC'})
axes[1].set_title('AIC by Lag and Method', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Method')
axes[1].set_ylabel('Lag (p)')

plt.suptitle('Canada VAR — Model Selection Overview', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_rmse_aic.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Heatmap saved.")
""")

add_code(nb_can, """\
# ============================================================
# VISUALIZATION — Forecast vs Actual for best lag per method
# ============================================================
best_p_per_method = df_all.loc[df_all.groupby('Method')['RMSE'].idxmin(), ['Method', 'p']]
test_dates = list(range(len(actual_test_levels)))

fig, axes = plt.subplots(d, 1, figsize=(14, 3 * d), sharex=True)
if d == 1: axes = [axes]

for col_idx, col in enumerate(COL_NAMES):
    ax = axes[col_idx]
    ax.plot(test_dates, actual_test_levels[:, col_idx], 'k-o', linewidth=2,
            markersize=5, label='Actual', zorder=5)
    for _, row in best_p_per_method.iterrows():
        key = (int(row['p']), row['Method'])
        if key in all_predictions:
            preds = all_predictions[key]['preds']
            ax.plot(test_dates, preds[:, col_idx], '--', linewidth=1.4,
                    label=f"{row['Method']} (p={int(row['p'])})", alpha=0.8)
    ax.set_title(f'Variable: {col}', fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

fig.suptitle('Forecast vs Actual — Best Lag per Method', fontsize=14, fontweight='bold')
plt.xlabel('Test Step')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'forecast_vs_actual.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Forecast plot saved.")
""")

with open('c:\\Users\\adity\\OneDrive\\Desktop\\Stats for AIML\\bayesian-regression-with-structured-priors\\canada_var_pipeline_library.ipynb', 'w') as f:
    json.dump(nb_can, f, indent=2)

# =========================================================
# 2. SIMULATION PIPELINE
# =========================================================
nb_sim = new_notebook()
add_md(nb_sim, "# VAR Simulation Pipeline — Library Implementation\nEvaluating Scikit-Learn/PyMC shrinkage priors across Scenarios")
add_code(nb_sim, """\


import os
os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64"
# CPU required: PyMC NUTS sampler is incompatible with CUDA backend

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import warnings
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from scipy import stats
import pymc as pm
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

N_REPLICATIONS = 5
T_TOTAL = 200
T_TEST = 20
BURNIN = 50
SPARSITY_PROB = 0.3
COEF_RANGE = (-0.4, 0.4)
SIGMA_DIAG = 0.1

LAMBDA_RIDGE = 0.1
CI_ALPHA = 0.05
MCMC_TUNE = 1000
MCMC_DRAWS = 1000
MCMC_CHAINS = 2
SEED = 123

SCENARIOS = {
    1: {'name': 'Low-dim (d=3, p=4)', 'd': 3, 'p_fit': 4, 'seed': 101},
    2: {'name': 'High-dim (d=20, p=1)', 'd': 20, 'p_fit': 1, 'seed': 202},
    3: {'name': 'High-dim overfit (d=20, p=4)', 'd': 20, 'p_fit': 4, 'seed': 303},
}

def gen_var_data(d, T, burnin=BURNIN, rng=None):
    if rng is None: rng = np.random.RandomState()
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if rng.uniform() > SPARSITY_PROB:
                A[i, j] = rng.uniform(*COEF_RANGE)
    eigvals = np.linalg.eigvals(A)
    max_mod = np.max(np.abs(eigvals))
    if max_mod >= 1.0: A = A / (1.1 * max_mod)
    Sigma = SIGMA_DIAG * np.eye(d)
    Y = np.zeros((T + burnin, d))
    Y[0, :] = rng.randn(d)
    for t in range(1, T + burnin):
        Y[t, :] = rng.multivariate_normal(A @ Y[t - 1, :], Sigma)
    return {'Y': Y[burnin:, :], 'A_true': A, 'Sigma_true': Sigma}

def make_var_design(Y, p):
    T, d = Y.shape
    X = np.zeros((T - p, d * p))
    Y_out = Y[p:, :]
    for t in range(p, T):
        lags = []
        for lag_i in range(1, p + 1):
            lags.extend(Y[t - lag_i, :])
        X[t - p, :] = lags
    return X, Y_out

def get_true_B(A_true, d, p_fit):
    B_true = np.zeros((d, d * p_fit))
    B_true[:, :d] = A_true
    return B_true
""")

add_code(nb_sim, """\
class VARModel(ABC):
    @property
    @abstractmethod
    def name(self): pass
    @abstractmethod
    def fit(self, X_train, Y_train, d, p_fit): pass
    @abstractmethod
    def get_coefficients(self): pass
    @abstractmethod
    def get_intervals(self, alpha=0.05): pass

class SklearnRidgeVAR(VARModel):
    def __init__(self, alpha=LAMBDA_RIDGE):
        self.alpha = alpha
        self._name = 'SklearnRidge'
    @property
    def name(self): return self._name
    def fit(self, X_train, Y_train, d, p_fit):
        self.model = MultiOutputRegressor(Ridge(alpha=self.alpha, fit_intercept=False))
        self.model.fit(X_train, Y_train)
        self.B_hat_ = np.vstack([est.coef_ for est in self.model.estimators_])
        # Analytic approx for fast runtime
        resid = Y_train - self.model.predict(X_train)
        sigma2 = np.var(resid, axis=0)
        XtX_inv = np.linalg.pinv(X_train.T @ X_train)
        se = np.sqrt(np.clip(np.outer(sigma2, np.diag(XtX_inv)), 0, None))
        z_val = stats.norm.ppf(1 - CI_ALPHA / 2)
        self.B_lower_ = self.B_hat_ - z_val * se
        self.B_upper_ = self.B_hat_ + z_val * se
        return self
    def get_coefficients(self): return self.B_hat_
    def get_intervals(self, alpha=0.05): return self.B_lower_, self.B_upper_

""" + """\
class SklearnOLSVAR(VARModel):
    def __init__(self):
        self._name = 'SklearnOLS'
    @property
    def name(self): return self._name
    def fit(self, X_train, Y_train, d, p_fit):
        self.model = MultiOutputRegressor(LinearRegression(fit_intercept=False))
        self.model.fit(X_train, Y_train)
        self.B_hat_ = np.vstack([est.coef_ for est in self.model.estimators_])
        resid = Y_train - self.model.predict(X_train)
        sigma2 = np.var(resid, axis=0)
        XtX_inv = np.linalg.pinv(X_train.T @ X_train)
        se = np.sqrt(np.clip(np.outer(sigma2, np.diag(XtX_inv)), 0, None))
        z_val = stats.norm.ppf(1 - CI_ALPHA / 2)
        self.B_lower_ = self.B_hat_ - z_val * se
        self.B_upper_ = self.B_hat_ + z_val * se
        return self
    def get_coefficients(self): return self.B_hat_
    def get_intervals(self, alpha=0.05): return self.B_lower_, self.B_upper_

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
                pi = pm.Beta("pi", 1, 1)
                sigma_slab = pm.HalfNormal("sigma_slab", sigma=1)
                slab = pm.Normal.dist(mu=0, sigma=sigma_slab)
                spike = pm.Normal.dist(mu=0, sigma=1e-4)
                w = pm.math.stack([pi, 1-pi])
                components = [slab, spike]
                B = pm.Mixture("B", w=w, comp_dists=components, shape=(d, q))

            sigma = pm.HalfCauchy("sigma", beta=2.5, shape=d)
            mu = pm.math.dot(X_train, B.T)
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y_train)
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
""")

add_code(nb_sim, """\
MODEL_REGISTRY = {
    'OLS': lambda: SklearnOLSVAR(),
    'Ridge': lambda: SklearnRidgeVAR(),
    'Normal': lambda: PyMCBayesianVAR('Normal'),
    'Lasso': lambda: PyMCBayesianVAR('Lasso'),
    'Horseshoe': lambda: PyMCBayesianVAR('Horseshoe'),
    'Spike and Slab': lambda: PyMCBayesianVAR('SpikeSlab'),
}

def compute_aic_bic(Y_train, pred_train, d, p):
    T = Y_train.shape[0]
    eps = Y_train - pred_train
    Sigma = eps.T @ eps / T
    Sigma += np.eye(d) * 1e-8
    det_S = np.linalg.det(Sigma)
    if det_S <= 0: det_S = 1e-8
    log_det = np.log(det_S)
    k = d * d * p
    return float(T * log_det + 2 * k), float(T * log_det + np.log(T) * k)

class MetricsEngine:
    @staticmethod
    def forecast_rmse(B, Ytrain, Ytest, p):
        Y_for_forecast = np.vstack([Ytrain[-p:, :], Ytest])
        Ttest = Ytest.shape[0]
        rmse_vec = []
        for i in tqdm(range(p, p + Ttest - 1), desc="Simulated Forecast", leave=False):
            lags = []
            for lag_j in range(p): lags.extend(Y_for_forecast[i - lag_j, :])
            pred = np.array(lags).reshape(1, -1) @ B.T
            actual = Y_for_forecast[i + 1, :]
            rmse_vec.append(np.sqrt(np.mean((actual - pred.flatten())**2)))
        return np.mean(rmse_vec)
    
    @staticmethod
    def param_metrics(B_est, B_lower, B_upper, B_true, tol=1e-3):
        b_e = B_est.flatten()
        b_lo = B_lower.flatten()
        b_hi = B_upper.flatten()
        b_t = B_true.flatten()
        
        zero_mask = np.abs(b_t) < tol
        nz_mask = ~zero_mask
        
        metrics = {}
        # Overall
        metrics['param_rmse'] = float(np.sqrt(np.mean((b_e - b_t)**2)))
        metrics['coverage'] = float(np.mean((b_t >= b_lo) & (b_t <= b_hi)))
        metrics['int_length'] = float(np.mean(b_hi - b_lo))
        
        # Zero
        if np.sum(zero_mask) > 0:
            metrics['param_rmse_zero'] = float(np.sqrt(np.mean((b_e[zero_mask] - b_t[zero_mask])**2)))
            metrics['coverage_zero'] = float(np.mean((b_t[zero_mask] >= b_lo[zero_mask]) & (b_t[zero_mask] <= b_hi[zero_mask])))
            metrics['int_length_zero'] = float(np.mean(b_hi[zero_mask] - b_lo[zero_mask]))
        else:
            metrics['param_rmse_zero'] = np.nan
            metrics['coverage_zero'] = np.nan
            metrics['int_length_zero'] = np.nan
            
        # NonZero
        if np.sum(nz_mask) > 0:
            metrics['param_rmse_nonzero'] = float(np.sqrt(np.mean((b_e[nz_mask] - b_t[nz_mask])**2)))
            metrics['coverage_nonzero'] = float(np.mean((b_t[nz_mask] >= b_lo[nz_mask]) & (b_t[nz_mask] <= b_hi[nz_mask])))
            metrics['int_length_nonzero'] = float(np.mean(b_hi[nz_mask] - b_lo[nz_mask]))
        else:
            metrics['param_rmse_nonzero'] = np.nan
            metrics['coverage_nonzero'] = np.nan
            metrics['int_length_nonzero'] = np.nan
            
        # Sparsity metrics
        is_zero_est = np.abs(b_e) < tol
        TP_zero = np.sum(zero_mask & is_zero_est)
        FN_zero = np.sum(zero_mask & ~is_zero_est)
        metrics['sparsity_recovery_rate'] = float(TP_zero / max(1, (TP_zero + FN_zero)))

        is_nz_est = np.abs(b_e) >= tol
        FP_nz = np.sum(zero_mask & is_nz_est)
        TP_nz = np.sum(nz_mask & is_nz_est)
        metrics['false_discovery_rate'] = float(FP_nz / max(1, (FP_nz + TP_nz)))
        
        return metrics

def run_single_replication(scen_id, rep, rng):
    cfg = SCENARIOS[scen_id]
    d, p_fit = cfg['d'], cfg['p_fit']
    gen_out = gen_var_data(d, T_TOTAL, burnin=BURNIN, rng=rng)
    Yfull = gen_out['Y']
    Ttrain = T_TOTAL - T_TEST
    Ytrain, Ytest = Yfull[:Ttrain, :], Yfull[Ttrain:, :]
    X_train, Y_train = make_var_design(Ytrain, p_fit)
    B_true = get_true_B(gen_out['A_true'], d, p_fit)
    
    results = []
    
    # Compute OLS first to calculate Shrinkage Ratio
    ols_model = SklearnOLSVAR().fit(X_train, Y_train, d, p_fit)
    B_ols = ols_model.get_coefficients()
    mean_abs_B_ols = np.mean(np.abs(B_ols))
    
    for name, factory in MODEL_REGISTRY.items():
        if name == 'OLS':
            model = ols_model
            B_hat = B_ols
            B_lo, B_hi = model.get_intervals()
        else:
            try:
                model = factory().fit(X_train, Y_train, d, p_fit)
                B_hat = model.get_coefficients()
                B_lo, B_hi = model.get_intervals()
            except Exception as e:
                print(f"    [{name}] FAILED: {e} — skipping")
                continue
            
        pred_train = X_train @ B_hat.T
        aic, bic = compute_aic_bic(Y_train, pred_train, d, p_fit)
        f_rmse = MetricsEngine.forecast_rmse(B_hat, Ytrain, Ytest, p_fit)
        mets = MetricsEngine.param_metrics(B_hat, B_lo, B_hi, B_true)
        shrinkage = np.mean(np.abs(B_hat)) / max(1e-8, mean_abs_B_ols)
        
        row = {
            'scenario': scen_id, 'scenario_name': SCENARIOS[scen_id]['name'],
            'rep': rep, 'model': name,
            'forecast_rmse': f_rmse, 'shrinkage_ratio': shrinkage,
            'AIC': aic, 'BIC': bic
        }
        row.update(mets)
        results.append(row)
    return results

all_results = []
for s_id in SCENARIOS.keys():
    rng = np.random.RandomState(SCENARIOS[s_id]['seed'])
    scen_name = SCENARIOS[s_id]['name']
    print(f"Running Scenario {s_id}: {scen_name}")
    for r in tqdm(range(1, N_REPLICATIONS + 1), desc="Replications"):
        t0 = time.time()
        res = run_single_replication(s_id, r, rng)
        all_results.extend(res)
        elapsed = round(time.time() - t0, 1)
        print(f"  Rep {r}/{N_REPLICATIONS} Done in {elapsed}s")

df = pd.DataFrame(all_results)

# --- Save results ---
OUTPUT_DIR = os.path.join(os.getcwd(), 'simulation_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, 'simulation_metrics.csv')
df.to_csv(csv_path, index=False)
print("Results saved to", csv_path)

df.head(10)
""")

add_code(nb_sim, """\
# ============================================================
# VISUALIZATION — Metric comparison across Models x Scenarios
# ============================================================
import seaborn as sns

if len(df) > 0:
    df_avg = df.groupby(['scenario_name', 'model'])[['forecast_rmse', 'param_rmse', 'coverage',
                                                       'sparsity_recovery_rate', 'false_discovery_rate']].mean().reset_index()

    scenarios = df_avg['scenario_name'].unique()
    n_scen = len(scenarios)
    fig, axes = plt.subplots(n_scen, 2, figsize=(16, 5 * n_scen))
    if n_scen == 1: axes = axes.reshape(1, -1)

    for idx, scen in enumerate(scenarios):
        scen_df = df_avg[df_avg['scenario_name'] == scen].sort_values('forecast_rmse')
        colors = ['#2196F3' if 'PyMC' not in m else '#FF5722' for m in scen_df['model']]

        ax0 = axes[idx, 0]
        ax0.barh(scen_df['model'], scen_df['forecast_rmse'], color=colors)
        ax0.set_title(f'{scen}\\nForecast RMSE (lower=better)', fontsize=11)
        ax0.set_xlabel('Forecast RMSE')
        ax0.grid(axis='x', alpha=0.3)

        ax1 = axes[idx, 1]
        ax1.barh(scen_df['model'], scen_df['param_rmse'], color=colors)
        ax1.set_title(f'{scen}\\nParam RMSE vs B_true (lower=better)', fontsize=11)
        ax1.set_xlabel('Param RMSE')
        ax1.grid(axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    handles = [Patch(color='#2196F3', label='Frequentist'), Patch(color='#FF5722', label='Bayesian')]
    fig.legend(handles=handles, loc='upper right', fontsize=11)
    plt.suptitle('VAR Simulation — Shrinkage Prior Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'simulation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Simulation comparison chart saved.")
else:
    print("No results to plot.")
""")

add_code(nb_sim, """\
# ============================================================
# VISUALIZATION — Sparsity Recovery and CI Coverage heatmap
# ============================================================
if len(df) > 0:
    df_avg = df.groupby(['scenario_name', 'model'])[['coverage', 'sparsity_recovery_rate',
                                                       'false_discovery_rate', 'shrinkage_ratio']].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    piv_cov = df_avg.pivot(index='model', columns='scenario_name', values='coverage')
    sns.heatmap(piv_cov, annot=True, fmt='.2f', cmap='Greens', ax=axes[0],
                linewidths=0.5, vmin=0, vmax=1)
    axes[0].set_title('CI Coverage Rate (target=0.95)', fontsize=12, fontweight='bold')

    piv_spar = df_avg.pivot(index='model', columns='scenario_name', values='sparsity_recovery_rate')
    sns.heatmap(piv_spar, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Sparsity Recovery Rate (higher=better)', fontsize=12, fontweight='bold')

    plt.suptitle('Uncertainty Quantification — Coverage & Sparsity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'simulation_coverage_sparsity.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Coverage/sparsity chart saved.")
""")

with open('c:\\Users\\adity\\OneDrive\\Desktop\\Stats for AIML\\bayesian-regression-with-structured-priors\\var_simulation_pipeline_library.ipynb', 'w') as f:
    json.dump(nb_sim, f, indent=2)

print("Notebooks generated successfully.")
