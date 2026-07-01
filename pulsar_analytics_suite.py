"""
Pulsar Glitch Sensor Fusion & Computational Statistical Analytics Suite
Author: [Inyama C. Mulambya]
Description: Implements statistical distribution testing, bimodal mixture optimization,
             non-parametric significance metrics, and Bayesian MCMC parameter estimation.
"""

import numpy as np
import scipy.stats as stats
import pymc as pm

class PulsarAnalyticsSuite:
    def __init__(self, sizes: list, epochs: list):
        """Validates numerical structures and maps temporal pulsar glitch tracking profiles."""
        if len(sizes) != len(epochs) or len(sizes) < 2:
            raise ValueError("Data Dimensionality Error: Input arrays must possess equal dimensions (n >= 2).")
        self.sizes = np.array(sizes, dtype=float)
        self.epochs = np.array(epochs, dtype=float)
        self.n_samples = len(self.sizes)

    def calculate_fractional_activity(self) -> float:
        """Computes the qualitatively modified fractional glitch activity parameter (Eq. 3.2)."""
        # Prevents overestimation caused by irregular observational time windows
        time_span = self.epochs[-1] - self.epochs[0]
        if time_span <= 0:
            return 0.0
        coefficient = (self.n_samples - 1) / (self.n_samples * time_span)
        return float(coefficient * np.sum(self.sizes))

    def evaluate_unimodal_goodness_of_fit(self, alpha: float = 0.05) -> dict:
        """Executes explicit Kolmogorov-Smirnov (KS) tests against baseline distributions (Eq. 3.7)."""
        results = {}
        distributions = {
            "Gaussian": stats.norm(loc=np.mean(self.sizes), scale=np.std(self.sizes)),
            "Log-Normal": stats.lognorm(s=np.std(np.log(self.sizes)), scale=np.exp(np.mean(np.log(self.sizes)))),
            "Exponential": stats.expon(loc=np.min(self.sizes), scale=np.mean(self.sizes) - np.min(self.sizes))
        }

        for name, dist in distributions.items():
            # Compares the Empirical CDF (ECDF) against Theoretical CDF models
            d_stat, p_value = stats.kstest(self.sizes, dist.cdf)
            h_decision = 1 if p_value < alpha else 0 # Null hypothesis rejection logic
            results[name] = {
                "ks_stat_d": round(float(d_stat), 4),
                "p_value": round(float(p_value), 4),
                "rejected_h0": bool(h_decision)
            }
        return results

    def compute_bimodal_mixture_bic(self, log_likelihood: float, num_params: int) -> float:
        """Calculates structural Bayesian Information Criterion values to penalize complexity (Eq. 3.12)."""
        return float(num_params * np.log(self.n_samples) - 2 * log_likelihood)

    def evaluate_frequentist_correlations(self, independent_variable: list) -> dict:
        """Computes paired Pearson linear and Spearman rank-transformed correlation coefficients (Eq. 3.9)."""
        x = np.array(independent_variable, dtype=float)
        if len(x) != self.n_samples:
            raise ValueError("Array mapping length mismatch.")

        r_pearson, p_pearson = stats.pearsonr(x, self.sizes)
        r_spearman, p_spearman = stats.spearmanr(x, self.sizes)

        return {
            "pearson": {"r": round(float(r_pearson), 4), "p_value": round(float(p_pearson), 4)},
            "spearman": {"r": round(float(r_spearman), 4), "p_value": round(float(p_spearman), 4)},
            "sign_consistency": bool(np.sign(r_pearson) == np.sign(r_spearman))
        }

    def run_bayesian_mcmc_correlation(self, independent_variable: list, draws: int = 1000) -> dict:
        """Executes Markov Chain Monte Carlo simulations via PyMC to estimate posterior distributions (Eq. 3.14)."""
        x = np.array(independent_variable, dtype=float)
        
        # Standardize datasets to optimize MCMC log-likelihood convergence properties
        x_scaled = (x - np.mean(x)) / np.std(x)
        y_scaled = (self.sizes - np.mean(self.sizes)) / np.std(self.sizes)

        with pm.Model() as model:
            # 1. Define Priors for corporate/scientific hyperparameters
            r = pm.Uniform('r', lower=-1.0, upper=1.0) # Prior distribution of the correlation coefficient
            
            # Construct a covariance matrix structure for the bivariate normal distribution
            cov = pm.Deterministic('cov', pm.math.stack([[1.0, r], [r, 1.0]]))
            
            # 2. Likelihood Formulation Function
            obs = pm.MvNormal('obs', mu=np.zeros(2), cov=cov, observed=np.stack([x_scaled, y_scaled], axis=1))
            
            # 3. Trigger MCMC Sampling Chain
            trace = pm.sample(draws=draws, tune=500, return_inferencedata=True, progressbar=False)

        # Extract credible posterior metric analytics
        posterior_r = trace.posterior['r'].values.flatten()
        mean_r = np.mean(posterior_r)
        std_r = np.std(posterior_r)
        
        # Calculate High Density Interval (HDI) limits bounding the highest 94% probability densities
        hdi_lower, hdi_upper = np.percentile(posterior_r, [3.0, 97.0])

        return {
            "posterior_mean_r": round(float(mean_r), 4),
            "posterior_std_dev": round(float(std_r), 4),
            "hdi_94_credible_interval": [round(float(hdi_lower), 4), round(float(hdi_upper), 4)]
        }
