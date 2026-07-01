# 🌌 Pulsar Glitch Distribution & Bayesian Model Validation Suite

## 🔬 Project Overview & Computational Scope
This repository houses an advanced **Bayesian Model Validation and Statistical Selection Suite** engineered to characterize the underlying stochastics of pulsar glitch size profiles and inter-glitch time intervals. 

Utilizing empirical datasets cross-matched from the **Jodrell Bank Observatory (JBO)** and the **Australian Telescope National Facility (ATNF)**, the pipeline implements multi-model distribution fitting (Gaussian, Log-Normal, Exponential, Power-Law) alongside multi-modal mixture testing. It automates non-parametric goodness-of-fit validations (**Kolmogorov-Smirnov Tests**) and conducts advanced **Bayesian Parameter Estimation via Markov Chain Monte Carlo (MCMC)** simulations to quantify structural correlation bounds [^1, 2].

---

## 🛠️ Implemented Statistical Methodologies
* **Feature Engineering Optimization**: Implements the qualitatively modified fractional glitch activity parameter (\(a_g\)) to isolate and eliminate observational window irregularities [^1, 2].
* **Goodness-of-Fit Validation**: Automates Kolmogorov-Smirnov (D) statistic computation and maps categorical null-hypothesis (H₀) rejection logic [^1, 2].
* **Bayesian Model Selection**: Computes penalized complexity metrics via the Bayesian Information Criterion (BIC) to validate mixture model structures [^1, 2]:

\[BIC = k \ln(n) - 2\ln(\hat{L})\]

* **Bayesian Inference via MCMC**: Deploys multi-chain sampling via `PyMC` to evaluate posterior probability densities and map High-Density Intervals (HDI) for correlation parameters [^1, 2]:

\[P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) \cdot P(\theta)}{P(\mathcal{D})}\]

---

## 📦 Core Architecture Structure
```text
├── pulsar_analytics_suite.py  # Production Statistical Architecture Class
└── README.md                  # Computational Briefing Documentation (This File)
```

---

## ⚡ Technical Installation & Execution
Configure your local analytics environment via this command sequence:

### 1. Install Dependencies
```bash
pip install numpy scipy pymc
```

### 2. Execution Snippet
```python
import numpy as np
from pulsar_analytics_suite import PulsarAnalyticsSuite

# Sample Mock Telemetry: Glitch Size Scales & Chronological Epoch Array
mock_sizes = [1.2e-9, 4.5e-9, 8.1e-9, 2.3e-9, 9.4e-9, 3.1e-9, 5.2e-9, 1.1e-8, 6.7e-9, 2.9e-9]
mock_epochs = [51000.2, 51240.5, 51610.1, 51920.4, 52100.8, 52450.3, 52810.9, 53100.2, 53420.7, 53800.1]
mock_spin = [3.4, 3.2, 3.9, 3.1, 4.2, 3.5, 3.8, 4.9, 3.7, 3.3]

# Initialize Suite
suite = PulsarAnalyticsSuite(sizes=mock_sizes, epochs=mock_epochs)

# 1. Compute Fractional Activity
print(f"Activity Index: {suite.calculate_fractional_activity()}")

# 2. Run KS Goodness-of-Fit Tests
print(suite.evaluate_unimodal_goodness_of_fit())

# 3. Run PyMC MCMC Sampling for Posterior Hyperparameters
mcmc_results = suite.run_bayesian_mcmc_correlation(independent_variable=mock_spin, draws=1000)
print(mcmc_results)
```

---
_Note: Developed as a research-grade open-source suite for data scientist portfolios and advanced astronomical telemetry indexing._

