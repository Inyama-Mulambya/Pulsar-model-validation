# 🌌 Pulsar Glitch Distribution Model Selection

## 🔬 Scientific Context & Research Objective
Pulsars experience sudden, anomalous rotational speed-ups known as **glitches**, driven by internal superfluid pinning vortices and crustal fractures. Understanding the exact statistical distribution of glitch size scales provides critical structural data parameters regarding neutron star interior architectures.

This repository implements a **Bayesian Model Selection Suite** engineered to evaluate whether local micro-glitch events match a continuous single-component distribution or adhere to a complex **Bimodal Exponential Mixture** layout model. The selection architecture employs an automated **Expectation-Maximization (EM) loop** and isolates optimal fitting trends via **Bayesian Information Criterion (BIC)** comparisons.

---

## 🛠️ Statistical Methodologies & Math Models
* **Expectation-Maximization Algorithm**: Iteratively updates hidden latent responsibility variables to segment separate glitch scale clusters cleanly.
* **Bayesian Information Criterion (BIC)**: Penalizes model complexity vs parameter dimensions to guard against overfitting tendencies:
  \[\text{BIC} = k \cdot \ln(n) - 2 \cdot \ln(\hat{L})\]
* **Kass & Raftery Delta Evaluation**: Maps raw numerical ΔBIC values into categorical significance weights to verify selection robustness levels.

---

## 💻 Sample Execution Snippet
```python
import numpy as np
from pulsar_model_validation import PulsarGlitchValidator

# Simulate a synthetic pulsar micro-glitch size dataset profile
np.random.seed(42)
mock_glitches = np.concatenate([
    np.random.exponential(scale=1.5, size=80),
    np.random.exponential(scale=6.0, size=40)
])

# Initialize the validation suite
validator = PulsarGlitchValidator(glitch_sizes=mock_glitches)

# Execute model selection analysis
selection_summary = validator.evaluate_best_fit()
print(selection_summary)
```

---
_Note: Developed as an open-source analytical tool for computational astrophysics workflows._
