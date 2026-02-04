# NC-CSF: Negative Control Causal Forest

A Python implementation of Causal Forests with Negative Control adjustment for handling unmeasured confounding. Built on top of [EconML](https://github.com/py-why/EconML)'s `CausalForestDML`.

> **Quick Reference:**
> - **Models**: All estimators are defined in [`models.py`](models.py)
> - **Run Tests**: `python evaluate_performance.py`

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Theory](#theory)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)

---

## Overview

### The Problem: Unmeasured Confounding

Standard causal inference methods assume **no unmeasured confounding**—that all variables affecting both treatment and outcome are observed. When this assumption fails, treatment effect estimates are biased.

### The Solution: Negative Controls

**Negative controls** are proxy variables that allow us to adjust for unmeasured confounding:

- **Z (Negative Control Exposure)**: A variable affected by the unmeasured confounder U, but does NOT directly affect the outcome Y
- **W (Negative Control Outcome)**: A variable affected by the unmeasured confounder U, but is NOT directly affected by treatment A

```
       U (unmeasured)
      /|\
     / | \
    ↓  ↓  ↓
    Z  A→→Y
       ↑  ↑
       |  |
       W──┘
```

### What NC-CSF Does

NC-CSF combines:
1. **Bridge function estimation** to remove confounding bias using negative controls
2. **Causal Forest** to estimate heterogeneous treatment effects (CATE)

---

## Installation

```bash
# Required dependencies
pip install numpy scikit-learn econml

# Optional for experiments
pip install pandas matplotlib
```

---

## Quick Start

```python
from nc_csf import NCCausalForestDML, BaselineCausalForestDML
import numpy as np

# Your data
# X: covariates, A: treatment, Y: outcome, Z: NC exposure, W: NC outcome
X, A, Y, Z, W = load_your_data()

# Fit NC-CSF
nccsf = NCCausalForestDML(n_estimators=200, cv=5, random_state=42)
nccsf.fit(Y=Y, T=A, X=X, Z=Z, W=W)

# Estimate treatment effects
cate = nccsf.effect(X)
print(f"Average treatment effect: {cate.mean():.3f}")

# Compare with baseline (ignores negative controls)
baseline = BaselineCausalForestDML(n_estimators=200, random_state=42)
baseline.fit(Y=Y, T=A, X=X)
cate_biased = baseline.effect(X)
```

---

## Theory

### Standard DML (Double Machine Learning)

Standard CausalForestDML estimates CATE by:

1. **Nuisance estimation**: Learn E[Y|X,W] and E[T|X,W]
2. **Residualization**: Compute Y_res = Y - E[Y|X,W] and T_res = T - E[T|X,W]
3. **Final stage**: Solve moment equation locally via CausalForest

$$E[(Y_{res} - \tau(X) \cdot T_{res}) \cdot T_{res} \mid X] = 0$$

### NC-CSF Bridge Functions

NC-CSF replaces standard nuisance estimation with **bridge functions**:

| Function | Definition | Purpose |
|----------|------------|---------|
| $q(Z,X)$ | $P(A=1 \mid Z, X)$ | Action bridge: how NC exposure predicts treatment |
| $h_1(W,X)$ | $E[Y \mid W, X, A=1]$ | Outcome bridge for treated |
| $h_0(W,X)$ | $E[Y \mid W, X, A=0]$ | Outcome bridge for control |
| $m(Z,W,X)$ | $q \cdot h_1 + (1-q) \cdot h_0$ | Combined bridge function |

**Residuals become:**
- $Y_{res} = Y - m(Z,W,X)$
- $A_{res} = A - q(Z,X)$

These residuals remove the confounding bias from U, allowing unbiased CATE estimation.

### Why This Works (Intuition)

1. **Z captures U's effect on A**: By modeling P(A|Z,X), we capture the confounded part of treatment assignment
2. **W captures U's effect on Y**: By modeling E[Y|W,X,A], we capture the confounded part of the outcome
3. **Combining them**: The bridge function m(Z,W,X) predicts what Y would be due to confounding, so subtracting it removes the bias

---

## Architecture

### Class Hierarchy

```
econml._ortho_learner._OrthoLearner
    │
    └── econml.dml._rlearner._RLearner
            │
            └── econml.dml.dml._BaseDML
                    │
                    └── econml.dml.causal_forest.CausalForestDML
                            │
                            ├── NCCausalForestDML        ← NC bridge functions
                            ├── NCCausalForestDMLOracle  ← Uses true U (benchmark)
                            └── BaselineCausalForestDML  ← Ignores NC (comparison)
```

### Key Overrides

| Method | Parent | Override Purpose |
|--------|--------|------------------|
| `_gen_ortho_learner_model_nuisance()` | `_OrthoLearner` | Return `_NCModelNuisance` instead of standard `_ModelNuisance` |
| `fit()` | `CausalForestDML` | Accept Z parameter; call `_OrthoLearner.fit()` directly |

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      NCCausalForestDML                          │
├─────────────────────────────────────────────────────────────────┤
│  Inherits from: CausalForestDML                                 │
│                                                                 │
│  Custom Components:                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  _NCModelNuisance                                        │   │
│  │  ├── q_model: LogisticRegression  (P(A=1|Z,X))         │   │
│  │  ├── h1_model: RandomForestRegressor (E[Y|W,X,A=1])    │   │
│  │  └── h0_model: RandomForestRegressor (E[Y|W,X,A=0])    │   │
│  │                                                         │   │
│  │  Methods:                                               │   │
│  │  ├── train(Y, T, X, W, Z) → fits q, h1, h0             │   │
│  │  └── predict(Y, T, X, W, Z) → returns (Y_res, A_res)   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Inherited Components (from econml):                            │
│  ├── _crossfit(): K-fold cross-validation for nuisances        │
│  ├── CausalForest: Final stage CATE estimation                 │
│  └── effect(): Predict treatment effects                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Complete Pipeline Diagram

```
User calls:
    nccsf.fit(Y=Y, T=A, X=X, Z=Z_nc, W=W_nc)
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  NCCausalForestDML.fit() [@override]                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Validates X, Z, W are provided                                           │
│  • Reshapes Z, W to 2D arrays                                               │
│  • Calls _OrthoLearner.fit(self, Y, T, X=X, W=W_nc, Z=Z_nc, ...)           │
│    (bypasses CausalForestDML.fit which doesn't accept Z)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  _OrthoLearner.fit() [econml/_ortho_learner.py]                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  1. _prefit() →                                                             │
│     • Calls _gen_ortho_learner_model_nuisance() [OUR OVERRIDE]             │
│     • Returns _NCModelNuisance(q_model, h_model)                           │
│                                                                             │
│  2. _fit_nuisances() →                                                      │
│     • Creates K-fold splits                                                 │
│     • Calls _crossfit(nuisance_model, folds, Y, T, X=X, W=W_nc, Z=Z_nc)    │
└─────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  _crossfit() [econml/_ortho_learner.py]                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  For each fold k in K folds:                                                │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐ │
│    │  _NCModelNuisance.train(Y[train], T[train], X[train],              │ │
│    │                         W=W_nc[train], Z=Z_nc[train])              │ │
│    │  ───────────────────────────────────────────────────────────────── │ │
│    │  Inside train():                                                   │ │
│    │    • A = T (treatment)                                             │ │
│    │    • XZ = [X, Z_nc]  (features for q model)                       │ │
│    │    • XW = [X, W_nc]  (features for h models)                      │ │
│    │                                                                    │ │
│    │    • q_model.fit(XZ, A)      → learns P(A=1 | Z_nc, X)            │ │
│    │    • h1_model.fit(XW[A=1], Y[A=1]) → learns E[Y | W_nc, X, A=1]   │ │
│    │    • h0_model.fit(XW[A=0], Y[A=0]) → learns E[Y | W_nc, X, A=0]   │ │
│    └─────────────────────────────────────────────────────────────────────┘ │
│                               │                                             │
│                               ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐ │
│    │  _NCModelNuisance.predict(Y[val], T[val], X[val],                  │ │
│    │                           W=W_nc[val], Z=Z_nc[val])                │ │
│    │  ───────────────────────────────────────────────────────────────── │ │
│    │  Inside predict():                                                 │ │
│    │    • q_pred = q_model.predict_proba(XZ)[:, 1]  → P(A=1|Z_nc,X)    │ │
│    │    • h1_pred = h1_model.predict(XW)            → E[Y|W_nc,X,A=1]  │ │
│    │    • h0_pred = h0_model.predict(XW)            → E[Y|W_nc,X,A=0]  │ │
│    │                                                                    │ │
│    │    • m_pred = q_pred * h1_pred + (1-q_pred) * h0_pred             │ │
│    │              ↑ Bridge function combination                        │ │
│    │                                                                    │ │
│    │    • Y_res = Y - m_pred        (outcome residual)                 │ │
│    │    • A_res = A - q_pred        (treatment residual)               │ │
│    │                                                                    │ │
│    │    return (Y_res, A_res)       ← Out-of-sample residuals          │ │
│    └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Result: nuisances = (Y_res_all, A_res_all) for all samples                │
└─────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  _OrthoLearner._fit_final() → _CausalForestFinalWrapper.fit()              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Receives: Y, T, X, nuisances=(Y_res, A_res)                               │
│                                                                             │
│  CausalForest.fit(X, T_res=A_res, Y_res=Y_res)                             │
│                                                                             │
│  At each leaf, solves the moment equation:                                  │
│    E[(Y_res - τ(X) · A_res) · A_res | X ∈ leaf] = 0                        │
│                                                                             │
│    τ_leaf = Σ(Y_res · A_res) / Σ(A_res²)                                   │
│           = local weighted OLS of Y_res on A_res                           │
└─────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  User calls: nccsf.effect(X_new)                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Forest predicts τ(X_new) by averaging leaf values across trees          │
│  • Returns CATE estimates for each sample                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Transformation Summary

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| **User** | `Y, A, X, Z, W` | - | Raw observed data |
| **fit()** | Same | Reshaped arrays | Validation; reshape Z, W to 2D |
| **train()** | Train fold data | Fitted q, h1, h0 | Fit bridge function models |
| **predict()** | Val fold data | `(Y_res, A_res)` | Compute out-of-sample residuals |
| **_crossfit()** | Per-fold residuals | Full `(Y_res, A_res)` | Aggregate all residuals |
| **_fit_final()** | `X, Y_res, A_res` | Fitted CausalForest | Solve moment equation in leaves |
| **effect()** | `X_new` | `τ(X_new)` | Forest prediction of CATE |

### Variable Mapping

| NC Literature | econml Parameter | Description |
|--------------|------------------|-------------|
| Z (NC exposure) | `Z` | Proxy for U affecting treatment A |
| W (NC outcome) | `W` | Proxy for U affecting outcome Y |
| X (covariates) | `X` | Effect modifiers for heterogeneity |
| A (treatment) | `T` | Binary treatment indicator |
| Y (outcome) | `Y` | Observed outcome |

> **Note**: econml's `W` normally means "controls" and `Z` means "instruments", but we repurpose these parameter slots to pass our NC variables through the econml infrastructure.

---

## API Reference

### NCCausalForestDML

```python
class NCCausalForestDML(CausalForestDML):
    """
    Negative Control Causal Forest for CATE estimation under unmeasured confounding.
    """
    
    def __init__(
        self,
        q_model=None,           # Model for P(A=1|Z,X), default: LogisticRegression
        h_model=None,           # Model for E[Y|W,X,A], default: RandomForestRegressor
        n_estimators=100,       # Number of trees in CausalForest
        cv=3,                   # Number of cross-fitting folds
        min_samples_leaf=5,     # Minimum samples per leaf
        max_depth=None,         # Maximum tree depth
        random_state=None,      # Random seed
        **kwargs                # Additional CausalForestDML arguments
    ):
        ...
    
    def fit(
        self,
        Y,                      # Outcome array (n,)
        T,                      # Treatment array (n,) - binary
        *,
        X,                      # Covariates (n, p) - required
        Z,                      # NC exposure (n, d_z) - required
        W,                      # NC outcome (n, d_w) - required
        sample_weight=None,     # Sample weights
        inference=None          # Inference method
    ) -> self:
        """Fit the NC-CSF model."""
        ...
    
    def fit_nc(
        self, X, A, Y, Z, W, **kwargs
    ) -> self:
        """Convenience method with NC argument order."""
        ...
    
    def effect(
        self,
        X,                      # Covariates at which to estimate effects
        T0=0,                   # Baseline treatment
        T1=1                    # Target treatment
    ) -> np.ndarray:
        """Estimate CATE τ(X) = E[Y(1) - Y(0) | X]."""
        ...
    
    def const_marginal_effect(self, X) -> np.ndarray:
        """Alias for effect()."""
        ...
```

### NCCausalForestDMLOracle

```python
class NCCausalForestDMLOracle(CausalForestDML):
    """
    Oracle NC-CSF that uses true confounder U directly (for benchmarking).
    """
    
    def fit(
        self,
        Y, T, *, X,
        W,                      # TRUE confounder U (not NC outcome!)
        Z=None,                 # Ignored
        **kwargs
    ) -> self:
        ...
    
    def fit_oracle(self, X, A, Y, U, **kwargs) -> self:
        """Convenience method: fit(Y=Y, T=A, X=X, W=U)."""
        ...
```

### BaselineCausalForestDML

```python
class BaselineCausalForestDML(CausalForestDML):
    """
    Standard CausalForestDML that ignores negative controls (for comparison).
    """
    
    def fit_baseline(self, X, A, Y, Z=None, W=None, **kwargs) -> self:
        """Fit ignoring Z and W."""
        ...
```

### _NCModelNuisance (Internal)

```python
class _NCModelNuisance:
    """
    Nuisance model implementing bridge function estimation.
    Follows econml's interface for _crossfit().
    """
    
    def train(
        self, is_selecting, folds,
        Y, T, X=None, W=None, Z=None,
        sample_weight=None, groups=None
    ) -> self:
        """Fit q, h1, h0 models on training data."""
        ...
    
    def predict(
        self, Y, T, X=None, W=None, Z=None,
        sample_weight=None, groups=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (Y_res, A_res) residuals."""
        ...
```

---

## Evaluation

### Running Experiments

```bash
cd /Users/ryan/Desktop/causal\ rf/nc_csf
python evaluate_performance.py
```

### Expected Results

With synthetic data (confounding strength γ_u = 1.5):

| Model | RMSE | Improvement |
|-------|------|-------------|
| Baseline | ~0.37 | - |
| NC-CSF | ~0.27 | ~27% over baseline |
| Oracle | ~0.22 | Upper bound |

NC-CSF typically closes **60-75%** of the gap between baseline and oracle.

### Metrics

- **RMSE**: Root mean squared error vs true CATE
- **MAE**: Mean absolute error
- **Bias**: Mean(predicted - true)
- **Pearson r**: Correlation with true CATE

---

## Project Structure

```
nc_csf/
├── __init__.py              # Package exports
├── models.py                # Main implementation
│   ├── _NCModelNuisance     # Bridge function nuisance model
│   ├── NCCausalForestDML    # Main NC-CSF estimator
│   ├── NCCausalForestDMLOracle  # Oracle benchmark
│   └── BaselineCausalForestDML  # Baseline comparison
├── data_generation.py       # Synthetic data generation
├── evaluate_performance.py  # Evaluation experiments
└── README.md               # This file
```

---

## References

1. **Causal Forests**: Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of Statistics*.

2. **Double Machine Learning**: Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*.

3. **Negative Controls**: Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika*.

4. **EconML**: [https://github.com/py-why/EconML](https://github.com/py-why/EconML)

---

## License

MIT License
