"""
NC-CSF Model Implementations (Built on EconML)

This version maximally leverages econml's infrastructure by:
1. Implementing _NCModelNuisance with train()/predict() interface
2. Overriding _gen_ortho_learner_model_nuisance() to return our NC nuisance model
3. Letting econml's _crossfit() handle cross-fitting automatically

The only difference from CausalForestDML is the nuisance model - everything else
(cross-fitting, final stage fitting) is inherited.

Class Hierarchy:
    _OrthoLearner (econml._ortho_learner)
        └── _RLearner (econml.dml._rlearner)
            └── _BaseDML (econml.dml.dml)
                └── CausalForestDML (econml.dml.causal_forest)
                    ├── NCCausalForestDML (this file) - uses NC bridge functions
                    ├── NCCausalForestDMLOracle (this file) - uses true U
                    └── BaselineCausalForestDML (this file) - ignores NC

Key Overrides:
    - _gen_ortho_learner_model_nuisance(): Returns _NCModelNuisance instead of _ModelNuisance
    - fit(): Calls _OrthoLearner.fit() directly to pass Z parameter (NC exposure)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
from typing import override  # Python 3.12+

# EconML imports
from econml.dml import CausalForestDML
from econml._ortho_learner import _OrthoLearner
from econml.utilities import filter_none_kwargs

# Survival analysis imports
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter


# =============================================================================
# NC Nuisance Model (follows econml's _ModelNuisance interface)
# =============================================================================

class _NCModelNuisance:
    """
    NC nuisance model that follows econml's interface for _crossfit().
    
    Replaces the standard _ModelNuisance which estimates E[Y|X,W] and E[T|X,W].
    Instead estimates bridge functions:
        - q(Z,X) = P(A=1 | Z, X)  
        - h(W,X,a) = E[Y | W, X, A=a]
        - m(Z,W,X) = q * h(W,X,1) + (1-q) * h(W,X,0)
    
    Interface required by _crossfit():
        - train(is_selecting, folds, Y, T, X=X, W=W, Z=Z, ...)
        - predict(Y, T, X=X, W=W, Z=Z, ...)
    
    Note: econml's W and Z have different meanings than NC literature:
        - We use econml's Z for NC exposure (our Z)
        - We use econml's W for NC outcome (our W)
    """
    
    def __init__(self, q_model, h_model):
        """
        Parameters
        ----------
        q_model : estimator
            Model for P(A=1 | Z, X). Must have fit() and predict_proba().
        h_model : estimator  
            Model for E[Y | W, X, A=a]. Must have fit() and predict().
        """
        self._q_model_template = q_model
        self._h_model_template = h_model
        
        # Fitted models (set during train)
        self._q_model = None
        self._h1_model = None  # E[Y | W, X, A=1]
        self._h0_model = None  # E[Y | W, X, A=0]
    
    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, 
              sample_weight=None, groups=None):
        """
        Fit the NC bridge function models.
        
        Parameters (following econml's interface)
        ----------
        is_selecting : bool
            Whether this is model selection phase (we ignore this for NC)
        folds : list or None
            Cross-validation folds (we ignore this, _crossfit handles it)
        Y : array (n,)
            Outcome
        T : array (n,) or (n, d_t)  
            Treatment (we use the raw binary T, not one-hot)
        X : array (n, p)
            Covariates for heterogeneity
        W : array (n, d_w)
            In NC context: negative control OUTCOME proxy
        Z : array (n, d_z)
            In NC context: negative control EXPOSURE proxy
        sample_weight : array (n,), optional
            Sample weights
        groups : array, optional
            Group labels (ignored)
        """
        # Handle treatment - extract binary if one-hot encoded
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        if np.max(A) > 1:  # Likely one-hot, take first column
            A = T[:, 0] if T.ndim > 1 else T
        A = A.ravel()
        
        # Ensure Z and W are 2D
        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if W is not None and W.ndim == 1:
            W = W.reshape(-1, 1)
        
        # Build feature matrices
        # q model uses (X, Z) to predict A
        XZ = np.column_stack([X, Z]) if Z is not None else X
        # h models use (X, W) to predict Y
        XW = np.column_stack([X, W]) if W is not None else X
        
        # Fit q model: P(A=1 | Z, X)
        self._q_model = clone(self._q_model_template)
        if sample_weight is not None:
            self._q_model.fit(XZ, A, sample_weight=sample_weight)
        else:
            self._q_model.fit(XZ, A)
        
        # Fit h1 model: E[Y | W, X, A=1]
        treated_mask = A == 1
        self._h1_model = clone(self._h_model_template)
        if treated_mask.sum() > 10:
            Y_flat = Y.ravel()
            if sample_weight is not None:
                self._h1_model.fit(XW[treated_mask], Y_flat[treated_mask],
                                   sample_weight=sample_weight[treated_mask])
            else:
                self._h1_model.fit(XW[treated_mask], Y_flat[treated_mask])
        
        # Fit h0 model: E[Y | W, X, A=0]
        control_mask = A == 0
        self._h0_model = clone(self._h_model_template)
        if control_mask.sum() > 10:
            Y_flat = Y.ravel()
            if sample_weight is not None:
                self._h0_model.fit(XW[control_mask], Y_flat[control_mask],
                                   sample_weight=sample_weight[control_mask])
            else:
                self._h0_model.fit(XW[control_mask], Y_flat[control_mask])
        
        return self
    
    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        """
        Compute residuals Y_res and A_res using fitted bridge functions.
        
        Returns
        -------
        Y_res : array (n,)
            Y - m(Z,W,X) where m = q*h1 + (1-q)*h0
        A_res : array (n, 1)
            A - q(Z,X)
        """
        # Handle treatment
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        if np.max(A) > 1:
            A = T[:, 0] if T.ndim > 1 else T
        A = A.ravel()
        
        # Ensure Z and W are 2D
        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if W is not None and W.ndim == 1:
            W = W.reshape(-1, 1)
        
        # Build feature matrices
        XZ = np.column_stack([X, Z]) if Z is not None else X
        XW = np.column_stack([X, W]) if W is not None else X
        
        # Predict q(Z,X) = P(A=1 | Z, X)
        q_pred = self._q_model.predict_proba(XZ)[:, 1]
        q_pred = np.clip(q_pred, 0.01, 0.99)  # Avoid extreme propensities
        
        # Predict h1 and h0
        h1_pred = self._h1_model.predict(XW)
        h0_pred = self._h0_model.predict(XW)
        
        # Compute m(Z,W,X) = q * h1 + (1-q) * h0
        m_pred = q_pred * h1_pred + (1 - q_pred) * h0_pred
        
        # Compute residuals
        Y_res = Y.ravel() - m_pred
        A_res = (A - q_pred).reshape(-1, 1)  # Must be 2D for econml
        
        return Y_res, A_res


# =============================================================================
# NC-CSF using econml's _crossfit()
# =============================================================================

class NCCausalForestDML(CausalForestDML):
    """
    Negative Control Causal Survival Forest extending EconML's CausalForestDML.
    
    Inheritance: CausalForestDML -> _BaseDML -> _RLearner -> _OrthoLearner
    
    This implementation maximally leverages econml by:
    1. Overriding _gen_ortho_learner_model_nuisance() to return _NCModelNuisance
    2. Letting the parent class handle cross-fitting via _crossfit()
    3. Reusing the final CausalForest fitting entirely
    
    The only custom code is the nuisance model - everything else is econml.
    
    Parameters
    ----------
    q_model : estimator, default=None
        Model for P(A=1|Z,X). Must have fit() and predict_proba().
        If None, uses LogisticRegression.
    h_model : estimator, default=None
        Model for E[Y|W,X,A]. Must have fit() and predict().
        If None, uses RandomForestRegressor.
    **kwargs : dict
        Additional arguments passed to CausalForestDML.
        
    Variable Mapping (NC literature -> econml parameter slot):
    ---------------------------------------------------------
    NC Literature    econml Slot    Description
    -------------    -----------    -----------
    Z (NC exposure)  Z parameter    Proxy for U affecting treatment A
    W (NC outcome)   W parameter    Proxy for U affecting outcome Y
    X (covariates)   X parameter    Covariates for heterogeneity
    A (treatment)    T parameter    Binary treatment indicator
    Y (outcome)      Y parameter    Observed outcome
    
    Note: econml's W normally means "controls" and Z means "instruments",
    but we repurpose these slots to pass our NC variables through.
    """
    
    def __init__(
        self,
        q_model=None,
        h_model=None,
        **kwargs
    ):
        # NC-specific models
        self._q_model_template = q_model or LogisticRegression(max_iter=1000)
        self._h_model_template = h_model or RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20
        )
        
        # Force discrete treatment for NC-CSF
        kwargs['discrete_treatment'] = True
        # Use heterogeneity criterion consistent with GRF paper
        kwargs['criterion'] = 'het'
        
        # Don't pass model_y and model_t - we override nuisance estimation
        kwargs.pop('model_y', None)
        kwargs.pop('model_t', None)
        
        super().__init__(**kwargs)
    
    @override
    def _gen_ortho_learner_model_nuisance(self):
        """
        Override: _OrthoLearner._gen_ortho_learner_model_nuisance()
        
        Returns _NCModelNuisance instead of the standard _ModelNuisance.
        This is called by _OrthoLearner._fit_nuisances() which passes
        the result to _crossfit() for cross-validated nuisance estimation.
        """
        return _NCModelNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template
        )
    
    @override
    def fit(self, Y, T, *, X=None, W=None, Z=None, 
            sample_weight=None, freq_weight=None, sample_var=None,
            groups=None, cache_values=False, inference=None):
        """
        Override: CausalForestDML.fit() -> _OrthoLearner.fit()
        
        We override to:
        1. Accept Z parameter (CausalForestDML.fit() doesn't accept Z)
        2. Call _OrthoLearner.fit() directly to pass Z through
        
        The parent's fit() flow then handles everything:
        - _prefit() -> _gen_ortho_learner_model_nuisance() [our override]
        - _fit_nuisances() -> _crossfit() with our _NCModelNuisance
        - _fit_final() -> fit CausalForest on residuals
        
        Parameters (NC notation -> econml slot)
        ---------------------------------------
        Y : array (n,) - Observed outcome
        T : array (n,) - Treatment A
        X : array (n, p) - Covariates for heterogeneity
        Z : array (n, d_z) - NC exposure proxy (passed via econml's Z slot)
        W : array (n, d_w) - NC outcome proxy (passed via econml's W slot)
        """
        if X is None:
            raise ValueError("X is required for NC-CSF")
        if Z is None:
            raise ValueError("Z (negative control exposure) is required for NC-CSF")
        if W is None:
            raise ValueError("W (negative control outcome) is required for NC-CSF")
        
        # Convert to proper shapes
        Z = np.atleast_1d(np.asarray(Z))
        W = np.atleast_1d(np.asarray(W))
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        
        # Call _OrthoLearner.fit() directly (bypassing CausalForestDML.fit() which 
        # doesn't accept Z). This triggers the full fitting pipeline.
        return _OrthoLearner.fit(
            self, Y, T, X=X, W=W, Z=Z,
            sample_weight=sample_weight,
            freq_weight=freq_weight,
            sample_var=sample_var,
            groups=groups,
            cache_values=cache_values,
            inference=inference
        )
    
    def fit_nc(self, X, A, Y, Z, W, **kwargs):
        """
        Convenience method with NC-CSF natural argument order.
        
        Parameters
        ----------
        X : array (n, p)
            Covariates for heterogeneity.
        A : array (n,)
            Binary treatment indicator.
        Y : array (n,)
            Observed outcome.
        Z : array (n,) or (n, d_z)
            Negative control exposure.
        W : array (n,) or (n, d_w)
            Negative control outcome.
        **kwargs : dict
            Additional arguments passed to fit().
        """
        return self.fit(Y=Y, T=A, X=X, Z=Z, W=W, **kwargs)


# =============================================================================
# Oracle NC-CSF (uses true U instead of proxies)
# =============================================================================

class _OracleModelNuisance:
    """
    Oracle nuisance model that uses true confounder U directly.
    
    Instead of bridge functions with proxies, directly models:
        - q(U,X) = P(A=1 | U, X)
        - h(U,X,a) = E[Y | U, X, A=a]
        - m(U,X) = q * h(U,X,1) + (1-q) * h(U,X,0)
    """
    
    def __init__(self, q_model, h_model):
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_model = None
        self._h1_model = None
        self._h0_model = None
    
    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None,
              sample_weight=None, groups=None):
        """
        Fit oracle nuisances using U (passed as W).
        
        Note: We pass U through the W parameter since that's what econml expects.
        """
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        U = W  # U is passed as W
        
        if U is not None and U.ndim == 1:
            U = U.reshape(-1, 1)
        
        XU = np.column_stack([X, U]) if U is not None else X
        
        # Fit q model: P(A=1 | U, X)
        self._q_model = clone(self._q_model_template)
        self._q_model.fit(XU, A)
        
        # Fit h1: E[Y | U, X, A=1]
        treated_mask = A == 1
        self._h1_model = clone(self._h_model_template)
        if treated_mask.sum() > 10:
            self._h1_model.fit(XU[treated_mask], Y.ravel()[treated_mask])
        
        # Fit h0: E[Y | U, X, A=0]
        control_mask = A == 0
        self._h0_model = clone(self._h_model_template)
        if control_mask.sum() > 10:
            self._h0_model.fit(XU[control_mask], Y.ravel()[control_mask])
        
        return self
    
    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        U = W
        
        if U is not None and U.ndim == 1:
            U = U.reshape(-1, 1)
        
        XU = np.column_stack([X, U]) if U is not None else X
        
        q_pred = self._q_model.predict_proba(XU)[:, 1]
        q_pred = np.clip(q_pred, 0.01, 0.99)
        
        h1_pred = self._h1_model.predict(XU)
        h0_pred = self._h0_model.predict(XU)
        
        m_pred = q_pred * h1_pred + (1 - q_pred) * h0_pred
        
        Y_res = Y.ravel() - m_pred
        A_res = (A - q_pred).reshape(-1, 1)
        
        return Y_res, A_res


class NCCausalForestDMLOracle(CausalForestDML):
    """
    NC-CSF with oracle access to the true confounder U.
    
    Inheritance: CausalForestDML -> _BaseDML -> _RLearner -> _OrthoLearner
    
    Uses U directly instead of proxies Z and W, representing the best
    possible performance with perfect knowledge of the confounder.
    
    Variable Mapping:
    ----------------
    U (true confounder) is passed via econml's W parameter slot.
    Z parameter is not used (set to None).
    """
    
    def __init__(self, q_model=None, h_model=None, **kwargs):
        self._q_model_template = q_model or LogisticRegression(max_iter=1000)
        self._h_model_template = h_model or RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20
        )
        
        kwargs['discrete_treatment'] = True
        # Use heterogeneity criterion consistent with GRF paper
        kwargs['criterion'] = 'het'
        kwargs.pop('model_y', None)
        kwargs.pop('model_t', None)
        
        super().__init__(**kwargs)
    
    @override
    def _gen_ortho_learner_model_nuisance(self):
        """
        Override: _OrthoLearner._gen_ortho_learner_model_nuisance()
        
        Returns _OracleModelNuisance which uses true U directly.
        """
        return _OracleModelNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template
        )
    
    @override
    def fit(self, Y, T, *, X=None, W=None, Z=None,
            sample_weight=None, freq_weight=None, sample_var=None,
            groups=None, cache_values=False, inference=None):
        """
        Override: CausalForestDML.fit() -> _OrthoLearner.fit()
        
        Parameters (Oracle notation -> econml slot)
        -------------------------------------------
        Y : array - Observed outcome
        T : array - Treatment A
        X : array - Covariates for heterogeneity
        W : array - TRUE confounder U (passed via econml's W slot)
        Z : array - Ignored (not needed for oracle)
        """
        if X is None:
            raise ValueError("X is required for Oracle NC-CSF")
        if W is None:
            raise ValueError("W (true confounder U) is required for Oracle NC-CSF")
        
        W = np.atleast_1d(np.asarray(W))
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        
        # Call _OrthoLearner.fit() directly
        return _OrthoLearner.fit(
            self, Y, T, X=X, W=W, Z=None,
            sample_weight=sample_weight,
            freq_weight=freq_weight,
            sample_var=sample_var,
            groups=groups,
            cache_values=cache_values,
            inference=inference
        )
    
    def fit_oracle(self, X, A, Y, U, **kwargs):
        """Convenience method: fit(Y=Y, T=A, X=X, W=U)"""
        return self.fit(Y=Y, T=A, X=X, W=U, **kwargs)


# =============================================================================
# Baseline CausalForestDML (for comparison)
# =============================================================================

class BaselineCausalForestDML(CausalForestDML):
    """
    Baseline CausalForestDML that IGNORES negative controls.
    
    Demonstrates the bias from unmeasured confounding when
    negative controls are not used.
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault('model_y', RandomForestRegressor(n_estimators=100, min_samples_leaf=20))
        kwargs.setdefault('model_t', RandomForestClassifier(n_estimators=100, min_samples_leaf=20))
        kwargs.setdefault('discrete_treatment', True)
        # Use heterogeneity criterion consistent with GRF paper
        kwargs.setdefault('criterion', 'het')
        
        super().__init__(**kwargs)
    
    def fit_baseline(self, X, A, Y, Z=None, W=None, verbose=False, **kwargs):
        """
        Fit baseline CausalForestDML (ignores Z and W).
        """
        # verbose is ignored (for API compatibility)
        return self.fit(Y=Y, T=A, X=X, **kwargs)


# =============================================================================
# Kaplan-Meier IPCW helper
# =============================================================================

def _fit_kaplan_meier_censoring(time, event):
    """
    Fit unconditional Kaplan-Meier for the censoring distribution.
    
    Under C ⊥ (T, W, A, X, Z), a single KM on the full dataset
    consistently estimates S^C(t).
    
    Parameters
    ----------
    time : array (n,)
        Observed times Y_i = min(T_i, C_i).
    event : array (n,)
        Event indicator Delta_i (1=event, 0=censored).
    
    Returns
    -------
    km_times : array
        Sorted unique event times.
    km_surv : array
        S^C evaluated at km_times.
    """
    kmf = KaplanMeierFitter()
    # Flip indicator: censoring is the "event" for S^C
    kmf.fit(durations=time, event_observed=1 - event)
    # Extract survival function on the timeline
    km_times = kmf.survival_function_.index.values
    km_surv = kmf.survival_function_.values.ravel()
    return km_times, km_surv


def _evaluate_sc(time_points, km_times, km_surv, clip_min=0.01):
    """
    Evaluate S^C(t) at arbitrary time points via step-function interpolation.
    
    Parameters
    ----------
    time_points : array (n,)
        Times at which to evaluate S^C.
    km_times : array
        KM timeline (from _fit_kaplan_meier_censoring).
    km_surv : array
        KM survival values.
    clip_min : float
        Minimum value to avoid division by near-zero.
    
    Returns
    -------
    sc : array (n,)
        S^C evaluated at each time point, clipped from below.
    """
    # Step function: S^C(t) = S^C(t_k) where t_k is the largest KM time <= t
    idx = np.searchsorted(km_times, time_points, side='right') - 1
    idx = np.clip(idx, 0, len(km_surv) - 1)
    sc = km_surv[idx]
    return np.clip(sc, clip_min, 1.0)


def _compute_ipcw_pseudo_outcome(Y_time, Delta, km_times, km_surv):
    """
    Compute IPCW pseudo-outcome: Ỹ_i = Y_i * Delta_i / S^C(Y_i).
    
    Under independent censoring:
        E[Ỹ_i | X_i, A_i, W_i, Z_i] = E[T_i | X_i, A_i, W_i, Z_i]
    
    Parameters
    ----------
    Y_time : array (n,)
        Observed time.
    Delta : array (n,)
        Event indicator.
    km_times, km_surv : arrays
        Kaplan-Meier censoring survival function.
    
    Returns
    -------
    Y_tilde : array (n,)
        IPCW pseudo-outcome.
    """
    sc_at_y = _evaluate_sc(Y_time, km_times, km_surv)
    return Y_time * Delta / sc_at_y


def _compute_Q_from_S(S_hat, t_grid):
    """
    Compute Q(t) = E[T | T > t] via backward recursion from a survival function.
    
    Following GRF's causal_survival_forest (Cui et al. 2023):
        Q(t_k) = t_k + sum_{j>=k} S(t_j | v) * dt_{j+1} / S(t_k | v)
    
    Parameters
    ----------
    S_hat : array (n, G)
        Conditional survival function S(t_j | v_i) for each sample on the
        time grid.  Must be non-increasing across columns.
    t_grid : array (G,)
        Sorted time grid (unique event times).
    
    Returns
    -------
    Q_hat : array (n, G)
        Q(t_k | v_i) = E[T | T > t_k, v_i] for each sample and grid point.
    """
    n, G = S_hat.shape
    if G < 2:
        return np.full((n, G), t_grid[0] if G == 1 else 0.0)

    # dt[j] = t_{j} - t_{j-1}  (length G, first element = t_0)
    dt = np.diff(np.concatenate([[0.0], t_grid]))  # (G,)

    # dot_products[i, j] = S(t_j | v_i) * (t_{j+1} - t_j)  for j = 0..G-2
    dot_products = S_hat[:, :-1] * dt[1:]  # (n, G-1)

    # Cumulative sum from the right (backward recursion)
    # cum[i, k] = sum_{j=k}^{G-2} dot_products[i, j]
    cum = np.zeros((n, G))
    cum[:, 0] = dot_products.sum(axis=1)
    for k in range(1, G - 1):
        cum[:, k] = cum[:, k - 1] - dot_products[:, k - 1]
    # cum[:, G-1] stays 0

    # Q(t_k) = t_k + cum[k] / S(t_k)
    Q_hat = t_grid[None, :] + cum / np.maximum(S_hat, 1e-10)
    Q_hat[:, -1] = t_grid[-1]  # boundary: no mass beyond last grid point
    return Q_hat


def _fit_event_cox(Y_time, Delta, features, penalizer=0.01):
    """
    Fit a Cox PH model for event survival S(t | features).
    
    Parameters
    ----------
    Y_time : array (n,)
        Observed times.
    Delta : array (n,)
        Event indicator (1=event, 0=censored).
    features : array (n, p)
        Covariates [A, X, ...].
    penalizer : float
        L2 regularization strength.
    
    Returns
    -------
    cox : CoxPHFitter
        Fitted Cox model.
    col_names : list of str
        Column names used in the DataFrame.
    """
    n_ft = features.shape[1]
    col_names = [f'cxf{i}' for i in range(n_ft)]
    df = pd.DataFrame(features, columns=col_names)
    df['_duration'] = Y_time
    df['_event'] = Delta.astype(float)
    cox = CoxPHFitter(penalizer=penalizer)
    cox.fit(df, duration_col='_duration', event_col='_event')
    return cox, col_names


def _predict_S_on_grid(cox, col_names, features, t_grid):
    """
    Predict survival function on a time grid using a fitted Cox model.
    
    Parameters
    ----------
    cox : CoxPHFitter
        Fitted Cox PH model.
    col_names : list of str
        Column names matching the training DataFrame.
    features : array (n, p)
        Covariates at which to predict.
    t_grid : array (G,)
        Time grid for evaluation.
    
    Returns
    -------
    S_hat : array (n, G)
        S(t_k | v_i) for each sample i and grid point k.
    """
    pred_df = pd.DataFrame(features, columns=col_names)
    surv = cox.predict_survival_function(pred_df, times=t_grid)
    # surv is a DataFrame with shape (G, n); rows=times, cols=samples
    return np.clip(surv.values.T, 1e-10, 1.0)  # (n, G)


def _compute_ipcw_3term_Y_res(
    Y_time, Delta, m_pred, Q_hat, t_grid,
    km_times, km_surv
):
    """
    Compute the full 3-term IPCW-augmented outcome residual (CSF eq 11).

    First computes the doubly-robust pseudo-outcome (replacing censored T):

        Ỹ_i = [Δ_i Y_i + (1-Δ_i) Q_{A_i}(Y_i)] / S^C(Y_i)
               − Σ_{t_k ≤ Y_i} [dΛ^C(t_k) / S^C(t_k)] · Q_{A_i}(t_k)

    Then forms the R-learner residual: Y_res = Ỹ − m.

    IMPORTANT: m is subtracted *outside* the IPCW mechanism to avoid
    numerical errors from discretising the identity 1/S^C = 1 + ∫ dΛ^C/S^C
    on a finite time grid.

    Parameters
    ----------
    Y_time : array (n,)
        Observed time.
    Delta : array (n,)
        Event indicator (1=event, 0=censored).
    m_pred : array (n,)
        Predicted marginal outcome E[T | X, ...].
    Q_hat : array (n, G)
        E[T | T > t_k, A=A_i, v_i] on the time grid, already assembled
        per individual's treatment arm.
    t_grid : array (G,)
        Time grid.
    km_times, km_surv : arrays
        Censoring KM survival function.

    Returns
    -------
    Y_res : array (n,)
        IPCW-augmented outcome residual.
    """
    n = len(Y_time)
    G = len(t_grid)

    # --- S^C evaluated at each sample's Y_i and on the grid ---
    sc_at_Y = _evaluate_sc(Y_time, km_times, km_surv)       # (n,)
    sc_grid = _evaluate_sc(t_grid, km_times, km_surv)        # (G,)

    # --- Q evaluated at each sample's Y_i ---
    Y_idx = np.searchsorted(t_grid, Y_time, side='right') - 1
    Y_idx = np.clip(Y_idx, 0, G - 1)
    Q_at_Y = Q_hat[np.arange(n), Y_idx]                     # (n,)

    # --- Pseudo-outcome (WITHOUT subtracting m inside IPCW) ---
    # Term 1: [Δ Y + (1-Δ) Q(Y)] / S^C(Y)
    numerator = Delta * Y_time + (1 - Delta) * Q_at_Y
    term1 = numerator / sc_at_Y                              # (n,)

    # --- Augmentation integral ---
    # dΛ^C on the grid (cumulative-hazard increments of the censoring dist.)
    log_sc = -np.log(np.maximum(sc_grid, 1e-10))             # Λ^C(t_k)
    dLambda_C = np.diff(np.concatenate([[0.0], log_sc]))     # (G,)

    # Weight at each grid point: dΛ^C(t_k) / S^C(t_k)
    grid_weight = dLambda_C / np.maximum(sc_grid, 1e-10)     # (G,)

    # Integrand: grid_weight[k] * Q[i,k]   (Q only, NOT Q-m)
    integrand = grid_weight[None, :] * Q_hat                 # (n, G)

    # Sum up to Y_i: mask[i,k] = 1 iff t_grid[k] <= Y_i
    mask = np.arange(G)[None, :] <= Y_idx[:, None]           # (n, G)
    term2 = (integrand * mask).sum(axis=1)                   # (n,)

    # Pseudo-outcome Ỹ = term1 - term2
    Y_pseudo = term1 - term2

    # R-learner residual: subtract m *directly* (not through IPCW)
    Y_res = Y_pseudo - m_pred

    # Winsorise to limit influence of extreme IPCW values
    lo, hi = np.percentile(Y_res, [1, 99])
    Y_res = np.clip(Y_res, lo, hi)

    return Y_res


# =============================================================================
# NC Survival Nuisance Model (3-term IPCW, extension of _NCModelNuisance)
# =============================================================================

class _NCSurvivalModelNuisance:
    """
    NC nuisance model for right-censored survival outcomes using the
    full 3-term IPCW-augmented doubly-robust score (CSF eq 11).
    
    Under C ⊥ (T, W, A, X, Z) (independent censoring), this model:
    1. Fits unconditional Kaplan-Meier for S^C(t)
    2. Fits bridge functions q(Z,X) and h(W,X,a) using IPCW pseudo-outcomes
    3. Fits per-arm Cox PH models: S_a(t | X, W, Z) → Q_a(s|X,W,Z)
       following eq (11): Q_a(s|v) = E[T | T>s, A=a, v]
    4. Computes the 3-term IPCW-augmented outcome residual
    5. Returns (Y_res, A − q) to the CausalForest final stage
    
    Y is passed as a 2-column array [time, event] packed by NCSurvivalForestDML.
    """
    
    MAX_GRID = 500  # cap on time grid size for efficiency
    
    def __init__(self, q_model, h_model):
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_model = None
        self._h1_model = None
        self._h0_model = None
        self._km_times = None
        self._km_surv = None
        self._event_cox_1 = None   # Cox for treated arm
        self._event_cox_0 = None   # Cox for control arm
        self._cox_col_names = None
        self._t_grid = None
    
    def _unpack_Y(self, Y):
        """Unpack [time, event] from the packed Y array."""
        Y = np.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] >= 2:
            return Y[:, 0], Y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event] for survival models")
    
    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None,
              sample_weight=None, groups=None):
        """
        Fit NC bridge functions on IPCW pseudo-outcomes and per-arm Cox models.
        
        Steps:
        1. Unpack Y -> (time, event)
        2. Fit KM censoring model on (time, event)
        3. Compute Ỹ = time * event / S^C(time) for h training
        4. Fit q model: P(A=1 | Z, X)  [unchanged from non-survival]
        5. Fit h1, h0 on (W, X) -> Ỹ   [Ỹ replaces Y]
        6. Fit per-arm Cox PH: S_1(t | X,W,Z) and S_0(t | X,W,Z)
        7. Build time grid from unique event times
        """
        # Unpack survival data
        Y_time, Delta = self._unpack_Y(Y)
        
        # Handle treatment
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        if np.max(A) > 1:
            A = T[:, 0] if T.ndim > 1 else T
        A = A.ravel()
        
        # Step 1: Fit unconditional KM for censoring
        self._km_times, self._km_surv = _fit_kaplan_meier_censoring(Y_time, Delta)
        
        # Step 2: IPCW pseudo-outcome for h training
        Y_tilde = _compute_ipcw_pseudo_outcome(Y_time, Delta,
                                                self._km_times, self._km_surv)
        
        # Ensure Z and W are 2D
        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if W is not None and W.ndim == 1:
            W = W.reshape(-1, 1)
        
        # Build feature matrices
        XZ = np.column_stack([X, Z]) if Z is not None else X
        XW = np.column_stack([X, W]) if W is not None else X
        
        # Fit q model: P(A=1 | Z, X) — unaffected by censoring
        self._q_model = clone(self._q_model_template)
        if sample_weight is not None:
            self._q_model.fit(XZ, A, sample_weight=sample_weight)
        else:
            self._q_model.fit(XZ, A)
        
        # Fit h1 model: E[Ỹ | W, X, A=1]
        treated_mask = A == 1
        self._h1_model = clone(self._h_model_template)
        if treated_mask.sum() > 10:
            if sample_weight is not None:
                self._h1_model.fit(XW[treated_mask], Y_tilde[treated_mask],
                                   sample_weight=sample_weight[treated_mask])
            else:
                self._h1_model.fit(XW[treated_mask], Y_tilde[treated_mask])
        
        # Fit h0 model: E[Ỹ | W, X, A=0]
        control_mask = A == 0
        self._h0_model = clone(self._h_model_template)
        if control_mask.sum() > 10:
            if sample_weight is not None:
                self._h0_model.fit(XW[control_mask], Y_tilde[control_mask],
                                   sample_weight=sample_weight[control_mask])
            else:
                self._h0_model.fit(XW[control_mask], Y_tilde[control_mask])
        
        # Step 6: Per-arm Cox PH for Q_a (CSF eq 11)
        # Features for Cox: (X, W, Z) — no A column; separate model per arm
        surv_features = np.column_stack([X, W, Z])
        self._event_cox_1, self._cox_col_names = _fit_event_cox(
            Y_time[treated_mask], Delta[treated_mask],
            surv_features[treated_mask]
        )
        self._event_cox_0, _ = _fit_event_cox(
            Y_time[control_mask], Delta[control_mask],
            surv_features[control_mask]
        )
        
        # Step 7: Build time grid (all unique observed times for dΛ^C accuracy)
        all_times = np.sort(np.unique(Y_time))
        if len(all_times) > self.MAX_GRID:
            idx = np.linspace(0, len(all_times) - 1, self.MAX_GRID, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times
        
        return self
    
    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        """
        Compute 3-term IPCW-augmented residuals using fitted nuisances.
        
        Uses per-arm Q_a(s|X,W,Z) following CSF eq (11).
        
        Returns
        -------
        Y_res : array (n,)
            IPCW-augmented outcome residual.
        A_res : array (n, 1)
            A - q(Z,X)
        """
        # Unpack survival data
        Y_time, Delta = self._unpack_Y(Y)
        
        # Handle treatment
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        if np.max(A) > 1:
            A = T[:, 0] if T.ndim > 1 else T
        A = A.ravel()
        
        # Ensure Z and W are 2D
        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if W is not None and W.ndim == 1:
            W = W.reshape(-1, 1)
        
        # Build feature matrices
        XZ = np.column_stack([X, Z]) if Z is not None else X
        XW = np.column_stack([X, W]) if W is not None else X
        
        # --- Bridge function predictions ---
        q_pred = self._q_model.predict_proba(XZ)[:, 1]
        q_pred = np.clip(q_pred, 0.01, 0.99)
        h1_pred = self._h1_model.predict(XW)
        h0_pred = self._h0_model.predict(XW)
        m_pred = q_pred * h1_pred + (1 - q_pred) * h0_pred
        
        # --- Per-arm Q_a via Cox PH (CSF eq 11) ---
        surv_features = np.column_stack([X, W, Z])
        S_hat_1 = _predict_S_on_grid(
            self._event_cox_1, self._cox_col_names, surv_features, self._t_grid
        )
        S_hat_0 = _predict_S_on_grid(
            self._event_cox_0, self._cox_col_names, surv_features, self._t_grid
        )
        Q_hat_1 = _compute_Q_from_S(S_hat_1, self._t_grid)
        Q_hat_0 = _compute_Q_from_S(S_hat_0, self._t_grid)
        
        # Assemble Q_{A_i}: each individual uses Q for their treatment arm
        treated = (A == 1)
        Q_hat = np.where(treated[:, None], Q_hat_1, Q_hat_0)
        
        # --- 3-term IPCW Y_res ---
        Y_res = _compute_ipcw_3term_Y_res(
            Y_time, Delta, m_pred, Q_hat, self._t_grid,
            self._km_times, self._km_surv
        )
        
        A_res = (A - q_pred).reshape(-1, 1)
        
        return Y_res, A_res


# =============================================================================
# NC Survival CausalForestDML
# =============================================================================

class NCSurvivalForestDML(CausalForestDML):
    """
    Negative Control Causal Survival Forest with IPCW for right-censored data.
    
    Extends NCCausalForestDML to handle censored survival outcomes using
    Inverse Probability of Censoring Weighting (IPCW).
    
    Assumption: C ⊥ (T, W, A, X, Z)  (independent censoring)
    
    Under this assumption:
    - S^C(t) is estimated by unconditional Kaplan-Meier
    - The IPCW pseudo-outcome Ỹ_i = Y_i * Δ_i / S^C(Y_i) satisfies
      E[Ỹ_i | X,A,W,Z] = E[T_i | X,A,W,Z]
    - Bridge functions h(W,X,a) are fit on Ỹ instead of Y
    - q(Z,X) is unaffected by censoring
    
    Parameters
    ----------
    q_model : estimator, default=None
        Model for P(A=1|Z,X). Must have fit() and predict_proba().
    h_model : estimator, default=None
        Model for E[Ỹ|W,X,A]. Must have fit() and predict().
    **kwargs : dict
        Additional arguments passed to CausalForestDML.
    """
    
    def __init__(self, q_model=None, h_model=None, **kwargs):
        self._q_model_template = q_model or LogisticRegression(max_iter=1000)
        self._h_model_template = h_model or RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20
        )
        
        kwargs['discrete_treatment'] = True
        kwargs['criterion'] = 'het'
        kwargs.pop('model_y', None)
        kwargs.pop('model_t', None)
        
        super().__init__(**kwargs)
    
    @override
    def _gen_ortho_learner_model_nuisance(self):
        return _NCSurvivalModelNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template
        )
    
    @override
    def fit(self, Y, T, *, Delta=None, X=None, W=None, Z=None,
            sample_weight=None, freq_weight=None, sample_var=None,
            groups=None, cache_values=False, inference=None):
        """
        Fit NC Survival CausalForest with IPCW.
        
        Parameters
        ----------
        Y : array (n,)
            Observed time = min(T, C).
        T : array (n,)
            Treatment A (binary).
        Delta : array (n,)
            Event indicator (1=event, 0=censored).
        X : array (n, p)
            Covariates for heterogeneity.
        Z : array (n, d_z)
            Negative control exposure proxy.
        W : array (n, d_w)
            Negative control outcome proxy.
        """
        if X is None:
            raise ValueError("X is required for NC Survival Forest")
        if Z is None:
            raise ValueError("Z (negative control exposure) is required")
        if W is None:
            raise ValueError("W (negative control outcome) is required")
        if Delta is None:
            raise ValueError("Delta (event indicator) is required for survival models")
        
        # Convert to arrays
        Y = np.asarray(Y).ravel()
        Delta = np.asarray(Delta).ravel()
        Z = np.atleast_1d(np.asarray(Z))
        W = np.atleast_1d(np.asarray(W))
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        
        # Pack [time, event] into Y for the nuisance model
        Y_packed = np.column_stack([Y, Delta])
        
        return _OrthoLearner.fit(
            self, Y_packed, T, X=X, W=W, Z=Z,
            sample_weight=sample_weight,
            freq_weight=freq_weight,
            sample_var=sample_var,
            groups=groups,
            cache_values=cache_values,
            inference=inference
        )
    
    def fit_survival(self, X, A, time, event, Z, W, **kwargs):
        """
        Convenience method with natural survival argument names.
        
        Parameters
        ----------
        X : array (n, p) - Covariates
        A : array (n,)   - Treatment
        time : array (n,) - Observed time
        event : array (n,) - Event indicator
        Z : array (n,)   - NC exposure
        W : array (n,)   - NC outcome
        """
        return self.fit(Y=time, T=A, Delta=event, X=X, Z=Z, W=W, **kwargs)


# =============================================================================
# Oracle Survival CausalForestDML (uses true U with IPCW)
# =============================================================================

class _OracleSurvivalModelNuisance:
    """
    Oracle nuisance model for censored survival data using true U.
    
    Same 3-term IPCW-augmented score as _NCSurvivalModelNuisance but:
    - q, h models condition on (U, X) instead of bridge proxies
    - Per-arm Cox models for Q_a condition on (X, U)
    """
    
    MAX_GRID = 500
    
    def __init__(self, q_model, h_model):
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_model = None
        self._h1_model = None
        self._h0_model = None
        self._km_times = None
        self._km_surv = None
        self._event_cox_1 = None
        self._event_cox_0 = None
        self._cox_col_names = None
        self._t_grid = None
    
    def _unpack_Y(self, Y):
        Y = np.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] >= 2:
            return Y[:, 0], Y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event] for survival models")
    
    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None,
              sample_weight=None, groups=None):
        Y_time, Delta = self._unpack_Y(Y)
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        U = W  # U is passed via the W parameter slot
        
        # Fit KM censoring model
        self._km_times, self._km_surv = _fit_kaplan_meier_censoring(Y_time, Delta)
        Y_tilde = _compute_ipcw_pseudo_outcome(Y_time, Delta,
                                                self._km_times, self._km_surv)
        
        if U is not None and U.ndim == 1:
            U = U.reshape(-1, 1)
        XU = np.column_stack([X, U]) if U is not None else X
        
        # q model: P(A=1 | U, X)
        self._q_model = clone(self._q_model_template)
        self._q_model.fit(XU, A)
        
        # h1: E[Ỹ | U, X, A=1]
        treated_mask = A == 1
        self._h1_model = clone(self._h_model_template)
        if treated_mask.sum() > 10:
            self._h1_model.fit(XU[treated_mask], Y_tilde[treated_mask])
        
        # h0: E[Ỹ | U, X, A=0]
        control_mask = A == 0
        self._h0_model = clone(self._h_model_template)
        if control_mask.sum() > 10:
            self._h0_model.fit(XU[control_mask], Y_tilde[control_mask])
        
        # Per-arm Cox PH for Q_a(s|X,U) (CSF eq 11)
        surv_features = XU  # (X, U)
        self._event_cox_1, self._cox_col_names = _fit_event_cox(
            Y_time[treated_mask], Delta[treated_mask],
            surv_features[treated_mask]
        )
        self._event_cox_0, _ = _fit_event_cox(
            Y_time[control_mask], Delta[control_mask],
            surv_features[control_mask]
        )
        
        # Time grid (all unique times for dΛ^C accuracy)
        all_times = np.sort(np.unique(Y_time))
        if len(all_times) > self.MAX_GRID:
            idx = np.linspace(0, len(all_times) - 1, self.MAX_GRID, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times
        
        return self
    
    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        Y_time, Delta = self._unpack_Y(Y)
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        U = W
        
        if U is not None and U.ndim == 1:
            U = U.reshape(-1, 1)
        XU = np.column_stack([X, U]) if U is not None else X
        
        q_pred = self._q_model.predict_proba(XU)[:, 1]
        q_pred = np.clip(q_pred, 0.01, 0.99)
        h1_pred = self._h1_model.predict(XU)
        h0_pred = self._h0_model.predict(XU)
        m_pred = q_pred * h1_pred + (1 - q_pred) * h0_pred
        
        # Per-arm Q_a on grid
        S_hat_1 = _predict_S_on_grid(
            self._event_cox_1, self._cox_col_names, XU, self._t_grid
        )
        S_hat_0 = _predict_S_on_grid(
            self._event_cox_0, self._cox_col_names, XU, self._t_grid
        )
        Q_hat_1 = _compute_Q_from_S(S_hat_1, self._t_grid)
        Q_hat_0 = _compute_Q_from_S(S_hat_0, self._t_grid)
        
        treated = (A == 1)
        Q_hat = np.where(treated[:, None], Q_hat_1, Q_hat_0)
        
        # 3-term IPCW Y_res
        Y_res = _compute_ipcw_3term_Y_res(
            Y_time, Delta, m_pred, Q_hat, self._t_grid,
            self._km_times, self._km_surv
        )
        A_res = (A - q_pred).reshape(-1, 1)
        
        return Y_res, A_res


class NCSurvivalForestDMLOracle(CausalForestDML):
    """
    Oracle NC Survival Forest using true U with IPCW for censored data.
    
    Same as NCCausalForestDMLOracle but handles censoring via IPCW.
    U is passed via the W parameter slot.
    """
    
    def __init__(self, q_model=None, h_model=None, **kwargs):
        self._q_model_template = q_model or LogisticRegression(max_iter=1000)
        self._h_model_template = h_model or RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20
        )
        
        kwargs['discrete_treatment'] = True
        kwargs['criterion'] = 'het'
        kwargs.pop('model_y', None)
        kwargs.pop('model_t', None)
        
        super().__init__(**kwargs)
    
    @override
    def _gen_ortho_learner_model_nuisance(self):
        return _OracleSurvivalModelNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template
        )
    
    @override
    def fit(self, Y, T, *, Delta=None, X=None, W=None, Z=None,
            sample_weight=None, freq_weight=None, sample_var=None,
            groups=None, cache_values=False, inference=None):
        """
        Parameters
        ----------
        Y : array - Observed time
        T : array - Treatment A
        Delta : array - Event indicator
        X : array - Covariates
        W : array - TRUE confounder U
        """
        if X is None:
            raise ValueError("X is required for Oracle Survival Forest")
        if W is None:
            raise ValueError("W (true confounder U) is required")
        if Delta is None:
            raise ValueError("Delta (event indicator) is required for survival models")
        
        Y = np.asarray(Y).ravel()
        Delta = np.asarray(Delta).ravel()
        W = np.atleast_1d(np.asarray(W))
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        
        Y_packed = np.column_stack([Y, Delta])
        
        return _OrthoLearner.fit(
            self, Y_packed, T, X=X, W=W, Z=None,
            sample_weight=sample_weight,
            freq_weight=freq_weight,
            sample_var=sample_var,
            groups=groups,
            cache_values=cache_values,
            inference=inference
        )
    
    def fit_oracle(self, X, A, time, event, U, **kwargs):
        """Convenience: fit(Y=time, T=A, Delta=event, X=X, W=U)"""
        return self.fit(Y=time, T=A, Delta=event, X=X, W=U, **kwargs)


# =============================================================================
# Baseline Survival CausalForestDML (ignores NC, uses IPCW)
# =============================================================================

class _BaselineSurvivalModelNuisance:
    """
    Baseline nuisance model for censored survival data ignoring NC.
    
    Uses the 3-term IPCW-augmented score with standard DML nuisances:
    - model_t: P(A=1 | X)
    - model_y: E[T | X]  estimated via IPCW pseudo-outcome
    - Per-arm Cox models on X for Q_a computation (CSF eq 11)
    """
    
    MAX_GRID = 500
    
    def __init__(self, model_y, model_t):
        self._model_y_template = model_y
        self._model_t_template = model_t
        self._model_y = None
        self._model_t = None
        self._km_times = None
        self._km_surv = None
        self._event_cox_1 = None
        self._event_cox_0 = None
        self._cox_col_names = None
        self._t_grid = None
    
    def _unpack_Y(self, Y):
        Y = np.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] >= 2:
            return Y[:, 0], Y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event] for survival models")
    
    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None,
              sample_weight=None, groups=None):
        Y_time, Delta = self._unpack_Y(Y)
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        
        # KM censoring
        self._km_times, self._km_surv = _fit_kaplan_meier_censoring(Y_time, Delta)
        Y_tilde = _compute_ipcw_pseudo_outcome(Y_time, Delta,
                                                self._km_times, self._km_surv)
        
        # model_t: P(A=1 | X)
        self._model_t = clone(self._model_t_template)
        if sample_weight is not None:
            self._model_t.fit(X, A, sample_weight=sample_weight)
        else:
            self._model_t.fit(X, A)
        
        # model_y: E[Ỹ | X]
        self._model_y = clone(self._model_y_template)
        if sample_weight is not None:
            self._model_y.fit(X, Y_tilde, sample_weight=sample_weight)
        else:
            self._model_y.fit(X, Y_tilde)
        
        # Per-arm Cox PH for Q_a(s|X) (CSF eq 11)
        treated_mask = A == 1
        control_mask = A == 0
        self._event_cox_1, self._cox_col_names = _fit_event_cox(
            Y_time[treated_mask], Delta[treated_mask], X[treated_mask]
        )
        self._event_cox_0, _ = _fit_event_cox(
            Y_time[control_mask], Delta[control_mask], X[control_mask]
        )
        
        # Time grid (all unique times for dΛ^C accuracy)
        all_times = np.sort(np.unique(Y_time))
        if len(all_times) > self.MAX_GRID:
            idx = np.linspace(0, len(all_times) - 1, self.MAX_GRID, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times
        
        return self
    
    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        Y_time, Delta = self._unpack_Y(Y)
        A = T.ravel() if T.ndim == 1 else T[:, 0].ravel()
        
        y_pred = self._model_y.predict(X)
        t_pred = self._model_t.predict_proba(X)[:, 1]
        t_pred = np.clip(t_pred, 0.01, 0.99)
        
        # Per-arm Q_a on grid
        S_hat_1 = _predict_S_on_grid(
            self._event_cox_1, self._cox_col_names, X, self._t_grid
        )
        S_hat_0 = _predict_S_on_grid(
            self._event_cox_0, self._cox_col_names, X, self._t_grid
        )
        Q_hat_1 = _compute_Q_from_S(S_hat_1, self._t_grid)
        Q_hat_0 = _compute_Q_from_S(S_hat_0, self._t_grid)
        
        treated = (A == 1)
        Q_hat = np.where(treated[:, None], Q_hat_1, Q_hat_0)
        
        # 3-term IPCW Y_res (m_pred = E[Ỹ|X] here, marginal over A)
        Y_res = _compute_ipcw_3term_Y_res(
            Y_time, Delta, y_pred, Q_hat, self._t_grid,
            self._km_times, self._km_surv
        )
        A_res = (A - t_pred).reshape(-1, 1)
        
        return Y_res, A_res


class BaselineSurvivalForestDML(CausalForestDML):
    """
    Baseline CausalForestDML for censored survival data, ignoring NC.
    
    Uses IPCW pseudo-outcomes with standard E[Ỹ|X] and P(A|X) nuisances.
    Demonstrates bias from unmeasured confounding under censoring.
    """
    
    def __init__(self, model_y=None, model_t=None, **kwargs):
        self._model_y_template = model_y or RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20
        )
        self._model_t_template = model_t or RandomForestClassifier(
            n_estimators=100, min_samples_leaf=20
        )
        
        kwargs['discrete_treatment'] = True
        kwargs['criterion'] = 'het'
        kwargs.pop('model_y', None)
        kwargs.pop('model_t', None)
        
        super().__init__(**kwargs)
    
    @override
    def _gen_ortho_learner_model_nuisance(self):
        return _BaselineSurvivalModelNuisance(
            model_y=self._model_y_template,
            model_t=self._model_t_template
        )
    
    @override
    def fit(self, Y, T, *, Delta=None, X=None, W=None, Z=None,
            sample_weight=None, freq_weight=None, sample_var=None,
            groups=None, cache_values=False, inference=None):
        """
        Fit baseline survival CausalForest (ignores Z and W).
        
        Parameters
        ----------
        Y : array - Observed time
        T : array - Treatment A
        Delta : array - Event indicator
        X : array - Covariates
        """
        if X is None:
            raise ValueError("X is required")
        if Delta is None:
            raise ValueError("Delta (event indicator) is required for survival models")
        
        Y = np.asarray(Y).ravel()
        Delta = np.asarray(Delta).ravel()
        
        Y_packed = np.column_stack([Y, Delta])
        
        return _OrthoLearner.fit(
            self, Y_packed, T, X=X, W=None, Z=None,
            sample_weight=sample_weight,
            freq_weight=freq_weight,
            sample_var=sample_var,
            groups=groups,
            cache_values=cache_values,
            inference=inference
        )
    
    def fit_baseline(self, X, A, time, event, Z=None, W=None, **kwargs):
        """Convenience: fit(Y=time, T=A, Delta=event, X=X). Ignores Z,W."""
        return self.fit(Y=time, T=A, Delta=event, X=X, **kwargs)
