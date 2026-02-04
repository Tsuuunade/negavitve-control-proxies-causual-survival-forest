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
        
        super().__init__(**kwargs)
    
    def fit_baseline(self, X, A, Y, Z=None, W=None, verbose=False, **kwargs):
        """
        Fit baseline CausalForestDML (ignores Z and W).
        """
        # verbose is ignored (for API compatibility)
        return self.fit(Y=Y, T=A, X=X, **kwargs)
