import numpy as np
from .pipeline import NCCSFPipeline
from .forest.forest import HonestCausalForest

class NCCausalSurvivalForest:
    """
    User-facing estimator for Negative Control Causal Survival Forests.
    Wraps the Nuisance Pipeline and the Custom Honest Causal Forest together.
    """
    def __init__(self, target="rmst", rmst_horizon=None, n_estimators=200, min_samples_leaf=10, honesty=True, n_jobs=1):
        self.pipeline = NCCSFPipeline(target=target, rmst_horizon=rmst_horizon, n_jobs=n_jobs)
        self.forest = HonestCausalForest(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            honesty=honesty,
        )
        # We can also pass n_jobs to the final forest if we update forest.py, 
        # but the bottleneck issue on macOS is usually in sksurv.
        self.is_fitted = False
        
    def fit(self, X, W, Z, A, Y, delta):
        """
        Fit the NCCSF Model.
        
        X: (n_samples, n_features) Core Covariates
        W: (n_samples, n_out_proxies) Outcome Proxies
        Z: (n_samples, n_trt_proxies) Treatment Proxies
        A: (n_samples,) Binary Treatment indicator
        Y: (n_samples,) Observed Event Time min(T, C)
        delta: (n_samples,) Event indicator (1=Event, 0=Censored)
        """

        X = np.asarray(X)
        if W is not None: W = np.asarray(W)
        if Z is not None: Z = np.asarray(Z)
        A = np.asarray(A)
        Y = np.asarray(Y)
        delta = np.asarray(delta)
        print('Starting Nuisance Estimation Phase...')
        Gamma, H = self.pipeline.generate_pseudo_responses(X, W, Z, A, Y, delta)
        
        print("Training Final Causal Forest...")
        self.forest.fit(X, Gamma, H)
        
        self.is_fitted = True
        print("Model Fitted Successfully.")
        return self
        
    def predict(self, X):
        """
        Predict the Conditional Average Treatment Effect (CATE) on Survival Time 
        (bounded by RMST horizon L) for new patients based purely on X.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        X = np.asarray(X)
        return self.forest.predict(X)
