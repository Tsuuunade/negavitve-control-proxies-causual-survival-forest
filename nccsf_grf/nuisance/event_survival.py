import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

class EventSurvivalModel:
    def __init__(self, n_estimators=100, min_samples_leaf=15, n_jobs=-1):
        """
        Estimates S_e(t | X, W, Z), the survival probability of the TRUE EVENT.
        This is required exclusively to compute the K_tau nuisance functions, 
        specifically E[min(T,L) | T > t, V].
        """
        self.rsf = RandomSurvivalForest(
            n_estimators=n_estimators, 
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs  # Sometimes n_jobs=-1 hangs on macOS
        )

    def fit(self, covariates, Y, delta):
        """
        Fit the true event model. 
        """
        event_indicator = delta.astype(bool)
        y_surv = Surv.from_arrays(event=event_indicator, time=Y)
        self.rsf.fit(covariates, y_surv)
        
    def predict_survival(self, covariates):
        """
        Returns S_e(t) evaluated at the model's unique times.
        """
        surv_funcs = self.rsf.predict_survival_function(covariates, return_array=True)
        return surv_funcs, self.rsf.unique_times_
