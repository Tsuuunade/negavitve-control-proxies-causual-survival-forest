import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

class CensoringModel:
    def __init__(self, n_estimators=100, min_samples_leaf=15, n_jobs=-1):
        """
        Estimates S^C_A(t | X, W, Z) and lambda^C_A(t | X, W, Z) using 
        a Random Survival Forest based on the logic that Censoring is the 'Event'.
        """
        self.rsf = RandomSurvivalForest(
            n_estimators=n_estimators, 
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs  # Sometimes n_jobs=-1 hangs on macOS
        )
        self.failure_times = None

    def fit(self, covariates, Y, delta):
        """
        Fit the censoring model. 
        NOTE: Delta is inverted here! Censoring becomes the event (delta==0 becomes 1).
        """
        censoring_event = (delta == 0).astype(bool)
        
        # sksurv format
        y_surv = Surv.from_arrays(event=censoring_event, time=Y)
        
        self.rsf.fit(covariates, y_surv)
        self.failure_times = self.rsf.unique_times_
        
    def predict_surv_and_hazard(self, covariates):
        """
        Returns S^C(t) and lambda^C(t) evaluated at self.failure_times
        """
        surv_funcs = self.rsf.predict_survival_function(covariates, return_array=True)
        chf_funcs = self.rsf.predict_cumulative_hazard_function(covariates, return_array=True)
        
        # Hazard jumps lambda(t_k) = \Lambda(t_k) - \Lambda(t_{k-1})
        n_samples, n_times = chf_funcs.shape
        hazard_jumps = np.zeros_like(chf_funcs)
        
        hazard_jumps[:, 0] = chf_funcs[:, 0]
        for k in range(1, n_times):
            hazard_jumps[:, k] = chf_funcs[:, k] - chf_funcs[:, k-1]
            
        # Optional safeguard: hazard cannot be negative
        hazard_jumps = np.clip(hazard_jumps, 0, None)
        
        return surv_funcs, hazard_jumps, self.failure_times
