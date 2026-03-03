import numpy as np
from .nuisance.survival import CensoringModel
from .nuisance.event_survival import EventSurvivalModel
from .nuisance.proxies import TreatmentProxyModel, OutcomeProxyModel, compute_ipcw_outcome
from .commons import apply_rmst_truncation, compute_risk_set_expectations
from .splitting.score import compute_grf_orthogonal_scores

class NCCSFPipeline:
    """
    Coordinates the Generalized Random Forest nuisance estimation phase 
    for the Negative Control Causal Survival Forest.
    """
    def __init__(self, target="rmst", rmst_horizon=None, n_jobs=1):
        self.target = target
        if self.target == "unrestricted":
            self.L = np.inf
        else:
            # Fallback to infinity if rmst is asked but no horizon provided
            self.L = rmst_horizon if rmst_horizon is not None else np.inf
            
        self.censoring_model = CensoringModel(n_jobs=n_jobs)
        self.event_model = EventSurvivalModel(n_jobs=n_jobs)
        self.q_model = TreatmentProxyModel()
        self.h_model = OutcomeProxyModel()
        
    def generate_pseudo_responses(self, X, W, Z, A, Y, delta):
        # 1. Truncate target to RMST horizon L
        Y_star, delta_star = apply_rmst_truncation(Y, delta, self.L)
        
        # We need all covariates combined for survival forest estimates

        to_stack = [X, A.reshape(-1, 1)]
        if W is not None: to_stack.append(W)
        if Z is not None: to_stack.append(Z)
        V = np.hstack(tuple(to_stack))

        n_samples = X.shape[0]
        
        # 2. Fit Censoring survival model (S^C, lambda^C)
        print("Fitting Censoring Model...")
        self.censoring_model.fit(V, Y_star, delta_star)
        surv_C_matrix, hazard_C_matrix, fail_times_C = self.censoring_model.predict_surv_and_hazard(V)
        
        # Get S^C evaluated exactly at Y_i for IPCW computation
        # (Find closest t_k <= Y_i)
        S_c_Y = np.zeros(n_samples)
        for i in range(n_samples):
            idx = np.searchsorted(fail_times_C, Y_star[i], side='right') - 1
            idx = max(0, idx)
            S_c_Y[i] = surv_C_matrix[i, idx]
            
        # 3. Fit True Event survival model (S_e)
        print("Fitting Event Survival Model...")
        self.event_model.fit(V, Y_star, delta_star)
        surv_E_matrix, fail_times_E = self.event_model.predict_survival(V)
        
        # 4. Proxy/Bridging Phase
        print("Fitting Treatment & Outcome Proxies...")
        self.q_model.fit(Z, X, A)
        q_preds = self.q_model.predict_proba(Z, X)
        
        Y_ipcw = compute_ipcw_outcome(Y_star, delta_star, S_c_Y)
        self.h_model.fit(W, X, A, Y_ipcw)
        h_0, h_1 = self.h_model.predict(W, X, A=None)
        
        # Mixture m(X,W,Z)
        m_val = q_preds * h_1 + (1 - q_preds) * h_0
        
        # Residualized treatment D_i
        D = A - q_preds
        
        # 5. Compute Risk Set Nuisances K_gamma, K_H
        # E[T | T > t_k, V]
        E_T_given_gt_t = compute_risk_set_expectations(surv_E_matrix, fail_times_E, self.L)
        
        # Note: we need to evaluate K at the failure_times_C grid for integral alignment
        # To do this safely, we interpolate E_T_given_gt_t onto fail_times_C
        K_gamma = np.zeros_like(surv_C_matrix)
        K_H_matrix = np.zeros_like(surv_C_matrix)
        
        for i in range(n_samples):
            # Interpolate the expectations to the censoring time grid
            # (step-function interpolation is usually exact enough for survival curves)
            E_T_interp = np.interp(fail_times_C, fail_times_E, E_T_given_gt_t[i, :])
            
            # K_Gamma = D * (E[T | T>t] - m)
            K_gamma[i, :] = D[i] * (E_T_interp - m_val[i])
            # K_H = D^2 (constant over time)
            K_H_matrix[i, :] = (D[i] ** 2)
            
        # 6. Compute Final GRF Orthogonal Scores
        print("Computing Final GRF Negative Control Pseudo-Responses...")
        Gamma_i, H_i = compute_grf_orthogonal_scores(
            Y_star, A, D, m_val, delta_star,
            surv_C_matrix, hazard_C_matrix, K_gamma, K_H_matrix, fail_times_C
        )
        
        return Gamma_i, H_i

