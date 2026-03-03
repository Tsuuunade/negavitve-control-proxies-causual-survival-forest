import numpy as np

def apply_rmst_truncation(Y, delta, L):
    """
    Truncates survival data to the Restricted Mean Survival Time (RMST) horizon L.
    If L is None or infinity, returns the data unmodified (unrestricted).
    """
    if L is None or np.isinf(L):
        return Y.copy(), delta.copy()
        
    Y_star = np.minimum(Y, L)
    
    delta_star = delta.copy()
    # If they survived past L, we fully observed their path to the horizon L.
    delta_star[Y >= L] = 1 
    
    return Y_star, delta_star

def compute_risk_set_expectations(S_e_matrix, failure_times, L):
    """
    Computes E[min(T, L) | T > t_k] for every patient and every time step t_k.
    If L is None or infinity, computes unrestricted expected time.
    """
    n_samples, n_times = S_e_matrix.shape
    E_T_given_T_gt_t = np.zeros_like(S_e_matrix)
    
    if L is None or np.isinf(L):
        bounded_times = failure_times
        L_cap = np.inf
    else:
        bounded_times = np.minimum(failure_times, L)
        L_cap = L
        
    dt = np.diff(bounded_times)
    
    # Vectorized integration:
    # integral_matrix[i, k] = \sum_{j=k}^{n_times-2} S_e_matrix[i, j] * dt[j]
    
    # First compute S_e_matrix * dt for all points
    # dt has length n_times - 1
    S_dt = S_e_matrix[:, :-1] * dt[np.newaxis, :]
    
    # The integral from t_k to L is just the reverse cumulative sum of S_dt
    # flipped along the time axis
    integral_from_k = np.cumsum(S_dt[:, ::-1], axis=1)[:, ::-1]
    
    # Pad to match shape (integral from the last time point is 0)
    integral_matrix = np.zeros_like(S_e_matrix)
    integral_matrix[:, :-1] = integral_from_k
    
    # Calculate expected time: t_k + (integral / S_t_k)
    # Prevent division by zero
    S_safe = np.maximum(S_e_matrix, 1e-10)
    
    E_T_given_T_gt_t = bounded_times[np.newaxis, :] + (integral_matrix / S_safe)
    
    # Cap at L (cannot expect to survive past RMST horizon)
    if not np.isinf(L_cap):
        E_T_given_T_gt_t = np.minimum(E_T_given_T_gt_t, L_cap)
        
        # Explicitly set expectations to L for times >= L
        mask_past_L = bounded_times >= L_cap
        if np.any(mask_past_L):
            E_T_given_T_gt_t[:, mask_past_L] = L_cap
            
    return E_T_given_T_gt_t
