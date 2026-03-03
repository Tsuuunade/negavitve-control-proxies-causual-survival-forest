import numpy as np

def compute_grf_orthogonal_scores(Y, A, D, m_val, delta, surv_prob_C, hazard_jumps_C, K_gamma, K_H, failure_times):
    """
    Computes \Gamma_i and H_i for the gradient tree splits following GRF principles.
    Vectorized for performance.
    """
    n_samples = Y.shape[0]
    
    # 1. Compute the pseudo-responses for the COMPLETE case
    uncensored_gamma = D * (Y - m_val)
    uncensored_H = D**2
    
    # Find indices for Y_i in the discrete time grid
    # searchsorted side='right' - 1 gets the largest t_k <= Y_i
    idx_Y = np.searchsorted(failure_times, Y, side='right') - 1
    idx_Y = np.clip(idx_Y, 0, len(failure_times) - 1)
    
    # Get values at exactly Y_i
    S_c_Y = surv_prob_C[np.arange(n_samples), idx_Y]
    S_c_Y = np.maximum(S_c_Y, 1e-10)
    
    K_gamma_Y = K_gamma[np.arange(n_samples), idx_Y]
    K_H_Y = K_H[np.arange(n_samples), idx_Y]
    
    # Create mask for valid integration times t_k <= Y_i
    time_mask = np.arange(len(failure_times))[np.newaxis, :] <= idx_Y[:, np.newaxis]
    
    # Integral parts
    S_c_safe = np.maximum(surv_prob_C, 1e-10)
    
    integrand_gamma = (hazard_jumps_C / S_c_safe) * K_gamma
    integrand_H = (hazard_jumps_C / S_c_safe) * K_H
    
    # Zero out integrals strictly past Y_i
    integrand_gamma = np.where(time_mask, integrand_gamma, 0.0)
    integrand_H = np.where(time_mask, integrand_H, 0.0)
    
    integral_gamma = np.sum(integrand_gamma, axis=1)
    integral_H = np.sum(integrand_H, axis=1)
    
    # Piece 1: Complete case divided by S_c(Y_i)
    term1_gamma = (delta / S_c_Y) * uncensored_gamma
    term1_H = (delta / S_c_Y) * uncensored_H
    
    # Piece 2: Censored padding
    term2_gamma = ((1 - delta) / S_c_Y) * K_gamma_Y
    term2_H = ((1 - delta) / S_c_Y) * K_H_Y
    
    # Total DR Score Components
    Gamma_array = term1_gamma + term2_gamma - integral_gamma
    H_array = term1_H + term2_H - integral_H
        
    return Gamma_array, H_array
