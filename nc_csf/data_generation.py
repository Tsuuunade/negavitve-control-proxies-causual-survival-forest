"""
Synthetic Data Generation for NC-CSF Experiments

Standard notation:
  X : observed covariates
  U : unobserved confounder
  A : treatment (binary)
  Z : NC variable / proxy (continuous)
  W : NC outcome / proxy (continuous)
  Y : observed outcome (time)
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_intercept_for_prevalence(
    linpred_no_intercept: np.ndarray,
    target_prevalence: float,
    max_iter: int = 60,
) -> float:
    """Bisection for b0 so mean(sigmoid(b0 + linpred)) ~= target_prevalence."""
    lo, hi = -20.0, 20.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = sigmoid(mid + linpred_no_intercept).mean()
        if p < target_prevalence:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def weibull_ph_time_paper(u01: np.ndarray, k: float, lam: float, eta: np.ndarray) -> np.ndarray:
    """
    Sampling consistent with:
      T | (X,U,A) ~ Weibull(k, scale = lam * exp(-eta/k))
      T = scale * (-log(U01))^(1/k)
    """
    u01 = np.clip(u01, 1e-12, 1 - 1e-12)
    scale = lam * np.exp(-eta / k)
    return scale * (-np.log(u01)) ** (1.0 / k)


@dataclass
class SynthConfig:
    n: int = 5000
    p_x: int = 10
    seed: int = 123

    # Treatment A: P(A=1 | X,U)
    a_prevalence: float = 0.5
    gamma_u_in_a: float = 1.0  # U -> A strength

    # Event time model (Weibull Cox PH)
    k_t: float = 1.5
    lam_t: float = 0.4
    tau_log_hr: float = -0.6   # treatment effect (log hazard ratio)
    beta_u_in_t: float = 0.8   # U -> event time

    # Censoring time model (Weibull Cox PH)
    k_c: float = 1.2
    lam_c: Optional[float] = None
    beta_u_in_c: float = 0.3
    target_censor_rate: float = 0.35
    max_censor_calib_iter: int = 60
    censor_lam_lo: float = 1e-8
    censor_lam_hi: float = 1e6

    # Optional admin censoring
    admin_censor_time: Optional[float] = None

    # Proxies:
    # Z = aZ*U + X'bZ + epsZ
    aZ: float = 1.0
    sigma_z: float = 0.8

    # W = aW*U + X'bW + epsW
    aW: float = 1.0
    sigma_w: float = 0.8


@dataclass
class SynthParams:
    b_z: np.ndarray
    b_w: np.ndarray
    beta_t: np.ndarray


def generate_synthetic_nc_cox(cfg: SynthConfig) -> Tuple[pd.DataFrame, pd.DataFrame, SynthParams]:
    """
    Generate synthetic data with negative control structure.
    
    Returns:
        observed_df: DataFrame with observed variables (time, event, A, W, Z, X0..Xp)
        truth_df: DataFrame with ground truth including U and potential outcomes
        params: SynthParams with regression coefficients
    """
    rng = np.random.default_rng(cfg.seed)
    n, p = cfg.n, cfg.p_x

    # 1) Sample X and U
    X = rng.normal(size=(n, p))
    U = rng.normal(size=n)

    # 2) Build proxies Z and W (continuous)
    b_z = rng.normal(scale=0.3, size=p)
    b_w = rng.normal(scale=0.3, size=p)

    Z = cfg.aZ * U + X @ b_z + rng.normal(scale=cfg.sigma_z, size=n)
    W_nc = cfg.aW * U + X @ b_w + rng.normal(scale=cfg.sigma_w, size=n)

    # 3) Treatment A from logistic model of (X,U)
    alpha = rng.normal(scale=0.5, size=p)
    linpred = X @ alpha + cfg.gamma_u_in_a * U
    b0 = calibrate_intercept_for_prevalence(linpred, cfg.a_prevalence)
    p_a = sigmoid(b0 + linpred)
    A = rng.binomial(1, p_a, size=n).astype(int)

    # 4) Potential event times T0,T1 (shared uniform u_t)
    beta_t = rng.normal(scale=0.4, size=p)
    u_t = rng.random(n)

    eta_t0 = X @ beta_t + cfg.beta_u_in_t * U + cfg.tau_log_hr * 0.0
    eta_t1 = X @ beta_t + cfg.beta_u_in_t * U + cfg.tau_log_hr * 1.0

    T0 = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t0)
    T1 = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t1)

    # 5) Censoring times
    beta_c = rng.normal(scale=0.3, size=p)
    u_c = rng.random(n)
    eta_c = X @ beta_c + cfg.beta_u_in_c * U

    T_obs_for_calib = np.where(A == 1, T1, T0)
    lam_c_used = cfg.lam_c

    if lam_c_used is None:
        lo, hi = float(cfg.censor_lam_lo), float(cfg.censor_lam_hi)
        for _ in range(cfg.max_censor_calib_iter):
            mid = 0.5 * (lo + hi)
            C_mid = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=mid, eta=eta_c)
            censor_rate_mid = (C_mid < T_obs_for_calib).mean()
            if censor_rate_mid < cfg.target_censor_rate:
                hi = mid
            else:
                lo = mid
        lam_c_used = 0.5 * (lo + hi)

    C0 = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=lam_c_used, eta=eta_c)
    C1 = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=lam_c_used, eta=eta_c)

    # 6) Realized T,C and observed (time,event)
    T = np.where(A == 1, T1, T0)
    C = np.where(A == 1, C1, C0)

    time = np.minimum(T, C)
    event = (T <= C).astype(int)

    if cfg.admin_censor_time is not None:
        admin = float(cfg.admin_censor_time)
        cens_by_admin = admin < time
        time = np.where(cens_by_admin, admin, time)
        event = np.where(cens_by_admin, 0, event).astype(int)

    # 7) DataFrames
    X_cols = {f"X{j}": X[:, j] for j in range(p)}

    observed_df = pd.DataFrame({
        "time": time,
        "event": event,
        "A": A,
        "W": W_nc,
        "Z": Z,
        **X_cols,
    })

    truth_df = observed_df.copy()
    truth_df.insert(0, "U", U)
    truth_df["T0"] = T0
    truth_df["T1"] = T1
    truth_df["C0"] = C0
    truth_df["C1"] = C1
    truth_df["T"] = T
    truth_df["C"] = C
    truth_df.attrs["lam_c_used"] = lam_c_used

    params = SynthParams(b_z=b_z, b_w=b_w, beta_t=beta_t)
    return observed_df, truth_df, params


def add_ground_truth_cate(
    observed_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    cfg: SynthConfig,
    params: SynthParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add ground truth CATE columns to DataFrames.
    
    Adds to truth_df:
      - CATE_XU_eq7: E[T(1)-T(0) | X, U] (oracle)
      - ITE_T1_minus_T0: sample ITE
    """
    obs = observed_df.copy()
    tru = truth_df.copy()

    x_cols = sorted([c for c in obs.columns if c.startswith("X")], key=lambda s: int(s[1:]))
    X = obs[x_cols].to_numpy()

    k = float(cfg.k_t)
    lam = float(cfg.lam_t)
    tau = float(cfg.tau_log_hr)
    beta_u = float(cfg.beta_u_in_t)

    G = math.gamma(1.0 + 1.0 / k)
    xb = X @ params.beta_t

    # Eq.(7): E[T(1)-T(0) | X, U] (oracle)
    U = tru["U"].to_numpy()
    cate_xu = (
        lam * G
        * np.exp(-(1.0 / k) * (xb + beta_u * U))
        * (np.exp(-tau / k) - 1.0)
    )

    # Sample ITE
    ite = tru["T1"].to_numpy() - tru["T0"].to_numpy()

    tru["CATE_XU_eq7"] = cate_xu
    tru["ITE_T1_minus_T0"] = ite

    return obs, tru
