#!/usr/bin/env python3
"""
================================================================================
Comprehensive 8-Variant Benchmark
================================================================================

GROUP A – Oracle (observe U_i directly)
  A1  Oracle  all true S^C, λ^C, K_τ, q, r
  A2  Oracle  true S^C, λ^C, K_τ  |  estimated q, r
  A3  Oracle  all estimated

GROUP B – CSF baselines (no U, no proxy bridge)
  B1  Naive CSF    X only
  B2  Augmented    X with W,Z appended as extra covariates (no bridge)

GROUP C – NC-CSF (our proposal, negative-control bridge with W, Z)
  C1  NC-CSF  all true S^C, λ^C, K_τ, q, r
  C2  NC-CSF  true S^C, λ^C, K_τ  |  estimated q, r
  C3  NC-CSF  all estimated

Nuisance notation
-----------------
  S, λ     Censoring survival / hazard  (Weibull PH in the DGP)
  K_τ      Risk-set expectation E[T | T>t, V]  (from event survival)
  q        Treatment bridge   P(A=1 | Z, X)  [or propensity e(X,U) for oracle]
  r (= h)  Outcome bridge     E[Y_ipcw | W, X, A]  [or E[T_a|X,U] for oracle]

"True" means computed analytically from the known Weibull / logistic DGP.
"Estimated" means fitted with RandomSurvivalForest / LogisticRegression / RF.
================================================================================
"""

import math
import time
import numpy as np
import pandas as pd

from nc_csf.data_generation import (
    generate_synthetic_nc_cox,
    SynthConfig,
    add_ground_truth_cate,
    sigmoid,
    calibrate_intercept_for_prevalence,
    weibull_ph_time_paper,
)
from nccsf_grf.forest.forest import HonestCausalForest
from nccsf_grf.commons import compute_risk_set_expectations
from nccsf_grf.splitting.score import compute_grf_orthogonal_scores
from nccsf_grf.nuisance.survival import CensoringModel
from nccsf_grf.nuisance.event_survival import EventSurvivalModel
from nccsf_grf.nuisance.proxies import (
    TreatmentProxyModel,
    OutcomeProxyModel,
    compute_ipcw_outcome,
)


# ====================================================================
# 1.  Recover every DGP-internal parameter by replaying the RNG
# ====================================================================
def recover_dgp_internals(cfg):
    """
    Re-derive all internal random coefficients (alpha, b0, beta_c, …)
    that are not exposed by generate_synthetic_nc_cox, by replaying the
    exact same numpy RNG sequence with the same seed.
    """
    rng = np.random.default_rng(cfg.seed)
    n, p = cfg.n, cfg.p_x

    X      = rng.normal(size=(n, p))
    U      = rng.normal(size=n)
    b_z    = rng.normal(scale=0.3, size=p)
    b_w    = rng.normal(scale=0.3, size=p)
    _eps_z = rng.normal(scale=cfg.sigma_z, size=n)     # consumed but unused here
    _eps_w = rng.normal(scale=cfg.sigma_w, size=n)

    alpha   = rng.normal(scale=0.5, size=p)
    linpred = X @ alpha + cfg.gamma_u_in_a * U
    b0      = calibrate_intercept_for_prevalence(linpred, cfg.a_prevalence)
    p_a     = sigmoid(b0 + linpred)
    A       = rng.binomial(1, p_a, size=n).astype(int)

    beta_t = rng.normal(scale=0.4, size=p)
    u_t    = rng.random(n)

    eta_t0 = X @ beta_t + cfg.beta_u_in_t * U
    eta_t1 = eta_t0 + cfg.tau_log_hr
    T0     = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t0)
    T1     = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t1)

    beta_c = rng.normal(scale=0.3, size=p)
    u_c    = rng.random(n)
    eta_c  = X @ beta_c + cfg.beta_u_in_c * U
    T_obs  = np.where(A == 1, T1, T0)

    lam_c = cfg.lam_c
    if lam_c is None:
        lo, hi = float(cfg.censor_lam_lo), float(cfg.censor_lam_hi)
        for _ in range(cfg.max_censor_calib_iter):
            mid = 0.5 * (lo + hi)
            C_mid = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=mid, eta=eta_c)
            if (C_mid < T_obs).mean() < cfg.target_censor_rate:
                hi = mid
            else:
                lo = mid
        lam_c = 0.5 * (lo + hi)

    return dict(
        X=X, U=U, A=A,
        alpha=alpha, b0=b0,
        beta_t=beta_t, beta_c=beta_c,
        b_z=b_z, b_w=b_w,
        lam_c=lam_c,
    )


# ====================================================================
# 2.  Analytic (true) nuisance helpers
# ====================================================================
def _weibull_scale(lam, k, eta):
    """Per-obs Weibull scale: lam * exp(-eta / k)."""
    return lam * np.exp(-eta / k)


def true_censoring_on_grid(X, U, Y_star, time_grid, cfg, beta_c, lam_c):
    """
    Exact Weibull censoring S^C and hazard jumps on *time_grid*.
    Also returns S^C(Y_i) evaluated continuously (not from grid).
    """
    eta_c   = X @ beta_c + cfg.beta_u_in_c * U
    scale_c = _weibull_scale(lam_c, cfg.k_c, eta_c)            # (n,)

    # Vectorised:  (n, K)
    t_over_s = time_grid[np.newaxis, :] / scale_c[:, np.newaxis]
    cum_haz  = t_over_s ** cfg.k_c
    surv_C   = np.exp(-cum_haz)

    # Hazard jumps  ΔΛ(t_k) = Λ(t_k) − Λ(t_{k-1})
    hazard_jumps        = np.zeros_like(cum_haz)
    hazard_jumps[:, 0]  = cum_haz[:, 0]
    hazard_jumps[:, 1:] = np.diff(cum_haz, axis=1)
    hazard_jumps        = np.clip(hazard_jumps, 0, None)

    # S^C at each Y_i (continuous Weibull, no grid discretisation)
    S_c_Y = np.exp(-(Y_star / scale_c) ** cfg.k_c)

    return surv_C, hazard_jumps, S_c_Y


def true_event_surv_on_grid(X, U, A, time_grid, cfg, beta_t):
    """
    Exact Weibull event survival S_e(t | X_i, U_i, A_i) on *time_grid*.
    Each subject's curve conditions on their actual treatment.
    """
    eta_t   = X @ beta_t + cfg.beta_u_in_t * U + cfg.tau_log_hr * A
    scale_t = _weibull_scale(cfg.lam_t, cfg.k_t, eta_t)

    t_over_s = time_grid[np.newaxis, :] / scale_t[:, np.newaxis]
    surv_E   = np.exp(-t_over_s ** cfg.k_t)
    return surv_E


# ---- propensity / bridge analytic values ----------------------------

def true_propensity_oracle(X, U, dgp, cfg):
    """e(X, U) = P(A = 1 | X, U)  — exact logistic from DGP."""
    return sigmoid(dgp['b0'] + X @ dgp['alpha'] + cfg.gamma_u_in_a * U)


def true_propensity_nc(Z, X, dgp, cfg):
    """
    q(Z, X) = P(A = 1 | Z, X)  via probit approximation.

    Posterior:  U | Z, X  ~  N(mu, var)
       var  = sigma_z^2 / (sigma_z^2 + aZ^2)
       mu   = aZ * (Z − X b_z) / (sigma_z^2 + aZ^2)
    """
    Z = np.asarray(Z).ravel()
    sz2, az2 = cfg.sigma_z ** 2, cfg.aZ ** 2
    var_post = sz2 / (sz2 + az2)
    mu_post  = cfg.aZ * (Z - X @ dgp['b_z']) / (sz2 + az2)

    loc = dgp['b0'] + X @ dgp['alpha'] + cfg.gamma_u_in_a * mu_post
    return sigmoid(loc / np.sqrt(1 + np.pi / 8 * cfg.gamma_u_in_a ** 2 * var_post))


def true_outcome_oracle(X, U, cfg, dgp):
    """
    E[T_a | X, U] for a = 0, 1  — Weibull closed form.
    Returns (h_0, h_1).
    """
    G   = math.gamma(1.0 + 1.0 / cfg.k_t)
    xbu = X @ dgp['beta_t'] + cfg.beta_u_in_t * U
    h_0 = cfg.lam_t * G * np.exp(-xbu / cfg.k_t)
    h_1 = cfg.lam_t * G * np.exp(-(xbu + cfg.tau_log_hr) / cfg.k_t)
    return h_0, h_1


def true_outcome_nc(W, X, cfg, dgp):
    """
    h_a(W, X) = E[ E[T_a | X, U] | W, X ]  via MGF of posterior.

    Posterior:  U | W, X  ~  N(mu_w, var_w)
       var_w = sigma_w^2 / (sigma_w^2 + aW^2)
       mu_w  = aW * (W − X b_w) / (sigma_w^2 + aW^2)

    E[exp(c U) | W, X] = exp(c mu_w + c^2 var_w / 2)   with c = -beta_u / k_t
    """
    W = np.asarray(W).ravel()
    sw2, aw2 = cfg.sigma_w ** 2, cfg.aW ** 2
    var_w = sw2 / (sw2 + aw2)
    mu_w  = cfg.aW * (W - X @ dgp['b_w']) / (sw2 + aw2)

    G   = math.gamma(1.0 + 1.0 / cfg.k_t)
    c   = -cfg.beta_u_in_t / cfg.k_t
    mgf = np.exp(c * mu_w + 0.5 * c ** 2 * var_w)
    xb  = X @ dgp['beta_t']

    h_0 = cfg.lam_t * G * np.exp(-xb / cfg.k_t) * mgf
    h_1 = cfg.lam_t * G * np.exp(-(xb + cfg.tau_log_hr) / cfg.k_t) * mgf
    return h_0, h_1


# ====================================================================
# 3.  Flexible pseudo-response builder
# ====================================================================

def build_pseudo_responses(
    X, U, W, Z, A, Y, delta,
    cfg, dgp,
    *,
    mode,          # "oracle" | "nc" | "naive" | "augmented"
    true_surv,     # use exact Weibull  S^C, λ^C, K_τ
    true_qr,       # use exact propensity q and outcome h (= r)
):
    """
    Compute (Gamma_i, H_i, X_forest) doubly-robust pseudo-responses.

    mode
    ----
    oracle     X_forest = [X, U],  survival models see (X, U, A)
    nc         X_forest = X,       proxy bridge via (Z → q,  W → h)
    naive      X_forest = X,       no U / W / Z
    augmented  X_forest = [X,W,Z], W,Z as plain covariates, no bridge
    """
    Y_star  = Y.copy()
    delta_s = delta.copy()
    n       = X.shape[0]

    # ── build X_forest and V_surv (covariate set for survival RSFs) ──
    if mode == "oracle":
        X_forest = np.hstack([X, U.reshape(-1, 1)])
        V_surv   = np.hstack([X, U.reshape(-1, 1), A.reshape(-1, 1)])
    elif mode == "nc":
        X_forest = X
        V_surv   = np.hstack([X, A.reshape(-1, 1), W, Z])
    elif mode == "naive":
        X_forest = X
        V_surv   = np.hstack([X, A.reshape(-1, 1)])
    elif mode == "augmented":
        X_forest = np.hstack([X, W, Z])
        V_surv   = np.hstack([X, W, Z, A.reshape(-1, 1)])
    else:
        raise ValueError(mode)

    # ── Censoring  S^C, λ^C ─────────────────────────────────────────
    time_grid = np.sort(np.unique(Y_star))

    if true_surv:
        surv_C, hazard_C, S_c_Y = true_censoring_on_grid(
            X, U, Y_star, time_grid, cfg, dgp['beta_c'], dgp['lam_c'],
        )
        fail_times_C = time_grid
        # Standard IPCW weight stabilisation: floor survival at 2.5 %
        # to prevent astronomic inverse-probability weights when the
        # analytic Weibull S^C → 0 at late time points.
        SURV_FLOOR = 0.025
        surv_C = np.maximum(surv_C, SURV_FLOOR)
        S_c_Y  = np.maximum(S_c_Y,  SURV_FLOOR)
    else:
        print("    Fitting Censoring RSF ...")
        cm = CensoringModel(n_jobs=1)
        cm.fit(V_surv, Y_star, delta_s)
        surv_C, hazard_C, fail_times_C = cm.predict_surv_and_hazard(V_surv)
        S_c_Y = np.array([
            surv_C[i, max(0, np.searchsorted(fail_times_C, Y_star[i], side='right') - 1)]
            for i in range(n)
        ])

    # ── Event survival  S_e  →  K_τ ─────────────────────────────────
    if true_surv:
        surv_E       = true_event_surv_on_grid(X, U, A, time_grid, cfg, dgp['beta_t'])
        fail_times_E = time_grid
    else:
        print("    Fitting Event Survival RSF ...")
        em = EventSurvivalModel(n_jobs=1)
        em.fit(V_surv, Y_star, delta_s)
        surv_E, fail_times_E = em.predict_survival(V_surv)

    E_T_given_gt_t = compute_risk_set_expectations(surv_E, fail_times_E, np.inf)

    # ── Propensity  q ────────────────────────────────────────────────
    if true_qr:
        if mode == "oracle":
            q = true_propensity_oracle(X, U, dgp, cfg)
        elif mode == "nc":
            q = true_propensity_nc(Z, X, dgp, cfg)
        else:
            # naive / augmented: marginal E_U[P(A=1|X,U)] via probit
            g = cfg.gamma_u_in_a
            q = sigmoid(
                (dgp['b0'] + X @ dgp['alpha'])
                / np.sqrt(1 + np.pi / 8 * g ** 2)
            )
    else:
        qm = TreatmentProxyModel()
        if mode == "oracle":
            print("    Fitting Propensity (X, U) ...")
            qm.fit(None, X_forest, A)
            q = qm.predict_proba(None, X_forest)
        elif mode == "nc":
            print("    Fitting Treatment Proxy (Z, X) ...")
            qm.fit(Z, X, A)
            q = qm.predict_proba(Z, X)
        else:                                     # naive / augmented
            print("    Fitting Propensity ...")
            qm.fit(None, X_forest, A)
            q = qm.predict_proba(None, X_forest)

    # ── IPCW outcome ─────────────────────────────────────────────────
    S_c_Y_safe = np.maximum(S_c_Y, 1e-10)
    Y_ipcw     = compute_ipcw_outcome(Y_star, delta_s, S_c_Y_safe)

    # ── Outcome bridge  h_0, h_1  (= r_0, r_1) ─────────────────────
    if true_qr:
        if mode == "oracle":
            h_0, h_1 = true_outcome_oracle(X, U, cfg, dgp)
        elif mode == "nc":
            h_0, h_1 = true_outcome_nc(W, X, cfg, dgp)
        else:
            # naive / augmented: marginal E_U[E[T_a|X,U]] via MGF(U~N(0,1))
            G   = math.gamma(1.0 + 1.0 / cfg.k_t)
            c   = -cfg.beta_u_in_t / cfg.k_t
            mgf = np.exp(0.5 * c ** 2)
            xb  = X @ dgp['beta_t']
            h_0 = cfg.lam_t * G * np.exp(-xb / cfg.k_t) * mgf
            h_1 = cfg.lam_t * G * np.exp(-(xb + cfg.tau_log_hr) / cfg.k_t) * mgf
    else:
        hm = OutcomeProxyModel()
        if mode == "oracle":
            print("    Fitting Outcome Model (X, U) ...")
            hm.fit(None, X_forest, A, Y_ipcw)
            h_0, h_1 = hm.predict(None, X_forest)
        elif mode == "nc":
            print("    Fitting Outcome Proxy (W, X) ...")
            hm.fit(W, X, A, Y_ipcw)
            h_0, h_1 = hm.predict(W, X)
        else:
            print("    Fitting Outcome Model ...")
            hm.fit(None, X_forest, A, Y_ipcw)
            h_0, h_1 = hm.predict(None, X_forest)

    # ── Assemble Doubly-Robust GRF scores ────────────────────────────
    m = q * h_1 + (1 - q) * h_0
    D = A - q

    K_gamma = np.zeros_like(surv_C)
    K_H     = np.zeros_like(surv_C)
    for i in range(n):
        E_interp = np.interp(fail_times_C, fail_times_E, E_T_given_gt_t[i, :])
        K_gamma[i, :] = D[i] * (E_interp - m[i])
        K_H[i, :]     = D[i] ** 2

    Gamma, H = compute_grf_orthogonal_scores(
        Y_star, A, D, m, delta_s,
        surv_C, hazard_C, K_gamma, K_H, fail_times_C,
    )

    return Gamma, H, X_forest


# ====================================================================
# 4.  Evaluate one variant
# ====================================================================
def evaluate_variant(name, Gamma, H, X_forest, true_cate, n_trees=200):
    t0     = time.time()
    forest = HonestCausalForest(
        n_estimators=n_trees, min_samples_leaf=10, honesty=True,
    )
    forest.fit(X_forest, Gamma, H)
    preds  = forest.predict(X_forest)
    fit_t  = time.time() - t0

    bias = np.mean(preds - true_cate)
    mae  = np.mean(np.abs(preds - true_cate))
    rmse = np.sqrt(np.mean((preds - true_cate) ** 2))
    corr = np.corrcoef(preds, true_cate)[0, 1]

    return dict(
        name=name, bias=bias, mae=mae, rmse=rmse, corr=corr,
        mean_pred=np.mean(preds), std_pred=np.std(preds), forest_time=fit_t,
    )


# ====================================================================
# 5.  Main
# ====================================================================
def main():
    # ── generate data ────────────────────────────────────────────────
    cfg = SynthConfig(n=2000, p_x=5, seed=42)
    obs_df, truth_df, params = generate_synthetic_nc_cox(cfg)
    obs_df, truth_df = add_ground_truth_cate(obs_df, truth_df, cfg, params)

    dgp = recover_dgp_internals(cfg)

    x_cols    = [f"X{j}" for j in range(cfg.p_x)]
    X         = obs_df[x_cols].values
    W         = obs_df[["W"]].values            # (n, 1)
    Z         = obs_df[["Z"]].values            # (n, 1)
    A         = obs_df["A"].values
    Y         = obs_df["time"].values
    delta     = obs_df["event"].values
    U         = truth_df["U"].values
    true_cate = truth_df["CATE_XU_eq7"].values

    print("=" * 80)
    print("  8-Variant Benchmark:  Oracle | CSF Baseline | NC-CSF")
    print("=" * 80)
    print(f"  n = {cfg.n},  p = {cfg.p_x},  event rate = {delta.mean():.2f}")
    print(f"  True CATE   mean = {true_cate.mean():.4f},  "
          f"std = {true_cate.std():.4f}")
    print()

    # ── variant specs ────────────────────────────────────────────────
    variants = [
        # ---- Oracle (group A) ----
        ("A1  Oracle (all true)",
         dict(mode="oracle",    true_surv=True,  true_qr=True)),
        ("A2  Oracle (true surv, est q/r)",
         dict(mode="oracle",    true_surv=True,  true_qr=False)),
        ("A3  Oracle (all estimated)",
         dict(mode="oracle",    true_surv=False, true_qr=False)),
        # ---- CSF baselines (group B) ----
        ("B1  Naive CSF (X only)",
         dict(mode="naive",     true_surv=False, true_qr=False)),
        ("B2  Augmented (X+W+Z, no bridge)",
         dict(mode="augmented", true_surv=False, true_qr=False)),
        # ---- NC-CSF (group C) ----
        ("C1  NC-CSF (all true)",
         dict(mode="nc",        true_surv=True,  true_qr=True)),
        ("C2  NC-CSF (true surv, est q/r)",
         dict(mode="nc",        true_surv=True,  true_qr=False)),
        ("C3  NC-CSF (all estimated)",
         dict(mode="nc",        true_surv=False, true_qr=False)),
    ]

    results = []
    for idx, (name, kw) in enumerate(variants, 1):
        print(f"{'─' * 60}")
        print(f"  [{idx}/8]  {name}")
        print(f"{'─' * 60}")

        wall_t0 = time.time()
        Gamma, H, X_forest = build_pseudo_responses(
            X, U, W, Z, A, Y, delta, cfg, dgp, **kw,
        )
        nuis_t = time.time() - wall_t0
        print(f"    Nuisance phase : {nuis_t:.1f}s")

        row = evaluate_variant(name, Gamma, H, X_forest, true_cate)
        row['nuis_time'] = nuis_t
        row['total_time'] = nuis_t + row['forest_time']
        results.append(row)

        print(f"    Forest fit     : {row['forest_time']:.1f}s")
        print(f"    Bias  = {row['bias']:+.4f}")
        print(f"    RMSE  = {row['rmse']:.4f}")
        print(f"    Corr  = {row['corr']:.4f}")
        print()

    # ── summary table ────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"{'Variant':<40s} {'Bias':>8s} {'RMSE':>8s} "
          f"{'MAE':>8s} {'Corr':>8s} {'Time':>7s}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<40s} {r['bias']:>+8.4f} {r['rmse']:>8.4f} "
              f"{r['mae']:>8.4f} {r['corr']:>8.4f} {r['total_time']:>6.1f}s")
    print("=" * 90)

    # ── save CSV ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv("data/benchmark_8variant_results.csv", index=False)
    print("\nResults saved to data/benchmark_8variant_results.csv")


if __name__ == "__main__":
    main()
