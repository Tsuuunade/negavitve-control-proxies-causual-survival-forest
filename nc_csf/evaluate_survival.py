"""
Evaluation of NC-CSF Survival Models (with censoring via IPCW).

Compares:
  1. NCSurvivalForestDML      — NC bridge functions + IPCW
  2. NCSurvivalForestDMLOracle — True U + IPCW
  3. BaselineSurvivalForestDML — Ignores NC + IPCW
  4. NCCausalForestDML         — NC bridge functions, NO censoring adjustment (uses raw Y)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from data_generation import SynthConfig, generate_synthetic_nc_cox, add_ground_truth_cate
from models import (
    NCSurvivalForestDML,
    NCSurvivalForestDMLOracle,
    BaselineSurvivalForestDML,
    NCCausalForestDML,
)


def evaluate_model(true_cate, pred_cate, name):
    """Compute evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(true_cate, pred_cate))
    mae = mean_absolute_error(true_cate, pred_cate)
    bias = np.mean(pred_cate - true_cate)
    pearson_r, _ = pearsonr(true_cate, pred_cate)
    spearman_r, _ = spearmanr(true_cate, pred_cate)
    slope = np.cov(true_cate, pred_cate)[0, 1] / np.var(true_cate)

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "Pearson r": pearson_r,
        "Spearman r": spearman_r,
        "Slope": slope,
    }


def run_survival_experiment(seed=42, n=3000, gamma_u=1.5, censor_rate=0.35):
    """Run one experiment comparing all survival models."""
    cfg = SynthConfig(
        n=n,
        p_x=10,
        seed=seed,
        gamma_u_in_a=gamma_u,
        target_censor_rate=censor_rate,
    )
    obs_df, truth_df, params = generate_synthetic_nc_cox(cfg)
    obs_df, truth_df = add_ground_truth_cate(obs_df, truth_df, cfg, params)

    x_cols = sorted([c for c in obs_df.columns if c.startswith("X")], key=lambda s: int(s[1:]))
    X = obs_df[x_cols].values
    A = obs_df["A"].values
    time = obs_df["time"].values
    event = obs_df["event"].values
    Z = obs_df["Z"].values
    W = obs_df["W"].values
    U = truth_df["U"].values
    true_cate = truth_df["CATE_XU_eq7"].values

    actual_censor_rate = 1.0 - event.mean()

    # Train/test split
    split = train_test_split(
        X, A, time, event, Z, W, U, true_cate,
        test_size=0.3, random_state=seed,
    )
    X_tr, X_te = split[0], split[1]
    A_tr, A_te = split[2], split[3]
    time_tr, time_te = split[4], split[5]
    event_tr, event_te = split[6], split[7]
    Z_tr, Z_te = split[8], split[9]
    W_tr, W_te = split[10], split[11]
    U_tr, U_te = split[12], split[13]
    cate_tr, cate_te = split[14], split[15]

    forest_kwargs = dict(n_estimators=200, min_samples_leaf=20, cv=5, random_state=seed)
    results = []

    # --- 1) NC Survival Forest (IPCW) ---
    print("    Fitting NCSurvivalForestDML...")
    nc_surv = NCSurvivalForestDML(**forest_kwargs)
    nc_surv.fit_survival(X_tr, A_tr, time_tr, event_tr, Z_tr, W_tr)
    pred_nc_surv = nc_surv.effect(X_te).ravel()
    results.append(evaluate_model(cate_te, pred_nc_surv, "NC-CSF (IPCW)"))

    # --- 2) Oracle Survival Forest (IPCW) ---
    print("    Fitting NCSurvivalForestDMLOracle...")
    oracle_surv = NCSurvivalForestDMLOracle(**forest_kwargs)
    oracle_surv.fit_oracle(X_tr, A_tr, time_tr, event_tr, U_tr)
    pred_oracle = oracle_surv.effect(X_te).ravel()
    results.append(evaluate_model(cate_te, pred_oracle, "Oracle (IPCW)"))

    # --- 3) Baseline Survival Forest (IPCW, no NC) ---
    print("    Fitting BaselineSurvivalForestDML...")
    base_surv = BaselineSurvivalForestDML(**forest_kwargs)
    base_surv.fit_baseline(X_tr, A_tr, time_tr, event_tr)
    pred_baseline = base_surv.effect(X_te).ravel()
    results.append(evaluate_model(cate_te, pred_baseline, "Baseline (IPCW)"))

    # --- 4) NC-CSF without censoring adjustment (uses raw observed time) ---
    print("    Fitting NCCausalForestDML (no IPCW)...")
    nc_raw = NCCausalForestDML(**forest_kwargs)
    nc_raw.fit(Y=time_tr, T=A_tr, X=X_tr, Z=Z_tr, W=W_tr)
    pred_nc_raw = nc_raw.effect(X_te).ravel()
    results.append(evaluate_model(cate_te, pred_nc_raw, "NC-CSF (no IPCW)"))

    return pd.DataFrame(results), actual_censor_rate, cate_te


def main():
    print("=" * 74)
    print("NC-CSF SURVIVAL MODEL EVALUATION (IPCW for Censoring)")
    print("=" * 74)

    # ---- Experiment 1: Multiple seeds ----
    print("\n1. STABILITY ACROSS SEEDS (n=3000, gamma_u=1.5, censor_rate=0.35)")
    print("-" * 74)

    all_results = []
    for seed in [42, 123, 456]:
        print(f"  Seed {seed}:")
        df, cr, _ = run_survival_experiment(seed=seed, n=3000, gamma_u=1.5, censor_rate=0.35)
        df["seed"] = seed
        df["censor_rate"] = cr
        all_results.append(df)

    combined = pd.concat(all_results)
    summary = combined.groupby("Model").agg({
        "RMSE": ["mean", "std"],
        "Bias": ["mean", "std"],
        "Pearson r": ["mean", "std"],
    }).round(4)
    print("\nAggregated (mean +/- std):")
    print(summary.to_string())

    # ---- Experiment 2: Varying censoring rate ----
    print("\n\n2. VARYING CENSORING RATE (n=3000, gamma_u=1.5, seed=42)")
    print("-" * 74)

    cr_results = []
    for cr in [0.0, 0.15, 0.35, 0.50]:
        print(f"  Target censor_rate={cr}:")
        df, actual_cr, _ = run_survival_experiment(seed=42, n=3000, gamma_u=1.5, censor_rate=cr)
        df["target_cr"] = cr
        df["actual_cr"] = actual_cr
        cr_results.append(df)

    cr_combined = pd.concat(cr_results)
    print("\nRMSE by Censoring Rate:")
    rmse_pivot = cr_combined.pivot(index="target_cr", columns="Model", values="RMSE")
    print(rmse_pivot.round(4).to_string())

    print("\nPearson r by Censoring Rate:")
    r_pivot = cr_combined.pivot(index="target_cr", columns="Model", values="Pearson r")
    print(r_pivot.round(4).to_string())

    print("\nBias by Censoring Rate:")
    bias_pivot = cr_combined.pivot(index="target_cr", columns="Model", values="Bias")
    print(bias_pivot.round(4).to_string())

    # ---- Experiment 3: Detailed single run ----
    print("\n\n3. DETAILED METRICS (n=3000, gamma_u=1.5, censor_rate=0.35, seed=42)")
    print("-" * 74)
    detail_df, cr, cate_te = run_survival_experiment(seed=42, n=3000, gamma_u=1.5, censor_rate=0.35)
    print(f"  Actual censor rate: {cr:.2%}")
    print(detail_df.round(4).to_string(index=False))

    print("\n  True CATE distribution:")
    print(f"    mean={cate_te.mean():.4f}  std={cate_te.std():.4f}  "
          f"min={cate_te.min():.4f}  max={cate_te.max():.4f}")

    print("\n" + "=" * 74)
    print("EVALUATION COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()
