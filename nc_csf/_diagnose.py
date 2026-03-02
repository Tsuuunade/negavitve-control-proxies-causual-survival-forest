"""Diagnose why all survival models produce negative Pearson r."""
import numpy as np
from data_generation import SynthConfig, generate_synthetic_nc_cox, add_ground_truth_cate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from models import (
    _fit_kaplan_meier_censoring, _evaluate_sc, _compute_ipcw_pseudo_outcome,
    _fit_event_cox, _predict_S_on_grid, _compute_Q_from_S, _compute_ipcw_3term_Y_res
)

cfg = SynthConfig(n=3000, p_x=10, seed=42, gamma_u_in_a=1.5, target_censor_rate=0.35)
obs_df, truth_df, params = generate_synthetic_nc_cox(cfg)
obs_df, truth_df = add_ground_truth_cate(obs_df, truth_df, cfg, params)

x_cols = sorted([c for c in obs_df.columns if c.startswith("X")], key=lambda s: int(s[1:]))
X = obs_df[x_cols].values
A = obs_df["A"].values
Y_time = obs_df["time"].values
Delta = obs_df["event"].values
Z = obs_df[["Z"]].values
W = obs_df[["W"]].values
T_true = truth_df["T"].values
true_cate = truth_df["CATE_XU_eq7"].values

print(f"n={len(X)}, censor_rate={1-Delta.mean():.2%}")
print(f"True CATE: mean={true_cate.mean():.4f}, std={true_cate.std():.4f}")
print(f"True T: mean={T_true.mean():.4f}, std={T_true.std():.4f}")
print(f"Observed Y: mean={Y_time.mean():.4f}, std={Y_time.std():.4f}")
print(f"  Y|A=1: mean={Y_time[A==1].mean():.4f}")
print(f"  Y|A=0: mean={Y_time[A==0].mean():.4f}")
print(f"True T|A=1: mean={T_true[A==1].mean():.4f}")
print(f"True T|A=0: mean={T_true[A==0].mean():.4f}")
print(f"True ATE (T|A=1 - T|A=0): {T_true[A==1].mean() - T_true[A==0].mean():.4f}")

print("\n" + "="*60)
print("STEP 1: Kaplan-Meier for censoring")
print("="*60)
km_times, km_surv = _fit_kaplan_meier_censoring(Y_time, Delta)
sc_at_Y = _evaluate_sc(Y_time, km_times, km_surv)
print(f"S^C(Y): mean={sc_at_Y.mean():.4f}, min={sc_at_Y.min():.4f}, max={sc_at_Y.max():.4f}")

print("\n" + "="*60)
print("STEP 2: Simple IPCW pseudo-outcome")
print("="*60)
Y_simple = _compute_ipcw_pseudo_outcome(Y_time, Delta, km_times, km_surv)
print(f"Y_tilde_simple: mean={Y_simple.mean():.4f}, std={Y_simple.std():.4f}")
print(f"  A=1: mean={Y_simple[A==1].mean():.4f}")
print(f"  A=0: mean={Y_simple[A==0].mean():.4f}")
print(f"  ATE from simple IPCW: {Y_simple[A==1].mean() - Y_simple[A==0].mean():.4f}")

print("\n" + "="*60)
print("STEP 3: Bridge functions (fitted on full data)")
print("="*60)
XZ = np.column_stack([X, Z])
XW = np.column_stack([X, W])

q_model = LogisticRegression(max_iter=1000)
q_model.fit(XZ, A)
q_pred = np.clip(q_model.predict_proba(XZ)[:, 1], 0.01, 0.99)

h1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42)
h0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42)
h1.fit(XW[A==1], Y_simple[A==1])
h0.fit(XW[A==0], Y_simple[A==0])
h1_pred = h1.predict(XW)
h0_pred = h0.predict(XW)
m_pred = q_pred * h1_pred + (1 - q_pred) * h0_pred

print(f"q: mean={q_pred.mean():.4f}, std={q_pred.std():.4f}")
print(f"h1: mean={h1_pred.mean():.4f}, std={h1_pred.std():.4f}")
print(f"h0: mean={h0_pred.mean():.4f}, std={h0_pred.std():.4f}")
print(f"h1-h0: mean={np.mean(h1_pred - h0_pred):.4f}")
print(f"m: mean={m_pred.mean():.4f}, std={m_pred.std():.4f}")

print("\n" + "="*60)
print("STEP 4: Per-arm Cox PH for Q_a")
print("="*60)
surv_features = np.column_stack([X, W, Z])
treated = A == 1
control = A == 0

cox_1, col_names = _fit_event_cox(Y_time[treated], Delta[treated], surv_features[treated])
cox_0, _ = _fit_event_cox(Y_time[control], Delta[control], surv_features[control])

event_times = np.sort(np.unique(Y_time[Delta == 1]))
all_times = np.sort(np.unique(Y_time))
if len(all_times) > 500:
    idx = np.linspace(0, len(all_times) - 1, 500, dtype=int)
    all_times = all_times[idx]
t_grid = all_times
G = len(t_grid)

S_hat_1 = _predict_S_on_grid(cox_1, col_names, surv_features, t_grid)
S_hat_0 = _predict_S_on_grid(cox_0, col_names, surv_features, t_grid)
Q_hat_1 = _compute_Q_from_S(S_hat_1, t_grid)
Q_hat_0 = _compute_Q_from_S(S_hat_0, t_grid)
Q_hat = np.where(treated[:, None], Q_hat_1, Q_hat_0)

print(f"t_grid: {G} points, range [{t_grid[0]:.4f}, {t_grid[-1]:.4f}]")
print(f"S_1 at first grid pt: mean={S_hat_1[:,0].mean():.4f}")
print(f"S_0 at first grid pt: mean={S_hat_0[:,0].mean():.4f}")
print(f"S_1 at last grid pt:  mean={S_hat_1[:,-1].mean():.4f}")
print(f"S_0 at last grid pt:  mean={S_hat_0[:,-1].mean():.4f}")
print(f"Q_1(t_0): mean={Q_hat_1[:,0].mean():.4f}, std={Q_hat_1[:,0].std():.4f}")
print(f"Q_0(t_0): mean={Q_hat_0[:,0].mean():.4f}, std={Q_hat_0[:,0].std():.4f}")
print(f"Q_1(t_0) should ≈ E[T|A=1] = {T_true[A==1].mean():.4f}")
print(f"Q_0(t_0) should ≈ E[T|A=0] = {T_true[A==0].mean():.4f}")

# Q at each individual's Y_i
Y_idx = np.clip(np.searchsorted(t_grid, Y_time, side='right') - 1, 0, G - 1)
Q_at_Y = Q_hat[np.arange(len(Y_time)), Y_idx]
print(f"\nQ(Y_i): mean={Q_at_Y.mean():.4f}, std={Q_at_Y.std():.4f}")
print(f"  censored: mean Q(Y)={Q_at_Y[Delta==0].mean():.4f}, mean Y={Y_time[Delta==0].mean():.4f}")
print(f"  Q(Y) > Y for censored? {(Q_at_Y[Delta==0] > Y_time[Delta==0]).mean():.2%}")

print("\n" + "="*60)
print("STEP 5: 3-term pseudo-outcome")
print("="*60)
Y_res = _compute_ipcw_3term_Y_res(
    Y_time, Delta, m_pred, Q_hat, t_grid, km_times, km_surv
)
Y_pseudo = Y_res + m_pred  # the 3-term pseudo-outcome before subtracting m

print(f"3-term pseudo-outcome (Ỹ = Y_res + m):")
print(f"  mean={Y_pseudo.mean():.4f}, std={Y_pseudo.std():.4f}")
print(f"  min={Y_pseudo.min():.4f}, max={Y_pseudo.max():.4f}")
print(f"  A=1: mean={Y_pseudo[A==1].mean():.4f}")
print(f"  A=0: mean={Y_pseudo[A==0].mean():.4f}")
print(f"  ATE from 3-term: {Y_pseudo[A==1].mean() - Y_pseudo[A==0].mean():.4f}")
print(f"  True ATE: {T_true[A==1].mean() - T_true[A==0].mean():.4f}")

print(f"\nY_res (after subtracting m):")
print(f"  mean={Y_res.mean():.4f}, std={Y_res.std():.4f}")
print(f"  min={Y_res.min():.4f}, max={Y_res.max():.4f}")
print(f"  pct 1/50/99: {np.percentile(Y_res,1):.4f} / {np.median(Y_res):.4f} / {np.percentile(Y_res,99):.4f}")

# Decompose into terms
sc_at_Y = _evaluate_sc(Y_time, km_times, km_surv)
sc_grid = _evaluate_sc(t_grid, km_times, km_surv)
Q_at_Y_vals = Q_hat[np.arange(len(Y_time)), Y_idx]

# Term 1: [ΔY + (1-Δ)Q(Y)] / S^C(Y)
numerator = Delta * Y_time + (1 - Delta) * Q_at_Y_vals
term1 = numerator / sc_at_Y

# Term 2: augmentation integral (Q only, no m)
log_sc = -np.log(np.maximum(sc_grid, 1e-10))
dLambda_C = np.diff(np.concatenate([[0.0], log_sc]))
grid_weight = dLambda_C / np.maximum(sc_grid, 1e-10)
integrand = grid_weight[None, :] * Q_hat
mask = np.arange(G)[None, :] <= Y_idx[:, None]
term2 = (integrand * mask).sum(axis=1)

Y_pseudo_check = term1 - term2

print(f"\nTerm decomposition (m subtracted OUTSIDE):")
print(f"  term1 [ΔY+(1-Δ)Q(Y)]/S^C: mean={term1.mean():.4f}, std={term1.std():.4f}")
print(f"  term2 Σ Q·dΛ^C/S^C:        mean={term2.mean():.4f}, std={term2.std():.4f}")
print(f"  Ỹ = term1 - term2:         mean={Y_pseudo_check.mean():.4f}, std={Y_pseudo_check.std():.4f}")
print(f"  Y_res = Ỹ - m:             mean={(Y_pseudo_check - m_pred).mean():.4f}")
print(f"  (should match Y_res mean={Y_res.mean():.4f})")

print("\n" + "="*60)
print("STEP 6: R-learner signal check")
print("="*60)
T_res = A - q_pred
signal = np.mean(Y_res * T_res) / np.mean(T_res**2)
print(f"T_res: mean={T_res.mean():.4f}, std={T_res.std():.4f}")
print(f"E[Y_res * T_res] / E[T_res^2] = {signal:.4f}")
print(f"True mean CATE = {true_cate.mean():.4f}")
print(f"Pearson r(Y_res * T_res, true_cate) = {np.corrcoef(Y_res * T_res, true_cate)[0,1]:.4f}")

# Check: what about using raw Y instead of 3-term?
Y_res_raw = Y_time - m_pred
signal_raw = np.mean(Y_res_raw * T_res) / np.mean(T_res**2)
print(f"\nUsing raw Y (no IPCW):")
print(f"  E[Y_res_raw * T_res] / E[T_res^2] = {signal_raw:.4f}")

# Check: what about using simple IPCW?
Y_res_simple = Y_simple - m_pred
signal_simple = np.mean(Y_res_simple * T_res) / np.mean(T_res**2)
print(f"\nUsing simple IPCW pseudo-outcome:")
print(f"  E[Y_res_simple * T_res] / E[T_res^2] = {signal_simple:.4f}")

# Check: identity 1/S^C(Y) = 1 + integral(dLambda^C/S^C)
lhs = 1.0 / sc_at_Y
rhs_terms = np.zeros_like(Y_time)
for i in range(len(Y_time)):
    rhs_terms[i] = 1.0 + grid_weight[mask[i]].sum()
print(f"\nIdentity check: 1/S^C(Y) vs 1 + Σ dΛ^C/S^C:")
print(f"  LHS mean: {lhs.mean():.4f}")
print(f"  RHS mean: {rhs_terms.mean():.4f}")
print(f"  Max |diff|: {np.max(np.abs(lhs - rhs_terms)):.6f}")
