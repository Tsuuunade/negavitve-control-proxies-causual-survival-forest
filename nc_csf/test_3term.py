"""Quick functional test of the 3-term IPCW survival models."""
import numpy as np
from data_generation import SynthConfig, generate_synthetic_nc_cox, add_ground_truth_cate
from models import NCSurvivalForestDML, NCSurvivalForestDMLOracle, BaselineSurvivalForestDML

cfg = SynthConfig(n=500, p_x=5, seed=42, gamma_u_in_a=1.5, target_censor_rate=0.35)
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

print(f"n={len(X)}, censor_rate={1 - event.mean():.2%}")
print(f"True CATE: mean={true_cate.mean():.4f} std={true_cate.std():.4f}")

fkw = dict(n_estimators=100, min_samples_leaf=20, cv=2, random_state=42)

# --- NC-CSF (3-term IPCW) ---
print("\n1. NCSurvivalForestDML (3-term IPCW)...")
m1 = NCSurvivalForestDML(**fkw)
m1.fit_survival(X, A, time, event, Z, W)
p1 = m1.effect(X).ravel()
print(f"   Pred: mean={p1.mean():.4f} std={p1.std():.4f}")
print(f"   RMSE={np.sqrt(np.mean((p1-true_cate)**2)):.4f}  r={np.corrcoef(true_cate,p1)[0,1]:.4f}")

# --- Oracle (3-term IPCW) ---
print("\n2. NCSurvivalForestDMLOracle (3-term IPCW)...")
m2 = NCSurvivalForestDMLOracle(**fkw)
m2.fit_oracle(X, A, time, event, U)
p2 = m2.effect(X).ravel()
print(f"   Pred: mean={p2.mean():.4f} std={p2.std():.4f}")
print(f"   RMSE={np.sqrt(np.mean((p2-true_cate)**2)):.4f}  r={np.corrcoef(true_cate,p2)[0,1]:.4f}")

# --- Baseline (3-term IPCW) ---
print("\n3. BaselineSurvivalForestDML (3-term IPCW)...")
m3 = BaselineSurvivalForestDML(**fkw)
m3.fit_baseline(X, A, time, event)
p3 = m3.effect(X).ravel()
print(f"   Pred: mean={p3.mean():.4f} std={p3.std():.4f}")
print(f"   RMSE={np.sqrt(np.mean((p3-true_cate)**2)):.4f}  r={np.corrcoef(true_cate,p3)[0,1]:.4f}")

print("\nSUCCESS - all 3 models completed")
