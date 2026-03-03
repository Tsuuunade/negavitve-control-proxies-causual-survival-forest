import time
import numpy as np
import pandas as pd
from nc_csf.data_generation import generate_synthetic_nc_cox, SynthConfig, add_ground_truth_cate
from nccsf_grf import NCCausalSurvivalForest

def test_nccsf_grf_performance():
    print("=" * 60)
    print("Performance & Time Test: Negative Control Causal Survival Forest (GRF)")
    print("=" * 60)
    
    # 1. Generate Synthetic Data
    n_samples = 2000
    p_covariates = 5
    print(f"Generating synthetic dataset with N={n_samples}, P={p_covariates}...")
    cfg = SynthConfig(n=n_samples, p_x=p_covariates, seed=42)
    obs_df, truth_df, params = generate_synthetic_nc_cox(cfg)
    obs_df, truth_df = add_ground_truth_cate(obs_df, truth_df, cfg, params)
    obs_df.to_csv('data/latest_nc_synth.csv', index=False)
    truth_df.to_csv('data/latest_nc_truth.csv', index=False)
    print("Dataset saved to data/latest_nc_synth.csv")

    
    # Extract variables
    x_cols = [f"X{j}" for j in range(p_covariates)]
    X = obs_df[x_cols].values
    W = obs_df[["W"]].values
    Z = obs_df[["Z"]].values
    A = obs_df["A"].values
    Y = obs_df["time"].values
    delta = obs_df["event"].values
    
    true_cate = truth_df["CATE_XU_eq7"].values
    
    # Determine RMST Horizon L (e.g., 90th percentile of observed times)
    L = np.percentile(Y, 90)
    print(f"Data generated. Mean Event Rate: {np.mean(delta):.2f}, RMST Horizon L={L:.2f}")
    
    # Initialize the model
    # Use smaller tree counts to make the test run moderately fast, 
    # but still show valid statistical performance
    model = NCCausalSurvivalForest(
        target="unrestricted",
        n_estimators=100, 
        min_samples_leaf=20, 
        honesty=True
    )
    
    # 2. Benchmark Fit Time
    print("\n--- Starting Model Training ---")
    start_time = time.time()
    
    try:
        model.fit(X, W, Z, A, Y, delta)
        fit_time = time.time() - start_time
        print(f"--- Training Complete in {fit_time:.2f} seconds ---")
        
        # 3. Benchmark Prediction Time
        print("\n--- Starting CATE Prediction ---")
        pred_start_time = time.time()
        cate_preds = model.predict(X)
        pred_time = time.time() - pred_start_time
        print(f"--- Prediction Complete in {pred_time:.2f} seconds ---")
        
        # 4. Statistical Performance Analysis
        # Since GRF calculates RMST effects (bounded by L) and truth is unbounded expected 
        # difference, they won't match 1:1 perfectly, but they should be highly correlated.
        bias = np.mean(cate_preds - true_cate)
        mae = np.mean(np.abs(cate_preds - true_cate))
        mse = np.mean((cate_preds - true_cate)**2)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(cate_preds, true_cate)[0, 1]
        
        print("\n--- Detailed Statistical Performance (In-Sample) ---")
        print(f"Mean Predicted CATE (RMST):  {np.mean(cate_preds):.4f} (StdDev: {np.std(cate_preds):.4f})")
        print(f"Mean True CATE (Unbounded):  {np.mean(true_cate):.4f} (StdDev: {np.std(true_cate):.4f})")
        print(f"Bias (Pred - True):          {bias:.4f}")
        print(f"Mean Absolute Error (MAE):   {mae:.4f}")
        print(f"Mean Squared Error (MSE):    {mse:.4f}")
        print(f"Root Mean Sq Error (RMSE):   {rmse:.4f}")
        print(f"Pearson Correlation:         {correlation:.4f}")
        print("============================================================")
        
    except Exception as e:
        print(f"\n[ERROR] Model training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nccsf_grf_performance()
