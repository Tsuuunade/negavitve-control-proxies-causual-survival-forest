"""
Comprehensive evaluation of NC-CSF performance.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from data_generation import SynthConfig, generate_synthetic_nc_cox, add_ground_truth_cate
from models import NCCausalForestDML, NCCausalForestDMLOracle, BaselineCausalForestDML


def evaluate_model(true_cate, pred_cate, name):
    """Compute comprehensive evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(true_cate, pred_cate))
    mae = mean_absolute_error(true_cate, pred_cate)
    bias = np.mean(pred_cate - true_cate)
    pearson_r, _ = pearsonr(true_cate, pred_cate)
    spearman_r, _ = spearmanr(true_cate, pred_cate)
    
    # Relative error
    rel_rmse = rmse / np.std(true_cate)
    
    # Calibration: slope of pred vs true
    slope = np.cov(true_cate, pred_cate)[0, 1] / np.var(true_cate)
    
    return {
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'Bias': bias,
        'Pearson r': pearson_r,
        'Spearman r': spearman_r,
        'Rel RMSE': rel_rmse,
        'Slope': slope,
    }


def run_experiment(seed, gamma_u=1.5, n=2000):
    """Run single experiment."""
    cfg = SynthConfig(n=n, gamma_u_in_a=gamma_u, seed=seed)
    obs_df, truth_df, params = generate_synthetic_nc_cox(cfg)
    obs_df, truth_df = add_ground_truth_cate(obs_df, truth_df, cfg, params)
    
    x_cols = [c for c in obs_df.columns if c.startswith('X')]
    X = obs_df[x_cols].values
    A = obs_df['A'].values
    Y = obs_df['time'].values
    Z = obs_df['Z'].values
    W = obs_df['W'].values
    U = truth_df['U'].values
    true_cate = truth_df['CATE_XU_eq7'].values
    
    split = train_test_split(
        X, A, Y, Z, W, U, true_cate, test_size=0.3, random_state=seed
    )
    X_tr, X_te = split[0], split[1]
    A_tr, A_te = split[2], split[3]
    Y_tr, Y_te = split[4], split[5]
    Z_tr, Z_te = split[6], split[7]
    W_tr, W_te = split[8], split[9]
    U_tr, U_te = split[10], split[11]
    cate_tr, cate_te = split[12], split[13]
    
    results = []
    
    # Baseline
    baseline = BaselineCausalForestDML(n_estimators=200, min_samples_leaf=20, random_state=seed)
    baseline.fit_baseline(X_tr, A_tr, Y_tr, verbose=False)
    pred_baseline = baseline.effect(X_te).ravel()
    results.append(evaluate_model(cate_te, pred_baseline, 'Baseline'))
    
    # NC-CSF (cv=5 replaces n_crossfit_splits=5)
    nccsf = NCCausalForestDML(n_estimators=200, min_samples_leaf=20, cv=5, random_state=seed)
    nccsf.fit(Y=Y_tr, T=A_tr, X=X_tr, Z=Z_tr, W=W_tr)
    pred_nccsf = nccsf.effect(X_te).ravel()
    results.append(evaluate_model(cate_te, pred_nccsf, 'NC-CSF'))
    
    # Oracle (pass U through W parameter)
    oracle = NCCausalForestDMLOracle(n_estimators=200, min_samples_leaf=20, cv=5, random_state=seed)
    oracle.fit(Y=Y_tr, T=A_tr, X=X_tr, W=U_tr)
    pred_oracle = oracle.effect(X_te).ravel()
    results.append(evaluate_model(cate_te, pred_oracle, 'Oracle'))
    
    return pd.DataFrame(results), cate_te, pred_baseline, pred_nccsf, pred_oracle
    
    return pd.DataFrame(results), cate_te, pred_baseline, pred_nccsf, pred_oracle


def main():
    print('=' * 70)
    print('NC-CSF COMPREHENSIVE EVALUATION')
    print('=' * 70)
    
    # 1. Multiple seeds evaluation
    print('\n1. STABILITY ACROSS RANDOM SEEDS (gamma_u=1.5, n=2000)')
    print('-' * 70)
    
    all_results = []
    for seed in [42, 123, 456, 789, 1011]:
        print(f'  Running seed {seed}...')
        df, _, _, _, _ = run_experiment(seed, gamma_u=1.5, n=2000)
        df['seed'] = seed
        all_results.append(df)
    
    combined = pd.concat(all_results)
    summary = combined.groupby('Model').agg({
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'Bias': ['mean', 'std'],
        'Pearson r': ['mean', 'std'],
    }).round(4)
    print('\nAggregated Results (mean ± std):')
    print(summary.to_string())
    
    # 2. Varying confounding strength
    print('\n\n2. VARYING CONFOUNDING STRENGTH (n=2000, seed=42)')
    print('-' * 70)
    
    gamma_results = []
    for gamma in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        print(f'  Running gamma_u={gamma}...')
        df, _, _, _, _ = run_experiment(42, gamma_u=gamma, n=2000)
        df['gamma_u'] = gamma
        gamma_results.append(df)
    
    gamma_combined = pd.concat(gamma_results)
    print('\nRMSE by Confounding Strength:')
    rmse_pivot = gamma_combined.pivot(index='gamma_u', columns='Model', values='RMSE')
    print(rmse_pivot.round(4).to_string())
    
    print('\nImprovement of NC-CSF over Baseline:')
    for gamma in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        baseline_rmse = gamma_combined[(gamma_combined['gamma_u'] == gamma) & 
                                       (gamma_combined['Model'] == 'Baseline')]['RMSE'].values[0]
        nccsf_rmse = gamma_combined[(gamma_combined['gamma_u'] == gamma) & 
                                    (gamma_combined['Model'] == 'NC-CSF')]['RMSE'].values[0]
        oracle_rmse = gamma_combined[(gamma_combined['gamma_u'] == gamma) & 
                                     (gamma_combined['Model'] == 'Oracle')]['RMSE'].values[0]
        improvement = (baseline_rmse - nccsf_rmse) / baseline_rmse * 100
        if baseline_rmse != oracle_rmse:
            oracle_gap_closed = (baseline_rmse - nccsf_rmse) / (baseline_rmse - oracle_rmse) * 100
        else:
            oracle_gap_closed = 100
        print(f'  gamma_u={gamma}: NC-CSF improves {improvement:.1f}% over baseline, '
              f'closes {oracle_gap_closed:.1f}% of oracle gap')
    
    # 3. Detailed metrics
    print('\n\n3. DETAILED METRICS (gamma_u=1.5, seed=42)')
    print('-' * 70)
    detail_df, cate_te, pred_baseline, pred_nccsf, pred_oracle = run_experiment(42, gamma_u=1.5, n=2000)
    print(detail_df.to_string(index=False))
    
    # 4. Distribution analysis
    print('\n\n4. PREDICTION DISTRIBUTION ANALYSIS')
    print('-' * 70)
    print(f'True CATE:     mean={cate_te.mean():.4f}, std={cate_te.std():.4f}, '
          f'min={cate_te.min():.4f}, max={cate_te.max():.4f}')
    print(f'Baseline pred: mean={pred_baseline.mean():.4f}, std={pred_baseline.std():.4f}, '
          f'min={pred_baseline.min():.4f}, max={pred_baseline.max():.4f}')
    print(f'NC-CSF pred:   mean={pred_nccsf.mean():.4f}, std={pred_nccsf.std():.4f}, '
          f'min={pred_nccsf.min():.4f}, max={pred_nccsf.max():.4f}')
    print(f'Oracle pred:   mean={pred_oracle.mean():.4f}, std={pred_oracle.std():.4f}, '
          f'min={pred_oracle.min():.4f}, max={pred_oracle.max():.4f}')
    
    # 5. Error quantiles
    print('\n\n5. ABSOLUTE ERROR QUANTILES')
    print('-' * 70)
    for name, pred in [('Baseline', pred_baseline), ('NC-CSF', pred_nccsf), ('Oracle', pred_oracle)]:
        errors = np.abs(cate_te - pred)
        print(f'{name:10s}: 50th={np.percentile(errors, 50):.4f}, '
              f'75th={np.percentile(errors, 75):.4f}, '
              f'90th={np.percentile(errors, 90):.4f}, '
              f'95th={np.percentile(errors, 95):.4f}')
    
    print('\n' + '=' * 70)
    print('EVALUATION COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
