"""
How to use:

    from synthetic_dataset_generator import (
        SyntheticCausalSurvivalDatasetGenerator,
        SyntheticCausalSurvivalParams,
    )

    gen = SyntheticCausalSurvivalDatasetGenerator(
        SyntheticCausalSurvivalParams(seed=123, n=5000, p=20)
    )
    data = gen.generate()
    df = data["observed"]

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticCausalSurvivalParams:
    """
    Notes:
    - Other coefficients/constants (e.g., a_z/a_v, sparsity of b_z/b_v/alpha/beta_t,
      Weibull shape, baseline scale, etc.) are fixed to the same defaults used in
      the original notebook.
    """

    seed: int = 123
    n: int = 5000
    p: int = 20

    # Main knobs
    sigma_z: float = 0.6
    sigma_v: float = 0.6
    gamma_u: float = 0.2
    beta_u: float = 0.2
    tau: float = 0.2
    target_censor_rate: float = 0.0
    use_random_censoring: bool = False


def generate_synthetic_dataset(
    *,
    sigma_z: float = 0.6,
    sigma_v: float = 0.6,
    gamma_u: float = 0.2,
    beta_u: float = 0.2,
    tau: float = 0.2,
    target_censor_rate: float = 0.0,
    use_random_censoring: bool = False,
    seed: int = 123,
    n: int = 5000,
    p: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper for quick dataset generation.

    Exposes the most commonly tuned parameters directly in the function signature.
    For full control, instantiate `SyntheticCausalSurvivalDatasetGenerator` with
    `SyntheticCausalSurvivalParams`.
    """

    params = SyntheticCausalSurvivalParams(
        seed=seed,
        n=n,
        p=p,
        sigma_z=sigma_z,
        sigma_v=sigma_v,
        gamma_u=gamma_u,
        beta_u=beta_u,
        tau=tau,
        target_censor_rate=target_censor_rate,
        use_random_censoring=use_random_censoring,
    )
    return SyntheticCausalSurvivalDatasetGenerator(params, verbose=verbose).generate()


class SyntheticCausalSurvivalDatasetGenerator:
    """Generates the synthetic causal survival dataset.

    The generation procedure follows the notebook step-by-step.

    `generate()` returns a dict with:
      - "observed": DataFrame with X covariates, proxies (Z,V), treatment W, and (Y, Delta)
      - "hidden": DataFrame with latent U and potential outcomes (T0, T1) plus factual T
      - "arrays": raw numpy arrays (X, U, Z, V, W, T0, T1, T, C, Y, Delta)
      - "model": coefficients/vectors used (b_z, b_v, alpha, beta_t)
      - "meta": misc generation metadata (seed, achieved censoring rate, etc.)
    """

    def __init__(self, params: Optional[SyntheticCausalSurvivalParams] = None, *, verbose: bool = False):
        self.params = params or SyntheticCausalSurvivalParams()
        self.verbose = verbose

        if self.params.n <= 0:
            raise ValueError("n must be positive")
        if self.params.p <= 0:
            raise ValueError("p must be positive")
        if not (0.0 <= self.params.target_censor_rate <= 1.0):
            raise ValueError("target_censor_rate must be in [0, 1]")
        if self.params.sigma_z < 0 or self.params.sigma_v < 0:
            raise ValueError("sigma_z and sigma_v must be non-negative")

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def generate(self) -> Dict[str, Any]:
        p = self.params
        rng = np.random.default_rng(p.seed)

        # Fixed defaults (match the notebook)
        a_z = 1.5
        a_v = 1.5
        k_proxy = 5
        proxy_coeff_value = 0.2

        b0 = 0.0
        k_alpha = 5
        alpha_coeff_value = 0.1

        k_weib = 1.5
        lambda0 = 1500.0
        k_beta = 5
        beta_coeff_value = 0.1

        # Random censoring defaults (only used when use_random_censoring=True)
        rand_censor_scale = 5000.0

        # ------------------
        # Draw observed X
        X = rng.normal(loc=0.0, scale=1.0, size=(p.n, p.p))
        if self.verbose:
            print("X shape:", X.shape)

        # Draw latent U
        U = rng.normal(loc=0.0, scale=1.0, size=p.n)
        if self.verbose:
            print("U shape:", U.shape)

        # ------------------
        # Proxy coefficient vectors b_z, b_v
        b_z = np.zeros(p.p)
        b_v = np.zeros(p.p)
        proxy_idx_z = rng.choice(p.p, size=k_proxy, replace=False) if k_proxy > 0 else np.array([], dtype=int)
        proxy_idx_v = rng.choice(p.p, size=k_proxy, replace=False) if k_proxy > 0 else np.array([], dtype=int)
        if k_proxy > 0:
            b_z[proxy_idx_z] = proxy_coeff_value
            b_v[proxy_idx_v] = proxy_coeff_value

        # Generate proxies
        eps_z = rng.normal(loc=0.0, scale=p.sigma_z, size=p.n)
        eps_v = rng.normal(loc=0.0, scale=p.sigma_v, size=p.n)
        Z = a_z * U + (X @ b_z) + eps_z
        V = a_v * U + (X @ b_v) + eps_v

        if self.verbose:
            print("Corr(Z, U):", float(np.corrcoef(Z, U)[0, 1]))
            print("Corr(V, U):", float(np.corrcoef(V, U)[0, 1]))

        # ------------------
        # Treatment assignment
        alpha = np.zeros(p.p)
        alpha_idx = rng.choice(p.p, size=k_alpha, replace=False) if k_alpha > 0 else np.array([], dtype=int)
        if k_alpha > 0:
            alpha[alpha_idx] = alpha_coeff_value

        lin_treat = b0 + X @ alpha + p.gamma_u * U
        p_treat = self._sigmoid(lin_treat)
        W = rng.binomial(n=1, p=p_treat, size=p.n).astype(int)

        if self.verbose:
            print("Treatment prevalence (mean W):", float(W.mean()))

        # ------------------
        # Potential event times under w in {0,1}
        beta_t = np.zeros(p.p)
        beta_idx = rng.choice(p.p, size=k_beta, replace=False) if k_beta > 0 else np.array([], dtype=int)
        if k_beta > 0:
            beta_t[beta_idx] = beta_coeff_value

        eta0 = X @ beta_t + p.beta_u * U + p.tau * 0.0
        eta1 = X @ beta_t + p.beta_u * U + p.tau * 1.0

        # Weibull scale parameterization used in the notebook
        scale0 = lambda0 * np.exp(-eta0 / k_weib)
        scale1 = lambda0 * np.exp(-eta1 / k_weib)

        u_latent = rng.uniform(0.0, 1.0, size=p.n)
        u_latent = np.clip(u_latent, 1e-12, 1.0 - 1e-12)
        base = (-np.log(1.0 - u_latent)) ** (1.0 / k_weib)
        T0 = scale0 * base
        T1 = scale1 * base

        # Reveal factual time
        T = np.where(W == 1, T1, T0)

        # ------------------
        # Right censoring (administrative + optional random)
        c_admin = float(np.quantile(T, 1.0 - p.target_censor_rate))
        C = np.full(p.n, c_admin, dtype=float)

        if p.use_random_censoring:
            C_rand = rng.exponential(scale=rand_censor_scale, size=p.n)
            C = np.minimum(C, C_rand)

        Y = np.minimum(T, C)
        Delta = (T <= C).astype(int)
        censor_rate = float(1.0 - Delta.mean())

        if self.verbose:
            print(f"Calibrated administrative censoring time c_admin = {c_admin:.3f}")
            print(f"Achieved censoring rate = {censor_rate:.3%} (target {p.target_censor_rate:.0%})")

        # ------------------
        # Assemble DataFrames
        X_cols = [f"X{j + 1}" for j in range(p.p)]
        df_X = pd.DataFrame(X, columns=X_cols)

        df = pd.concat(
            [
                df_X,
                pd.DataFrame(
                    {
                        "Z": Z,
                        "V": V,
                        "W": W,
                        "Y": Y,
                        "Delta": Delta,
                    }
                ),
            ],
            axis=1,
        )

        df_hidden = pd.DataFrame({"U": U, "T0": T0, "T1": T1, "T": T})

        return {
            "observed": df,
            "hidden": df_hidden,
            "arrays": {
                "X": X,
                "U": U,
                "Z": Z,
                "V": V,
                "W": W,
                "T0": T0,
                "T1": T1,
                "T": T,
                "C": C,
                "Y": Y,
                "Delta": Delta,
            },
            "model": {
                "b_z": b_z,
                "b_v": b_v,
                "alpha": alpha,
                "beta_t": beta_t,
                "proxy_idx_z": proxy_idx_z,
                "proxy_idx_v": proxy_idx_v,
                "alpha_idx": alpha_idx,
                "beta_idx": beta_idx,
            },
            "meta": {
                "seed": p.seed,
                "n": p.n,
                "p": p.p,
                "c_admin": c_admin,
                "achieved_censor_rate": censor_rate,
            },
        }


if __name__ == "__main__":
    gen = SyntheticCausalSurvivalDatasetGenerator(verbose=True)
    data = gen.generate()
    print("Observed shape:", data["observed"].shape)
    print(data["observed"].head())
