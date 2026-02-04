"""
NC-CSF: Negative Control Causal Survival Forest (Built on EconML)

A Python implementation of Causal Survival Forests with Negative Controls
for handling unmeasured confounding, built on top of econml's CausalForest.

Classes:
    NCCausalForestDML: Main NC-CSF implementation using bridge functions
    NCCausalForestDMLOracle: Oracle version with true confounder U
    BaselineCausalForestDML: Standard CausalForestDML for comparison
    BridgeFunctionEstimator: Cross-fitted bridge function estimation
"""

from .data_generation import (
    SynthConfig,
    SynthParams,
    generate_synthetic_nc_cox,
    add_ground_truth_cate,
)

from .models import (
    # Main classes (econml-based)
    NCCausalForestDML,
    NCCausalForestDMLOracle,
    BaselineCausalForestDML,
    BridgeFunctionEstimator,
    # Legacy aliases
    NCCausalSurvivalForest,
    NCCSFCorrectBridge,
)

__all__ = [
    # Data generation
    'SynthConfig',
    'SynthParams',
    'generate_synthetic_nc_cox',
    'add_ground_truth_cate',
    # Main model classes
    'NCCausalForestDML',
    'NCCausalForestDMLOracle',
    'BaselineCausalForestDML',
    'BridgeFunctionEstimator',
    # Legacy aliases
    'NCCausalSurvivalForest',
    'NCCSFCorrectBridge',
]

