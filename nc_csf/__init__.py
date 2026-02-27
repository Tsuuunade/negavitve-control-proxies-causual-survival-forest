"""
NC-CSF: Negative Control Causal Survival Forest (Built on EconML)

A Python implementation of Causal Survival Forests with Negative Controls
for handling unmeasured confounding, built on top of econml's CausalForest.

Classes (non-survival):
    NCCausalForestDML: NC-CSF using bridge functions
    NCCausalForestDMLOracle: Oracle version with true confounder U
    BaselineCausalForestDML: Standard CausalForestDML (ignores NC)

Classes (survival / censored):
    NCSurvivalForestDML: NC-CSF with IPCW for censored data
    NCSurvivalForestDMLOracle: Oracle version with IPCW
    BaselineSurvivalForestDML: Baseline with IPCW (ignores NC)
"""

from .data_generation import (
    SynthConfig,
    SynthParams,
    generate_synthetic_nc_cox,
    add_ground_truth_cate,
)

from .models import (
    # Non-survival classes
    NCCausalForestDML,
    NCCausalForestDMLOracle,
    BaselineCausalForestDML,
    # Survival (IPCW) classes
    NCSurvivalForestDML,
    NCSurvivalForestDMLOracle,
    BaselineSurvivalForestDML,
)

__all__ = [
    # Data generation
    'SynthConfig',
    'SynthParams',
    'generate_synthetic_nc_cox',
    'add_ground_truth_cate',
    # Non-survival model classes
    'NCCausalForestDML',
    'NCCausalForestDMLOracle',
    'BaselineCausalForestDML',
    # Survival model classes
    'NCSurvivalForestDML',
    'NCSurvivalForestDMLOracle',
    'BaselineSurvivalForestDML',
]

