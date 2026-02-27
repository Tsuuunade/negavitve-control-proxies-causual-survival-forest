# Bias Analysis Report: Why the Baseline Model Fails


### Structural Equations

```
Z = U + X'β_z + noise           (proxy for U)
W = U + X'β_w + noise           (proxy for U)
P(A=1) = sigmoid(X'α + γ_u·U)   (treatment depends on U)
log h(t) = ... + β_u·U + τ·A    (Weibull Cox PH hazard model)
```

**Note**: In Cox PH, higher hazard → event happens sooner → **lower survival time Y**.

### Key Parameters

| Parameter | Value | In Model | Effect on Hazard | Effect on Y |
|-----------|-------|----------|------------------|-------------|
| `gamma_u_in_a` | 1.5 | U → A | - | High U → more likely treated |
| `beta_u_in_t` | 0.8 | U → hazard | ×2.2 per unit U | High U → **lower Y** (worse) |
| `tau_log_hr` | -0.6 | A → hazard | ×0.55 (45% reduction) | Treatment → **higher Y** (better) |

**Interpretation**:
- `beta_u_in_t = +0.8`: Positive coefficient in Cox PH → higher hazard → shorter survival → **lower Y**
- `tau_log_hr = -0.6`: Negative coefficient in Cox PH → lower hazard → longer survival → **higher Y** (treatment helps)

---

## The Bias Mechanism

### Step 1: Selection into Treatment

| U value | P(A=1) |
|---------|--------|
| U = -1 | ~18% |
| U = +1 | ~82% |

**High-U patients are treated more often.**

### Step 2: Outcome Depends on U

| U value | Expected Y |
|---------|------------|
| U = -1 | High (good) |
| U = +1 | Low (bad) |

**High-U patients have worse outcomes.**

### Step 3: The Composition Problem

| Group | Composition | Mean Y |
|-------|-------------|--------|
| Treated (A=1) | Mostly high-U | Low |
| Control (A=0) | Mostly low-U | High |

**Treated group looks worse—not because treatment hurts, but because sicker patients are treated.**

---

## 4. Numerical Example

1000 patients, 50% treated:

| U | Count | P(A=1) | # Treated | # Control | Y(0) | Y(1) |
|---|-------|--------|-----------|-----------|------|------|
| -1 | 500 | 18% | 90 | 410 | 5.0 | 6.5 |
| +1 | 500 | 82% | 410 | 90 | 2.0 | 3.0 |

**Observed means:**
- Treated: (90×6.5 + 410×3.0)/500 = **3.63**
- Control: (410×5.0 + 90×2.0)/500 = **4.46**

**Estimates:**
| | Value |
|--|-------|
| Baseline τ̂ | 3.63 - 4.46 = **-0.83**|
| True τ | **+1.25**|
