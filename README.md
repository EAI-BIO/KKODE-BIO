> **LEGAL DISCLAIMER & INTENDED USE NOTICE**  
> K-KODE is a biostatistical calculation engine provided strictly for **Research Use Only (RUO)**. It is NOT a medical device, clinical decision-support system, or diagnostic instrument, and it has not been evaluated, cleared, or approved by the US FDA, Health Canada, EMA, or any medical regulatory authority. This software is intended exclusively for retrospective data analysis, exploratory statistical modeling, and trial protocol simulation. It must not be used for direct patient diagnosis, treatment planning, or clinical management.

---

# K-KODE APEX ENGINE (v55.0)
**Protocol-Grade Biostatistical Engine for Longitudinal Retinal Biomarker Decay & Clinical Trial Powering**

Developed by **EAI-BIO** for USH2A and inherited retinal degeneration trial planning.

---

## Executive Summary

Legacy clinical trial tools force curved biological cell loss onto flat linear rulers, creating mathematical software noise that inflates required cohort sizes. K-KODE Apex (v55.0) resolves this by fitting true biological decay curves through automated 4-model AIC/AICc competition, preserving fast-progressing floor-censored patients using dual Tobit Maximum Likelihood Estimation and Hierarchical Bayesian modeling, and empirically validating statistical power through Monte Carlo trial simulations.

---

## Core Engine Capabilities

* **Generalized Endpoint Engine:** Native modeling for structural endpoints (`ez_width_mm`, `ez_area_mm2`) alongside functional primary endpoints (`static_perimetry_sensitivity_db`, `microperimetry_sensitivity_db`) as recommended by the RUSH2A study group.
* **Dual-Stage Floor-Censoring Mechanics:** Eliminates floor-and-drop survivorship bias by evaluating floor points ($0.05\text{ mm}$) using Gaussian CDF likelihoods (Tobit MLE) at the individual level and PyMC MCMC (`pm.Censored`) at the cohort level—retaining 100% of patient visits.
* **Automated 4-Model Competition:** Competes Linear, Square-Root, Log-Exponential, and Power-Law decay curves on every trajectory, aggregating results via Akaike Weights ($w_i$) to isolate true biological progression velocity.
* **Finite-Difference Hessian Standard Errors:** Computes exact asymptotic parameter standard errors from central finite differences of the log-likelihood surface rather than relying on optimizer approximations.
* **Identifiability & Protocol Stability Guards:** Hardcoded rules require $\ge 2$ uncensored visits per patient to prevent single-point slope explosions ($\lambda \to \infty$) and apply an $n=4$ AIC fallback for standard 4-visit annual trial designs where small-sample AICc denominators equal zero.
* **Simulation-Validated Sizing Engine:** Combines closed-form $z$-bounds with vectorized Monte Carlo two-arm trial simulations to empirically measure statistical power (80%) under actual between-patient ($\tau_b^2$) and residual ($\sigma^2$) variance.

---

## Architectural Platform Comparison

| Capability / Metric | Legacy Trial Software (Cytel East, nQuery) | Ocular Imaging Platforms (RetinAI, Voxeleron) | Custom R / SAS Scripts | **K-KODE v55.0 Apex Engine** |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Scope** | Generic trial sample sizing | Single-scan image segmentation | Ad-hoc script modeling | **Longitudinal decay & protocol $N$ optimization** |
| **Decay Shape Fitting** | Forced linear ($\Delta y = -c \cdot t$) | N/A (Static scans) | Manual setup required | **Automated 4-Model AIC/AICc Competition** |
| **Patient Floor Logic** | Excludes or clamps zero values | Visible area only | Complex manual coding | **Tobit MLE Likelihood (Gaussian CDF)** |
| **Cohort Floor Logic** | Excludes floor-censored visits | N/A | Excludes floor-censored visits | **Hierarchical Bayesian Tobit (`pm.Censored`)** |
| **Endpoint Compatibility** | Unstructured metric | Image metrics only | Single metric script | **Generalized (EZ Width/Area, Perimetry dB)** |
| **Trial Powering Method** | Static $z$-score lookup tables | N/A | Manual script execution | **Dual Engine: Closed-Form + Monte Carlo Simulation** |
| **Pipeline Reliability** | Point-and-click GUI | Single-scan analysis | Fails on 1-visit or $n=4$ edge cases | **Identifiability Guarded ($<0.2\text{s}$ execution)** |

---

## Quick Start

```python
import pandas as pd
from kkode_engine import KKodeApexEngine

# 1. Load raw longitudinal visit dataset
df = pd.read_csv("patient_retinal_data.csv")

# 2. Initialize K-KODE v55.0 for any longitudinal endpoint
engine = KKodeApexEngine(
    data_source=df,
    endpoint_column="static_perimetry_sensitivity_db",  # Supports 'ez_width_mm', 'ez_area_mm2', etc.
    eye_column="eye",
    measurement_floor=0.05,
    higher_is_better=True
)

# 3. Execute complete single-command analysis pipeline
results = engine.run_full_analysis(
    target_power=0.80,
    alpha=0.05,
    therapeutic_efficacy=0.30,
    run_simulation=True,
    n_sims_per_candidate=150
)

# 4. Print executive biostatistical summary report
print(engine.generate_report())
