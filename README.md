## LEGAL DISCLAIMER & INTENDED USE NOTICE

K-KODE is a biostatistical calculation engine provided strictly for Research Use Only (RUO). It is NOT a medical device, clinical decision-support system, or diagnostic instrument, and it has not been evaluated, cleared, or approved by the US FDA, Health Canada, EMA, or any medical regulatory authority. This software is intended exclusively for retrospective data analysis, exploratory statistical modeling, and trial protocol simulation. It must not be used for direct patient diagnosis, treatment planning, or clinical management.

# K-KODE ENGINE (v55.0)

**Protocol-Grade Biostatistical Engine for Longitudinal Retinal Biomarker Decay & Clinical Trial Powering**

Developed by EAI-BIO for USH2A and inherited retinal degeneration trial planning.

## Executive Summary

Legacy clinical trial tools force curved biological decay onto flat linear assumptions, inflating the cohort sizes a trial actually needs. K-KODE (v55.0) fits the true shape of that decay directly: automated 4-model competition selects the best-supported functional form per patient, a dual-stage Tobit/Bayesian censoring engine keeps fast-progressing patients in the analysis instead of excluding them once they cross the measurement floor, and every sample-size estimate is cross-checked with a full Monte Carlo trial simulation — not just a formula. No other tool in this space combines all four of those in one pipeline.

## Core Engine Capabilities

**Generalized Endpoint Engine** — Native modeling for structural endpoints (`ez_width_mm`, `ez_area_mm2`) alongside functional primary endpoints (`static_perimetry_sensitivity_db`, `microperimetry_sensitivity_db`), matching the endpoint priorities RUSH2A's own investigators recommended.

**Dual-Stage Floor-Censoring Mechanics** — Eliminates naive floor-and-drop bias by evaluating measurement-floor visits with a censored (Tobit) likelihood at the individual-patient level, and a full hierarchical Bayesian model (PyMC, `pm.Censored`) at the population level. Every patient with at least two measurable visits stays in the analysis, even after crossing the floor — a substantial reduction in excluded data compared to standard exclusion-based approaches.

**Automated 4-Model Competition** — Competes Linear, Square-Root, Log-Exponential, and Power-Law decay curves on every patient trajectory, aggregating results via Akaike weights to identify the best-supported biological progression shape — validated against ground-truth-generated data to correctly recover the true underlying model.

**Finite-Difference Hessian Standard Errors** — Computes parameter standard errors from a central finite-difference Hessian of the log-likelihood surface at the fitted optimum, rather than trusting an optimizer's internal approximation.

**Identifiability & Protocol Stability Guards** — Requires ≥2 uncensored visits per patient before fitting, preventing single-point slope explosions, and applies a validated small-sample AIC fallback for standard 4-visit annual trial designs, where the AICc correction is otherwise mathematically undefined.

**Simulation-Validated Sizing Engine** — Combines a fast closed-form z-based estimate with a vectorized Monte Carlo two-arm trial simulation, empirically measuring statistical power under the cohort's actual between-patient and residual variance rather than relying on formula assumptions alone.

**Performance** — Per-patient decay fitting runs in roughly 0.1 seconds; a full standard analysis (model competition, decay fitting, population modeling, and simulation-validated sample sizing) completes on a 60-patient cohort in under 10 seconds. The optional hierarchical Bayesian population model — the piece that goes further than any comparable tool by modeling censored data at the cohort level — runs its full MCMC sampler in under two minutes.

## Architectural Platform Comparison

| Capability / Metric | Generic Trial-Sizing Software (e.g. East, nQuery) | Ocular Imaging Platforms (e.g. RetinAI, Voxeleron) | Custom R / SAS Scripts | K-KODE Engine v55.0 |
|---|---|---|---|---|
| Primary Scope | Trial sample sizing for standard designs | Image segmentation & measurement extraction | One-off, analyst-built modeling | Longitudinal decay modeling **and** protocol N optimization, in one pipeline |
| Decay Shape Fitting | Built for standard/user-specified designs; no automated biological-curve competition | Not applicable — measures a single scan, doesn't model change over time | Possible, but requires manual setup for every new dataset | Automated 4-model AIC/AICc competition, run per patient by default |
| Patient Floor Logic | Not domain-specific | Not applicable | Typically excludes floor values unless custom-coded | Tobit MLE censored likelihood, built in |
| Cohort Floor Logic | Not domain-specific | Not applicable | Typically excludes floor-censored visits unless custom-coded | Hierarchical Bayesian censored model (PyMC) available for heavy censoring |
| Endpoint Compatibility | General-purpose, not disease-specific | Image-derived metrics only | Whatever the analyst codes, one dataset at a time | Generalized across any longitudinal endpoint via configuration |
| Trial Powering Method | Closed-form formulas | Not applicable — no trial-powering function | Manual, per-analyst | Closed-form estimate **plus** Monte Carlo simulation validation |
| Reusability | Productized, general-purpose | Productized for imaging workflows | Rebuilt per study, not reusable | Config-driven and reusable across cohorts and endpoints out of the box |

*Named platforms are referenced by publicly documented capabilities as of this writing; specific feature sets are set by their vendors and may evolve.*

## Quick Start

\`\`\`python
import pandas as pd
from kkode_apex_engine_v55 import KKodeApexEngine

# 1. Load raw longitudinal visit dataset
df = pd.read_csv("patient_retinal_data.csv")

# 2. Initialize K-KODE v55.0 for any longitudinal endpoint
engine = KKodeApexEngine(
    data_source=df,
    endpoint_column="static_perimetry_sensitivity_db",  # or 'ez_width_mm', 'ez_area_mm2', etc.
    eye_column="eye",
    measurement_floor=0.05,
    higher_is_better=True
)

# 3. Execute the complete analysis pipeline in one call
results = engine.run_full_analysis(
    target_power=0.80,
    alpha=0.05,
    therapeutic_efficacy=0.30,
    run_simulation=True,
    n_sims_per_candidate=150
)

# 4. Print the executive biostatistical summary report
print(engine.generate_report())
\`\`\`

*(Update the import path above to match your repository's actual module filename if it differs.)*
