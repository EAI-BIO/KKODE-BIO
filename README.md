LEGAL DISCLAIMER & INTENDED USE NOTICE:
K-KODE is a biostatistical calculation engine provided strictly for Research Use Only (RUO). It is NOT a medical device, clinical decision-support system, or diagnostic instrument, and it has not been evaluated, cleared, or approved by the US FDA, Health Canada, EMA, or any medical regulatory authority. This software is intended exclusively for retrospective data analysis, exploratory statistical modeling, and trial protocol simulation. It must not be used for direct patient diagnosis, treatment planning, or clinical management.

K-KODE APEX ENGINE (v55.0) — EAI-BIO
An open-source, protocol-grade biostatistical pipeline for non-linear longitudinal biomarker decay modeling and clinical trial sample size optimization in USH2A and retinal degeneration trial planning.

Architectural Overview
K-KODE Apex (v55.0) bridges raw imaging and functional scan exports directly to simulation-validated clinical trial protocols. Unlike legacy tools that force curved biological cell death onto a flat ruler line, K-KODE automatically competes non-linear functional forms, preserves floor-censored fast-progressing patients via Tobit Maximum Likelihood Estimation and Hierarchical Bayesian modeling, and calculates sample sizes using empirical Monte Carlo trial simulations.

Key Capabilities
Generalized Endpoint Engine: Models structural primary/secondary outcomes (ez_width_mm, ez_area_mm2) alongside functional primary outcomes (static_perimetry_sensitivity_db, microperimetry_sensitivity_db) as recommended by the RUSH2A study group.

Dual Floor-Censoring Mechanics: Eliminates floor-and-drop survivorship bias by evaluating floor points (0.05 mm) using Gaussian CDF likelihoods (Tobit MLE) at the individual level and PyMC MCMC (pm.Censored) at the population level—preserving 100% of patient visits.

Automated 4-Model Competition: Competes Linear, Square-Root, Log-Exponential, and Power-Law decay curves on every trajectory, aggregating results via Akaike Weights (w 
i
​
 ) to identify the true biological decay shape.

Finite-Difference Hessian Uncertainties: Computes exact asymptotic parameter standard errors (SE) from central finite differences of the log-likelihood surface rather than relying on optimizer approximations.

Identifiability & Protocol Stability Guards: Hardcoded rules require ≥2 uncensored visits per patient to prevent single-point slope explosions (λ→∞) and apply an n=4 AIC fallback for standard 4-visit annual trial designs where small-sample AICc denominators equal zero.

Simulation-Validated Sizing Engine: Combines closed-form z-bounds with vectorized Monte Carlo two-arm trial simulations to empirically measure statistical power (80%) under actual between-patient (τ 
b
2
​
 ) and residual (σ 
2
 ) variance.

Market Platform Comparison
Capability / Feature	Legacy Trial Software (Cytel East, nQuery)	Ocular Imaging Platforms (RetinAI, Voxeleron)	Custom R / SAS Scripts	K-KODE v55.0 Apex Engine
Primary Focus	General trial sample sizing	Single-visit image segmentation	Ad-hoc biostatistics	Longitudinal decay modeling & protocol N optimization
Decay Curve Fitting	Forced linear (Δy=−c⋅t)	N/A (Static scans)	Manual custom coding	Automated 4-Model AIC/AICc Competition
Patient Floor Handling	Excludes or clamps zero values	Visible area only	Complex manual setup	Tobit MLE Likelihood (Gaussian CDF)
Cohort Floor Handling	Excludes floor-censored visits	N/A	Excludes floor-censored visits	Hierarchical Bayesian Tobit (pm.Censored)
Endpoint Compatibility	Generic metric	Image metrics only	Single metric script	Generalized (EZ width/area, Perimetry dB)
Trial Powering Method	Static z-score lookup tables	N/A	Manual script execution	Dual Engine: Closed-Form + Monte Carlo Simulation
Pipeline Reliability	Point-and-click GUI	Single-scan analysis	Fails on 1-visit or n=4 edge cases	Identifiability Guarded (<0.2s execution)
Quick Start
Python
import pandas as pd
from kkode_engine import KKodeApexEngine

# 1. Load raw longitudinal visit dataset
df = pd.read_csv("patient_retinal_data.csv")

# 2. Initialize K-KODE v55.0 for any longitudinal endpoint
engine = KKodeApexEngine(
    data_source=df,
    endpoint_column="static_perimetry_sensitivity_db",  # Works with 'ez_width_mm', 'ez_area_mm2', etc.
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

# 4. Print executive biostatistical summary
print(engine.generate_report())
