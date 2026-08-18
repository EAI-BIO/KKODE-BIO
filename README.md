> **LEGAL DISCLAIMER & INTENDED USE NOTICE:**  
> K-KODE is a biostatistical calculation engine provided strictly for **Research Use Only (RUO)**. It is **NOT** a medical device, clinical decision-support system, or diagnostic instrument, and it has not been evaluated, cleared, or approved by the US FDA, Health Canada, EMA, or any medical regulatory authority. This software is intended exclusively for retrospective data analysis, exploratory statistical modeling, and trial protocol simulation. It must not be used for direct patient diagnosis, treatment planning, or clinical management.

# K-KODE (OcularExponentialOptimizer)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAI-BIO/KKODE-BIO/blob/main/demo.ipynb)

An open-source, production-grade biostatistical pipeline for longitudinal Ellipsoid Zone (EZ) optical coherence tomography (OCT) tracking in USH2A-related retinal degeneration.

## Architectural Overview
K-KODE automates clinical dataset ingestion, converts irregular visit dates into continuous fractional-year time vectors ($t$), and applies log-exponential transformations ($\ln(y) = \ln(y_0) - \lambda t$) to isolate constant annual percentage loss rates ($\lambda$). This decouples disease progression velocity from baseline tissue area noise and calculates statistical power for two-sample parallel-group trial designs.

## Key Capabilities
* **Log-Exponential Decay Modeling:** Eliminates heteroscedasticity and prevents linear boundary errors (predicting negative tissue).
* **Bilateral Trajectory Separation:** Isolates right (OD) and left (OS) eye vectors to eliminate pooled variance bias across correlated organ pairs.
* **Non-Parametric Bootstrap Engine:** Executes 2,000 Monte Carlo resamples to derive empirical 95% confidence bounds for sample size estimation.
* **Automated Audit Logging:** Detects malformed dates, floor-clipped values ($0\,\mu\text{m}$ bounds), and missing fields with transparent logging.

## Trial Design Impact: Legacy Linear vs. K-KODE Engine

| Parameter / Metric | Standard Linear Model ($y = y_0 - \beta t$) | K-KODE Log-Exponential ($\ln(y) = \ln(y_0) - \lambda t$) | Biostatistical Advantage |
| :--- | :--- | :--- | :--- |
| **Model Assumptions** | Absolute loss/year ($\mu\text{m}/\text{yr}$) | Proportional decay rate ($\lambda \%/\text{yr}$) | Reflects biological cell loss curves |
| **Baseline Confounding** | Correlated with baseline EZ size | Decoupled ($\rho \approx -0.10$) | Removes baseline measurement bias |
| **Required Cohort ($1-\beta=0.80$)** | ~73 patients / arm | ~38 patients / arm | **~48% cohort size reduction** |
| **Boundary Conditions** | Predicts impossible negative area | Asymptotic zero-floor tracking | Mathematically realistic projection |

## Quick Start
```python
import pandas as pd
from OcularExponentialOptimizer import OcularExponentialOptimizer

# Load raw longitudinal visit data
df = pd.read_csv("patient_oct_data.csv")

# Initialize pipeline with explicit eye column designation
optimizer = OcularExponentialOptimizer(df, eye_column="eye")
optimizer.clean_and_transform()

# Fit exponential decay models per patient/eye trajectory
for pid in optimizer.unique_group_ids():
    metrics = optimizer.model_exponential_decay(pid)

# Compute required sample size via bootstrap engine (80% power, alpha=0.05, 30% efficacy)
sample_size_results = optimizer.compute_required_sample_size(
    target_power=0.80, 
    alpha=0.05, 
    therapeutic_efficacy=0.30
)
