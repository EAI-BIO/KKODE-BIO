> **LEGAL DISCLAIMER & INTENDED USE NOTICE:**  
> K-KODE is a biostatistical calculation tool provided strictly for **Research Use Only (RUO)**. It is **NOT** a medical device, clinical decision-support tool, or diagnostic instrument, and it has not been evaluated or cleared by the FDA, Health Canada, or any medical regulatory body. This software is intended solely for retrospective data analysis and mathematical modeling. It must not be used for patient diagnosis, clinical treatment planning, or direct patient care.

# K-KODE (OcularExponentialOptimizer)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAI-BIO/KKODE-BIO/blob/main/demo.ipynb)

An open-source, production-grade biostatistical engine designed for longitudinal Ellipsoid Zone (EZ) optical coherence tomography (OCT) tracking in USH2A retinopathy.

## Architectural Overview
K-KODE automates data cleansing, converts irregular appointment dates into continuous fractional-year vectors, applies log-transformations to decouple decay velocity ($\ln(y) = \ln(y_0) - \lambda t$) from baseline measurement noise, and projects two-sample clinical trial sample sizes.

## Key Capabilities
* **Log-Exponential Decay Modeling:** Bypasses linear regression flaws to isolate true annual tissue loss velocity.
* **Bilateral Eye Tracking:** Isolates right (OD) and left (OS) eye trajectories to prevent pooled variance bias.
* **95% Bootstrap Confidence Intervals:** Runs 2,000 resamples to output statistical uncertainty bounds for 1:1 parallel-group trial designs.
* **Transparent Data Quality Logging:** Audits and logs floor-clipped values, malformed dates, and missing fields without silent data loss.

## Quick Start
```python
import pandas as pd
from OcularExponentialOptimizer import OcularExponentialOptimizer

# Load raw longitudinal visit data
df = pd.read_csv("patient_oct_data.csv")

# Initialize and run pipeline
optimizer = OcularExponentialOptimizer(df, eye_column="eye")
optimizer.clean_and_transform()

# Process per-patient regressions and compute trial power
for pid in optimizer.unique_group_ids():
    metrics = optimizer.model_exponential_decay(pid)

sample_size_results = optimizer.compute_required_sample_size(
    target_power=0.80, 
    alpha=0.05, 
    therapeutic_efficacy=0.30
)
```

## Attribution & License
Developed and maintained by **Élite Architecture Intelligence Inc. (E.A.I. Bio™)**.

Engineered by **Eric Fitzgerald** (Founder & CEO, E.A.I. Inc.).

Licensed under the **Apache License 2.0**. Free for academic, clinical, and commercial research use globally.
