# K-KODE (OcularExponentialOptimizer)

An open-source, production-grade biostatistical engine designed for longitudinal Ellipsoid Zone (EZ) optical coherence tomography (OCT) tracking in USH2A retinopathy.

## Architectural Overview
K-KODE automates data cleansing, converts irregular appointment dates into continuous fractional-year vectors, applies log-transformations to decouple decay velocity ($\ln(y) = \ln(y_0) - \lambda t$) from baseline measurement noise, and projects two-sample clinical trial sample sizes[cite: 1].

## Key Capabilities
* **Log-Exponential Decay Modeling:** Bypasses linear regression flaws to isolate true annual tissue loss velocity[cite: 1].
* **Bilateral Eye Tracking:** Isolates right (OD) and left (OS) eye trajectories to prevent pooled variance bias[cite: 1].
* **95% Bootstrap Confidence Intervals:** Runs 2,000 resamples to output statistical uncertainty bounds for 1:1 parallel-group trial designs[cite: 1].
* **Transparent Data Quality Logging:** Audits and logs floor-clipped values, malformed dates, and missing fields without silent data loss[cite: 1].

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
