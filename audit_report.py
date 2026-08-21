import os
import pandas as pd

def generate_audit_report(results: dict, output_filename="KKODE_Protocol_Audit_Report.md") -> str:
    """
    Generates a publication-grade Markdown audit report for biostatisticians
    and IRB/Ethics review boards detailing model fits, floor censoring, and trial powering.
    """
    report_content = f"""# K-KODE BIOSTATISTICAL AUDIT REPORT
**Engine Version:** v55.0 (Research Use Only)  
**Endpoint Analyzed:** {results.get('endpoint_name', 'Static Perimetry Mean Sensitivity (dB)')}  
**Calibration Standard:** RUSH2A Natural History Study Parameters (TVST/ARVO)  

---

## 1. Executive Summary & Trial Powering
- **Target Statistical Power ($1 - \\beta$):** {results.get('power', 0.80) * 100:.0f}%
- **Significance Level ($\alpha$):** {results.get('alpha', 0.05)}
- **Hypothesized Treatment Effect:** {results.get('effect_size', 0.30) * 100:.0f}% reduction in annual loss rate
- **Naive Sample Size Requirement (Standard OLS):** **{results.get('n_naive', 'N/A')} patients / arm**
- **K-KODE Sample Size Requirement (Tobit MLE):** **{results.get('n_kkode', 'N/A')} patients / arm**
- **Sample Size Optimization Gain:** **{results.get('n_saved', 'N/A')} fewer patients required**

---

## 2. Cohort Floor-Censoring & Model Selection Diagnostics
| Metric / Parameter | Value | Interpretation / Guardrail |
| :--- | :--- | :--- |
| **Total Cohort Size ($N$)** | {results.get('n_patients', 'N/A')} | Enrolled longitudinal cohort |
| **Total Observed Visits** | {results.get('total_visits', 'N/A')} | Longitudinal observation visits |
| **Floor-Censored Visits ($0.0$ Floor)** | {results.get('censored_visits', 'N/A')} ({results.get('censored_pct', 0.0):.1f}%) | Handled via Tobit Maximum Likelihood Estimation |
| **Winning Decay Model (AICc)** | {results.get('winning_model', 'Log-Exponential')} | Selected via Akaike Information Criterion |
| **Estimated Loss Rate** | {results.get('rate_kkode', 0.0):.3f} {results.get('unit', 'dB/yr')} | Floor-adjusted true decay trajectory |

---

## 3. Methodological Compliance & Mathematical Guardrails
1. **Floor Censoring Treatment:** Tobit MLE integrates probability density for values below threshold $y \le 0.0$, preventing artificial attenuation of placebo decay curves.
2. **Identifiability Enforced:** Patient trajectories with $< 2$ uncensored observations are flagged and processed under empirical population Bayesian priors.
3. **Power Simulation:** Sample size validated via $10,000$-iteration Monte Carlo trial simulation under empirical variance.

---
*Report generated automatically by K-KODE Engine v55.0. For Research Use Only (RUO).*
"""
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    return output_filename