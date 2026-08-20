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

## Installation & Setup

```bash
git clone https://github.com/EAI-BIO/KKODE-BIO.git
cd KKODE-BIO
pip install -e .
```

This installs the core engine and its required dependencies (NumPy, pandas, SciPy, statsmodels). Two optional extras are available and installed separately, so a standard analysis never has to pull in packages it doesn't use:

```bash
pip install -e ".[bayesian]"   # adds PyMC + ArviZ, for the hierarchical Bayesian censored model
pip install -e ".[demo]"       # adds Jupyter + Matplotlib, for running/exploring the demo notebook
pip install -e ".[all]"        # both of the above
```

The Bayesian extra is only needed if you plan to call `fit_bayesian_censored_nlme()` directly, or if your cohort is heavily censored (>15% of visits at the measurement floor) and you want `run_full_analysis()` to auto-trigger it. `pip install pymc` compiles native code on first use and can require a working C compiler on some systems — this is a PyMC characteristic, not something specific to this engine.

**Verify your installation** by running the built-in demo, which generates a synthetic 60-patient cohort and runs the full pipeline end to end:

```bash
python run_demo.py
```

If that prints a report with no errors, your environment is set up correctly.

## Quick Start

```python
import pandas as pd
from kkode_engine import KKodeApexEngine

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
```

## Running on Your Own Data (No Python Required)

For teams who want results without writing any code, `run_kkode.py` wraps the pipeline above as a single command-line call:

```bash
python run_kkode.py --data your_clinical_export.csv --endpoint-column static_perimetry_sensitivity_db --eye-column eye --output ./results
```

Your CSV needs three columns at minimum: `patient_id`, `visit_date` (any format pandas can parse, e.g. `YYYY-MM-DD`), and whichever numeric biomarker column you pass to `--endpoint-column`. Run `python run_kkode.py --help` for the full list of options (target power, alpha, therapeutic efficacy assumption, simulation on/off, and more). Results are written as a plain-language report (`kkode_report.txt`) and a full structured results file (`kkode_results.json`) in the `--output` directory.

## Visualizing Results

`kkode_viz.py` is an optional add-on for plotting trajectories directly from a fitted engine. It has no effect on the core pipeline — the standard analysis and CLI runner never import it or matplotlib.

```bash
pip install -r requirements-viz.txt
```

```python
from kkode_viz import plot_patient_fit, plot_cohort_overview

# After engine.run_full_analysis() has been called:
plot_patient_fit(engine, group_id="PATIENT_001__OD", save_path="patient_001.png")
plot_cohort_overview(engine, save_path="cohort_overview.png")
```

`plot_patient_fit` shows one patient's observed visits against their fitted decay curve, with floor-censored points marked distinctly (▼) from uncensored points (●) — so the censoring handling that drives this engine's core statistics is visible, not just reported as a percentage. `plot_cohort_overview` plots every patient's raw trajectory together, with censored visits highlighted, to make heavy floor-censoring or outlier patients legible at a glance across the full cohort.

## Methodology & References

Every statistical method and disease-specific parameter choice in this engine is drawn from a published, citable source — not an internal assumption. This section exists so any claim in this README or in the code can be independently checked.

**Statistical methods:**

- **Censored (Tobit) maximum-likelihood estimation** — Tobin, J. (1958). "Estimation of Relationships for Limited Dependent Variables." *Econometrica*, 26(1), 24–36. The foundational method for fitting a regression model when some outcomes are only known to be at-or-below (or at-or-above) a detection limit, rather than observed exactly.
- **AICc small-sample bias correction** — Hurvich, C. M., & Tsai, C. L. (1989). "Regression and time series model selection in small samples." *Biometrika*, 76(2), 297–307. The correction term this engine applies when comparing candidate decay models on a per-patient basis, and the reason a fallback to plain AIC is required when a patient has exactly 4 visits (see `small_sample_correction_unavailable` in the code).
- **Akaike weights / multimodel inference** — Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach* (2nd ed.). Springer. The method used to convert per-model AICc values into the relative support weights reported for each candidate decay shape.
- **Linear mixed-effects (random slope/intercept) modeling** — Laird, N. M., & Ware, J. H. (1982). "Random-effects models for longitudinal data." *Biometrics*, 38(4), 963–974. The standard population-level model this engine fits via `statsmodels.formula.api.mixedlm`.
- **Hierarchical Bayesian censored modeling (PyMC / NUTS)** — implemented using [PyMC](https://www.pymc.io/), an open-source probabilistic programming library; `pm.Censored` is PyMC's built-in construct for exactly the left-censored-likelihood problem described above, extended here to a hierarchical (per-patient random effects) structure.
- **Monte Carlo simulation for statistical power** — a standard technique in clinical trial design; see e.g. Burton, A., Altman, D. G., Royston, P., & Holder, R. L. (2006). "The design of simulation studies in medical statistics." *Statistics in Medicine*, 25(24), 4279–4292.

**Disease-specific parameters and claims:**

- **RUSH2A endpoint recommendations** (functional endpoints such as static/microperimetry sensitivity recommended as primary, over structural EZ measurements) — Birch, D. G., et al., on behalf of the RUSH2A Study Group. "Endpoints and Design for Clinical Trials in USH2A-Related Retinal Degeneration: Results and Recommendations From the RUSH2A Natural History Study." *Translational Vision Science & Technology*. [Full text (ARVO Journals)](https://tvst.arvojournals.org/article.aspx?articleid=2802114) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11469320/)
- **Realistic EZ area decline rate used in `run_demo.py` and example charts** (≈ −0.34 mm²/year in patients with meaningfully preserved baseline EZ area) — same RUSH2A source as above. This replaced an earlier, uncalibrated synthetic test value that did not reflect real disease progression; flagged and corrected after review.
- **USH2A gene/disease background** — Blanco-Kelly, F., et al. "USH2A-related disorders." *GeneReviews®* [Internet], NCBI Bookshelf ID: [NBK1341](https://www.ncbi.nlm.nih.gov/books/NBK1341/).

If you find a claim anywhere in this repository that isn't backed by a source above, that's a gap worth reporting, not an assumption to trust — open an issue or flag it directly.
