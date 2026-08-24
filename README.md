# K-KODE

**A biostatistical engine for USH2A research, decay modeling, and clinical trial design — turning raw longitudinal patient data into rigorously validated trial sample sizes.**

`EAI-BIO` · Elite Architecture Intelligence Inc. · Research Use Only (RUO)

---

Rare-disease trials fight math that generic tools weren't built for: cohorts too small for standard statistics, biology that decays in curves instead of straight lines, and fast progressors who get silently dropped the moment their measurement crosses the instrument's floor. K-KODE was built to solve exactly those three problems for USH2A retinal degeneration trial design — and it does so entirely with cited, peer-reviewed methods. Nothing here is a novel statistical claim; the contribution is packaging established techniques (censored MLE, AICc model competition, hierarchical Bayesian NLME, Monte Carlo-validated power) into one pipeline that runs on a raw CSV.

## What It Does

| Stage | Method | Why it matters |
|---|---|---|
| Data cleaning | Automated audit of missing fields, bad dates, non-numeric values, duplicate visits | Every dropped row is counted and reported — nothing silently disappears |
| Per-patient decay fit | Censored (Tobit-style) MLE across 4 competing functional forms, selected via AICc | Patients who cross the measurement floor stay in the model instead of being dropped or clamped |
| Population estimate | Standard mixed-effects NLME **and** hierarchical Bayesian censored NLME (PyMC/NUTS) | Two independent estimates; the Bayesian model eliminates floor-censoring bias entirely |
| Correlated random effects | LKJ-Cholesky prior linking baseline severity to progression rate | Captures a real biological relationship that independent-effects models can't represent |
| Sample size | Closed-form formula **and** Monte Carlo trial simulation | The number isn't just computed — it's been stress-tested against a simulated version of the actual trial |

Every method above traces to a specific, cited, peer-reviewed source. See [Methodology & References](#methodology--references) — every citation individually verified against the publisher record.

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Key Capabilities](#key-capabilities)
- [What This Version Deliberately Does Not Claim To Do](#what-this-version-deliberately-does-not-claim-to-do)
- [Installation](#installation)
- [Input Data Format](#input-data-format)
- [Usage](#usage)
- [Methodology & References](#methodology--references)
- [Output: Understanding the Report](#output-understanding-the-report)
- [Validation Status & Disclaimers](#validation-status--disclaimers)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact / Collaboration](#contact--collaboration)

---

## Why This Exists

K-KODE began as one father's response to his son's diagnosis with Usher syndrome type 2A — and grew into a rigorously sourced engineering effort to give the USH2A research community a better tool for trial planning. Every method in this pipeline traces to a specific, peer-reviewed source; every citation has been individually verified against the publisher record; and the code has been stress-tested against edge cases, with real issues found and fixed along the way.

It's offered open-source to the USH2A research and clinical community as a planning aid — built to real engineering standards, and offered in the spirit of open collaboration with the biostatisticians and clinicians who know this disease best. Feedback, corrections, and partnership are genuinely welcome, and any issue identified will be addressed promptly.

## Key Capabilities

### 1. Generalized Endpoint Support
Not hardcoded to EZ width. Any longitudinal numeric endpoint — EZ width, EZ area, static perimetry sensitivity, microperimetry sensitivity — passes in via `endpoint_column=`. This matters because RUSH2A's own published recommendation is that **functional endpoints (perimetry sensitivity)**, not structural EZ measurements, should be the *primary* efficacy outcome — EZ area is mainly an enrollment criterion.

> **Source:** Maguire MG, Birch DG, et al., for the REDI Working Group / Foundation Fighting Blindness Clinical Consortium. "Endpoints and Design for Clinical Trials in USH2A-Related Retinal Degeneration." *Translational Vision Science & Technology*, October 1, 2024; 13(10):15. DOI: [10.1167/tvst.13.10.15](https://doi.org/10.1167/tvst.13.10.15)

### 2. Proper Censored-Data Handling (Tobit-Style MLE) — Not Floor-and-Drop
Every candidate model is fit per patient via maximum likelihood using a left-censored Gaussian likelihood: points above the floor contribute a normal density; points at or below it contribute the normal CDF. A censored point becomes real information — "this patient was *at least* this far progressed" — instead of being discarded.

> **Source:** Tobin J. "Estimation of Relationships for Limited Dependent Variables." *Econometrica*, January 1958; 26(1):24–36. DOI: [10.2307/1907382](https://doi.org/10.2307/1907382)

### 3. Four Candidate Functional Forms, Competed Per Patient via AICc
Linear, Square-Root, Log-Exponential, and Power-Law decay curves are fit using the *same* censored likelihood, so the comparison is apples-to-apples. Small-sample-corrected AIC and Akaike weights determine the best-supported shape.

> **Source:** Hurvich CM, Tsai CL. "Regression and Time Series Model Selection in Small Samples." *Biometrika*, June 1989; 76(2):297–307. DOI: [10.1093/biomet/76.2.297](https://doi.org/10.1093/biomet/76.2.297)
> Akaike weights: Burnham KP, Anderson DR. *Model Selection and Multimodel Inference* (2nd ed). Springer; 2002.

### 4. Numerical-Hessian Standard Errors
Parameter uncertainty is computed from a central finite-difference Hessian of the negative log-likelihood at the fitted optimum, rather than trusting the optimizer's internal approximation.

### 4b. Identifiability Guard
A two-parameter censored fit with only one uncensored point is not identified — the intercept can absorb any slope choice, letting the optimizer chase an unbounded "improvement" by driving the slope toward infinity. Fixed by requiring at least two uncensored points before a censored fit is attempted.

### 5. Simulation-Validated Sample Size
Many two-arm trials are simulated under the fitted population parameters, with a mixed-effects model fit to each, empirically measuring statistical power at candidate sample sizes — rather than relying purely on a closed-form formula's assumptions.

> **Source:** Burton A, Altman DG, Royston P, Holder RL. "The Design of Simulation Studies in Medical Statistics." *Statistics in Medicine*, December 30, 2006; 25(24):4279–4292. DOI: [10.1002/sim.2673](https://doi.org/10.1002/sim.2673)

### 6. Hierarchical Bayesian Censored NLME (PyMC)
A full population-level MCMC sampler (NUTS) using `pm.Censored` estimates population decay and between-patient variance directly from *every* observation, including floor-censored points — eliminating cohort-level survivorship bias under heavy censoring.

> **Source:** Laird NM, Ware JH. "Random-Effects Models for Longitudinal Data." *Biometrics*, December 1982; 38(4):963–974. DOI: [10.2307/2529876](https://doi.org/10.2307/2529876)
> Implementation: PyMC probabilistic programming library. https://www.pymc.io/

### 7. Correlated Random Effects
A patient's baseline severity and rate of progression are often related. A diagonal (independent) random-effects structure can't represent that relationship at all. `fit_bayesian_censored_nlme()` estimates the (intercept, slope) covariance jointly via an LKJ-Cholesky prior and reports the posterior mean correlation directly. Requires enough patients and visits per patient to be identifiable — on small/sparse cohorts, prefer `correlated_random_effects=False`.

> **Source:** Lewandowski D, Kurowicka D, Joe H. "Generating Random Correlation Matrices Based on Vines and Extended Onion Method." *Journal of Multivariate Analysis*, October 2009; 100(9):1989–2001. DOI: [10.1016/j.jmva.2009.04.008](https://doi.org/10.1016/j.jmva.2009.04.008)

### 8. Automatic Population-Parameter Fallback for Simulation
If the standard NLME optimizer fails to converge — which can happen on cohorts with small between-patient variance even when the data is perfectly reasonable — K-KODE automatically falls back to the Bayesian censored NLME rather than silently failing sample-size estimation. A borderline Bayesian fit automatically retries with more MCMC effort before giving up, without loosening the convergence bar itself.

### 9. Plateau-Aware Floor Detection (available, not yet auto-wired)
A `_detect_plateau_floor()` helper distinguishes a genuine instrument floor (multiple visits from the same patient clustering flat at a low value) from a single low reading that's just a patient who started mild. Not yet called automatically — K-KODE still takes an explicit, user-supplied `measurement_floor`.

## What This Version Deliberately Does Not Claim To Do

- Does **not** ingest raw OCT images or perform retinal layer segmentation — assumes a reading center or imaging pipeline has already produced a numeric measurement per visit.
- Standard (non-Bayesian) NLME still excludes floor-censored rows at the population level — use the Bayesian censored NLME for heavy censoring.
- **Not** FDA-qualified or validated as a Drug Development Tool.
- Not yet tested against real or independently validated clinical cohort data — see [Validation Status](#validation-status--disclaimers).

## Installation

```bash
pip install -r requirements.txt
```

Core: `numpy`, `pandas`, `scipy`, `statsmodels`. Optional (Bayesian censored NLME): `pymc`, `arviz`.

## Input Data Format

One row per visit, as a CSV or pandas DataFrame:

| Column | Description |
|---|---|
| `patient_id` | Patient identifier |
| `visit_date` | Visit date (any format `pandas.to_datetime` can parse) |
| *(endpoint column)* | Numeric endpoint value — column name passed via `endpoint_column=` |
| *(eye column, optional)* | Eye identifier (OD/OS) — pass its name via `eye_column=` |

**Tracking both eyes?** Always pass `eye_column=` — otherwise both eyes on the same visit date get pooled into a single regression per patient, biasing the result. K-KODE's data quality report will warn you if it detects same-date duplicates with no `eye_column` set.

## Usage

```python
from kkode_engine import KKodeApexEngine

engine = KKodeApexEngine(
    data_source="my_cohort.csv",
    endpoint_column="ez_width_mm",
    eye_column="eye",             # optional
    measurement_floor=0.05,       # instrument/assay floor for this endpoint
)

results = engine.run_full_analysis(
    target_power=0.80,
    alpha=0.05,
    therapeutic_efficacy=0.30,    # assumed fractional slowing of decay rate
)

print(engine.generate_report())
```

`run_full_analysis()` runs the full pipeline: data cleaning → per-patient model competition → per-patient decay fits → population mixed-effects fit → closed-form sample size → simulated sample size (with automatic Bayesian fallback) → Bayesian censored NLME (auto-triggered under heavy censoring). Each stage is independently callable.

## Methodology & References

Every citation below was individually verified against the publisher/DOI record.

1. **Maguire MG, Birch DG, Duncan JL, et al.**, for the REDI Working Group and the Foundation Fighting Blindness Clinical Consortium Investigator Group. "Endpoints and Design for Clinical Trials in USH2A-Related Retinal Degeneration: Results and Recommendations From the RUSH2A Natural History Study." *Translational Vision Science & Technology.* October 1, 2024; 13(10):15. DOI: [10.1167/tvst.13.10.15](https://doi.org/10.1167/tvst.13.10.15)

2. **Tobin J.** "Estimation of Relationships for Limited Dependent Variables." *Econometrica.* January 1958; 26(1):24–36. DOI: [10.2307/1907382](https://doi.org/10.2307/1907382)

3. **Hurvich CM, Tsai CL.** "Regression and Time Series Model Selection in Small Samples." *Biometrika.* June 1989; 76(2):297–307. DOI: [10.1093/biomet/76.2.297](https://doi.org/10.1093/biomet/76.2.297)

4. **Burnham KP, Anderson DR.** *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach.* 2nd edition. New York: Springer; 2002.

5. **Burton A, Altman DG, Royston P, Holder RL.** "The Design of Simulation Studies in Medical Statistics." *Statistics in Medicine.* Published online August 31, 2006; print December 30, 2006; 25(24):4279–4292. DOI: [10.1002/sim.2673](https://doi.org/10.1002/sim.2673)

6. **Laird NM, Ware JH.** "Random-Effects Models for Longitudinal Data." *Biometrics.* December 1982; 38(4):963–974. DOI: [10.2307/2529876](https://doi.org/10.2307/2529876)

7. **Lewandowski D, Kurowicka D, Joe H.** "Generating Random Correlation Matrices Based on Vines and Extended Onion Method." *Journal of Multivariate Analysis.* October 2009; 100(9):1989–2001. DOI: [10.1016/j.jmva.2009.04.008](https://doi.org/10.1016/j.jmva.2009.04.008) — source of the LKJ-Cholesky prior used for correlated random effects (Capability 7).

8. **PyMC Development Team.** *PyMC: Probabilistic Programming in Python.* Software. https://www.pymc.io/ — cite the specific version pinned in `requirements.txt` if a version-locked citation is needed.

## Output: Understanding the Report

`generate_report()` returns a plain-language executive summary covering:

- **Data quality** — raw rows in, rows retained, how many were floor-censored (kept and modeled, not dropped), any heavy-censoring warning.
- **Best-supported functional form** — which decay shape won, and by how much.
- **Population decay rate** — from both the standard NLME and the Bayesian censored NLME (95% credible interval; correlation, if correlated random effects were used).
- **Sample size (closed-form)** — required N per arm, with a bootstrap 95% confidence interval.
- **Sample size (simulation-validated)** — required N per arm to empirically hit target power in Monte Carlo simulation.

The report closes with an explicit reminder: **this is a planning aid, not a finalized protocol.**

## Validation Status & Disclaimers

K-KODE is **Research Use Only**. As of this version:

- All testing to date uses synthetic, simulated cohort data — not real or de-identified USH2A patient data.
- No independent biostatistician review has been performed.
- No retrospective validation against published USH2A natural history data has been performed.
- Not FDA-qualified or validated as a Drug Development Tool.

If you're a biostatistician, clinician, or researcher and find something wrong, please open an issue or reach out directly — corrections are genuinely welcome and will be fixed free of charge.

## Roadmap

- Retrospective validation against real or published USH2A natural history data.
- Independent biostatistician review.
- Wiring `_detect_plateau_floor()` into the automatic cleaning pipeline for endpoints without a fixed, known instrument floor.

## Acknowledgments

Built for Kye. Named in part for Dr. Robert Koenekoop, McGill Ocular Genetics Laboratory.

## License

*(specify your chosen license here — e.g., MIT, Apache 2.0)*

## Contact / Collaboration

Eric Fitzgerald, Founder, Elite Architecture Intelligence Inc. (EAI-BIO)
Repository: `github.com/EAI-BIO/KKODE-BIO`
