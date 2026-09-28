# K-KODE

**Censored-data-aware biostatistics for rare-disease trial planning.**
Turn raw longitudinal patient data into transparent, uncertainty-quantified progression estimates and trial sample sizes.

EAI-BIO · Elite Architecture Intelligence Inc. · Research Use Only (RUO) · Apache 2.0

---

## The problem

Rare-disease trials fight math that generic tools weren't built for:

- **Cohorts too small** for standard statistics.
- **Biology that decays in curves**, not straight lines.
- **Fast progressors who get silently dropped** the moment a measurement crosses the instrument's floor.

K-KODE was built for these three problems, first for USH2A retinal degeneration. It packages established, peer-reviewed methods (censored maximum likelihood, information-criterion model competition, hierarchical Bayesian mixed-effects, Monte Carlo trial simulation) into one auditable pipeline that runs on a raw CSV. Nothing here is a novel statistical claim. The contribution is putting the right methods together, exposing every assumption, and reporting uncertainty honestly.

## What sets it apart

- **Nothing disappears silently.** Every dropped or flagged row is counted and reported.
- **Floor-censored readings stay in the model.** A reading at the instrument floor becomes information ("at least this progressed") instead of being clamped or discarded.
- **The decay shape is chosen by evidence.** Four competing functional forms, compared with small-sample-corrected AIC and a Jacobian correction so cross-model comparisons are valid.
- **Full Bayesian population model** with correlated random effects, so baseline severity and rate of progression can inform each other.
- **Sample sizes are stress-tested.** A closed-form estimate is checked against Monte Carlo simulation of the actual trial design.
- **Endpoint-agnostic.** EZ area or width, static perimetry, microperimetry, or any longitudinal numeric endpoint.
- **Open and auditable.** Every method traces to a cited source, and the code is Apache 2.0.

## Pipeline

| Stage | Method | Why it matters |
|---|---|---|
| Data cleaning | Automated audit of missing fields, bad dates, non-numeric values, duplicate visits | Every dropped row is counted and reported |
| Per-patient decay fit | Censored (Tobit-style) MLE across 4 competing functional forms, selected via AICc with a Jacobian correction | Patients who cross the measurement floor stay in the model, and forms are compared on equal footing |
| Population estimate | Standard mixed-effects NLME and hierarchical Bayesian censored NLME (PyMC / NUTS) | Two independent estimates; the Bayesian model addresses floor-censoring bias directly |
| Correlated random effects | LKJ-Cholesky prior linking baseline severity to progression rate | Captures a biological relationship that independent-effects models can't represent |
| Sample size | Closed-form formula and Monte Carlo trial simulation | The number is tested against a simulated version of the trial |

## Table of Contents

1. [Key Capabilities](#key-capabilities)
2. [What This Version Deliberately Does Not Claim To Do](#what-this-version-deliberately-does-not-claim-to-do)
3. [Installation](#installation)
4. [Input Data Format](#input-data-format)
5. [Usage](#usage)
6. [Methodology & References](#methodology--references)
7. [Output: Understanding the Report](#output-understanding-the-report)
8. [Validation Status & Integrity](#validation-status--integrity)
9. [Roadmap](#roadmap)
10. [Collaboration](#collaboration)
11. [Dedication](#dedication)
12. [License](#license)

## Why This Exists

K-KODE is a rigorously sourced engineering effort to give the USH2A research community a better tool for trial planning. It is offered open-source, in the spirit of collaboration with the biostatisticians and clinicians who know this disease best. Feedback, corrections, and partnership are welcome, and any issue identified will be addressed promptly.

## Key Capabilities

### 1. Generalized endpoint support
Not hardcoded to one measurement. Any longitudinal numeric endpoint (EZ width, EZ area, static perimetry sensitivity, microperimetry sensitivity) passes in via `endpoint_column=`. This matters because the RUSH2A investigators recommend functional endpoints (perimetry sensitivity) as the primary efficacy outcome, with EZ area mainly serving as an enrollment criterion.

*Source: Maguire MG, Birch DG, et al., for the REDI Working Group / Foundation Fighting Blindness Clinical Consortium. "Endpoints and Design for Clinical Trials in USH2A-Related Retinal Degeneration." Translational Vision Science & Technology, October 1, 2024; 13(10):15. DOI: 10.1167/tvst.13.10.15*

### 2. Proper censored-data handling (Tobit-style MLE)
Every candidate model is fit per patient by maximum likelihood with a left-censored Gaussian likelihood: points above the floor contribute a normal density, and points at or below it contribute the normal CDF. A censored point becomes real information instead of being discarded.

*Source: Tobin J. "Estimation of Relationships for Limited Dependent Variables." Econometrica, January 1958; 26(1):24-36. DOI: 10.2307/1907382*

### 3. Four candidate functional forms, competed per patient
Linear, Square-Root, Log-Exponential, and Power-Law curves are fit under the same censored likelihood. Small-sample-corrected AIC and Akaike weights rank the evidence. Because the forms are fit on different transformed scales of the endpoint, each likelihood includes the Jacobian term needed to compare them as likelihoods of the original measurement.

*Sources: Hurvich CM, Tsai CL. Biometrika, June 1989; 76(2):297-307. DOI: 10.1093/biomet/76.2.297. Burnham KP, Anderson DR. Model Selection and Multimodel Inference (2nd ed). Springer; 2002.*

### 4. Numerical-Hessian standard errors and an identifiability guard
Parameter uncertainty comes from a central finite-difference Hessian at the fitted optimum. A censored fit with fewer than two uncensored points is not identified, so it is refused rather than allowed to produce an unbounded slope.

### 5. Simulation-validated sample size
Many two-arm trials are simulated under the fitted population parameters, empirically measuring power at candidate sample sizes rather than relying only on a closed-form formula's assumptions.

*Source: Burton A, Altman DG, Royston P, Holder RL. "The Design of Simulation Studies in Medical Statistics." Statistics in Medicine, December 30, 2006; 25(24):4279-4292. DOI: 10.1002/sim.2673*

### 6. Hierarchical Bayesian censored NLME (PyMC)
A full population-level MCMC sampler (NUTS) using `pm.Censored` estimates population decay and between-patient variance from every observation, including floor-censored points.

*Sources: Laird NM, Ware JH. Biometrics, December 1982; 38(4):963-974. DOI: 10.2307/2529876. Implementation: PyMC, https://www.pymc.io/*

### 7. Correlated random effects
A patient's baseline severity and rate of progression are often related. `fit_bayesian_censored_nlme()` estimates the intercept-slope covariance jointly via an LKJ-Cholesky prior and reports the posterior mean correlation. On small or sparse cohorts, prefer `correlated_random_effects=False`.

*Source: Lewandowski D, Kurowicka D, Joe H. Journal of Multivariate Analysis, October 2009; 100(9):1989-2001. DOI: 10.1016/j.jmva.2009.04.008*

### 8. Automatic population-parameter fallback
If the standard NLME optimizer fails to converge, K-KODE falls back to the Bayesian censored NLME rather than silently failing sample-size estimation. A borderline Bayesian fit retries with more MCMC effort without loosening the convergence bar.

### 9. Plateau-aware floor detection (available, not yet auto-wired)
A `_detect_plateau_floor()` helper distinguishes a genuine instrument floor from a single low reading. It is not yet called automatically; K-KODE still takes an explicit `measurement_floor`.

## What This Version Deliberately Does Not Claim To Do

- Does not ingest raw OCT images or perform retinal layer segmentation. It assumes a reading center or imaging pipeline has already produced a numeric measurement per visit.
- The standard (non-Bayesian) NLME still excludes floor-censored rows at the population level. Use the Bayesian censored NLME for heavy censoring.
- Not FDA-qualified or validated as a Drug Development Tool. Research Use Only.

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
| (endpoint column) | Numeric endpoint value; the column name is passed via `endpoint_column=` |
| (eye column, optional) | Eye identifier (OD/OS); pass its name via `eye_column=` |

Tracking both eyes? Always pass `eye_column=`. Otherwise both eyes on the same visit date are pooled into a single regression per patient, which biases the result. The data-quality report warns you if it detects same-date duplicates with no `eye_column` set.

## Usage

```python
from kkode_engine import KKodeApexEngine

engine = KKodeApexEngine(
    data_source="my_cohort.csv",
    endpoint_column="ez_area_mm2",
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

`run_full_analysis()` runs the full pipeline: data cleaning, per-patient model competition, per-patient decay fits, population mixed-effects fit, closed-form sample size, simulated sample size (with automatic Bayesian fallback), and Bayesian censored NLME (auto-triggered under heavy censoring). Each stage is independently callable.

## Methodology & References

1. Maguire MG, Birch DG, Duncan JL, et al., for the REDI Working Group and the Foundation Fighting Blindness Clinical Consortium Investigator Group. "Endpoints and Design for Clinical Trials in USH2A-Related Retinal Degeneration: Results and Recommendations From the RUSH2A Natural History Study." *Translational Vision Science & Technology.* 2024; 13(10):15. DOI: 10.1167/tvst.13.10.15
2. Tobin J. "Estimation of Relationships for Limited Dependent Variables." *Econometrica.* 1958; 26(1):24-36. DOI: 10.2307/1907382
3. Hurvich CM, Tsai CL. "Regression and Time Series Model Selection in Small Samples." *Biometrika.* 1989; 76(2):297-307. DOI: 10.1093/biomet/76.2.297
4. Burnham KP, Anderson DR. *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach.* 2nd ed. Springer; 2002.
5. Burton A, Altman DG, Royston P, Holder RL. "The Design of Simulation Studies in Medical Statistics." *Statistics in Medicine.* 2006; 25(24):4279-4292. DOI: 10.1002/sim.2673
6. Laird NM, Ware JH. "Random-Effects Models for Longitudinal Data." *Biometrics.* 1982; 38(4):963-974. DOI: 10.2307/2529876
7. Lewandowski D, Kurowicka D, Joe H. "Generating Random Correlation Matrices Based on Vines and Extended Onion Method." *Journal of Multivariate Analysis.* 2009; 100(9):1989-2001. DOI: 10.1016/j.jmva.2009.04.008
8. PyMC Development Team. *PyMC: Probabilistic Programming in Python.* https://www.pymc.io/

## Output: Understanding the Report

`generate_report()` returns a plain-language executive summary covering:

- **Data quality:** raw rows in, rows retained, how many were floor-censored (kept and modeled, not dropped), and any heavy-censoring warning.
- **Best-supported functional form:** which decay shape led, and by how much.
- **Population decay rate:** from both the standard NLME and the Bayesian censored NLME (95% credible interval; correlation, if correlated random effects were used).
- **Sample size (closed-form):** required N per arm, with a bootstrap 95% confidence interval.
- **Sample size (simulation-validated):** required N per arm to empirically reach target power.

The report closes with an explicit reminder: this is a planning aid, not a finalized protocol.

## Validation Status & Integrity

K-KODE is Research Use Only. Here is exactly where validation stands.

**Completed**
- **Ground-truth testing on synthetic cohorts** with known, planted parameters. This testing exposed a model-selection bias (a missing Jacobian term in cross-model likelihood comparison). It was diagnosed, corrected, and documented in the changelog at the top of `kkode_engine.py`.
- **Initial retrospective run on the public RUSH2A 4-year natural history dataset** (EZ area): 125 participants, 249 eyes, 715 observations, including 27 sub-threshold readings that were kept and modeled rather than dropped. The Bayesian censored population model converged and estimated a decline in the direction and general magnitude of published RUSH2A estimates. Per-patient model competition on the corrected engine found no single functional form clearly preferred. Results are being re-confirmed at full sampling settings before any stronger claim is made.

**Not yet done**
- Independent biostatistician review.
- Replication on a second, independent dataset.
- Validation on functional endpoints (static perimetry, microperimetry).
- FDA qualification (not applicable to this RUO release).

> The source of the data is the Foundation Fighting Blindness Clinical Consortium, but the analyses, content and conclusions presented herein are solely the responsibility of the authors and may not reflect the views of the Foundation Fighting Blindness.

We publish this status plainly because trust in a trial-planning tool has to be earned in the open. If you are a biostatistician, clinician, or researcher and find something wrong, please open an issue or reach out directly. Corrections are welcome and will be fixed promptly.

## Roadmap

1. Re-run the full validation on the corrected engine at default sampling settings, and publish the audit trail.
2. Extend to functional endpoints: static perimetry and microperimetry sensitivity.
3. Covariate-adjusted and prognostic-enrichment sample sizing. Early exploratory analysis suggests baseline severity explains a meaningful share of between-patient variation in progression.
4. Independent replication on a second dataset, and independent biostatistician review.
5. Planned: machine-learning-assisted multi-endpoint progression forecasting with calibrated uncertainty, and cross-cohort data harmonization.
6. Wire plateau-aware floor detection into the automatic cleaning pipeline.

## Collaboration

We welcome biostatisticians, clinicians, trial sponsors, patient-advocacy organizations, and AI researchers. Open an issue, or contact:

**Eric Fitzgerald**, Founder, Elite Architecture Intelligence Inc. (EAI-BIO)
eric@eaiinc.ca · github.com/EAI-BIO/KKODE-BIO

## Dedication

K-KODE is personal, and I am on a mission to see it through.

It is dedicated to my family and to all families living with inherited vision loss, and in honour of the blind woman I was able to help save from a house fire, an act for which I was awarded the National Assembly of Quebec's Médaille du député for bravery and bravery honours from the City of Pincourt.

Every line of this project is built to help the research community find solutions as soon as possible.

With gratitude to the USH2A research and patient community.

## License

Apache License 2.0. See `LICENSE` for full terms.
