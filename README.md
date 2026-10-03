K-KODE

Censored-data biostatistics for rare-disease clinical trial planning. K-KODE turns raw longitudinal patient data into transparent, uncertainty-quantified progression estimates and trial sample sizes. Every method is established and cited, every assumption is exposed, every dropped row is counted, and every claim is backed by an automated test.

EAI-BIO · Elite Architecture Intelligence Inc. · Research Use Only (RUO) · Apache 2.0

Evidence to date: 44 automated tests on synthetic cohorts with planted ground truth, and one end-to-end run on the real RUSH2A natural-history dataset (EZ area: 715 readings, 125 participants, 249 eyes). The real-data run kept every censored reading, agreed with an independent analysis, and exposed three weaknesses that were fixed and tested. See Validation status for exactly what that does and does not show.

Start here
If you are...	Read this first
A patient, family member or funder	In plain language, then Questions people ask
A clinician or clinical researcher	In plain language, then For clinicians and trial teams
A biostatistician or data scientist	Pipeline, Key capabilities, Known limitations, Validation status
A developer who wants to run it	Installation, Input data format, Usage, Running the tests
Unsure of a term	Glossary
In plain language

For clinicians, patients, families and funders.

Some eye diseases, like USH2A retinal degeneration (a leading cause of combined hearing and vision loss), are rare. A clinical trial for a rare disease can only enrol a small number of people, so every measurement has to count. Planning such a trial means answering two questions: how fast does the disease usually progress, and how many patients are needed to show that a treatment slows it?

Standard tools struggle with this in three ways:

Small groups. There are few patients, so ordinary statistics become unreliable.
Curved decline. Vision loss rarely follows a straight line, and forcing a straight line misstates the rate.
Lost fast progressors. Every instrument has a limit. When a patient's reading falls below what it can measure, many tools throw that reading away. Those are often the patients who are progressing fastest, so discarding them makes the disease look slower than it is.

What K-KODE does. You give it a spreadsheet of patient measurements over time. It cleans the data and tells you exactly what it removed and why. It keeps the readings that hit the instrument's limit, treating them as "at least this much loss" instead of deleting them. For each eye it lets four different curve shapes compete and keeps the one the evidence supports best. It then estimates the typical rate of progression for the group, with honest error margins, and estimates how many patients a trial would need. Finally it checks that number against thousands of simulated trials.

What it will not do. It does not diagnose, treat, or predict what will happen to an individual patient. It is a planning aid for researchers and statisticians. It does not replace a biostatistician, and it refuses to give an answer when the data cannot support one, for example when a group of patients is improving or not changing.

How much to trust it. The methods are standard and published. The code is open for anyone to inspect, and it ships with an automated test suite. It has been tested two ways. First, on computer-generated patient data where the true answer is known in advance, where the engine recovers it. Second, it has been run end to end on real patient data from the RUSH2A natural-history study (EZ area from 125 participants), where it kept the measurements that hit the instrument's limit, agreed with a separate statistical analysis, and revealed three weaknesses that were then fixed. That real-data run is an engineering check, not a clinical validation, and it has not yet been independently reviewed. K-KODE is Research Use Only: it is not an approved medical device and must not be used to make clinical decisions.

For clinicians and trial teams

What goes in. A table with one row per visit: patient ID, eye (optional), visit date and a numeric endpoint. Examples are ellipsoid zone (EZ) area or width, static perimetry, microperimetry, or any other measurement that changes over time. You also state the instrument's measurement floor, and optionally a ceiling.

What comes out. A report with:

a data audit listing every row that was removed, de-duplicated or flagged
the best-supported progression curve for each eye, with how strongly the evidence favours it over the alternatives
the average rate of progression for the group, with a confidence or credible interval
the number of patients (not eyes) needed to detect a given treatment effect at a stated power, cross-checked by simulation

How to read it.

The average rate is only as representative as the cohort it came from. Check the cohort size and follow-up length printed in the report.
Intervals are wide for small cohorts. That is the tool being honest, not a malfunction.
If model competition shows no clear winner among the four curve shapes, the report says so. Treat the sample size as a range, not a single number.
The tool will refuse and say why if the data show improvement, no change, or too few points.

What it does not do. It does not diagnose, stage or predict outcomes for an individual patient, and it is not an approved medical device. It supports trial planning and natural-history analysis. A biostatistician should review any analysis before it informs a protocol.

The problem

Rare-disease trials fight math that generic tools weren't built for:

Cohorts too small for standard statistics.
Biology that decays in curves, not straight lines.
Fast progressors who get silently dropped the moment a measurement crosses the instrument's floor.

K-KODE was built for these problems, first for USH2A retinal degeneration. It packages established, peer-reviewed methods (censored maximum likelihood, information-criterion model competition, mixed-effects and hierarchical Bayesian modeling, Monte Carlo trial simulation) into one auditable pipeline that runs on a raw CSV. Nothing here is a novel statistical claim. The contribution is putting the right methods together, exposing every assumption, and reporting uncertainty honestly.

What sets it apart
Nothing disappears silently. Every dropped, de-duplicated or flagged row is counted and reported.
Censored readings stay in the model. A reading at the instrument floor (or ceiling) becomes information ("at least this much progression") instead of being clamped or discarded.
The decay shape is chosen by evidence, per eye. Four competing functional forms are compared with small-sample-corrected AIC and a Jacobian correction, so cross-model comparisons are valid.
Patients, not eyes, are the unit of analysis for sample size. Eyes of one patient are correlated, so sizing counts patients.
Three population estimates that cross-check each other: a standard mixed-effects model, a fast censored (Tobit) mixed model that keeps censored rows, and a hierarchical Bayesian censored model.
Sample sizes are stress-tested. A closed-form estimate is checked against Monte Carlo simulation of the trial design.
Degenerate inputs are refused, not answered. Improving or flat cohorts, too-few-points fits and non-positive data return a clear error instead of a number.
Endpoint-agnostic. EZ area or width, static perimetry, microperimetry, or any longitudinal numeric endpoint.
Tested and open. An automated test suite on synthetic data with planted ground truth ships with the code, and the code is Apache 2.0.
Pipeline
Raw CSV of visits
Data audit: every droppedrow counted
Per-eye curve fit: floor andceiling readings kept
Four curve shapes compete:best by evidence
Group progression rate:three cross-checkingestimates
Patient-level sample size
Simulation check of the trial
Plain report with uncertainty
Stage	Method	Why it matters
Data cleaning	Automated audit of missing fields, bad dates, non-numeric values, exact duplicates, same-date repeats	Every dropped row is counted and reported
Per-eye decay fit	Censored (Tobit-style) MLE with floor and optional ceiling, across 4 competing functional forms, selected via AICc with a Jacobian correction	Eyes that cross a measurement bound stay in the model, and forms are compared on equal footing
Population estimate	Standard mixed-effects NLME; censored (Tobit) mixed model with adaptive Gauss-Hermite integration; hierarchical Bayesian censored NLME (PyMC / NUTS)	Independent estimates that cross-check each other; two of the three keep censored rows
Correlated random effects	Baseline severity and progression rate modeled jointly	Captures a biological relationship that independent-effects models can't represent
Sample size	Closed-form formula (per patient) and Monte Carlo trial simulation	The number is tested against a simulated version of the trial
Glossary
Endpoint. The measurement a trial tracks over time, such as EZ area.
EZ (ellipsoid zone) area. The size of the retinal region where photoreceptors are still intact, seen on OCT scans. In USH2A it shrinks as the disease progresses.
Censoring (floor or ceiling). When a value falls past what the instrument can measure, we only know it is "below the floor" or "above the ceiling". K-KODE keeps that partial information instead of deleting the reading.
Tobit / censored maximum likelihood. The standard statistical method for fitting data that include censored readings.
AICc. A score that compares competing models while penalising complexity, with a correction for small samples. Lower is better.
Jacobian correction. A mathematical adjustment that makes scores comparable between models that describe the data on different scales (for example straight versus logarithmic).
Random effects / mixed model. A way to estimate the group average while letting each patient have their own starting point and rate.
Hierarchical Bayesian model. A mixed model that reports a full range of plausible values rather than one number, and can use prior knowledge.
Confidence or credible interval. The range of values consistent with the data. A wider range means more uncertainty.
Monte Carlo simulation. Running thousands of simulated trials on the computer to see how often a design would succeed.
Power and sample size. Power is the chance a trial detects a real treatment effect. Sample size is how many patients are needed to reach a chosen power, commonly 80%.
Research Use Only (RUO). Intended for research. Not approved for diagnosing or treating patients.
Questions people ask

Is this a medical device or a diagnostic tool? No. It is research software for planning studies. It is Research Use Only and has no regulatory approval.

Has it been tested on real patients? Yes, on real research data, but as an engineering check and not a clinical validation. It has been run end to end on the RUSH2A natural-history dataset (EZ area, 125 participants), which found and fixed three weaknesses. It has also been tested on computer-generated cohorts where the true answer is known, and that test suite is published with the code. The real-data results are unreviewed and are not reported here. Independent review is the next step.

Why does it keep readings other tools discard? A reading at the instrument's limit still says something: the disease progressed at least that far. Discarding it makes fast progressors vanish and the disease look slower than it is.

Can it tell me how my eyes (or my child's eyes) will progress? No. It estimates group-level behaviour for research planning. Questions about an individual belong with their eye-care team.

Who built it and why? Eric Fitzgerald, founder of Elite Architecture Intelligence Inc. in Montreal, built it as an open-source contribution to the USH2A research community. See the Dedication.

How can someone check the work? Read the code, run the tests, and read the Known limitations section. Corrections are welcome through the Collaboration section.

Table of contents

Start here · In plain language · For clinicians · Key capabilities · What this version does not claim · Known limitations · Installation · Input data format · Usage · Running the tests · Methodology & references · Output · Validation status · Glossary · Questions · Roadmap · Collaboration · Dedication · License

Why this exists

K-KODE is a rigorously sourced engineering effort to give the USH2A research community a better tool for trial planning. It is offered open-source, in the spirit of collaboration with the biostatisticians and clinicians who know this disease best. Feedback, corrections, and partnership are welcome, and any issue identified will be addressed promptly.

Key capabilities
1. Generalized endpoint support

Not hardcoded to one measurement. Any longitudinal numeric endpoint (EZ width, EZ area, static perimetry sensitivity, microperimetry sensitivity) passes in via endpoint_column=. This matters because the RUSH2A investigators prioritize functional measures (for example, rate of change of static-perimetry mean sensitivity) over structural ones, and report that a baseline EZ area of at least 3 mm² is needed to detect structural change.

Source: Maguire MG, Birch DG, et al., for the REDI Working Group / Foundation Fighting Blindness Clinical Consortium. "Endpoints and Design for Clinical Trials in USH2A-Related Retinal Degeneration: Results and Recommendations From the RUSH2A Natural History Study." Translational Vision Science & Technology. 2024; 13(10):15. DOI: 10.1167/tvst.13.10.15

2. Proper censored-data handling (Tobit-style MLE), floor and ceiling

Every candidate model is fit per eye by maximum likelihood with a censored Gaussian likelihood: points inside the measurement range contribute a normal density, points at or below the floor contribute the normal CDF, and (when measurement_ceiling= is given) points at or above the ceiling contribute the normal survival function. Both bounds are transformed onto each model's own scale, so the censored probability is identical across models. A censored point becomes real information instead of being discarded.

Source: Tobin J. "Estimation of Relationships for Limited Dependent Variables." Econometrica. 1958; 26(1):24-36. DOI: 10.2307/1907382

3. Four candidate functional forms, competed per eye

Linear, Square-Root, Log-Exponential, and Power-Law curves are fit under the same censored likelihood. Small-sample-corrected AIC and Akaike weights rank the evidence. Because the forms are fit on different transformed scales of the endpoint, each likelihood includes the Jacobian term needed to compare them as likelihoods of the original measurement; this is verified numerically in the test suite (the density integrates to one on every scale). Log-scale forms are declared inadmissible, rather than scored arbitrarily, when the floor is zero or below and censored rows exist.

Sources: Hurvich CM, Tsai CL. Biometrika. 1989; 76(2):297-307. DOI: 10.1093/biomet/76.2.297. Burnham KP, Anderson DR. Model Selection and Multimodel Inference (2nd ed). Springer; 2002.

4. Standard errors, small-sample intervals and an identifiability guard

Parameter uncertainty comes from a central finite-difference Hessian at the fitted optimum. Because each eye has few visits, per-eye slope intervals use a small-sample standard error and a t quantile (about 94-95% coverage of the true slope in planted-truth checks, versus about 82% for a plain 1.96 × SE interval). A fit with fewer than two uncensored points is not identified, so it is refused rather than allowed to produce an unbounded slope.

5. Patient-level, simulation-validated sample size

Sample size is computed per patient (eyes averaged), the unit a trial randomizes. Many two-arm trials are then simulated under the fitted population parameters, empirically measuring power at candidate sample sizes rather than relying only on a closed-form formula's assumptions. Warnings are raised when the cohort is small, the mean decline is not distinguishable from zero, or the required n is implausibly small.

Source: Burton A, Altman DG, Royston P, Holder RL. "The Design of Simulation Studies in Medical Statistics." Statistics in Medicine. 2006; 25(24):4279-4292. DOI: 10.1002/sim.2673

6. Censored population model (fast, frequentist)

fit_censored_population_model() fits a censored mixed model with correlated patient-level random intercept and slope, integrated out with adaptive Gauss-Hermite quadrature. It keeps floor- and ceiling-censored rows, returns Wald intervals, and gives a Jacobian-corrected AIC comparable across model forms. Both eyes of a patient share that patient's random effects. With model="Linear" it is a native-scale Tobit cross-check.

7. Hierarchical Bayesian censored NLME (PyMC)

A full population-level MCMC sampler (NUTS) using pm.Censored estimates population decay and between-patient variance from every observation, including censored points, with optional LKJ-Cholesky correlated random effects.

Sources: Laird NM, Ware JH. Biometrics. 1982; 38(4):963-974. DOI: 10.2307/2529876. Lewandowski D, Kurowicka D, Joe H. Journal of Multivariate Analysis. 2009; 100(9):1989-2001. DOI: 10.1016/j.jmva.2009.04.008. Implementation: PyMC, https://www.pymc.io/

8. Automatic population-parameter selection

For the sample-size simulation, population parameters come from the censored population model when any rows are censored (the standard model drops them), otherwise from the standard mixed-effects model, with the Bayesian model as a further fallback. A borderline Bayesian fit retries with more MCMC effort without loosening the convergence bar.

9. Plateau-aware floor detection (available, not yet auto-wired)

A _detect_plateau_floor() helper distinguishes a genuine instrument floor from a single low reading. It is not yet called automatically; K-KODE takes an explicit measurement_floor.

What this version deliberately does not claim to do
Does not ingest raw OCT images or perform retinal layer segmentation. It assumes a reading center or imaging pipeline has already produced a numeric measurement per visit.
The standard (statsmodels) mixed model still excludes censored rows. Use the censored population model or the Bayesian model for heavy censoring.
Not FDA-qualified or validated as a Drug Development Tool. Research Use Only.
The real-data run on RUSH2A is an exploratory engineering check, not a clinical validation, and its results are unreviewed (see Validation status).
Known limitations
Eye structure is approximate. In the censored population model both eyes share the patient's random effects; the Bayesian model treats each eye as an independent unit (this is flagged in its output).
Model discrimination has limits. At higher noise, Square-Root and Log-Exponential are the pair most often confused (see Validation status).
Simulated sample sizes are conservative. The default simulation schedule is sparser than typical natural-history schedules and patient-level slope variance absorbs eye-level variance, so simulated n runs above the closed-form n.
Efficacy is proportional slowing on the fitted model's scale. For transformed models that is not the same as a proportional reduction of the raw rate.
Sparse natural-history data limit form selection. When most eyes have only 1 to 5 visits, per-eye model weights are nearly tied. K-KODE then compares the four forms at the cohort level, which worked at low noise in synthetic tests (named the planted form 3/3) but confused neighbouring forms at higher noise. Read the selected form as "how curved the decline is", not as a biological mechanism.
Sample sizes can disagree. If the closed-form and simulated sample sizes differ by more than 2x, the report prints a warning and neither number should be quoted without statistical review.
Informative dropout is not modeled. Patients who progress faster may leave studies earlier; this is on the roadmap.
Installation
pip install -r requirements.txt

Core: numpy, pandas, scipy, statsmodels. Optional (Bayesian censored NLME): pymc, arviz.

Input data format

One row per visit, as a CSV or pandas DataFrame:

Column	Description
patient_id	Patient identifier
visit_date	Visit date (any format pandas.to_datetime can parse)
(endpoint column)	Numeric endpoint value; the column name is passed via endpoint_column=
(eye column, optional)	Eye identifier (OD/OS); pass its name via eye_column=

Tracking both eyes? Always pass eye_column=. Otherwise both eyes on the same visit date are pooled into a single regression per patient, which biases the result. The data-quality report warns you if it detects same-date duplicates. Time zero is each patient's first visit, shared by both eyes.

Usage
python
from kkode_engine import KKodeApexEngine

engine = KKodeApexEngine(
    data_source="my_cohort.csv",
    endpoint_column="ez_area_mm2",
    eye_column="eye",             # optional
    measurement_floor=0.05,       # instrument/assay floor for this endpoint
    measurement_ceiling=None,     # optional upper bound, e.g. 36 for dB endpoints
)

results = engine.run_full_analysis(
    target_power=0.80,
    alpha=0.05,
    therapeutic_efficacy=0.30,    # assumed fractional slowing of decay rate
)

print(engine.generate_report())

run_full_analysis() runs the full pipeline: data cleaning, per-eye model competition, per-eye decay fits, standard mixed-effects fit, censored population model, closed-form sample size, simulated sample size, and the Bayesian censored model (auto-triggered under heavy censoring). Each stage is independently callable.

Running the tests
python test_kkode_engine.py

The suite (44 tests) uses synthetic cohorts with planted ground truth. A passing run prints one PASS line per test and ends with 44/44 passed; it takes several minutes because the population-model and simulation tests are the slow ones. The Bayesian model is smoke-tested separately and requires PyMC.

Methodology & references
Maguire MG, Birch DG, Duncan JL, et al., for the REDI Working Group and the Foundation Fighting Blindness Clinical Consortium Investigator Group. "Endpoints and Design for Clinical Trials in USH2A-Related Retinal Degeneration: Results and Recommendations From the RUSH2A Natural History Study." Translational Vision Science & Technology. 2024; 13(10):15. DOI: 10.1167/tvst.13.10.15
Tobin J. "Estimation of Relationships for Limited Dependent Variables." Econometrica. 1958; 26(1):24-36. DOI: 10.2307/1907382
Hurvich CM, Tsai CL. "Regression and Time Series Model Selection in Small Samples." Biometrika. 1989; 76(2):297-307. DOI: 10.1093/biomet/76.2.297
Burnham KP, Anderson DR. Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach. 2nd ed. Springer; 2002.
Burton A, Altman DG, Royston P, Holder RL. "The Design of Simulation Studies in Medical Statistics." Statistics in Medicine. 2006; 25(24):4279-4292. DOI: 10.1002/sim.2673
Laird NM, Ware JH. "Random-Effects Models for Longitudinal Data." Biometrics. 1982; 38(4):963-974. DOI: 10.2307/2529876
Lewandowski D, Kurowicka D, Joe H. "Generating Random Correlation Matrices Based on Vines and Extended Onion Method." Journal of Multivariate Analysis. 2009; 100(9):1989-2001. DOI: 10.1016/j.jmva.2009.04.008
PyMC Development Team. PyMC: Probabilistic Programming in Python. https://www.pymc.io/
Output: understanding the report

generate_report() returns a plain-language executive summary covering:

Data quality: raw rows in, rows retained, rows dropped and why, how many were floor- or ceiling-censored (kept and modeled, not dropped), and any warnings.
Best-supported functional form: which decay shape led, by how much, and how many patients (not just eyes) it rests on.
Population decay rate: from the standard mixed model, the censored population model (with a 95% confidence interval), and the Bayesian model (95% credible interval) when run.
Sample size (closed-form): required n per arm, with a bootstrap 95% confidence interval and any warnings.
Sample size (simulation-validated): required n per arm to empirically reach target power.

The report closes with an explicit reminder: this is a planning aid, not a finalized protocol.

Validation status & integrity

K-KODE is Research Use Only. Here is exactly where validation stands.

Completed

Ground-truth testing on synthetic cohorts with known, planted parameters, backed by an automated test suite (44 tests). This testing exposed a model-selection bias (a missing Jacobian term in cross-model likelihood comparison), which was diagnosed, corrected and documented in the changelog at the top of kkode_engine.py. A later line-by-line audit found and fixed further issues, also documented there.
Measured behavior on synthetic data (planted truth; settings are in the test suite and changelog):
Model competition named the true decay form in 32 of 32 cohorts at low noise and 23 of 32 at higher noise; Square-Root and Log-Exponential were the confusable pair.
The closed-form sample size delivered 76-87% true power against an 80% target.
Per-eye 95% slope intervals covered the true slope about 94-95% of the time.
The censored population model recovered the planted decline, variance components and standard error, and was less biased than dropping censored rows.
Pipeline exercised end to end on the RUSH2A natural history dataset (EZ area: 715 readings, 125 participants, 249 eyes; obtained from the Foundation Fighting Blindness / Jaeb Center under a data use agreement). The first run on the corrected engine kept all 27 sub-threshold readings, and its log-scale decline estimate agreed with an independent mixed-effects analysis. It also exposed three weaknesses (form selection when most eyes have few visits, no warning when the two sample-size methods disagree, and a missing interval when a variance component sits near zero), which were fixed and are covered by tests. These are exploratory, unreviewed engineering results. No real-data decline rates or sample sizes are reported here, and they will not be until independent statistical review and confirmation of the data-use-agreement terms.

Not yet done

Independent review and replication of the exploratory RUSH2A run (the run has been done once on the current build; its results are unreviewed and unreported).
Independent biostatistician review.
Replication on a second, independent dataset.
Validation on functional endpoints (static perimetry, microperimetry).
FDA qualification (not applicable to this RUO release).

The source of the RUSH2A data is the Foundation Fighting Blindness Clinical Consortium, but the analyses, content and conclusions presented herein are solely the responsibility of the authors and may not reflect the views of the Foundation Fighting Blindness.

We publish this status plainly because trust in a trial-planning tool has to be earned in the open. If you are a biostatistician, clinician, or researcher and find something wrong, please open an issue or reach out directly. Corrections are welcome and will be fixed promptly.

Roadmap
Independent review of the exploratory RUSH2A run, and publication of the audit trail once the data use agreement terms are confirmed.
Extend validation to functional endpoints: static perimetry and microperimetry sensitivity.
Sensitivity-dependent measurement noise and an informative-dropout sensitivity analysis.
Spline or fractional-polynomial decay as a fifth competing form.
Covariate-adjusted and prognostic-enrichment sample sizing.
Independent replication on a second dataset, and independent biostatistician review.
Planned: machine-learning-assisted multi-endpoint progression forecasting with calibrated uncertainty, and cross-cohort data harmonization.
Wire plateau-aware floor detection into the automatic cleaning pipeline.
Collaboration

We welcome biostatisticians, clinicians, trial sponsors, patient-advocacy organizations, and AI researchers. Open an issue, or contact:

Eric Fitzgerald, Founder, Elite Architecture Intelligence Inc. (EAI-BIO) eric@eaiinc.ca · github.com/EAI-BIO/KKODE-BIO

Dedication

K-KODE is personal, and I am on a mission to see it through.

It is dedicated to my family and to all families living with inherited vision loss, and in honour of the blind woman I was able to help save from a house fire, an act for which I was awarded the National Assembly of Quebec's Médaille du député for bravery and bravery honours from the City of Pincourt.

Every line of this project is built to help the research community find solutions as soon as possible.

With gratitude to the USH2A research and patient community.

License

Apache License 2.0. See LICENSE for full terms.
