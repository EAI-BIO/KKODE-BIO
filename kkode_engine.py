"""
K-KODE ENGINE (v55.0)
Generalized longitudinal biomarker decay engine + censored-data-aware
model competition + simulation-validated clinical trial sample sizing,
for USH2A / RUSH2A-style retinal degeneration trial planning.
 
============================================================================
CHANGELOG, AUDIT 2: fixes found in a line-by-line code review
============================================================================
The Jacobian correction below (Audit 1) was verified correct and is
unchanged. A second review found the following, all fixed in this build,
each covered by a regression test in test_kkode_engine.py:
 
1. ZERO-READING GUARD (fit_censored_model). The guard `np.any(y <= 0)`
   looked at ALL observations, including floor-censored ones. Any eye with
   a reading at or below zero was rejected by the Square-Root,
   Log-Exponential and Power-Law fits and therefore "won" the model
   competition by default as Linear-only. The guard now checks only
   UNCENSORED observations, and transformed values are computed on a
   clamped copy of y so censored zeros never produce log(0) = -inf.
 
2. POWER-LAW DECAY RATE (run_per_patient_decay). decay_rate was None for
   Power-Law, so if Power-Law won the competition, every per-patient rate
   was discarded and the closed-form sample size failed. decay_rate is now
   -slope for every model. For Linear it is endpoint-units/year; for the
   transformed forms it is a rate on that model's own scale (per log-unit,
   per sqrt-unit, per log-time). Because the closed-form sample size depends
   only on the ratio (between-unit SD / mean rate), it is scale-free and
   remains valid for all four forms.
 
3. POWER-LAW PREDICTOR IN THE BAYESIAN MODEL (fit_bayesian_censored_nlme).
   The Bayesian model used raw time as the predictor for every model form.
   For Power-Law it must use log(t + epsilon), exactly as the per-patient
   fit and (since Audit 1) the standard mixed-effects fit do. Fixed.
 
4. POWER-LAW PREDICTOR IN THE SIMULATION (_simulate_power). Simulated visits
   used raw time while the fitted slope is on log(t + epsilon). The
   simulation now builds its predictor with _prepare_predictor(), matching
   the fit. NOTE: with Power-Law, the baseline visit (t = 0) sits at
   log(epsilon), a very high-leverage point; interpret simulated power for
   Power-Law with extra caution.
 
5. EYES TREATED AS INDEPENDENT PATIENTS (sample size). Each eye was an
   independent unit, so 125 patients / 249 eyes was counted as 249
   independent observations. Eyes from the same patient are correlated, so
   this overstates information. Now:
   - compute_closed_form_sample_size(unit="auto") aggregates to ONE decay
     rate per patient (mean of that patient's eyes) whenever eye_column is
     set, so n is per PATIENT, the unit a trial randomizes. Pass
     unit="eye" to reproduce the old behavior.
   - fit_mixed_effects_nlme(nested_eyes=True) groups by patient with an
     eye-level variance component. If that model fails to fit it falls back
     to the old grouping and says so in the results.
   - The Bayesian model still treats each eye as its own group. That is
     stated in its output (eyes_treated_as_independent) rather than hidden.
 
6. INVENTED INTERCEPT SD (_get_population_parameters_for_simulation). The
   standard-NLME path set tau_intercept = 3 * tau_slope, a made-up number.
   It now uses the fitted random-intercept variance. (The simulation uses
   per-patient slopes, so power results are unaffected; the number was
   simply wrong.)
 
7. MINOR: optimizer failure check now rejects non-finite objective values;
   a warning is raised when the mean decay rate is not positive (no
   measurable decline, so a sample size is meaningless); the report footer
   now matches the changelog (the earlier Square-Root headline is retired;
   a RUSH2A re-run under the corrected build is still PENDING).
 
What this audit does NOT change: the Jacobian-corrected likelihood, the
censored MLE, parameter estimates, standard errors, or the Bayesian priors.
 
============================================================================
CHANGELOG, AUDIT 1: v55.0 (original) -> v55.0 (corrected build)
============================================================================
CRITICAL FIX -- Jacobian correction in per-patient model competition.
 
THE BUG (v55.0 and earlier): run_model_competition() / fit_censored_model()
compare four candidate functional forms -- Linear, Square-Root,
Log-Exponential, Power-Law -- by fitting a censored Gaussian likelihood to
DIFFERENT TRANSFORMED SCALES of the same endpoint (raw y for Linear,
sqrt(y) for Square-Root, log(y) for the other two), then comparing their
AICc scores directly. This comparison is only valid if each model's
log-likelihood is expressed as a likelihood of the ORIGINAL, untransformed
endpoint y -- which requires adding the log of the Jacobian determinant,
log|dg/dy|, where g is that model's transform. v55.0's _neg_log_likelihood
omitted this term entirely.
 
WHY THIS MATTERS: because sqrt() and log() compress the value of y
(their derivatives are <1 for y>1, so log|dg/dy| < 0), the omitted term
was systematically most negative for Square-Root and Log-Exponential /
Power-Law, and exactly zero for Linear (since d(y)/dy = 1). Leaving it
out therefore silently inflated the apparent log-likelihood -- and hence
lowered the apparent AICc -- of the Square-Root and Log-Exponential /
Power-Law models relative to Linear, independent of which model actually
generated the data. Verified empirically: on synthetic cohorts generated
under a KNOWN, planted Linear ground truth, v55.0's model competition
selected Square-Root for 90/90 simulated patients (100% wrong, and not by
chance -- a systematic, directional bias). Adding the missing Jacobian
term corrected this to a plurality-correct Linear recovery on the same
data. See kkode_v55_audit_report.md (accompanying this file) for the full
diagnostic trail, including the zero-noise sanity check that first
isolated this and the noise/decline-magnitude sweeps that characterize it.
 
THE FIX: _neg_log_likelihood now accepts and adds a per-observation
log-Jacobian term, computed from the ORIGINAL (untransformed) endpoint
value of each uncensored observation:
    Linear:            log|dy/dy|       = 0
    Square-Root:       log|d(sqrt y)/dy| = -log(2 * sqrt(y))
    Log-Exponential:   log|d(log y)/dy|  = -log(y)
    Power-Law:         log|d(log y)/dy|  = -log(y)   (Power-Law's y-transform
                                                        is the same log(y) as
                                                        Log-Exponential; its
                                                        x-transform, log(t),
                                                        does not require a
                                                        y-Jacobian term)
This term is a function of the DATA only, not of the fitted parameters
(a, b, sigma) -- so it does NOT shift the location of the MLE for any
SINGLE model (parameter estimates, decay rates, and sample-size
calculations from v55.0 runs are unaffected by this bug and require no
correction). It only matters, and was only missing, for the CROSS-MODEL
AICc comparison in run_model_competition() / fit_censored_model(), which
is exactly where it was silently biasing which functional form "wins."
Censored (floor-hitting) observations contribute a CDF term, not a
density, and correctly required no Jacobian correction in either version
-- only the uncensored/density terms were affected.
 
IMPORTANT CAVEAT (UPDATED): this fix was first validated on SYNTHETIC
data with a known, planted ground truth (see audit report). The original
internal validation report's headline finding -- "K-KODE independently
identified Square-Root" -- was produced by the bugged competition logic,
which was predisposed toward Square-Root regardless of the true pattern.
The earlier Square-Root headline is therefore RETIRED. A re-run of this
corrected build on the RUSH2A data has NOT yet been done in this repo;
until it is, no functional-form conclusion on real data should be quoted.
Population decay-rate estimates for a single, given model are unaffected
by the Jacobian bug.
 
AUDIT 3: (1) censored-row likelihood uses logcdf instead of a
clip at 1e-12 that replaced real tail probabilities with a constant;
(2) Log-Exponential/Power-Law fits are declared inadmissible when
measurement_floor <= 0 and any row is censored (log 0 is undefined);
(3) time zero is now the patient's first visit, shared by both eyes;
(4) model-competition output reports patients represented, not just eyes;
(5) removed unverified validation claims from this docstring and the
report footer. 
 
AUDIT 4: ADDED ceiling (upper) censoring (measurement_ceiling=)
through per-eye fits, model competition, data-quality report, statsmodels
filter and the Bayesian model (smoke-tested once: 4.98 vs true 5.0, short chains flagged r_hat 1.056);
ADDED fit_censored_population_model(): fast censored mixed model with
correlated patient random intercept+slope integrated by adaptive Gauss-Hermite;
ADDED source='censored_mle' for the simulation.
 
AUDIT 5: degenerate-input refusals (improving or flat cohorts
are refused by the simulation as well as the closed form; exact duplicate
rows are dropped and reported; same-date different-value rows are warned;
very small required n and a non-significant mean decline are flagged);
per-eye slope intervals now use a small-sample SE and t quantile (coverage
of the old 1.96*SE interval was ~82% on 7-visit eyes; the new interval
covered 94-95% in planted-truth checks).
MEASURED CALIBRATION (synthetic, planted truth; see test suite): model
competition named the true form 32/32 at low noise and 23/32 at higher
noise (Square-Root and Log-Exponential are the confusable pair); the
closed-form sample size gave 76-87% true power against an 80% target; the
simulated sample size is conservative (95-97% true power) because its
default 4-visit schedule is sparser than the data and patient-level slope
variance absorbs eye-level variance.
 
============================================================================
ORIGINAL v55.0 DOCUMENTATION (retained below)
============================================================================
KEY CAPABILITIES:
1. GENERALIZED ENDPOINT SUPPORT. Not hardcoded to EZ width. Any
   longitudinal numeric endpoint (EZ width, EZ area, static perimetry
   sensitivity, microperimetry sensitivity) can be passed in via
   endpoint_column=. This matters because RUSH2A's own published
   recommendations prioritize functional measures (e.g. rate of change of
   static-perimetry mean sensitivity) over structural ones, and report that
   a baseline EZ area of at least 3 mm^2 is needed to detect structural
   change (so EZ area is most useful as an eligibility criterion). A tool
   that only understands EZ width is modeling the field's secondary
   endpoint.
   SOURCE: Maguire MG, Birch DG, et al. (REDI Working Group / Foundation
   Fighting Blindness Clinical Consortium). "Endpoints and Design for
   Clinical Trials in USH2A-Related Retinal Degeneration: Results and
   Recommendations From the RUSH2A Natural History Study." Transl Vis Sci
   Technol. 2024; 13(10):15. DOI: 10.1167/tvst.13.10.15
   (title, authors, volume/issue and DOI verified Oct 2026.)
2. PROPER CENSORED-DATA HANDLING (Tobit-style MLE), not floor-and-drop.
   SOURCE: Tobin J (1958). "Estimation of Relationships for Limited
   Dependent Variables." Econometrica, 26(1), 24-36.
3. FOUR CANDIDATE FUNCTIONAL FORMS competed per patient via AICc
   (WITH JACOBIAN CORRECTION -- see changelog above).
   SOURCE (AICc): Hurvich CM, Tsai CL (1989). Biometrika, 76(2), 297-307.
   SOURCE (Akaike weights): Burnham KP, Anderson DR (2002). Model
   Selection and Multimodel Inference (2nd ed). Springer.
4. NUMERICAL-HESSIAN STANDARD ERRORS.
4b. IDENTIFIABILITY GUARD (>=2 uncensored points required per patient fit).
5. SIMULATION-VALIDATED SAMPLE SIZE (Monte Carlo trial simulation).
   SOURCE: Burton A, Altman DG, Royston P, Holder RL (2006). Stat Med,
   25(24), 4279-4292.
6. HIERARCHICAL BAYESIAN CENSORED NLME (PyMC), with CORRELATED random
   effects (LKJ-Cholesky prior) as the default.
   SOURCE: Laird NM, Ware JH (1982). Biometrics, 38(4), 963-974.
 
WHAT THIS VERSION DELIBERATELY DOES NOT CLAIM TO DO:
  - It does not ingest raw OCT images or do retinal layer segmentation.
  - Standard NLME (non-Bayesian) still excludes floor-censored rows at the
    population level; use fit_bayesian_censored_nlme() for heavy censoring.
  - It is not FDA-qualified or validated as a Drug Development Tool.
  - Sample sizes are planning estimates from a model of natural-history
    decline. Closed-form uses the SD of ESTIMATED per-unit slopes, which
    includes estimation noise, so it is mildly conservative.
  - Simulated "therapeutic efficacy" is a proportional reduction of the
    slope ON THE FITTED MODEL'S SCALE. For transformed models that is not
    the same as a proportional reduction of the raw rate.
  - Validation so far is synthetic data with planted truth only. A
    re-run on RUSH2A (access via Data Use Agreement; not public) and
    independent clinical/biostatistical review are outstanding.
  - Ceiling censoring is supported only when measurement_ceiling is given.
"""
import os
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from typing import Union, Dict, Any, List, Optional, Tuple
 
logger = logging.getLogger("KKODE_Apex_v55")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
 
MIN_POINTS_FOR_TOBIT_FIT = 4          # 3 free params (a, b, sigma) -> need >=1 df
MIN_POINTS_FOR_MODEL_COMPETITION = 4  # matches MIN_POINTS_FOR_TOBIT_FIT
MIN_COHORT_FOR_SAMPLE_SIZE = 3
SMALL_COHORT_WARNING_THRESHOLD = 10
HEAVY_CENSORING_WARNING_FRACTION = 0.15
POWER_LAW_TIME_OFFSET_YEARS = 1.0 / 365.25  # avoids ln(0) at baseline for power-law
FLOOR_DETECTION_QUANTILE = 0.02  # used only by _detect_plateau_floor(), see docstring there
JACOBIAN_Y_EPSILON = 1e-9  # numerical floor to avoid log(0) in Jacobian terms
TRANSFORM_Y_EPSILON = 1e-6  # clamp so sqrt/log never see <= 0 on censored rows
POSITIVE_ONLY_MODELS = ("Log-Exponential", "Power-Law", "Square-Root")
 
 
# ======================================================================
# Model definitions & functional transformations
# ======================================================================
def _transform_y(model: str, y: np.ndarray) -> np.ndarray:
    if model in ("Linear",):
        return y
    if model in ("Square-Root",):
        return np.sqrt(y)
    if model in ("Log-Exponential", "Power-Law"):
        return np.log(y)
    raise ValueError(f"Unknown model: {model}")
 
 
def _prepare_predictor(model: str, t: np.ndarray) -> np.ndarray:
    if model == "Power-Law":
        return np.log(np.asarray(t, dtype=float) + POWER_LAW_TIME_OFFSET_YEARS)
    return np.asarray(t, dtype=float)
 
 
def _floor_in_transformed_space(model: str, floor_value: float) -> float:
    if model == "Linear":
        return floor_value
    if model == "Square-Root":
        return np.sqrt(max(floor_value, 0.0))
    FLOOR_LOG_EPSILON = 1e-6
    return np.log(max(floor_value, FLOOR_LOG_EPSILON))
 
 
def _ceiling_in_transformed_space(model: str, ceiling_value: float) -> float:
    """Upper (ceiling) censoring bound on the model's transformed scale.
    Same monotone map as the data, so P(Y >= ceiling) is identical on every
    scale and AICc stays comparable across models."""
    if model == "Linear":
        return ceiling_value
    if model == "Square-Root":
        return float(np.sqrt(max(ceiling_value, 0.0)))
    return float(np.log(max(ceiling_value, TRANSFORM_Y_EPSILON)))
 
 
def _log_jacobian_dgdy(model: str, y_raw: np.ndarray) -> np.ndarray:
    """
    Returns log|dg/dy| for the given model's y-transform g, evaluated at the
    ORIGINAL (untransformed) endpoint values y_raw. This is the correction
    term needed to make AICc/log-likelihood comparable across models fit on
    different transformed scales of the same response -- see module
    changelog (Audit 1).
 
    Only defined/needed for UNCENSORED observations (censored observations
    contribute a CDF term to the likelihood, not a density, and are
    correctly unaffected by this correction).
    """
    y_safe = np.maximum(y_raw, JACOBIAN_Y_EPSILON)
    if model == "Linear":
        return np.zeros_like(y_safe)
    if model == "Square-Root":
        return -np.log(2.0 * np.sqrt(y_safe))
    if model in ("Log-Exponential", "Power-Law"):
        return -np.log(y_safe)
    raise ValueError(f"Unknown model: {model}")
 
 
def _detect_plateau_floor(df: pd.DataFrame, value_column: str,
                           subject_column: str = "patient_id",
                           date_column: str = "visit_date") -> Optional[float]:
    """
    Detect a genuine measurement floor by looking for subjects who plateau --
    multiple visits clustered tightly together near a low value over time --
    rather than just taking the low percentile of all values in the dataset.
    Ported from TK-KODE for future use if K-KODE moves from a fixed,
    user-supplied measurement_floor to floor auto-detection. Not currently
    called from clean_and_transform().
    """
    candidate = float(df[value_column].quantile(FLOOR_DETECTION_QUANTILE))
    band = max(float(df[value_column].std()) * 0.05, 1e-6)
 
    use_last_k = 2
    plateau_subject_count = 0
    near_total = 0
    for _, group in df.sort_values(date_column).groupby(subject_column):
        last_vals = group[value_column].tail(use_last_k)
        if len(last_vals) < use_last_k:
            continue
        is_flat = (last_vals.max() - last_vals.min()) <= band
        is_near_floor = abs(last_vals.mean() - candidate) <= band * 2
        if is_flat and is_near_floor:
            plateau_subject_count += 1
            near_total += len(last_vals)
 
    min_subjects_for_floor = 2
    min_fraction_for_floor = 0.03
    if (
        plateau_subject_count >= min_subjects_for_floor
        and near_total / max(len(df), 1) >= min_fraction_for_floor
    ):
        return candidate
    return None
 
 
# ======================================================================
# Per-patient censored (Tobit-style) MLE fit
# ======================================================================
def _neg_log_likelihood(params: np.ndarray, x: np.ndarray, g_y: np.ndarray,
                         is_censored: np.ndarray, g_floor: float,
                         log_jacobian: Optional[np.ndarray] = None,
                         is_upper: Optional[np.ndarray] = None,
                         g_ceiling: Optional[float] = None) -> float:
    """
    Censored Gaussian negative log-likelihood on the model's transformed
    scale. log_jacobian (log|dg/dy| per UNCENSORED observation, in the
    original y scale) is added so likelihoods from different transforms are
    comparable as likelihoods of the same, original y. Passing None (or
    zeros, as for Linear) leaves the model's own fit unchanged.
    """
    a, b, log_sigma = params
    sigma = np.exp(log_sigma)
    pred = a + b * x
    resid = g_y - pred
    ll = 0.0
    uncensored = ~is_censored
    if is_upper is not None and np.any(is_upper):
        uncensored = uncensored & ~is_upper
        ll += np.sum(stats.norm.logsf((g_ceiling - pred[is_upper]) / sigma))
    if np.any(uncensored):
        r = resid[uncensored]
        ll += np.sum(stats.norm.logpdf(r, loc=0.0, scale=sigma))
        if log_jacobian is not None:
            ll += np.sum(log_jacobian)
    if np.any(is_censored):
        z = (g_floor - pred[is_censored]) / sigma
        # logcdf is numerically stable in the tails; the old clip at 1e-12
        # silently replaced real tail probabilities with an arbitrary constant.
        ll += np.sum(stats.norm.logcdf(z))
    return -ll
 
 
def fit_censored_model(t: np.ndarray, y: np.ndarray, floor_value: float, model: str,
                       ceiling_value: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Fits one candidate functional form to one patient's (or eye's) data via
    censored (Tobit-style) maximum likelihood, with the Jacobian correction
    so the returned AIC/AICc is comparable across candidate models.
 
    Audit 2 change: the positivity guard now applies ONLY to uncensored
    observations, and transforms run on a clamped copy of y so a censored
    zero can never produce log(0).
    """
    n = len(t)
    if n < MIN_POINTS_FOR_TOBIT_FIT:
        return None
    if model == "Power-Law" and np.any(t < 0):
        return None
 
    if ceiling_value is not None and not ceiling_value > floor_value:
        raise ValueError("ceiling_value must be greater than floor_value.")
    is_censored = y <= floor_value
    is_upper = (y >= ceiling_value) if ceiling_value is not None else np.zeros(len(y), dtype=bool)
    is_density = ~(is_censored | is_upper)   # rows that contribute a density term
    if model in POSITIVE_ONLY_MODELS and np.any(y[is_density] <= 0):
        return None
    # Audit 3: a log-scale model cannot generate readings at or below a
    # floor <= 0 (log 0 = -inf), so a censored row has probability ~0 under
    # it. Rather than let an arbitrary epsilon decide the AICc, such fits are
    # declared inadmissible. Use a floor > 0 (the instrument's true minimum
    # resolution) if log-scale models should compete on floor-hitting eyes.
    if model in ("Log-Exponential", "Power-Law") and floor_value <= 0 and np.any(is_censored):
        return None
 
    x = _prepare_predictor(model, t)
    y_clamped = np.maximum(y, max(floor_value, TRANSFORM_Y_EPSILON))
    g_y = _transform_y(model, y_clamped)
    g_floor = _floor_in_transformed_space(model, floor_value)
    g_ceiling = _ceiling_in_transformed_space(model, ceiling_value) if ceiling_value is not None else None
 
    if np.max(x) == np.min(x):
        return None
 
    # IDENTIFIABILITY GUARD: requires >=2 uncensored points so slope is pinned down
    if np.sum(is_density) < 2:
        return None
 
    log_jacobian = _log_jacobian_dgdy(model, y[is_density])
 
    fit_mask = is_density
    try:
        init_slope, init_intercept, _, _, _ = stats.linregress(x[fit_mask], g_y[fit_mask])
        if not np.isfinite(init_slope) or not np.isfinite(init_intercept):
            init_slope, init_intercept = 0.0, float(np.mean(g_y))
    except Exception:
        init_slope, init_intercept = 0.0, float(np.mean(g_y))
 
    resid0 = g_y[fit_mask] - (init_intercept + init_slope * x[fit_mask])
    init_sigma = max(float(np.std(resid0)), 1e-3) if len(resid0) > 1 else 0.1
    x0 = np.array([init_intercept, init_slope, np.log(init_sigma)])
 
    try:
        res = minimize(
            _neg_log_likelihood, x0, args=(x, g_y, is_censored, g_floor, log_jacobian, is_upper, g_ceiling),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000, "maxfev": 4000},
        )
        if res.fun is None or not np.isfinite(res.fun):
            return None
    except Exception:
        return None
 
    a_hat, b_hat, log_sigma_hat = res.x
    sigma_hat = np.exp(log_sigma_hat)
    neg_ll = res.fun
    k = 3  # a, b, sigma
    aic = 2 * k + 2 * neg_ll
 
    small_sample_correction_unavailable = (n - k - 1) <= 0
    if small_sample_correction_unavailable:
        aicc = aic
    else:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
 
    se_a, se_b = _numerical_hessian_se(res.x, x, g_y, is_censored, g_floor, log_jacobian,
                                       is_upper=is_upper, g_ceiling=g_ceiling)
 
    # Audit 5: the raw ML/Hessian SE ignores that sigma and the intercept are
    # estimated from very few points per eye (coverage of its 1.96*SE interval
    # was ~82% on 7-visit eyes). Provide a small-sample SE (ML scale inflated by
    # sqrt(n/(n-2))) and a t-based 95% interval on the uncensored count.
    n_dens = int(np.sum(is_density))
    se_b_adj, slope_ci95 = None, None
    if se_b is not None and n_dens > 2:
        se_b_adj = float(se_b * np.sqrt(n_dens / (n_dens - 2)))
        tq = float(stats.t.ppf(0.975, df=n_dens - 2))
        slope_ci95 = [float(b_hat - tq * se_b_adj), float(b_hat + tq * se_b_adj)]
 
    return {
        "model": model,
        "n_observations": int(n),
        "n_censored": int(np.sum(is_censored)),
        "n_ceiling_censored": int(np.sum(is_upper)),
        "intercept": float(a_hat),
        "slope": float(b_hat),
        "sigma": float(sigma_hat),
        "slope_se": se_b,
        "slope_se_small_sample": se_b_adj,
        "slope_ci95": slope_ci95,
        "neg_log_likelihood": float(neg_ll),
        "aic": float(aic),
        "aicc": float(aicc),
        "small_sample_correction_unavailable": small_sample_correction_unavailable,
        "converged": bool(res.success),
        "jacobian_corrected": True,
    }
 
 
def _numerical_hessian_se(x0: np.ndarray, x: np.ndarray, g_y: np.ndarray,
                           is_censored: np.ndarray, g_floor: float,
                           log_jacobian: Optional[np.ndarray] = None,
                           eps: float = 1e-4, is_upper: Optional[np.ndarray] = None,
                           g_ceiling: Optional[float] = None) -> Tuple[Optional[float], Optional[float]]:
    """
    Central finite-difference Hessian inverse for asymptotic parameter SEs.
    The Jacobian term is constant in the parameters, so it does not change
    the SEs; it is threaded through only so this calls the same likelihood.
    """
    n_params = len(x0)
    H = np.zeros((n_params, n_params))
 
    def f(p):
        return _neg_log_likelihood(p, x, g_y, is_censored, g_floor, log_jacobian, is_upper, g_ceiling)
 
    for i in range(n_params):
        for j in range(n_params):
            if j < i:
                H[i, j] = H[j, i]
                continue
            pi_p, pi_m = x0.copy(), x0.copy()
            if i == j:
                pi_p[i] += eps
                pi_m[i] -= eps
                H[i, j] = (f(pi_p) - 2 * f(x0) + f(pi_m)) / (eps ** 2)
            else:
                pp, pm, mp, mm = x0.copy(), x0.copy(), x0.copy(), x0.copy()
                pp[i] += eps; pp[j] += eps
                pm[i] += eps; pm[j] -= eps
                mp[i] -= eps; mp[j] += eps
                mm[i] -= eps; mm[j] -= eps
                H[i, j] = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * eps ** 2)
    try:
        cov = np.linalg.inv(H)
        diag = np.diag(cov)
        if np.any(diag < 0):
            return None, None
        se = np.sqrt(diag)
        return float(se[0]), float(se[1])
    except np.linalg.LinAlgError:
        return None, None
 
 
CANDIDATE_MODELS = ["Linear", "Square-Root", "Log-Exponential", "Power-Law"]
 
 
class KKodeApexEngine:
    """
    K-KODE ENGINE v55.0
    Generalized longitudinal biomarker decay + trial sample-size engine.
    """
    REQUIRED_BASE_COLUMNS = ['patient_id', 'visit_date']
    ENGINE_VERSION = "v55.0"
    BUILD_TAG = "jacobian-fix + audit-5"
 
    def __init__(self, data_source: Union[str, pd.DataFrame], endpoint_column: str,
                 eye_column: Optional[str] = None, measurement_floor: float = 0.05,
                 higher_is_better: bool = True, measurement_ceiling: Optional[float] = None):
        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Target file path not found: {data_source}")
            self.raw_df: pd.DataFrame = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.raw_df = data_source.copy()
        else:
            raise TypeError("Data source must be a file path string or a pandas DataFrame.")
 
        required = self.REQUIRED_BASE_COLUMNS + [endpoint_column]
        missing = [c for c in required if c not in self.raw_df.columns]
        if missing:
            raise ValueError(f"Input is missing required column(s): {missing}. Expected: {required}")
 
        self.endpoint_column = endpoint_column
        self.eye_column = eye_column
        self.measurement_floor = measurement_floor
        if measurement_ceiling is not None and not measurement_ceiling > measurement_floor:
            raise ValueError("measurement_ceiling must be greater than measurement_floor.")
        self.measurement_ceiling = measurement_ceiling
        self.higher_is_better = higher_is_better
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.data_quality_report: Dict[str, Any] = {}
        self.per_patient_fits: Dict[str, Dict[str, Any]] = {}
        self.model_selection_results: Dict[str, Any] = {}
        self.mixed_effects_results: Dict[str, Any] = {}
        self.sample_size_closed_form: Dict[str, Any] = {}
        self.sample_size_simulated: Dict[str, Any] = {}
        self.bayesian_censored_nlme_results: Dict[str, Any] = {}
        self.censored_population_results: Dict[str, Any] = {}
 
    def clean_and_transform(self) -> pd.DataFrame:
        n_start = len(self.raw_df)
        df = self.raw_df.copy()
        col = self.endpoint_column
 
        n_missing = df[['patient_id', 'visit_date', col]].isna().any(axis=1).sum()
        df = df.dropna(subset=['patient_id', 'visit_date', col]).copy()
 
        parsed_dates = pd.to_datetime(df['visit_date'], errors='coerce')
        n_bad_dates = parsed_dates.isna().sum()
        df = df.assign(visit_date=parsed_dates).dropna(subset=['visit_date'])
 
        numeric_val = pd.to_numeric(df[col], errors='coerce')
        n_non_numeric = numeric_val.isna().sum()
        df = df.assign(**{col: numeric_val}).dropna(subset=[col])
 
        n_at_or_below_floor = int((df[col] <= self.measurement_floor).sum())
        n_at_or_above_ceiling = (int((df[col] >= self.measurement_ceiling).sum())
                                 if self.measurement_ceiling is not None else 0)
 
        if self.eye_column and self.eye_column in df.columns:
            df = df.assign(group_id=df['patient_id'].astype(str) + "__" + df[self.eye_column].astype(str))
        else:
            df = df.assign(group_id=df['patient_id'].astype(str))
 
        # Audit 5: byte-identical repeats of the same patient/eye/date/value are
        # data-entry artifacts and would double-count information; drop them
        # (and say so). Same date with DIFFERENT values is kept and warned.
        exact_dup = df.duplicated(subset=['group_id', 'visit_date', col], keep='first')
        n_exact_dup = int(exact_dup.sum())
        df = df[~exact_dup].copy()
        dup_mask = df.duplicated(subset=['group_id', 'visit_date'], keep=False)
        n_dup = int(dup_mask.sum())
 
        # Audit 3: time zero is the PATIENT's first visit, shared by both
        # eyes. (It was each eye's own first visit, so an eye missing its
        # baseline visit got a different time origin than its fellow eye.)
        patient_baseline = df.groupby('patient_id')['visit_date'].transform('min')
        df = df.assign(_baseline=patient_baseline)
 
        processed = []
        n_single = 0
        for gid, group in df.groupby('group_id'):
            group = group.sort_values('visit_date').copy()
            if len(group) < 2:
                n_single += 1
            baseline = group['_baseline'].iloc[0]
            group['years_from_baseline'] = (group['visit_date'] - baseline).dt.days / 365.25
            processed.append(group)
 
        if processed:
            self.clean_df = pd.concat(processed, ignore_index=True)
        else:
            self.clean_df = df.assign(years_from_baseline=pd.Series(dtype=float))
 
        censoring_fraction = (n_at_or_below_floor / n_start) if n_start else 0.0
        total_censoring_fraction = ((n_at_or_below_floor + n_at_or_above_ceiling) / n_start) if n_start else 0.0
        self.data_quality_report = {
            "engine_version": self.ENGINE_VERSION,
            "build_tag": self.BUILD_TAG,
            "endpoint_column": col,
            "rows_in_raw_input": n_start,
            "rows_dropped_missing_required_fields": int(n_missing),
            "rows_dropped_unparseable_date": int(n_bad_dates),
            "rows_dropped_non_numeric_endpoint": int(n_non_numeric),
            "rows_at_or_below_measurement_floor": n_at_or_below_floor,
            "floor_censoring_fraction_of_raw_input": round(censoring_fraction, 4),
            "rows_at_or_above_measurement_ceiling": n_at_or_above_ceiling,
            "total_censoring_fraction_of_raw_input": round(total_censoring_fraction, 4),
            "note_censored_rows_are_kept": (
                "Floor-censored rows are KEPT in clean_df and handled via a censored "
                "(Tobit-style) likelihood in per-patient model fitting, not dropped."
            ),
            "patients_with_only_one_visit": int(n_single),
            "rows_retained_for_modeling": int(len(self.clean_df)),
            "duplicate_same_date_rows": n_dup,
            "exact_duplicate_rows_dropped": n_exact_dup,
        }
        if not self.clean_df.empty:
            self.data_quality_report["n_patients"] = int(self.clean_df["patient_id"].nunique())
            self.data_quality_report["n_modeling_units"] = int(self.clean_df["group_id"].nunique())
        if n_dup and not self.eye_column:
            self.data_quality_report["duplicate_same_date_warning"] = (
                "Same patient_id + visit_date rows exist. If this tracks both eyes, pass "
                "eye_column= so each eye is modeled separately."
            )
        elif n_dup:
            self.data_quality_report["duplicate_same_date_warning"] = (
                f"{n_dup} rows share a patient/eye/date with a DIFFERENT value (repeat "
                "measurements). They are all kept and treated as independent readings; "
                "average them per visit first if they are replicates."
            )
        if total_censoring_fraction > HEAVY_CENSORING_WARNING_FRACTION:
            self.data_quality_report["heavy_censoring_warning"] = (
                f"{total_censoring_fraction:.1%} of raw rows were at/beyond a measurement bound. "
                "Per-patient fits correct for this via censored MLE. Automated PyMC Bayesian "
                "Censored NLME is recommended to eliminate population-level floor bias."
            )
        for k, v in self.data_quality_report.items():
            logger.info(f"{k}: {v}")
        return self.clean_df
 
    def unique_group_ids(self) -> List[str]:
        if self.clean_df.empty:
            self.clean_and_transform()
        if self.clean_df.empty or 'group_id' not in self.clean_df.columns:
            return []
        return sorted(self.clean_df['group_id'].unique().tolist())
 
    def run_model_competition(self) -> Dict[str, Any]:
        """Per-patient censored-MLE AICc competition across candidate forms
        (Jacobian-corrected likelihood; see changelog)."""
        if self.clean_df.empty:
            self.clean_and_transform()
        weight_sums = {m: 0.0 for m in CANDIDATE_MODELS}
        win_counts = {m: 0 for m in CANDIDATE_MODELS}
        n_evaluated = 0
        evaluated_patients = set()
        n_used_uncorrected_aic = 0
        n_fewer_than_four_models = 0
        skipped = []
        for gid, group in self.clean_df.groupby("group_id"):
            group = group.sort_values("years_from_baseline")
            t = group["years_from_baseline"].values
            y = group[self.endpoint_column].values
            if len(t) < MIN_POINTS_FOR_MODEL_COMPETITION:
                skipped.append(gid)
                continue
            fits = {}
            for m in CANDIDATE_MODELS:
                try:
                    f = fit_censored_model(t, y, self.measurement_floor, m, self.measurement_ceiling)
                    if f is not None and np.isfinite(f["aicc"]):
                        fits[m] = f
                except Exception:
                    continue
            if not fits:
                skipped.append(gid)
                continue
            if len(fits) < len(CANDIDATE_MODELS):
                n_fewer_than_four_models += 1
            if any(f.get("small_sample_correction_unavailable") for f in fits.values()):
                n_used_uncorrected_aic += 1
            aiccs = {m: f["aicc"] for m, f in fits.items()}
            min_aicc = min(aiccs.values())
            deltas = {m: v - min_aicc for m, v in aiccs.items()}
            raw_w = {m: np.exp(-0.5 * d) for m, d in deltas.items()}
            wsum = sum(raw_w.values())
            weights = {m: w / wsum for m, w in raw_w.items()}
            winner = min(aiccs, key=aiccs.get)
            win_counts[winner] += 1
            for m, w in weights.items():
                weight_sums[m] += w
            n_evaluated += 1
            evaluated_patients.add(str(group["patient_id"].iloc[0]))
 
        if n_evaluated == 0:
            self.model_selection_results = {
                "error": f"No patients had >= {MIN_POINTS_FOR_MODEL_COMPETITION} points to run model competition.",
                "patients_skipped": len(skipped),
            }
            return self.model_selection_results
 
        mean_weights = {m: w / n_evaluated for m, w in weight_sums.items()}
        win_fraction = {m: c / n_evaluated for m, c in win_counts.items()}
        overall_winner = max(mean_weights, key=mean_weights.get)
        self.model_selection_results = {
            "method": (
                "Per-patient Jacobian-corrected censored-MLE AICc competition across Linear, "
                "Square-Root, Log-Exponential, and Power-Law forms, aggregated by mean Akaike "
                "weight and win-fraction across patients."
            ),
            "patients_evaluated": n_evaluated,  # modeling units (eyes if eye_column set)
            "patients_represented": len(evaluated_patients),
            "patients_skipped_insufficient_data": len(skipped),
            "patients_with_fewer_than_four_models_fit": n_fewer_than_four_models,
            "patients_using_uncorrected_aic": n_used_uncorrected_aic,
            "mean_akaike_weight_by_model": {k: float(v) for k, v in mean_weights.items()},
            "win_fraction_by_model": {k: float(v) for k, v in win_fraction.items()},
            "overall_best_supported_model": overall_winner,
            "jacobian_correction_applied": True,
        }
        if n_used_uncorrected_aic > 0:
            self.model_selection_results["uncorrected_aic_note"] = (
                f"{n_used_uncorrected_aic} of {n_evaluated} patients had exactly 4 observations, "
                "where AICc's denominator is 0 - plain AIC was used as fallback."
            )
        return self.model_selection_results
 
    def run_per_patient_decay(self, model: str = "Log-Exponential") -> Dict[str, Dict[str, Any]]:
        """Fits candidate model to every patient/eye via Tobit MLE.
 
        decay_rate is -slope for every model form (see changelog item 2)."""
        if self.clean_df.empty:
            self.clean_and_transform()
        results = {}
        for gid in self.unique_group_ids():
            group = self.clean_df[self.clean_df["group_id"] == gid].sort_values("years_from_baseline")
            t = group["years_from_baseline"].values
            y = group[self.endpoint_column].values
            pid = str(group["patient_id"].iloc[0])
            try:
                fit = fit_censored_model(t, y, self.measurement_floor, model, self.measurement_ceiling)
                if fit is None:
                    results[gid] = {"status": f"Skipped {gid}: insufficient/degenerate data for {model} fit.",
                                    "patient_id": pid}
                    continue
                fit["decay_rate"] = -fit["slope"]
                fit["patient_id"] = pid
                results[gid] = fit
            except Exception as e:
                results[gid] = {"status": f"Skipped {gid}: fit error: {str(e)}", "patient_id": pid}
        self.per_patient_fits = results
        return results
 
    def _valid_decay_rates(self, unit: str = "eye") -> np.ndarray:
        """Per-unit decay rates. unit='eye' returns one rate per fitted
        group (eye, or patient when no eye column). unit='patient' averages
        a patient's eyes into one rate per patient."""
        valid = [
            v for v in self.per_patient_fits.values()
            if isinstance(v, dict) and v.get("decay_rate") is not None and np.isfinite(v.get("decay_rate"))
        ]
        if unit == "patient":
            by_patient: Dict[str, List[float]] = {}
            for v in valid:
                by_patient.setdefault(v["patient_id"], []).append(v["decay_rate"])
            return np.array([float(np.mean(r)) for r in by_patient.values()])
        return np.array([v["decay_rate"] for v in valid])
 
    def fit_mixed_effects_nlme(self, model: str = "Log-Exponential",
                                nested_eyes: Optional[bool] = None) -> Dict[str, Any]:
        """Population-level mixed-effects fit (statsmodels), uncensored rows only
        (floor- and ceiling-censored rows are excluded; use
        fit_censored_population_model() to keep them).
 
        Uses _prepare_predictor(), the same predictor construction as the
        per-patient fit (Audit 1). If an eye column exists and nested_eyes is
        True (default), groups by PATIENT with an eye-level variance
        component; on failure falls back to grouping by eye and reports it."""
        if self.clean_df.empty:
            self.clean_and_transform()
        if nested_eyes is None:
            nested_eyes = bool(self.eye_column)
        _in_range = self.clean_df[self.endpoint_column] > self.measurement_floor
        if self.measurement_ceiling is not None:
            _in_range &= self.clean_df[self.endpoint_column] < self.measurement_ceiling
        uncensored = self.clean_df[_in_range].copy()
        if uncensored.empty or uncensored["group_id"].nunique() < MIN_COHORT_FOR_SAMPLE_SIZE:
            self.mixed_effects_results = {"error": "Not enough uncensored data to fit population mixed-effects model."}
            return self.mixed_effects_results
        uncensored["_transformed_endpoint"] = _transform_y(model, uncensored[self.endpoint_column].values)
        uncensored["_predictor"] = _prepare_predictor(model, uncensored["years_from_baseline"].values)
        uncensored["_patient"] = uncensored["patient_id"].astype(str)
 
        def _summarize(mfit, structure: str, n_units: int, note: Optional[str] = None) -> Dict[str, Any]:
            cov_re = mfit.cov_re
            slope_var = float(cov_re.iloc[1, 1]) if cov_re.shape[0] > 1 else None
            intercept_var = float(cov_re.iloc[0, 0])
            out = {
                "model_form": model,
                "total_patients_modeled": int(uncensored["_patient"].nunique()),
                "total_modeling_units": int(n_units),
                "total_observations": int(len(uncensored)),
                "population_mean_decay_rate": float(-mfit.params["_predictor"]),
                "population_intercept": float(mfit.params["Intercept"]),
                "between_patient_slope_variance": slope_var,
                "between_patient_intercept_variance": intercept_var,
                "residual_error_variance_sigma2": float(mfit.scale),
                "fixed_effects_p_value": float(mfit.pvalues["_predictor"]),
                "model_converged": bool(mfit.converged),
                "predictor_used": ("log(t + epsilon)" if model == "Power-Law" else "years_from_baseline"),
                "random_effects_structure": structure,
            }
            if note:
                out["note"] = note
            if not mfit.converged:
                out["convergence_warning"] = "Optimizer did not fully converge; treat as approximate."
            return out
 
        results = None
        nested_failure = None
        if nested_eyes and self.eye_column:
            try:
                m = smf.mixedlm(
                    "_transformed_endpoint ~ _predictor", uncensored,
                    groups=uncensored["_patient"], re_formula="~_predictor",
                    vc_formula={"eye": "0 + C(group_id)"},
                )
                mfit = m.fit(disp=False)
                results = _summarize(
                    mfit, "patient random intercept+slope with eye-level variance component",
                    uncensored["group_id"].nunique(),
                )
            except Exception as e:
                nested_failure = str(e)
        if results is None:
            try:
                m = smf.mixedlm(
                    "_transformed_endpoint ~ _predictor", uncensored,
                    groups=uncensored["group_id"], re_formula="~_predictor",
                )
                mfit = m.fit(disp=False)
                note = None
                if nested_failure is not None:
                    note = ("Nested patient/eye model failed (" + nested_failure +
                            "); fell back to grouping by eye, which treats eyes as independent.")
                elif self.eye_column:
                    note = "nested_eyes=False: eyes are treated as independent units."
                results = _summarize(mfit, "random intercept+slope per modeling unit (eye or patient)",
                                     uncensored["group_id"].nunique(), note)
            except Exception as e:
                results = {"error": f"Mixed-effects model failed to fit: {str(e)}"}
        self.mixed_effects_results = results
        return self.mixed_effects_results
 
    def compute_closed_form_sample_size(self, target_power: float = 0.80, alpha: float = 0.05,
                                         therapeutic_efficacy: float = 0.30,
                                         n_bootstrap: int = 2000, random_seed: int = 42,
                                         unit: str = "auto") -> Dict[str, Any]:
        """Closed-form normal-approximation sample size.
 
        unit='auto' (default): per PATIENT (eyes averaged) when an eye column
        was supplied, otherwise per modeling unit. unit='eye' uses every eye
        as an independent unit (overstates information when eyes are
        correlated). Depends only on (SD / mean) of the decay rates, so it is
        unit-free for a GIVEN model form, but the ratio differs between
        model forms (rates live on different transformed scales)."""
        if unit not in ("auto", "patient", "eye"):
            self.sample_size_closed_form = {"error": "unit must be 'auto', 'patient' or 'eye'."}
            return self.sample_size_closed_form
        use_patient = (unit == "patient") or (unit == "auto" and bool(self.eye_column))
        rates = self._valid_decay_rates("patient" if use_patient else "eye")
        if len(rates) < MIN_COHORT_FOR_SAMPLE_SIZE:
            self.sample_size_closed_form = {"error": f"Need >= {MIN_COHORT_FOR_SAMPLE_SIZE} valid decay rates."}
            return self.sample_size_closed_form
        mean_l, std_l = float(np.mean(rates)), float(np.std(rates, ddof=1))
        if std_l == 0:
            self.sample_size_closed_form = {"error": "Zero variance across patients."}
            return self.sample_size_closed_form
        if mean_l <= 0:
            self.sample_size_closed_form = {
                "error": ("Mean decay rate is not positive (no measurable average decline), "
                          "so a sample size for slowing decline is not defined."),
                "mean_decay_rate": mean_l,
            }
            return self.sample_size_closed_form
        endpoint_scale = float(np.nanmax(np.abs(self.clean_df[self.endpoint_column].values))) if not self.clean_df.empty else 0.0
        if mean_l < 1e-6 * max(endpoint_scale, 1e-12):
            self.sample_size_closed_form = {
                "error": ("Mean decay rate is numerically indistinguishable from zero relative to the "
                          "endpoint's scale (flat data), so a sample size is not defined."),
                "mean_decay_rate": mean_l,
            }
            return self.sample_size_closed_form
        z_a = stats.norm.ppf(1 - alpha / 2)
        z_b = stats.norm.ppf(target_power)
 
        def req_n(m, s, eff):
            d = m * eff
            return np.inf if d == 0 else (2 * (z_a + z_b) ** 2 * (s ** 2)) / (d ** 2)
 
        n_req = req_n(mean_l, std_l, therapeutic_efficacy)
        rng = np.random.default_rng(random_seed)
        boots = []
        for _ in range(n_bootstrap):
            s = rng.choice(rates, size=len(rates), replace=True)
            sm_, ss_ = np.mean(s), np.std(s, ddof=1)
            if ss_ > 0 and sm_ > 0:
                boots.append(req_n(sm_, ss_, therapeutic_efficacy))
        boots = np.array([b for b in boots if np.isfinite(b)])
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(boots) else (None, None)
        self.sample_size_closed_form = {
            "method": "closed-form z-based normal approximation",
            "analysis_unit": "patient (eyes averaged)" if use_patient else "eye/modeling unit",
            "cohort_size_used": len(rates),
            "mean_decay_rate": mean_l,
            "between_unit_sd": std_l,
            "required_n_per_arm": int(np.ceil(n_req)) if np.isfinite(n_req) else None,
            "bootstrap_95pct_ci_per_arm": [int(np.ceil(ci[0])), int(np.ceil(ci[1]))] if ci[0] is not None else None,
        }
        if len(rates) < SMALL_COHORT_WARNING_THRESHOLD:
            self.sample_size_closed_form["provisional_estimate_warning"] = f"Based on only {len(rates)} units."
        t_stat = mean_l / (std_l / np.sqrt(len(rates)))
        p_decline = float(2 * stats.t.sf(abs(t_stat), df=len(rates) - 1))
        self.sample_size_closed_form["p_value_mean_decay_differs_from_zero"] = p_decline
        if p_decline > alpha:
            self.sample_size_closed_form["decline_not_established_warning"] = (
                "The cohort's mean decay rate is not statistically distinguishable from zero, "
                "so this sample size rests on an unreliable estimate of the effect to slow.")
        if self.sample_size_closed_form["required_n_per_arm"] is not None and self.sample_size_closed_form["required_n_per_arm"] < 10:
            self.sample_size_closed_form["very_small_n_warning"] = (
                "Required n per arm is very small; this usually means the fitted slopes are almost "
                "identical across units (estimation noise or real heterogeneity is being "
                "understated). Check the data before relying on it.")
        return self.sample_size_closed_form
 
    def _simulate_power(self, n_per_arm: int, pop_intercept: float, pop_slope: float,
                         tau_intercept: float, tau_slope: float, resid_sd: float,
                         visit_predictor: List[float], efficacy_reduction: float,
                         alpha: float, n_sims: int, seed: int) -> float:
        """Monte Carlo trial simulation using vectorized two-stage summary
        measures. visit_predictor is the model's PREDICTOR at each visit
        (raw years, or log(t + epsilon) for Power-Law)."""
        rng = np.random.default_rng(seed)
        treat_slope = pop_slope * (1 - efficacy_reduction)
        visit_arr = np.array(visit_predictor, dtype=float)
        successes, valid_sims = 0, 0
        for sim in range(n_sims):
            def arm_slopes(arm_pop_slope):
                b0 = rng.normal(pop_intercept, max(tau_intercept, 1e-6), size=n_per_arm)
                b1 = rng.normal(arm_pop_slope, max(tau_slope, 1e-6), size=n_per_arm)
                eps = rng.normal(0, max(resid_sd, 1e-6), size=(n_per_arm, len(visit_arr)))
                y = b0[:, None] + b1[:, None] * visit_arr[None, :] + eps
                t_mean = visit_arr.mean()
                t_centered = visit_arr - t_mean
                denom = np.sum(t_centered ** 2)
                y_centered = y - y.mean(axis=1, keepdims=True)
                return (y_centered @ t_centered) / denom
            control_slopes = arm_slopes(pop_slope)
            treat_slopes = arm_slopes(treat_slope)
            try:
                _, pval = stats.ttest_ind(control_slopes, treat_slopes, equal_var=False)
                if np.isfinite(pval):
                    valid_sims += 1
                    if pval < alpha:
                        successes += 1
            except Exception:
                continue
        return (successes / valid_sims) if valid_sims > 0 else 0.0
 
    def _get_population_parameters_for_simulation(self, source: str = "auto",
                                                    correlated_random_effects: bool = True) -> Dict[str, Any]:
        valid_sources = {"auto", "standard_nlme", "censored_mle", "bayesian"}
        if source not in valid_sources:
            return {"error": f"source must be one of {sorted(valid_sources)}, got '{source}'."}
 
        def _from_standard_nlme() -> Optional[Dict[str, Any]]:
            nlme = self.mixed_effects_results
            if not nlme or "error" in nlme or not nlme.get("model_converged"):
                return None
            tau_slope = float(np.sqrt(max(nlme.get("between_patient_slope_variance") or 0.0, 1e-8)))
            tau_int = float(np.sqrt(max(nlme.get("between_patient_intercept_variance") or 0.0, 1e-8)))
            return {
                "pop_intercept": nlme["population_intercept"],
                "pop_slope": -nlme["population_mean_decay_rate"],
                "tau_intercept": tau_int,
                "tau_slope": tau_slope,
                "resid_sd": float(np.sqrt(max(nlme["residual_error_variance_sigma2"], 1e-8))),
                "model_form": nlme["model_form"],
                "source_used": "standard_nlme",
            }
 
        def _from_censored_mle() -> Optional[Dict[str, Any]]:
            cm = self.censored_population_results
            if not cm or "error" in cm:
                primary = self.model_selection_results.get("overall_best_supported_model", "Log-Exponential")
                cm = self.fit_censored_population_model(model=primary)
            if not cm or "error" in cm or not cm.get("converged"):
                return None
            return {
                "pop_intercept": cm["population_intercept"],
                "pop_slope": -cm["population_mean_decay_rate"],
                "tau_intercept": cm["between_patient_intercept_sd"],
                "tau_slope": cm["between_patient_slope_sd"],
                "resid_sd": cm["residual_sd"],
                "model_form": cm["model_form"],
                "source_used": "censored_mle",
            }
 
        def _from_bayesian(run_if_missing: bool) -> Optional[Dict[str, Any]]:
            bnlme = self.bayesian_censored_nlme_results
            if (not bnlme or "error" in bnlme) and run_if_missing:
                primary_model = self.model_selection_results.get("overall_best_supported_model", "Log-Exponential")
                bnlme = self.fit_bayesian_censored_nlme(
                    model=primary_model, correlated_random_effects=correlated_random_effects,
                )
                if bnlme and "error" not in bnlme and not bnlme.get("convergence_ok"):
                    bnlme = self.fit_bayesian_censored_nlme(
                        model=primary_model, draws=1500, tune=1500, chains=4,
                        target_accept=0.95, correlated_random_effects=correlated_random_effects,
                        random_seed=99,
                    )
            if not bnlme or "error" in bnlme or not bnlme.get("convergence_ok"):
                return None
            return {
                "pop_intercept": bnlme["population_intercept"],
                "pop_slope": -bnlme["population_mean_decay_rate"],
                "tau_intercept": bnlme["between_patient_intercept_sd"],
                "tau_slope": bnlme["between_patient_slope_sd"],
                "resid_sd": bnlme["residual_sd"],
                "model_form": bnlme["model_form"],
                "source_used": "bayesian_censored_nlme",
            }
 
        if not self.mixed_effects_results:
            self.fit_mixed_effects_nlme()
 
        if source == "standard_nlme":
            params = _from_standard_nlme()
            if params is None:
                return {"error": "Standard (statsmodels) mixed-effects model unavailable/didn't converge, and source='standard_nlme' was forced (no fallback)."}
            return params
 
        if source == "censored_mle":
            params = _from_censored_mle()
            if params is None:
                return {"error": "Censored population model unavailable/didn't converge, and source='censored_mle' was forced (no fallback)."}
            return params
 
        if source == "bayesian":
            params = _from_bayesian(run_if_missing=True)
            if params is None:
                return {"error": "Bayesian censored NLME unavailable/didn't converge, and source='bayesian' was forced (no fallback)."}
            return params
 
        # auto: when any rows are censored the statsmodels fit (which drops
        # them) is biased, so the censored fit goes first.
        dq = self.data_quality_report
        any_censoring = (dq.get("rows_at_or_below_measurement_floor", 0)
                         + dq.get("rows_at_or_above_measurement_ceiling", 0)) > 0
        order = [_from_censored_mle, _from_standard_nlme] if any_censoring else [_from_standard_nlme, _from_censored_mle]
        for fn in order:
            params = fn()
            if params is not None:
                return params
        params = _from_bayesian(run_if_missing=True)
        if params is not None:
            return params
        return {
            "error": (
                "Neither the standard mixed-effects model nor the Bayesian censored NLME "
                "converged on this cohort -- population parameters for simulation are "
                "unavailable. Try increasing draws/tune on fit_bayesian_censored_nlme, or "
                "rely on the closed-form sample size estimate instead."
            )
        }
 
    def compute_simulated_sample_size(self, target_power: float = 0.80, alpha: float = 0.05,
                                       therapeutic_efficacy: float = 0.30,
                                       visit_times: Optional[List[float]] = None,
                                       n_sims_per_candidate: int = 2000,
                                       max_search_iterations: int = 10,
                                       random_seed: int = 7,
                                       source: str = "auto",
                                       correlated_random_effects: bool = True) -> Dict[str, Any]:
        params = self._get_population_parameters_for_simulation(source, correlated_random_effects)
        if "error" in params:
            self.sample_size_simulated = params
            return self.sample_size_simulated
        pop_intercept = params["pop_intercept"]
        pop_slope = params["pop_slope"]
        if pop_slope >= 0:
            self.sample_size_simulated = {
                "error": ("The fitted population slope is not declining (decay rate <= 0), so a "
                          "trial of slowing decline cannot be simulated."),
                "population_parameter_source": params.get("source_used"),
            }
            return self.sample_size_simulated
        tau_slope = params["tau_slope"]
        tau_intercept = params["tau_intercept"]
        resid_sd = params["resid_sd"]
        source_used = params["source_used"]
        model_form = params.get("model_form", "Log-Exponential")
        if visit_times is None:
            visit_times = [0.0, 1.0, 2.0, 3.0]
        visit_predictor = _prepare_predictor(model_form, np.array(visit_times, dtype=float)).tolist()
        seed_n = self.sample_size_closed_form.get("required_n_per_arm") if self.sample_size_closed_form else None
        if not seed_n:
            self.compute_closed_form_sample_size(target_power, alpha, therapeutic_efficacy)
            seed_n = self.sample_size_closed_form.get("required_n_per_arm") or 30
        n_lo, n_hi = None, None
        n_current = max(int(seed_n), 5)
        history = []
        it = 0
        power_at_current = self._simulate_power(
            n_current, pop_intercept, pop_slope, tau_intercept, tau_slope, resid_sd,
            visit_predictor, therapeutic_efficacy, alpha, n_sims_per_candidate, random_seed + it,
        )
        history.append({"n_per_arm": n_current, "empirical_power": power_at_current})
        while it < max_search_iterations:
            it += 1
            if power_at_current < target_power:
                n_lo = n_current
                n_current = int(n_current * 1.6) + 1
            else:
                n_hi = n_current
                break
            power_at_current = self._simulate_power(
                n_current, pop_intercept, pop_slope, tau_intercept, tau_slope, resid_sd,
                visit_predictor, therapeutic_efficacy, alpha, n_sims_per_candidate, random_seed + it,
            )
            history.append({"n_per_arm": n_current, "empirical_power": power_at_current})
        if n_hi is None:
            self.sample_size_simulated = {"error": "Could not bracket target power within max iterations."}
            return self.sample_size_simulated
        if n_lo is None:
            n_lo = max(3, n_hi // 2)
        while (n_hi - n_lo) > 1 and it < max_search_iterations * 2:
            it += 1
            n_mid = (n_lo + n_hi) // 2
            p_mid = self._simulate_power(
                n_mid, pop_intercept, pop_slope, tau_intercept, tau_slope, resid_sd,
                visit_predictor, therapeutic_efficacy, alpha, n_sims_per_candidate, random_seed + it,
            )
            history.append({"n_per_arm": n_mid, "empirical_power": p_mid})
            if p_mid >= target_power:
                n_hi = n_mid
            else:
                n_lo = n_mid
        self.sample_size_simulated = {
            "method": "Monte Carlo trial simulation under fitted population parameters",
            "population_parameter_source": source_used,
            "model_form": model_form,
            "required_n_per_arm": n_hi,
            "target_power": target_power,
            "visit_schedule_years": visit_times,
            "n_sims_per_candidate": n_sims_per_candidate,
            "search_history": history,
            "population_parameters_used": {
                "pop_intercept": pop_intercept,
                "pop_slope_control": pop_slope,
                "between_patient_slope_sd": tau_slope,
                "residual_sd": resid_sd,
            },
        }
        if model_form == "Power-Law":
            self.sample_size_simulated["power_law_note"] = (
                "Simulated with predictor log(t + epsilon); the t=0 visit is a very "
                "high-leverage point, so interpret this estimate with extra caution."
            )
        return self.sample_size_simulated
 
    def fit_censored_population_model(self, model: str = "Log-Exponential", n_quad: int = 9) -> Dict[str, Any]:
        """Frequentist censored (Tobit) mixed model on the chosen model scale.
 
        Keeps floor- AND ceiling-censored rows (the statsmodels fit drops
        them). Patient-level random intercept and slope, correlated, are
        integrated out with an ADAPTIVE (per-patient re-centered and re-scaled)
        n_quad x n_quad Gauss-Hermite grid, vectorized over patients. Both eyes of a patient share that
        patient's random effects (so eyes are NOT treated as independent);
        eye-specific departures are absorbed into the residual, which makes
        this somewhat conservative about patient-level information.
        With model="Linear" this is the native-scale Tobit cross-check.
 
        The Jacobian is added for uncensored rows so the returned AIC is
        comparable across model forms. Fast alternative / cross-check for
        the slow PyMC model; it gives a point estimate with Wald intervals,
        not a posterior."""
        if self.clean_df.empty:
            self.clean_and_transform()
        df = self.clean_df
        col = self.endpoint_column
        if df.empty or df["patient_id"].nunique() < MIN_COHORT_FOR_SAMPLE_SIZE:
            self.censored_population_results = {"error": f"Need >= {MIN_COHORT_FOR_SAMPLE_SIZE} patients."}
            return self.censored_population_results
 
        y = df[col].values.astype(float)
        floor_v, ceil_v = self.measurement_floor, self.measurement_ceiling
        is_low = y <= floor_v
        is_up = (y >= ceil_v) if ceil_v is not None else np.zeros(len(y), dtype=bool)
        is_dens = ~(is_low | is_up)
        if model in POSITIVE_ONLY_MODELS and np.any(y[is_dens] <= 0):
            self.censored_population_results = {"error": f"{model}: uncensored non-positive readings."}
            return self.censored_population_results
        if model in ("Log-Exponential", "Power-Law") and floor_v <= 0 and np.any(is_low):
            self.censored_population_results = {
                "error": f"{model} is inadmissible with measurement_floor <= 0 and floor-censored rows."}
            return self.censored_population_results
        if is_dens.sum() < 6:
            self.censored_population_results = {"error": "Too few uncensored observations."}
            return self.censored_population_results
 
        g_y = _transform_y(model, np.maximum(y, max(floor_v, TRANSFORM_Y_EPSILON)))
        x_all = _prepare_predictor(model, df["years_from_baseline"].values)
        g_floor = _floor_in_transformed_space(model, floor_v)
        g_ceil = _ceiling_in_transformed_space(model, ceil_v) if ceil_v is not None else 0.0
        jac_sum = float(np.sum(_log_jacobian_dgdy(model, y[is_dens])))
 
        # pad observations into [patients, max_obs]
        codes, uniques = pd.factorize(df["patient_id"].astype(str))
        P = len(uniques)
        counts = np.bincount(codes, minlength=P)
        T = int(counts.max())
        pos = np.zeros(len(y), dtype=int)
        seen = np.zeros(P, dtype=int)
        for i, c in enumerate(codes):
            pos[i] = seen[c]
            seen[c] += 1
        def pad(v, fill=0.0, dtype=float):
            m = np.full((P, T), fill, dtype=dtype)
            m[codes, pos] = v
            return m
        X, G = pad(x_all), pad(g_y)
        M_dens, M_low, M_up = pad(is_dens, False, bool), pad(is_low, False, bool), pad(is_up, False, bool)
 
        nodes, w = np.polynomial.hermite.hermgauss(int(n_quad))
        zq = nodes * np.sqrt(2.0)
        logw = np.log(w / np.sqrt(np.pi))                      # probability weights (sum to 1)
        zg0 = np.repeat(zq, len(zq)); zg1 = np.tile(zq, len(zq))
        lwg = np.repeat(logw, len(zq)) + np.tile(logw, len(zq))   # [K]
        half_z2 = 0.5 * (zg0 ** 2 + zg1 ** 2)
        LOG2PI = np.log(2.0 * np.pi)
        warm = {"m": np.zeros((P, 2))}
 
        def make_loglik(theta):
            a_, b_, ls_, lta_, ltb_, ar_ = theta
            sig, ta, tb, rho = np.exp(ls_), np.exp(lta_), np.exp(ltb_), np.tanh(ar_)
            srho = np.sqrt(max(1.0 - rho ** 2, 1e-12))
 
            def loglik(v0, v1):
                """Sum over a patient's observations of log-lik, for standardized
                random effects v (shape [P, K']) -> [P, K']."""
                u0 = ta * v0
                u1 = tb * (rho * v0 + srho * v1)
                mu = (a_ + u0)[:, :, None] + (b_ + u1)[:, :, None] * X[:, None, :]
                ll = np.where(M_dens[:, None, :], stats.norm.logpdf(G[:, None, :], mu, sig), 0.0)
                ll = ll + np.where(M_low[:, None, :], stats.norm.logcdf((g_floor - mu) / sig), 0.0)
                if ceil_v is not None:
                    ll = ll + np.where(M_up[:, None, :], stats.norm.logsf((g_ceil - mu) / sig), 0.0)
                return ll.sum(axis=2)
            return loglik
 
        stencil = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], float)
 
        def derivs(loglik, m, h=1e-3):
            pts = m[:, None, :] + h * stencil[None, :, :]                       # [P,9,2]
            f = loglik(pts[:, :, 0], pts[:, :, 1]) - 0.5 * (pts ** 2).sum(axis=2)
            g = np.stack([(f[:, 1] - f[:, 2]) / (2 * h), (f[:, 3] - f[:, 4]) / (2 * h)], axis=1)
            Hm = np.empty((P, 2, 2))
            Hm[:, 0, 0] = (f[:, 1] - 2 * f[:, 0] + f[:, 2]) / h ** 2
            Hm[:, 1, 1] = (f[:, 3] - 2 * f[:, 0] + f[:, 4]) / h ** 2
            Hm[:, 0, 1] = Hm[:, 1, 0] = (f[:, 5] - f[:, 6] - f[:, 7] + f[:, 8]) / (4 * h ** 2)
            return f[:, 0], g, Hm
 
        def find_modes(loglik, m):
            for _ in range(12):
                f0, g, Hm = derivs(loglik, m)
                negH = -Hm
                # make the Newton matrix positive definite (ridge) so steps ascend
                ridge = np.maximum(0.0, 1e-3 - np.linalg.eigvalsh(negH)[:, 0])
                negH = negH + ridge[:, None, None] * np.eye(2)[None]
                step = np.linalg.solve(negH, g[:, :, None])[:, :, 0]
                scale = np.ones(P)
                for _ls in range(6):
                    cand = m + scale[:, None] * step
                    fc = loglik(cand[:, None, 0], cand[:, None, 1])[:, 0] - 0.5 * (cand ** 2).sum(axis=1)
                    bad = ~(fc >= f0 - 1e-12)
                    if not bad.any():
                        break
                    scale = np.where(bad, scale * 0.5, scale)
                cand = m + scale[:, None] * step
                m = np.where(np.isfinite(cand).all(axis=1)[:, None], cand, m)
                if np.max(np.abs(scale[:, None] * step)) < 1e-6:
                    break
            return m
 
        def nll(theta):
            loglik = make_loglik(theta)
            try:
                m = find_modes(loglik, warm["m"].copy())
                f0, g, Hm = derivs(loglik, m)
                negH = -Hm + 1e-8 * np.eye(2)[None]
                Lc = np.linalg.cholesky(np.linalg.inv(negH))                     # cov = L L^T
                warm["m"] = m
            except Exception:
                return 1e12
            logdetL = np.log(Lc[:, 0, 0]) + np.log(Lc[:, 1, 1])
            v0 = m[:, None, 0] + Lc[:, 0, 0][:, None] * zg0[None, :]
            v1 = m[:, None, 1] + Lc[:, 1, 0][:, None] * zg0[None, :] + Lc[:, 1, 1][:, None] * zg1[None, :]
            f = loglik(v0, v1) - 0.5 * (v0 ** 2 + v1 ** 2) - LOG2PI
            terms = lwg[None, :] + half_z2[None, :] + f
            mx = terms.max(axis=1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(terms - mx).sum(axis=1))
            lp = logdetL + LOG2PI + lse            # d/2 * log(2 pi) with d=2
            val = -float(np.sum(lp))
            return val if np.isfinite(val) else 1e12
 
        sl, ic, *_ = stats.linregress(x_all[is_dens], g_y[is_dens])
        res0 = g_y[is_dens] - (ic + sl * x_all[is_dens])
        s0 = max(float(np.std(res0)), 1e-3)
        x0 = np.array([ic, sl, np.log(s0 * 0.7), np.log(s0 * 0.7), np.log(max(abs(sl) * 0.3, 1e-3)), 0.0])
        bounds = [(None, None), (None, None), (-9, 6), (-9, 6), (-9, 6), (-3, 3)]
        try:
            res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        except Exception as e:
            self.censored_population_results = {"error": f"Optimization failed: {e}"}
            return self.censored_population_results
        a, b, ls, lta, ltb, ar = res.x
        # Wald SEs from a finite-difference Hessian of the NLL
        se = None
        try:
            n_p = len(res.x); H = np.zeros((n_p, n_p)); h = 1e-3
            f0 = nll(res.x)
            for i in range(n_p):
                for j in range(i, n_p):
                    if i == j:
                        pp, pm = res.x.copy(), res.x.copy(); pp[i] += h; pm[i] -= h
                        H[i, i] = (nll(pp) - 2 * f0 + nll(pm)) / h ** 2
                    else:
                        v = []
                        for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                            q = res.x.copy(); q[i] += si * h; q[j] += sj * h; v.append(nll(q))
                        H[i, j] = H[j, i] = (v[0] - v[1] - v[2] + v[3]) / (4 * h ** 2)
            cov = np.linalg.inv(H)
            if np.all(np.diag(cov) > 0):
                se = np.sqrt(np.diag(cov))
        except Exception:
            se = None
        k = 6
        n_obs = int(len(y))
        aic = 2 * k + 2 * (res.fun - jac_sum)
        out = {
            "model_form": model,
            "method": "censored (Tobit) mixed model, adaptive Gauss-Hermite integration over correlated patient random intercept+slope",
            "total_patients_modeled": int(P),
            "total_observations": n_obs,
            "n_floor_censored": int(is_low.sum()),
            "n_ceiling_censored": int(is_up.sum()),
            "population_intercept": float(a),
            "population_mean_decay_rate": float(-b),
            "population_mean_decay_rate_se": (float(se[1]) if se is not None else None),
            "population_mean_decay_rate_95ci": ([float(-b - 1.96 * se[1]), float(-b + 1.96 * se[1])]
                                                if se is not None else None),
            "residual_sd": float(np.exp(ls)),
            "between_patient_intercept_sd": float(np.exp(lta)),
            "between_patient_slope_sd": float(np.exp(ltb)),
            "intercept_slope_correlation": float(np.tanh(ar)),
            "aic_jacobian_corrected": float(aic),
            "converged": bool(res.success),
            "eyes_share_patient_random_effects": bool(self.eye_column),
            "n_quadrature_nodes_per_dim": int(n_quad),
        }
        if not res.success:
            out["convergence_warning"] = "Optimizer reported no convergence: " + str(res.message)
        self.censored_population_results = out
        return out
 
    def fit_bayesian_censored_nlme(self, model: str = "Log-Exponential", draws: int = 600,
                                   tune: int = 600, chains: int = 2, target_accept: float = 0.9,
                                   random_seed: int = 42,
                                   correlated_random_effects: bool = True) -> Dict[str, Any]:
        """Hierarchical Bayesian mixed-effects model using PyMC's pm.Censored.
 
        Fits a single, given model form directly (the Jacobian fix only
        matters for cross-model AICc comparison). Audit 2: uses the model's
        own predictor (log(t + epsilon) for Power-Law). Each eye/modeling
        unit is its own group, so eyes of one patient are treated as
        independent here (flagged in the output)."""
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            return {"error": "pymc/arviz not installed. Run: pip install pymc arviz"}
 
        if self.clean_df.empty:
            self.clean_and_transform()
        df = self.clean_df.copy()
        if df.empty or df["group_id"].nunique() < MIN_COHORT_FOR_SAMPLE_SIZE:
            return {"error": f"Need >= {MIN_COHORT_FOR_SAMPLE_SIZE} patients."}
 
        col = self.endpoint_column
        try:
            raw_clamped = np.maximum(df[col].values, max(self.measurement_floor, TRANSFORM_Y_EPSILON))
            g_y = _transform_y(model, raw_clamped)
        except Exception as e:
            return {"error": f"Could not transform endpoint: {e}"}
 
        g_floor = _floor_in_transformed_space(model, self.measurement_floor)
        is_censored = df[col].values <= self.measurement_floor
        g_ceiling = None
        g_y_recorded = np.where(is_censored, g_floor, g_y)
        if self.measurement_ceiling is not None:
            g_ceiling = _ceiling_in_transformed_space(model, self.measurement_ceiling)
            g_y_recorded = np.where(df[col].values >= self.measurement_ceiling, g_ceiling, g_y_recorded)
        t = _prepare_predictor(model, df["years_from_baseline"].values)
        patient_codes, patient_uniques = pd.factorize(df["group_id"])
        coords = {"patient": patient_uniques, "re_dim": ["intercept", "slope"]}
 
        try:
            with pm.Model(coords=coords) as _:
                mu_a = pm.Normal("mu_a", mu=float(np.mean(g_y_recorded)), sigma=3.0)
                mu_b = pm.Normal("mu_b", mu=0.0, sigma=1.5)
                sigma = pm.HalfNormal("sigma", sigma=0.75)
 
                if correlated_random_effects:
                    sd_dist = pm.HalfNormal.dist(sigma=[1.5, 0.75], shape=2)
                    chol, corr, stds = pm.LKJCholeskyCov(
                        "chol_cov", n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True,
                    )
                    z = pm.Normal("z_re", mu=0.0, sigma=1.0, dims=("patient", "re_dim"))
                    ab_i = pm.Deterministic(
                        "ab_i", pm.math.dot(z, chol.T), dims=("patient", "re_dim")
                    )
                    a_i = ab_i[:, 0]
                    b_i = ab_i[:, 1]
                else:
                    tau_a = pm.HalfNormal("tau_a", sigma=1.5)
                    tau_b = pm.HalfNormal("tau_b", sigma=0.75)
                    a_raw = pm.Normal("a_raw", mu=0.0, sigma=1.0, dims="patient")
                    b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, dims="patient")
                    a_i = pm.Deterministic("a_i", a_raw * tau_a, dims="patient")
                    b_i = pm.Deterministic("b_i", b_raw * tau_b, dims="patient")
 
                pred = (mu_a + a_i[patient_codes]) + (mu_b + b_i[patient_codes]) * t
                latent = pm.Normal.dist(mu=pred, sigma=sigma)
                pm.Censored("obs", latent, lower=g_floor, upper=g_ceiling, observed=g_y_recorded)
                trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                                  progressbar=False, random_seed=random_seed,
                                  cores=max(1, min(chains, os.cpu_count() or 1)))
        except Exception as e:
            return {"error": f"Bayesian MCMC sampling failed: {e}"}
 
        var_names_for_summary = ["mu_a", "mu_b", "sigma"]
        var_names_for_summary += ["chol_cov_stds"] if correlated_random_effects else ["tau_a", "tau_b"]
        summ = az.summary(trace, var_names=var_names_for_summary)
        mu_b_mean = float(trace.posterior["mu_b"].mean())
        try:
            mu_b_hdi_raw = az.hdi(trace.posterior["mu_b"], hdi_prob=0.95)
        except TypeError:
            mu_b_hdi_raw = az.hdi(trace.posterior["mu_b"], prob=0.95)
        if hasattr(mu_b_hdi_raw, "data_vars"):
            hdi_vals = mu_b_hdi_raw["mu_b"].values
        else:
            hdi_vals = mu_b_hdi_raw.values
        hdi_low, hdi_high = float(np.min(hdi_vals)), float(np.max(hdi_vals))
        r_hat_mu_b = float(summ.loc["mu_b", "r_hat"])
        n_divergences = int(trace.sample_stats["diverging"].sum()) if "diverging" in trace.sample_stats else None
 
        if correlated_random_effects:
            between_patient_intercept_sd = float(trace.posterior["chol_cov_stds"].sel(chol_cov_stds_dim_0=0).mean())
            between_patient_slope_sd = float(trace.posterior["chol_cov_stds"].sel(chol_cov_stds_dim_0=1).mean())
            re_correlation = float(trace.posterior["chol_cov_corr"].sel(
                chol_cov_corr_dim_0=0, chol_cov_corr_dim_1=1
            ).mean())
        else:
            between_patient_intercept_sd = float(trace.posterior["tau_a"].mean())
            between_patient_slope_sd = float(trace.posterior["tau_b"].mean())
            re_correlation = None
 
        mu_a_mean = float(trace.posterior["mu_a"].mean())
 
        result = {
            "model_form": model,
            "total_patients_modeled": int(df["patient_id"].nunique()),
            "total_modeling_units": int(df["group_id"].nunique()),
            "eyes_treated_as_independent": bool(self.eye_column),
            "total_observations": int(len(df)),
            "n_censored_observations_included": int(is_censored.sum() + (np.sum(df[col].values >= self.measurement_ceiling) if self.measurement_ceiling is not None else 0)),
            "population_intercept": mu_a_mean,
            "population_mean_decay_rate": -mu_b_mean,
            "population_mean_decay_rate_95pct_hdi": [-hdi_high, -hdi_low],
            "between_patient_intercept_sd": between_patient_intercept_sd,
            "between_patient_slope_sd": between_patient_slope_sd,
            "residual_sd": float(trace.posterior["sigma"].mean()),
            "r_hat_mu_b": r_hat_mu_b,
            "n_divergences": n_divergences,
            "convergence_ok": bool(r_hat_mu_b < 1.05 and (n_divergences or 0) == 0),
            "correlated_random_effects": correlated_random_effects,
            "intercept_slope_correlation": re_correlation,
            "random_effects_structure": (
                "correlated (LKJ-Cholesky prior on intercept/slope covariance)"
                if correlated_random_effects else
                "diagonal (independent intercept/slope variances)"
            ),
            "note": (
                "intercept_slope_correlation is the posterior mean correlation between a "
                "unit's baseline severity and its decay rate -- a negative value means units "
                "that start worse tend to progress faster (or slower, if positive); near zero "
                "means the two are effectively independent for this cohort. Requires enough "
                "units and visits per unit for this extra parameter to be identifiable -- on "
                "small/sparse cohorts, expect wide posterior uncertainty."
                if correlated_random_effects else ""
            ),
        }
        if not result["convergence_ok"]:
            result["convergence_warning"] = (
                "r_hat elevated and/or divergences present - increase draws/tune/target_accept "
                "before trusting this estimate for a real decision."
            )
        self.bayesian_censored_nlme_results = result
        return result
 
    def run_full_analysis(self, target_power: float = 0.80, alpha: float = 0.05,
                           therapeutic_efficacy: float = 0.30, run_simulation: bool = True,
                           n_sims_per_candidate: int = 150,
                           auto_run_bayesian_if_heavily_censored: bool = True,
                           correlated_random_effects: bool = True) -> Dict[str, Any]:
        """Single-entry execution pipeline."""
        self.clean_and_transform()
        model_sel = self.run_model_competition()
        primary_model = model_sel.get("overall_best_supported_model", "Log-Exponential")
        self.run_per_patient_decay(model=primary_model)
        self.fit_mixed_effects_nlme(model=primary_model)
        self.fit_censored_population_model(model=primary_model)
        self.compute_closed_form_sample_size(target_power, alpha, therapeutic_efficacy)
        if run_simulation:
            self.compute_simulated_sample_size(
                target_power, alpha, therapeutic_efficacy, n_sims_per_candidate=n_sims_per_candidate,
                source="auto", correlated_random_effects=correlated_random_effects,
            )
        censoring_fraction = self.data_quality_report.get("total_censoring_fraction_of_raw_input", 0.0)
        need_bayesian_for_censoring = (
            auto_run_bayesian_if_heavily_censored and censoring_fraction > HEAVY_CENSORING_WARNING_FRACTION
        )
        already_have_bayesian = bool(self.bayesian_censored_nlme_results)
        if need_bayesian_for_censoring and not already_have_bayesian:
            self.fit_bayesian_censored_nlme(
                model=primary_model, correlated_random_effects=correlated_random_effects,
            )
        bayesian_results = self.bayesian_censored_nlme_results if self.bayesian_censored_nlme_results else None
        return {
            "engine_version": self.ENGINE_VERSION,
            "build_tag": self.BUILD_TAG,
            "data_quality_report": self.data_quality_report,
            "model_selection_results": self.model_selection_results,
            "primary_model_used": primary_model,
            "mixed_effects_results": self.mixed_effects_results,
            "bayesian_censored_nlme_results": bayesian_results,
            "censored_population_model_results": self.censored_population_results,
            "sample_size_closed_form": self.sample_size_closed_form,
            "sample_size_simulated": self.sample_size_simulated,
        }
 
    def generate_report(self) -> str:
        """Generates plain-language executive summary."""
        dq = self.data_quality_report
        ms = self.model_selection_results
        nlme = self.mixed_effects_results
        cf = self.sample_size_closed_form
        sim = self.sample_size_simulated
        lines = []
        lines.append(f"K-KODE ENGINE {self.ENGINE_VERSION} ({self.BUILD_TAG}) REPORT - Endpoint: {self.endpoint_column}")
        lines.append("=" * 70)
        lines.append("")
        lines.append("DATA QUALITY")
        lines.append(f"  {dq.get('rows_in_raw_input', '?')} raw rows; {dq.get('rows_retained_for_modeling', '?')} retained.")
        if "n_patients" in dq:
            lines.append(f"  {dq['n_patients']} patients, {dq['n_modeling_units']} modeling units.")
        if dq.get("rows_at_or_above_measurement_ceiling", 0) > 0:
            lines.append(f"  {dq['rows_at_or_above_measurement_ceiling']} ceiling-censored rows kept and modeled.")
        if dq.get("rows_at_or_below_measurement_floor", 0) > 0:
            lines.append(
                f"  {dq['rows_at_or_below_measurement_floor']} floor-censored rows "
                f"({dq.get('floor_censoring_fraction_of_raw_input', 0):.1%}) kept and modeled with Tobit MLE."
            )
        if "heavy_censoring_warning" in dq:
            lines.append(f"  WARNING: {dq['heavy_censoring_warning']}")
        lines.append("")
        lines.append("BEST-SUPPORTED FUNCTIONAL FORM (Jacobian-corrected)")
        if "overall_best_supported_model" in ms:
            lines.append(f"  {ms['overall_best_supported_model']} (evaluated on {ms['patients_evaluated']} modeling units "
                         f"from {ms.get('patients_represented', '?')} patients; eyes of one patient are not independent).")
            for m, w in ms.get("mean_akaike_weight_by_model", {}).items():
                lines.append(f"    - {m}: mean Akaike weight {w:.3f}, won for {ms['win_fraction_by_model'].get(m, 0):.0%} of units")
            top = sorted(ms.get("mean_akaike_weight_by_model", {}).values(), reverse=True)
            if len(top) > 1 and (top[0] - top[1]) < 0.15:
                lines.append("  NOTE: weights are close; no single form is clearly preferred.")
        else:
            lines.append(f"  Not enough data to run model competition ({ms.get('error', 'unknown error')}).")
        lines.append("")
        lines.append("POPULATION DECAY RATE")
        if "population_mean_decay_rate" in nlme:
            lines.append(f"  {nlme['population_mean_decay_rate']:.4f} / year on the {nlme['model_form']} scale "
                         f"(Standard NLME; {nlme.get('random_effects_structure', '')}).")
            if "note" in nlme:
                lines.append(f"  NOTE: {nlme['note']}")
            if "convergence_warning" in nlme:
                lines.append(f"  NOTE: {nlme['convergence_warning']}")
        else:
            lines.append(f"  Unavailable: {nlme.get('error', 'unknown error')}")
        cpm = self.censored_population_results
        if cpm and "population_mean_decay_rate" in cpm:
            ci = cpm.get("population_mean_decay_rate_95ci")
            lines.append(
                f"  {cpm['population_mean_decay_rate']:.4f} / year (Censored population model, "
                f"{cpm['n_floor_censored']} floor + {cpm['n_ceiling_censored']} ceiling rows kept; "
                f"95% CI: {[round(v, 4) for v in ci] if ci else 'unavailable'})."
            )
            if cpm.get("convergence_warning"):
                lines.append(f"  NOTE: {cpm['convergence_warning']}")
        bnlme = self.bayesian_censored_nlme_results
        if bnlme and "population_mean_decay_rate" in bnlme:
            lines.append(
                f"  {bnlme['population_mean_decay_rate']:.4f} / year "
                f"(Bayesian Censored NLME, {bnlme.get('random_effects_structure', '')}; "
                f"95% HDI: {bnlme['population_mean_decay_rate_95pct_hdi']})."
            )
            if bnlme.get("eyes_treated_as_independent"):
                lines.append("  NOTE: the Bayesian model treats each eye as an independent unit.")
            if bnlme.get("intercept_slope_correlation") is not None:
                lines.append(
                    f"  Baseline-severity/decay-rate correlation: {bnlme['intercept_slope_correlation']:.3f}"
                )
            if not bnlme.get("convergence_ok"):
                lines.append(f"  NOTE: {bnlme.get('convergence_warning', '')}")
        lines.append("")
        lines.append("SAMPLE SIZE (Closed-Form)")
        if "required_n_per_arm" in cf and cf["required_n_per_arm"] is not None:
            lines.append(f"  {cf['required_n_per_arm']} / arm, unit: {cf.get('analysis_unit')} "
                         f"(95% CI: {cf.get('bootstrap_95pct_ci_per_arm')})")
            if "provisional_estimate_warning" in cf:
                lines.append(f"  NOTE: {cf['provisional_estimate_warning']}")
        else:
            lines.append(f"  Unavailable: {cf.get('error', 'unknown error')}")
        lines.append("")
        lines.append("SAMPLE SIZE (Simulation-Validated)")
        if sim and "required_n_per_arm" in sim:
            lines.append(f"  {sim['required_n_per_arm']} / arm empirically achieving {sim['target_power']:.0%} power.")
            src = sim.get("population_parameter_source")
            if src == "bayesian_censored_nlme":
                lines.append(
                    "  NOTE: population parameters came from the Bayesian censored NLME "
                    "(standard NLME didn't converge on this cohort)."
                )
            if "power_law_note" in sim:
                lines.append(f"  NOTE: {sim['power_law_note']}")
        else:
            lines.append(f"  Not run or unavailable: {(sim or {}).get('error', 'not run')}")
        lines.append("=" * 70)
        lines.append("This report is a planning aid, not a finalized protocol, and the engine is")
        lines.append("not FDA-qualified. Validation to date: synthetic data with planted truth.")
        lines.append("A re-run of this build on RUSH2A (data-use-agreement data) is pending.")
        lines.append("Independent clinical/biostatistical review is still outstanding. See the")
        lines.append("module docstring for what this engine does and does not do.")
        return "\n".join(lines)
 
 
if __name__ == "__main__":
    print(f"K-KODE Engine {KKodeApexEngine.ENGINE_VERSION} ({KKodeApexEngine.BUILD_TAG}) initialized successfully.")
 
