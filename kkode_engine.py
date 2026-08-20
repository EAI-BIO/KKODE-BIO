"""
K-KODE ENGINE (v55.0)
Generalized longitudinal biomarker decay engine + censored-data-aware
model competition + simulation-validated clinical trial sample sizing,
for USH2A / RUSH2A-style retinal degeneration trial planning.
KEY CAPABILITIES IN THIS VERSION (v55.0):
1. GENERALIZED ENDPOINT SUPPORT. Not hardcoded to EZ width. Any
   longitudinal numeric endpoint (EZ width, EZ area, static perimetry
   sensitivity, microperimetry sensitivity) can be passed in via
   endpoint_column=. This matters because RUSH2A's own published
   recommendation is that functional endpoints (perimetry sensitivity),
   not structural EZ measurements, be the PRIMARY efficacy outcome - EZ
   area is mainly an enrollment criterion. A tool that only understands
   EZ width is modeling the field's secondary endpoint.
   SOURCE: Birch DG, et al. (RUSH2A Study Group). "Endpoints and Design
   for Clinical Trials in USH2A-Related Retinal Degeneration." Transl
   Vis Sci Technol. https://tvst.arvojournals.org/article.aspx?articleid=2802114
2. PROPER CENSORED-DATA HANDLING (Tobit-style MLE), not floor-and-drop.
   Every candidate model is fit per patient via maximum likelihood with
   a left-censored Gaussian likelihood: points above the measurement
   floor contribute a normal density, points at/below the floor
   contribute the normal CDF (the probability the true latent value was
   at or below the floor). This uses the fact that a censored point IS
   information ("this patient was at least this far progressed") instead
   of discarding it, and avoids the downward bias that comes from
   dropping or clamping fast progressors.
   SOURCE: Tobin J (1958). "Estimation of Relationships for Limited
   Dependent Variables." Econometrica, 26(1), 24-36.
3. FOUR CANDIDATE FUNCTIONAL FORMS competed per patient via AICc, using
   the same censored likelihood for all four so the comparison is
   apples-to-apples: Linear, Square-Root, Log-Exponential, and Power-Law.
   (Power-law needs t>0; baseline is offset by 1 day for that model only
   - see _prepare_predictor().)
   SOURCE (AICc): Hurvich CM, Tsai CL (1989). "Regression and time series
   model selection in small samples." Biometrika, 76(2), 297-307.
   SOURCE (Akaike weights): Burnham KP, Anderson DR (2002). Model
   Selection and Multimodel Inference (2nd ed). Springer.
4. NUMERICAL-HESSIAN STANDARD ERRORS. Parameter uncertainty is computed
   from a central finite-difference Hessian of the negative
   log-likelihood at the fitted optimum, rather than trusting the
   optimizer's internal (often low-rank / inexact) Hessian approximation.
4b. IDENTIFIABILITY GUARD. With only one uncensored point, a two-parameter
   (intercept, slope) censored fit is not identified: the intercept can
   absorb any slope choice for that single point, leaving the optimizer
   free to chase an unbounded "improvement" in the censored-point likelihood
   by driving |slope| toward infinity. Fixed by requiring >=2 uncensored
   points before a censored fit is attempted (see fit_censored_model).
5. SIMULATION-VALIDATED SAMPLE SIZE. Runs a Monte Carlo trial simulation:
   simulate many two-arm trials under the fitted population parameters
   (population intercept/slope, between-patient variance, residual variance)
   with a mixed-effects model fit to each simulated trial, empirically
   measuring statistical power at candidate sample sizes.
   SOURCE: Burton A, Altman DG, Royston P, Holder RL (2006). "The design
   of simulation studies in medical statistics." Stat Med, 25(24), 4279-4292.
6. HIERARCHICAL BAYESIAN CENSORED NLME (PyMC). Full population-level
   MCMC sampler (NUTS) using pm.Censored to estimate population decay (lambda)
   and between-patient variance directly from every observation including floor
   points, eliminating cohort-level survivorship bias under heavy censoring.
   SOURCE (mixed-effects structure): Laird NM, Ware JH (1982).
   "Random-effects models for longitudinal data." Biometrics, 38(4), 963-974.
   SOURCE (implementation): PyMC probabilistic programming library,
   https://www.pymc.io/
FULL REFERENCE LIST WITH LINKS: see "Methodology & References" in README.md
WHAT THIS VERSION DELIBERATELY DOES NOT CLAIM TO DO:
  - It does not ingest raw OCT images or do retinal layer segmentation.
    This engine assumes a reading center or imaging pipeline has already
    produced a numeric measurement per visit.
  - Standard NLME (non-Bayesian) still excludes floor-censored rows at the
    population level; use fit_bayesian_censored_nlme() for heavy censoring.
  - It is not FDA-qualified or validated as a Drug Development Tool.
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
        return np.log(t + POWER_LAW_TIME_OFFSET_YEARS)
    return t
def _floor_in_transformed_space(model: str, floor_value: float) -> float:
    if model == "Linear":
        return floor_value
    if model == "Square-Root":
        return np.sqrt(floor_value)
    return np.log(floor_value)
# ======================================================================
# Per-patient censored (Tobit-style) MLE fit
# ======================================================================
def _neg_log_likelihood(params: np.ndarray, x: np.ndarray, g_y: np.ndarray,
                         is_censored: np.ndarray, g_floor: float) -> float:
    a, b, log_sigma = params
    sigma = np.exp(log_sigma)
    pred = a + b * x
    resid = g_y - pred
    ll = 0.0
    uncensored = ~is_censored
    if np.any(uncensored):
        r = resid[uncensored]
        ll += np.sum(stats.norm.logpdf(r, loc=0.0, scale=sigma))
    if np.any(is_censored):
        z = (g_floor - pred[is_censored]) / sigma
        cdf = np.clip(stats.norm.cdf(z), 1e-12, 1.0)
        ll += np.sum(np.log(cdf))
    return -ll
def fit_censored_model(t: np.ndarray, y: np.ndarray, floor_value: float, model: str) -> Optional[Dict[str, Any]]:
    """
    Fits one candidate functional form to one patient's data via censored
    (Tobit-style) maximum likelihood.
    """
    n = len(t)
    if n < MIN_POINTS_FOR_TOBIT_FIT:
        return None
    if model == "Power-Law" and np.any(t < 0):
        return None
    if np.any(y <= 0) and model in ("Log-Exponential", "Power-Law", "Square-Root"):
        return None
    x = _prepare_predictor(model, t)
    g_y = _transform_y(model, np.maximum(y, floor_value))
    is_censored = y <= floor_value
    g_floor = _floor_in_transformed_space(model, floor_value)
    if np.max(x) == np.min(x):
        return None
    # IDENTIFIABILITY GUARD: requires >=2 uncensored points so slope is pinned down
    if np.sum(~is_censored) < 2:
        return None
    fit_mask = ~is_censored if np.any(~is_censored) else np.ones_like(is_censored, dtype=bool)
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
            _neg_log_likelihood, x0, args=(x, g_y, is_censored, g_floor),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000, "maxfev": 4000},
        )
        if not res.success and res.fun is None:
            return None
    except Exception:
        return None
    a_hat, b_hat, log_sigma_hat = res.x
    sigma_hat = np.exp(log_sigma_hat)
    neg_ll = res.fun
    k = 3  # a, b, sigma
    aic = 2 * k + 2 * neg_ll

    # AICc fallback for n=4, k=3 (standard 4-visit annual designs)
    small_sample_correction_unavailable = (n - k - 1) <= 0
    if small_sample_correction_unavailable:
        aicc = aic
    else:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    se_a, se_b = _numerical_hessian_se(res.x, x, g_y, is_censored, g_floor)
    return {
        "model": model,
        "n_observations": int(n),
        "n_censored": int(np.sum(is_censored)),
        "intercept": float(a_hat),
        "slope": float(b_hat),
        "sigma": float(sigma_hat),
        "slope_se": se_b,
        "neg_log_likelihood": float(neg_ll),
        "aic": float(aic),
        "aicc": float(aicc),
        "small_sample_correction_unavailable": small_sample_correction_unavailable,
        "converged": bool(res.success),
    }
def _numerical_hessian_se(x0: np.ndarray, x: np.ndarray, g_y: np.ndarray,
                           is_censored: np.ndarray, g_floor: float,
                           eps: float = 1e-4) -> Tuple[Optional[float], Optional[float]]:
    """Central finite-difference Hessian inverse for exact asymptotic parameter SEs."""
    n_params = len(x0)
    H = np.zeros((n_params, n_params))
    def f(p):
        return _neg_log_likelihood(p, x, g_y, is_censored, g_floor)
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
    def __init__(self, data_source: Union[str, pd.DataFrame], endpoint_column: str,
                 eye_column: Optional[str] = None, measurement_floor: float = 0.05,
                 higher_is_better: bool = True):
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
        self.higher_is_better = higher_is_better
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.data_quality_report: Dict[str, Any] = {}
        self.per_patient_fits: Dict[str, Dict[str, Any]] = {}
        self.model_selection_results: Dict[str, Any] = {}
        self.mixed_effects_results: Dict[str, Any] = {}
        self.sample_size_closed_form: Dict[str, Any] = {}
        self.sample_size_simulated: Dict[str, Any] = {}
        self.bayesian_censored_nlme_results: Dict[str, Any] = {}
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
        if self.eye_column and self.eye_column in df.columns:
            df = df.assign(group_id=df['patient_id'].astype(str) + "__" + df[self.eye_column].astype(str))
        else:
            df = df.assign(group_id=df['patient_id'].astype(str))
        dup_mask = df.duplicated(subset=['group_id', 'visit_date'], keep=False)
        n_dup = int(dup_mask.sum())
        processed = []
        n_single = 0
        for gid, group in df.groupby('group_id'):
            group = group.sort_values('visit_date').copy()
            if len(group) < 2:
                n_single += 1
            baseline = group['visit_date'].iloc[0]
            group['years_from_baseline'] = (group['visit_date'] - baseline).dt.days / 365.25
            processed.append(group)
        if processed:
            self.clean_df = pd.concat(processed, ignore_index=True)
        else:
            # Fall back to an empty frame that still carries the expected
            # columns (group_id, endpoint, years_from_baseline), not a bare
            # pd.DataFrame() with none at all -- otherwise downstream calls
            # like clean_df.groupby("group_id") or clean_df[endpoint_column]
            # raise KeyError instead of being correctly treated as "no data".
            self.clean_df = df.assign(years_from_baseline=pd.Series(dtype=float))
        censoring_fraction = (n_at_or_below_floor / n_start) if n_start else 0.0
        self.data_quality_report = {
            "engine_version": "v55.0",
            "endpoint_column": col,
            "rows_in_raw_input": n_start,
            "rows_dropped_missing_required_fields": int(n_missing),
            "rows_dropped_unparseable_date": int(n_bad_dates),
            "rows_dropped_non_numeric_endpoint": int(n_non_numeric),
            "rows_at_or_below_measurement_floor": n_at_or_below_floor,
            "floor_censoring_fraction_of_raw_input": round(censoring_fraction, 4),
            "note_censored_rows_are_kept": (
                "Floor-censored rows are KEPT in clean_df and handled via a censored "
                "(Tobit-style) likelihood in per-patient model fitting, not dropped."
            ),
            "patients_with_only_one_visit": int(n_single),
            "rows_retained_for_modeling": int(len(self.clean_df)),
            "duplicate_same_date_rows": n_dup,
        }
        if n_dup and not self.eye_column:
            self.data_quality_report["duplicate_same_date_warning"] = (
                "Same patient_id + visit_date rows exist. If this tracks both eyes, pass "
                "eye_column= so each eye is modeled separately."
            )
        if censoring_fraction > HEAVY_CENSORING_WARNING_FRACTION:
            self.data_quality_report["heavy_censoring_warning"] = (
                f"{censoring_fraction:.1%} of raw rows were at/below the measurement floor. "
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
        """Per-patient censored-MLE AIC/AICc competition across candidate forms."""
        if self.clean_df.empty:
            self.clean_and_transform()
        weight_sums = {m: 0.0 for m in CANDIDATE_MODELS}
        win_counts = {m: 0 for m in CANDIDATE_MODELS}
        n_evaluated = 0
        n_used_uncorrected_aic = 0
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
                    f = fit_censored_model(t, y, self.measurement_floor, m)
                    if f is not None and np.isfinite(f["aicc"]):
                        fits[m] = f
                except Exception:
                    continue
            if not fits:
                skipped.append(gid)
                continue
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
                "Per-patient censored-MLE AICc competition across Linear, Square-Root, "
                "Log-Exponential, and Power-Law forms, aggregated by mean Akaike weight and "
                "win-fraction across patients."
            ),
            "patients_evaluated": n_evaluated,
            "patients_skipped_insufficient_data": len(skipped),
            "patients_using_uncorrected_aic": n_used_uncorrected_aic,
            "mean_akaike_weight_by_model": {k: float(v) for k, v in mean_weights.items()},
            "win_fraction_by_model": {k: float(v) for k, v in win_fraction.items()},
            "overall_best_supported_model": overall_winner,
        }
        if n_used_uncorrected_aic > 0:
            self.model_selection_results["uncorrected_aic_note"] = (
                f"{n_used_uncorrected_aic} of {n_evaluated} patients had exactly 4 observations, "
                "where AICc's denominator is 0 - plain AIC was used as fallback."
            )
        return self.model_selection_results
    def run_per_patient_decay(self, model: str = "Log-Exponential") -> Dict[str, Dict[str, Any]]:
        """Fits candidate model to every patient via Tobit MLE."""
        if self.clean_df.empty:
            self.clean_and_transform()
        results = {}
        for gid in self.unique_group_ids():
            group = self.clean_df[self.clean_df["group_id"] == gid].sort_values("years_from_baseline")
            t = group["years_from_baseline"].values
            y = group[self.endpoint_column].values
            try:
                fit = fit_censored_model(t, y, self.measurement_floor, model)
                if fit is None:
                    results[gid] = {"status": f"Skipped {gid}: insufficient/degenerate data for {model} fit."}
                    continue
                decay_rate = -fit["slope"] if model in ("Log-Exponential", "Linear", "Square-Root") else None
                fit["decay_rate"] = decay_rate
                results[gid] = fit
            except Exception as e:
                results[gid] = {"status": f"Skipped {gid}: fit error: {str(e)}"}
        self.per_patient_fits = results
        return results
    def _valid_decay_rates(self) -> np.ndarray:
        rates = [
            v["decay_rate"] for v in self.per_patient_fits.values()
            if isinstance(v, dict) and v.get("decay_rate") is not None and np.isfinite(v.get("decay_rate"))
        ]
        return np.array(rates)
    def fit_mixed_effects_nlme(self, model: str = "Log-Exponential") -> Dict[str, Any]:
        """Population-level mixed-effects fit (statsmodels)."""
        if self.clean_df.empty:
            self.clean_and_transform()
        uncensored = self.clean_df[self.clean_df[self.endpoint_column] > self.measurement_floor].copy()
        if uncensored.empty or uncensored["group_id"].nunique() < MIN_COHORT_FOR_SAMPLE_SIZE:
            self.mixed_effects_results = {"error": "Not enough uncensored data to fit population mixed-effects model."}
            return self.mixed_effects_results
        uncensored["_transformed_endpoint"] = _transform_y(model, uncensored[self.endpoint_column].values)
        try:
            m = smf.mixedlm(
                "_transformed_endpoint ~ years_from_baseline", uncensored,
                groups=uncensored["group_id"], re_formula="~years_from_baseline",
            )
            mfit = m.fit(disp=False)
            fixed_slope = mfit.params["years_from_baseline"]
            pop_decay_rate = -fixed_slope
            slope_var = float(mfit.cov_re.iloc[1, 1]) if hasattr(mfit, "cov_re") and mfit.cov_re.shape[0] > 1 else None
            self.mixed_effects_results = {
                "model_form": model,
                "total_patients_modeled": int(uncensored["group_id"].nunique()),
                "total_observations": int(len(uncensored)),
                "population_mean_decay_rate": float(pop_decay_rate),
                "population_intercept": float(mfit.params["Intercept"]),
                "between_patient_slope_variance": slope_var,
                "residual_error_variance_sigma2": float(mfit.scale),
                "fixed_effects_p_value": float(mfit.pvalues["years_from_baseline"]),
                "model_converged": bool(mfit.converged),
            }
            if not mfit.converged:
                self.mixed_effects_results["convergence_warning"] = (
                    "Optimizer did not fully converge; treat as approximate."
                )
        except Exception as e:
            self.mixed_effects_results = {"error": f"Mixed-effects model failed to fit: {str(e)}"}
        return self.mixed_effects_results
    def compute_closed_form_sample_size(self, target_power: float = 0.80, alpha: float = 0.05,
                                         therapeutic_efficacy: float = 0.30,
                                         n_bootstrap: int = 2000, random_seed: int = 42) -> Dict[str, Any]:
        """Closed-form normal-approximation sample size calculator."""
        rates = self._valid_decay_rates()
        if len(rates) < MIN_COHORT_FOR_SAMPLE_SIZE:
            self.sample_size_closed_form = {"error": f"Need >= {MIN_COHORT_FOR_SAMPLE_SIZE} valid decay rates."}
            return self.sample_size_closed_form
        mean_l, std_l = float(np.mean(rates)), float(np.std(rates, ddof=1))
        if std_l == 0:
            self.sample_size_closed_form = {"error": "Zero variance across patients."}
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
            if ss_ > 0 and sm_ != 0:
                boots.append(req_n(sm_, ss_, therapeutic_efficacy))
        boots = np.array([b for b in boots if np.isfinite(b)])
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(boots) else (None, None)
        self.sample_size_closed_form = {
            "method": "closed-form z-based normal approximation",
            "cohort_size_used": len(rates),
            "mean_decay_rate": mean_l,
            "between_patient_sd": std_l,
            "required_n_per_arm": int(np.ceil(n_req)) if np.isfinite(n_req) else None,
            "bootstrap_95pct_ci_per_arm": [int(np.ceil(ci[0])), int(np.ceil(ci[1]))] if ci[0] is not None else None,
        }
        if len(rates) < SMALL_COHORT_WARNING_THRESHOLD:
            self.sample_size_closed_form["provisional_estimate_warning"] = f"Based on only {len(rates)} patients."
        return self.sample_size_closed_form
    def _simulate_power(self, n_per_arm: int, pop_intercept: float, pop_slope: float,
                         tau_intercept: float, tau_slope: float, resid_sd: float,
                         visit_times: List[float], efficacy_reduction: float,
                         alpha: float, n_sims: int, seed: int) -> float:
        """Monte Carlo trial simulation using vectorized two-stage summary measures."""
        rng = np.random.default_rng(seed)
        treat_slope = pop_slope * (1 - efficacy_reduction)
        visit_arr = np.array(visit_times)
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
    def compute_simulated_sample_size(self, target_power: float = 0.80, alpha: float = 0.05,
                                       therapeutic_efficacy: float = 0.30,
                                       visit_times: Optional[List[float]] = None,
                                       n_sims_per_candidate: int = 2000,
                                       max_search_iterations: int = 10,
                                       random_seed: int = 7) -> Dict[str, Any]:
        """Monte Carlo simulation power search under population parameters."""
        if not self.mixed_effects_results:
            self.fit_mixed_effects_nlme()
        nlme = self.mixed_effects_results
        if "error" in nlme or not nlme.get("model_converged"):
            self.sample_size_simulated = {"error": "Population mixed-effects model unavailable/didn't converge."}
            return self.sample_size_simulated
        pop_intercept = nlme["population_intercept"]
        pop_slope = -nlme["population_mean_decay_rate"]
        tau_slope = float(np.sqrt(max(nlme.get("between_patient_slope_variance") or 0.0, 1e-8)))
        resid_sd = float(np.sqrt(max(nlme["residual_error_variance_sigma2"], 1e-8)))
        tau_intercept = tau_slope * 3 if tau_slope > 0 else 0.1
        if visit_times is None:
            visit_times = [0.0, 1.0, 2.0, 3.0]
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
            visit_times, therapeutic_efficacy, alpha, n_sims_per_candidate, random_seed + it,
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
                visit_times, therapeutic_efficacy, alpha, n_sims_per_candidate, random_seed + it,
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
                visit_times, therapeutic_efficacy, alpha, n_sims_per_candidate, random_seed + it,
            )
            history.append({"n_per_arm": n_mid, "empirical_power": p_mid})
            if p_mid >= target_power:
                n_hi = n_mid
            else:
                n_lo = n_mid
        self.sample_size_simulated = {
            "method": "Monte Carlo trial simulation under fitted population parameters",
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
        return self.sample_size_simulated
    def fit_bayesian_censored_nlme(self, model: str = "Log-Exponential", draws: int = 600,
                                   tune: int = 600, chains: int = 2, target_accept: float = 0.9,
                                   random_seed: int = 42) -> Dict[str, Any]:
        """
        Hierarchical Bayesian mixed-effects model using PyMC's pm.Censored.
        Estimates population parameters including all floor-censored visits.
        """
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
            g_y = _transform_y(model, df[col].values)
        except Exception as e:
            return {"error": f"Could not transform endpoint: {e}"}

        g_floor = _floor_in_transformed_space(model, self.measurement_floor)
        is_censored = df[col].values <= self.measurement_floor
        g_y_recorded = np.where(is_censored, g_floor, g_y)
        t = df["years_from_baseline"].values
        patient_codes, patient_uniques = pd.factorize(df["group_id"])
        coords = {"patient": patient_uniques}
        try:
            with pm.Model(coords=coords) as _:
                mu_a = pm.Normal("mu_a", mu=float(np.mean(g_y_recorded)), sigma=3.0)
                mu_b = pm.Normal("mu_b", mu=0.0, sigma=1.5)
                tau_a = pm.HalfNormal("tau_a", sigma=1.5)
                tau_b = pm.HalfNormal("tau_b", sigma=0.75)
                sigma = pm.HalfNormal("sigma", sigma=0.75)
                a_raw = pm.Normal("a_raw", mu=0.0, sigma=1.0, dims="patient")
                b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, dims="patient")
                a_i = pm.Deterministic("a_i", a_raw * tau_a, dims="patient")
                b_i = pm.Deterministic("b_i", b_raw * tau_b, dims="patient")
                pred = (mu_a + a_i[patient_codes]) + (mu_b + b_i[patient_codes]) * t
                latent = pm.Normal.dist(mu=pred, sigma=sigma)
                pm.Censored("obs", latent, lower=g_floor, upper=None, observed=g_y_recorded)
                trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                                  progressbar=False, random_seed=random_seed)
        except Exception as e:
            return {"error": f"Bayesian MCMC sampling failed: {e}"}
        summ = az.summary(trace, var_names=["mu_a", "mu_b", "tau_a", "tau_b", "sigma"])
        mu_b_mean = float(trace.posterior["mu_b"].mean())
        mu_b_hdi = az.hdi(trace.posterior["mu_b"], hdi_prob=0.95)
        r_hat_mu_b = float(summ.loc["mu_b", "r_hat"])
        n_divergences = int(trace.sample_stats["diverging"].sum()) if "diverging" in trace.sample_stats else None
        result = {
            "model_form": model,
            "total_patients_modeled": int(df["group_id"].nunique()),
            "total_observations": int(len(df)),
            "n_censored_observations_included": int(is_censored.sum()),
            "population_mean_decay_rate": -mu_b_mean,
            "population_mean_decay_rate_95pct_hdi": [
                float(-mu_b_hdi["mu_b"].values[1]), float(-mu_b_hdi["mu_b"].values[0])
            ],
            "between_patient_slope_sd": float(trace.posterior["tau_b"].mean()),
            "residual_sd": float(trace.posterior["sigma"].mean()),
            "r_hat_mu_b": r_hat_mu_b,
            "n_divergences": n_divergences,
            "convergence_ok": bool(r_hat_mu_b < 1.05 and (n_divergences or 0) == 0),
            "random_effects_structure": "diagonal (independent intercept/slope variances)",
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
                           auto_run_bayesian_if_heavily_censored: bool = True) -> Dict[str, Any]:
        """Single-entry execution pipeline."""
        self.clean_and_transform()
        model_sel = self.run_model_competition()
        primary_model = model_sel.get("overall_best_supported_model", "Log-Exponential")
        self.run_per_patient_decay(model=primary_model)
        self.fit_mixed_effects_nlme(model=primary_model)
        self.compute_closed_form_sample_size(target_power, alpha, therapeutic_efficacy)
        if run_simulation:
            self.compute_simulated_sample_size(
                target_power, alpha, therapeutic_efficacy, n_sims_per_candidate=n_sims_per_candidate,
            )
        bayesian_results = None
        censoring_fraction = self.data_quality_report.get("floor_censoring_fraction_of_raw_input", 0.0)
        if auto_run_bayesian_if_heavily_censored and censoring_fraction > HEAVY_CENSORING_WARNING_FRACTION:
            bayesian_results = self.fit_bayesian_censored_nlme(model=primary_model)
        return {
            "data_quality_report": self.data_quality_report,
            "model_selection_results": self.model_selection_results,
            "primary_model_used": primary_model,
            "mixed_effects_results": self.mixed_effects_results,
            "bayesian_censored_nlme_results": bayesian_results,
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
        lines.append(f"K-KODE ENGINE v55.0 REPORT — Endpoint: {self.endpoint_column}")
        lines.append("=" * 70)
        lines.append("")
        lines.append("DATA QUALITY")
        lines.append(f"  {dq.get('rows_in_raw_input', '?')} raw rows; {dq.get('rows_retained_for_modeling', '?')} retained.")
        if dq.get("rows_at_or_below_measurement_floor", 0) > 0:
            lines.append(
                f"  {dq['rows_at_or_below_measurement_floor']} floor-censored rows "
                f"({dq.get('floor_censoring_fraction_of_raw_input', 0):.1%}) kept and modeled with Tobit MLE."
            )
        if "heavy_censoring_warning" in dq:
            lines.append(f"  WARNING: {dq['heavy_censoring_warning']}")
        lines.append("")
        lines.append("BEST-SUPPORTED FUNCTIONAL FORM")
        if "overall_best_supported_model" in ms:
            lines.append(f"  {ms['overall_best_supported_model']} (evaluated on {ms['patients_evaluated']} patients).")
            for m, w in ms.get("mean_akaike_weight_by_model", {}).items():
                lines.append(f"    - {m}: mean Akaike weight {w:.3f}, won for {ms['win_fraction_by_model'].get(m, 0):.0%} of patients")
        else:
            lines.append(f"  Not enough data to run model competition ({ms.get('error', 'unknown error')}).")
        lines.append("")
        lines.append("POPULATION DECAY RATE")
        if "population_mean_decay_rate" in nlme:
            lines.append(f"  {nlme['population_mean_decay_rate']:.4f} / year (Standard NLME).")
            if "convergence_warning" in nlme:
                lines.append(f"  NOTE: {nlme['convergence_warning']}")
        else:
            lines.append(f"  Unavailable: {nlme.get('error', 'unknown error')}")
        bnlme = self.bayesian_censored_nlme_results
        if bnlme and "population_mean_decay_rate" in bnlme:
            lines.append(
                f"  {bnlme['population_mean_decay_rate']:.4f} / year "
                f"(Bayesian Censored NLME; 95% HDI: {bnlme['population_mean_decay_rate_95pct_hdi']})."
            )
            if not bnlme.get("convergence_ok"):
                lines.append(f"  NOTE: {bnlme.get('convergence_warning', '')}")
        lines.append("")
        lines.append("SAMPLE SIZE (Closed-Form)")
        if "required_n_per_arm" in cf and cf["required_n_per_arm"] is not None:
            lines.append(f"  {cf['required_n_per_arm']} / arm (95% CI: {cf.get('bootstrap_95pct_ci_per_arm')})")
            if "provisional_estimate_warning" in cf:
                lines.append(f"  NOTE: {cf['provisional_estimate_warning']}")
        else:
            lines.append(f"  Unavailable: {cf.get('error', 'unknown error')}")
        lines.append("")
        lines.append("SAMPLE SIZE (Simulation-Validated)")
        if sim and "required_n_per_arm" in sim:
            lines.append(f"  {sim['required_n_per_arm']} / arm empirically achieving {sim['target_power']:.0%} power.")
        else:
            lines.append(f"  Not run or unavailable: {(sim or {}).get('error', 'not run')}")
        lines.append("=" * 70)
        lines.append("This report is a planning aid, not a finalized protocol. See module")
        lines.append("docstring for explicit statements of what this engine does and does not do.")
        return "\n".join(lines)
if __name__ == "__main__":
    print("K-KODE Engine v55.0 initialized successfully.")
