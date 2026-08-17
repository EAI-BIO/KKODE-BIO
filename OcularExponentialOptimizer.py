"""
OcularExponentialOptimizer - Biostatistical engine for USH2A retinopathy
longitudinal Ellipsoid Zone (EZ) width tracking and clinical trial sample
size estimation.

IMPORTANT MODELING ASSUMPTION (verify before relying on this for trial design):
This engine assumes EZ width decays exponentially over time, i.e.
    ln(EZ_width(t)) = ln(EZ_width_0) - lambda * t
This assumption should be checked against the published USH2A natural
history literature (e.g. the RUSH2A EZ-width natural history study) before
being used for real trial-design decisions. If the literature instead
supports a different functional form for width specifically (as opposed to
EZ *area*, where a square-root transform is more common because area grows
with the square of a linear atrophy radius), the transform in
`clean_and_transform()` should be updated accordingly.

STATISTICAL ASSUMPTIONS made by compute_required_sample_size():
- Two-arm, 1:1 randomized, parallel-group design.
- Equal variance assumed between control and treatment arms (the natural
  history cohort's variance is used as the planning estimate for both).
- Normal-approximation (z-based) sample size formula, not a small-sample /
  t-distribution correction - fine for planning purposes, but should be
  re-run with a t-based or simulation-based method once real pilot data
  seeds an actual protocol.
- The treatment effect (delta) is expressed as a fraction of the *mean*
  natural-history decay rate. This is one reasonable way to parameterize
  "efficacy," but it is a modeling choice, not a measured quantity - it
  should be stated explicitly in any protocol document.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Dict, Any, List, Optional

logger = logging.getLogger("OcularExponentialOptimizer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

MIN_EZ_WIDTH_MM = 0.05  # floor to keep log() well-defined; see note in clean_and_transform()
MIN_COHORT_FOR_SAMPLE_SIZE = 3
# Below this, treat the sample-size point estimate as provisional/high-uncertainty
# and say so loudly rather than just handing back a bare number.
SMALL_COHORT_WARNING_THRESHOLD = 10


class OcularExponentialOptimizer:
    """
    Biostatistical engine for USH2A retinopathy longitudinal EZ width data.
    Implements log-transformed exponential decay modeling and two-sample
    clinical trial sample size estimation, with a bootstrap confidence
    interval so the sample-size number isn't presented as more certain
    than the underlying pilot cohort actually supports.
    """

    REQUIRED_COLUMNS = ['patient_id', 'visit_date', 'ez_width_mm']

    def __init__(self, data_source: Union[str, pd.DataFrame], eye_column: Optional[str] = None):
        """
        eye_column: name of an optional column identifying which eye (e.g. "OD"/"OS")
        each row belongs to. If your source data tracks both eyes per patient and this
        is left unset, two independent eye trajectories on the same visit date will be
        silently pooled into one regression per patient - see the duplicate-date warning
        in the data quality report. Pass the column name here to group by
        (patient_id, eye) instead, so each eye gets its own decay estimate.
        """
        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Target file path not found: {data_source}")
            self.raw_df: pd.DataFrame = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.raw_df = data_source.copy()
        else:
            raise TypeError("Data source must be a file path string or a pandas DataFrame.")

        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in self.raw_df.columns]
        if missing_cols:
            raise ValueError(
                f"Input is missing required column(s): {missing_cols}. "
                f"Expected columns: {self.REQUIRED_COLUMNS}"
                + (f" plus eye column '{eye_column}'" if eye_column else "")
            )
        if eye_column and eye_column not in self.raw_df.columns:
            raise ValueError(f"eye_column='{eye_column}' was specified but is not present in the input data.")

        self.eye_column = eye_column
        self.group_id_column = 'patient_id'
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.global_rates: List[float] = []
        self.per_patient_results: Dict[str, Dict[str, Any]] = {}
        self.data_quality_report: Dict[str, Any] = {}

    def clean_and_transform(self) -> pd.DataFrame:
        """
        Validates clinical inputs, calculates fractional-year time deltas,
        and applies a natural log transform to EZ width. Every row dropped
        along the way is counted and reported in self.data_quality_report,
        rather than disappearing silently.
        """
        n_start = len(self.raw_df)
        df = self.raw_df.copy()

        # --- Required-field completeness ---
        n_missing_required = df[['patient_id', 'visit_date', 'ez_width_mm']].isna().any(axis=1).sum()
        df = df.dropna(subset=['patient_id', 'visit_date', 'ez_width_mm']).copy()

        # --- Date parsing: coerce instead of crash, then drop + count failures ---
        parsed_dates = pd.to_datetime(df['visit_date'], errors='coerce')
        n_bad_dates = parsed_dates.isna().sum()
        df = df.assign(visit_date=parsed_dates).dropna(subset=['visit_date'])

        # --- Numeric coercion for EZ width, then explicit drop + count (not just an
        # implicit "NaN > threshold is False" side effect) ---
        numeric_width = pd.to_numeric(df['ez_width_mm'], errors='coerce')
        n_non_numeric = numeric_width.isna().sum()
        df = df.assign(ez_width_mm=numeric_width).dropna(subset=['ez_width_mm'])

        # --- Physical/measurement floor: EZ width must be a positive, log-safe value ---
        n_below_floor = (df['ez_width_mm'] <= MIN_EZ_WIDTH_MM).sum()
        df = df[df['ez_width_mm'] > MIN_EZ_WIDTH_MM]

        # --- Composite group key: patient alone, or patient+eye if eye_column is set ---
        if self.eye_column:
            df = df.assign(group_id=df['patient_id'].astype(str) + "__" + df[self.eye_column].astype(str))
        else:
            df = df.assign(group_id=df['patient_id'].astype(str))

        # --- Duplicate (group, visit_date) rows: same patient (and eye, if tracked) measured
        # "twice" on one date. This is either a data-entry duplicate, a repeat scan, or -
        # if eye_column was left unset on a dataset that actually tracks both eyes - two
        # separate eyes' measurements getting pooled into what looks like one time series.
        # Either way this silently biases the regression, so it's counted and surfaced
        # rather than passed through unnoticed.
        dup_mask = df.duplicated(subset=['group_id', 'visit_date'], keep=False)
        n_duplicate_date_rows = int(dup_mask.sum())

        processed_groups = []
        n_single_visit_patients = 0
        for gid, group in df.groupby('group_id'):
            group = group.sort_values(by='visit_date').copy()
            if len(group) < 2:
                n_single_visit_patients += 1
                # Still kept in clean_df (useful for a data inventory / future visits),
                # just can't produce a decay estimate on its own later.
            baseline_date = group['visit_date'].iloc[0]
            group['years_from_baseline'] = (group['visit_date'] - baseline_date).dt.days / 365.25
            group['log_ez_width'] = np.log(group['ez_width_mm'])
            processed_groups.append(group)

        self.clean_df = (
            pd.concat(processed_groups, ignore_index=True) if processed_groups else pd.DataFrame()
        )

        self.data_quality_report = {
            "rows_in_raw_input": n_start,
            "rows_dropped_missing_required_fields": int(n_missing_required),
            "rows_dropped_unparseable_date": int(n_bad_dates),
            "rows_dropped_non_numeric_ez_width": int(n_non_numeric),
            "rows_dropped_below_measurement_floor_mm": int(n_below_floor),
            "patients_with_only_one_visit": int(n_single_visit_patients),
            "rows_retained_for_modeling": int(len(self.clean_df)),
            "duplicate_same_date_rows": n_duplicate_date_rows,
        }
        if n_duplicate_date_rows and not self.eye_column:
            self.data_quality_report["duplicate_same_date_warning"] = (
                "Rows with the same patient_id and visit_date exist. If this data tracks "
                "both eyes, pass eye_column= when constructing the optimizer so each eye is "
                "modeled separately - otherwise both eyes are being pooled into one regression."
            )
        for k, v in self.data_quality_report.items():
            logger.info(f"{k}: {v}")

        return self.clean_df

    def unique_group_ids(self) -> List[str]:
        """Returns the IDs to iterate over - patient_id alone, or patient_id__eye
        composites if eye_column was set. Use these with model_exponential_decay()
        rather than raw patient_id values when eye_column is in use."""
        if self.clean_df.empty:
            self.clean_and_transform()
        if self.clean_df.empty or 'group_id' not in self.clean_df.columns:
            return []
        return sorted(self.clean_df['group_id'].unique().tolist())

    def model_exponential_decay(self, target_patient_id: str) -> Dict[str, Any]:
        """
        Fits ln(EZ) = ln(EZ_0) - lambda * t via OLS in log-space for one patient
        (or one patient+eye, if eye_column was set - see unique_group_ids()).
        Every result is retained (including negative/near-zero or low-confidence
        fits) and appended to self.global_rates - by design, NOT filtered by
        significance, to avoid selection bias in the pooled variance used for
        trial sample-size planning. Each result carries its own r_squared/p_value
        so a reviewer can see per-patient data quality without the pooled
        estimate silently discarding "inconvenient" patients.
        """
        if self.clean_df.empty:
            self.clean_and_transform()
        if self.clean_df.empty:
            return {"status": f"Skipped patient {target_patient_id}: no valid rows survived cleaning."}

        p_data = self.clean_df[self.clean_df['group_id'] == target_patient_id]
        if len(p_data) < 2:
            result = {"status": f"Skipped patient {target_patient_id}: insufficient longitudinal points (need >=2)."}
            self.per_patient_results[target_patient_id] = result
            return result

        t = p_data['years_from_baseline'].values
        ln_y = p_data['log_ez_width'].values

        if np.max(t) == np.min(t):
            result = {"status": f"Skipped patient {target_patient_id}: zero variance in time vector (all visits same date)."}
            self.per_patient_results[target_patient_id] = result
            return result

        try:
            slope, intercept, r_val, p_val, std_err = stats.linregress(t, ln_y)

            if np.isnan(slope) or np.isinf(slope):
                result = {"status": f"Skipped patient {target_patient_id}: non-finite slope from OLS fit."}
                self.per_patient_results[target_patient_id] = result
                return result

            decay_coefficient_lambda = -slope
            annual_loss_percentage = (1 - np.exp(slope)) * 100

            metrics = {
                "patient_id": target_patient_id,
                "observations": int(len(p_data)),
                "exponential_decay_coefficient_lambda": float(decay_coefficient_lambda),
                "annual_tissue_loss_percentage": float(annual_loss_percentage),
                "r_squared_fit": float(r_val ** 2),
                "p_value": float(p_val),
                "is_statistically_significant": bool(p_val < 0.05),
                "flag_low_confidence_fit": bool(len(p_data) < 3 or r_val ** 2 < 0.5),
            }

            # Intentionally unconditional: see docstring note on avoiding selection bias.
            self.global_rates.append(decay_coefficient_lambda)
            self.per_patient_results[target_patient_id] = metrics
            return metrics

        except Exception as e:
            result = {"status": f"Skipped patient {target_patient_id}: regression error: {str(e)}"}
            self.per_patient_results[target_patient_id] = result
            return result

    def compute_required_sample_size(
        self,
        target_power: float = 0.80,
        alpha: float = 0.05,
        therapeutic_efficacy: float = 0.30,
        n_bootstrap: int = 2000,
        random_seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Two-sample, 1:1 parallel-group sample size calculation using the
        pooled cohort's decay-rate variance as the planning estimate for both
        arms (see module-level docstring for the full list of assumptions).

        Adds a bootstrap confidence interval around the required-N estimate,
        since a point estimate derived from a small pilot cohort (as few as
        MIN_COHORT_FOR_SAMPLE_SIZE patients) is materially less certain than a
        single integer implies. Below SMALL_COHORT_WARNING_THRESHOLD patients,
        the result is explicitly flagged as provisional.
        """
        if len(self.global_rates) < MIN_COHORT_FOR_SAMPLE_SIZE:
            return {
                "error": (
                    f"Insufficient cohort size: {len(self.global_rates)} valid patient "
                    f"regressions, minimum {MIN_COHORT_FOR_SAMPLE_SIZE} required."
                )
            }

        rates_array = np.array(self.global_rates)
        mean_lambda = np.mean(rates_array)
        std_lambda = np.std(rates_array, ddof=1)

        if std_lambda == 0:
            return {"error": "Cohort variance is zero across all patients; cannot compute sample size."}

        if mean_lambda <= 0:
            logger.warning(
                "Mean decay coefficient (lambda) is zero or negative across the cohort. "
                "This can happen with a small/noisy pilot cohort where measurement error "
                "dominates true signal. Sample size estimate below is not reliable."
            )

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(target_power)
        delta = mean_lambda * therapeutic_efficacy

        def _required_n(mean_l, std_l, eff):
            d = mean_l * eff
            if d == 0:
                return np.inf
            return (2 * (z_alpha + z_beta) ** 2 * (std_l ** 2)) / (d ** 2)

        required_n_per_arm = _required_n(mean_lambda, std_lambda, therapeutic_efficacy)

        # --- Bootstrap CI on the required-N estimate itself ---
        rng = np.random.default_rng(random_seed)
        boot_ns = []
        for _ in range(n_bootstrap):
            sample = rng.choice(rates_array, size=len(rates_array), replace=True)
            s_mean, s_std = np.mean(sample), np.std(sample, ddof=1)
            if s_std > 0 and s_mean != 0:
                boot_ns.append(_required_n(s_mean, s_std, therapeutic_efficacy))
        boot_ns = np.array([n for n in boot_ns if np.isfinite(n)])

        ci_low, ci_high = (
            (float(np.percentile(boot_ns, 2.5)), float(np.percentile(boot_ns, 97.5)))
            if len(boot_ns) > 0 else (None, None)
        )

        result = {
            "cohort_size_used": len(self.global_rates),
            "cohort_mean_exponential_lambda": float(mean_lambda),
            "cohort_standard_deviation_lambda": float(std_lambda),
            "targeted_therapeutic_progression_slowing": f"{therapeutic_efficacy * 100:.0f}%",
            "required_sample_size_per_treatment_arm": int(np.ceil(required_n_per_arm)) if np.isfinite(required_n_per_arm) else None,
            "total_optimized_trial_cohort_size": int(np.ceil(required_n_per_arm) * 2) if np.isfinite(required_n_per_arm) else None,
            "sample_size_95pct_bootstrap_ci_per_arm": (
                [int(np.ceil(ci_low)), int(np.ceil(ci_high))] if ci_low is not None else None
            ),
            "assumptions": {
                "design": "two-arm, 1:1 randomized, parallel-group",
                "variance_source": "pooled natural-history cohort variance, assumed equal in both arms",
                "test_type": "normal-approximation (z-based) two-sample formula",
            },
        }

        if len(self.global_rates) < SMALL_COHORT_WARNING_THRESHOLD:
            result["provisional_estimate_warning"] = (
                f"Based on only {len(self.global_rates)} patients. Treat this as a rough "
                "planning number, not a protocol-ready sample size - the bootstrap CI above "
                "shows how much it could move with a larger pilot cohort."
            )

        return result

    def export_json(self, output_path: str) -> None:
        """Writes cleaning report + per-patient results to a single JSON file."""
        payload = {
            "data_quality_report": self.data_quality_report,
            "per_patient_results": self.per_patient_results,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"Exported results to {output_path}")


if __name__ == "__main__":
    print("=====================================================================")
    print("INITIALIZING OCULAR EXPONENTIAL OPTIMIZER (K-Kode)")
    print("=====================================================================")

    simulated_cohort = {
        'patient_id': [
            'PT_01', 'PT_01', 'PT_01', 'PT_01',
            'PT_02', 'PT_02', 'PT_02', 'PT_02',
            'PT_03', 'PT_03', 'PT_03', 'PT_03',
            'PT_04', 'PT_04', 'PT_04',
            'PT_05', 'PT_05', 'PT_05', 'PT_05',
            'PT_ERR', 'PT_ERR',           # same-day visits -> zero time variance
            'PT_BADDATE', 'PT_BADDATE',   # one malformed date -> must not crash the pipeline
            'PT_SINGLE',                  # only one visit -> can't fit a slope
        ],
        'visit_date': [
            '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01',
            '2022-03-15', '2023-03-20', '2024-04-02', '2025-03-15',
            '2022-06-01', '2023-05-28', '2024-06-05', '2025-06-01',
            '2022-02-10', '2023-02-14', '2024-02-20',
            '2022-04-01', '2023-04-10', '2024-04-05', '2025-04-12',
            '2026-01-01', '2026-01-01',
            '2022-01-01', 'not-a-real-date',
            '2023-01-01',
        ],
        'ez_width_mm': [
            7.50, 6.65, 5.90, 5.23,
            5.20, 4.68, 4.21, 3.79,
            8.10, 7.05, 6.13, 5.33,
            6.40, 5.95, 5.52,
            9.00, 8.10, 7.35, 6.55,
            4.00, 4.00,
            5.00, 5.00,
            6.00,
        ]
    }

    study_df = pd.DataFrame(simulated_cohort)
    optimizer = OcularExponentialOptimizer(study_df)

    print("\n[STEP 1] Cleaning + log-transforming (see data quality report below)...")
    optimizer.clean_and_transform()

    print("\n[STEP 2] Fitting per-patient exponential decay...")
    for pid in optimizer.unique_group_ids():
        res = optimizer.model_exponential_decay(pid)
        if "status" in res:
            print(f"  - {res['status']}")
        else:
            flag = " [LOW CONFIDENCE]" if res["flag_low_confidence_fit"] else ""
            print(
                f"  - {pid} -> lambda: {res['exponential_decay_coefficient_lambda']:.4f} "
                f"| annual loss: {res['annual_tissue_loss_percentage']:.2f}% "
                f"| R^2: {res['r_squared_fit']:.3f} | p: {res['p_value']:.4f}{flag}"
            )

    print("\n[STEP 3] Trial sample size + bootstrap CI...")
    trial_design = optimizer.compute_required_sample_size(target_power=0.80, alpha=0.05, therapeutic_efficacy=0.30)
    print("=====================================================================")
    if "error" in trial_design:
        print(f"[ERROR] {trial_design['error']}")
    else:
        print(f"Cohort size used:                 {trial_design['cohort_size_used']}")
        print(f"Mean lambda:                       {trial_design['cohort_mean_exponential_lambda']:.4f}")
        print(f"Std dev lambda:                    {trial_design['cohort_standard_deviation_lambda']:.4f}")
        print(f"Target efficacy:                   {trial_design['targeted_therapeutic_progression_slowing']}")
        print(f"Required N per arm:                {trial_design['required_sample_size_per_treatment_arm']}")
        print(f"95% bootstrap CI per arm:          {trial_design['sample_size_95pct_bootstrap_ci_per_arm']}")
        print(f"Total cohort:                      {trial_design['total_optimized_trial_cohort_size']}")
        if "provisional_estimate_warning" in trial_design:
            print(f"NOTE: {trial_design['provisional_estimate_warning']}")
    print("=====================================================================")

    optimizer.export_json("/home/claude/kkode/results.json")
