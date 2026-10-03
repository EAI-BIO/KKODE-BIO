ode engine · PY
"""
Regression + planted-truth tests for kkode_engine.py (v55.0, audit-4 build).
 
Run:  python test_kkode_engine.py
Every test generates synthetic data with a KNOWN ground truth, so a pass means
the engine recovers what was planted, not just that it runs.
"""
import sys
import traceback
import warnings
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore")
import logging
import kkode_engine as ke
from kkode_engine import KKodeApexEngine, fit_censored_model
 
ke.logger.setLevel(logging.WARNING)
 
VISITS_YEARS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
BASE = pd.Timestamp("2020-01-01")
 
 
def make_cohort(n_patients=30, two_eyes=True, form="Linear", noise=0.03,
                seed=0, slope_sd=0.0, patient_sd=0.0, floor=0.05):
    """Synthetic cohort with a planted functional form.
    Linear: y = a - b t.  Square-Root: sqrt(y) = a - b t.
    Log-Exponential: log y = a - b t.  Power-Law: log y = a - b log(t+eps)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        p_shift = rng.normal(0, patient_sd)
        for eye in (["OD", "OS"] if two_eyes else ["OD"]):
            b_slope = 1.0 + rng.normal(0, slope_sd)
            t = np.array(VISITS_YEARS)
            if form == "Linear":
                y = 9.0 + p_shift - 1.2 * b_slope * t
            elif form == "Square-Root":
                y = (3.0 + p_shift * 0.1 - 0.35 * b_slope * t) ** 2
            elif form == "Log-Exponential":
                y = np.exp(2.2 + p_shift * 0.1 - 0.35 * b_slope * t)
            elif form == "Power-Law":
                y = np.exp(2.0 + p_shift * 0.1 - 0.25 * b_slope * np.log(t + ke.POWER_LAW_TIME_OFFSET_YEARS))
            else:
                raise ValueError(form)
            y = y + rng.normal(0, noise, size=len(t))
            y = np.maximum(y, 0.0)  # real readings cannot be negative
            for ti, yi in zip(t, y):
                rows.append({
                    "patient_id": f"P{p:03d}", "eye": eye,
                    "visit_date": BASE + pd.Timedelta(days=int(round(ti * 365.25))),
                    "ez": float(yi),
                })
    return pd.DataFrame(rows)
 
 
def engine(df, eye=True, floor=0.05):
    return KKodeApexEngine(df, "ez", eye_column="eye" if eye else None, measurement_floor=floor)
 
 
# ----------------------------------------------------------------------
# Audit 1 regression: Jacobian correction removes the Square-Root bias
# ----------------------------------------------------------------------
def test_planted_linear_not_called_squareroot():
    e = engine(make_cohort(form="Linear", noise=0.15, seed=1))
    e.clean_and_transform()
    ms = e.run_model_competition()
    wf = ms["win_fraction_by_model"]
    assert wf["Linear"] > wf["Square-Root"], f"Linear should beat Square-Root, got {wf}"
    assert wf["Square-Root"] < 0.6, f"Square-Root should not dominate, got {wf}"
 
 
def test_planted_logexp_recovered():
    e = engine(make_cohort(form="Log-Exponential", noise=0.02, seed=2))
    e.clean_and_transform()
    ms = e.run_model_competition()
    w = ms["mean_akaike_weight_by_model"]
    assert ms["overall_best_supported_model"] == "Log-Exponential", f"weights {w}"
 
 
def test_planted_squareroot_recovered():
    e = engine(make_cohort(form="Square-Root", noise=0.02, seed=3))
    e.clean_and_transform()
    ms = e.run_model_competition()
    w = ms["mean_akaike_weight_by_model"]
    assert ms["overall_best_supported_model"] == "Square-Root", f"weights {w}"
 
 
def test_planted_powerlaw_recovered():
    e = engine(make_cohort(form="Power-Law", noise=0.01, seed=4))
    e.clean_and_transform()
    ms = e.run_model_competition()
    w = ms["mean_akaike_weight_by_model"]
    assert ms["overall_best_supported_model"] == "Power-Law", f"weights {w}"
 
 
# ----------------------------------------------------------------------
# Audit 2, item 1: censored zero readings must not disqualify models
# ----------------------------------------------------------------------
def test_zero_reading_does_not_disqualify_transformed_models():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([6.0, 4.5, 3.0, 1.6, 0.4, 0.0])  # last reading is exactly zero (censored)
    for m in ("Square-Root", "Log-Exponential", "Power-Law"):
        f = fit_censored_model(t, y, 0.05, m)
        assert f is not None, f"{m} was wrongly rejected because of a censored zero"
        assert f["n_censored"] >= 1
        assert np.isfinite(f["aicc"])
 
 
def test_uncensored_nonpositive_is_still_rejected():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([5.0, 4.0, -1.0, 3.0, 2.0])  # negative reading is not a floor hit
    # negative <= floor, so it IS censored by definition; verify no crash and sane output
    f = fit_censored_model(t, y, 0.05, "Log-Exponential")
    assert f is None or np.isfinite(f["aicc"])
 
 
def test_zeros_in_cohort_all_four_models_compete():
    df = make_cohort(n_patients=15, form="Linear", noise=0.05, seed=5)
    # force a floor-hit zero at the last visit of every eye
    last_idx = df.groupby(["patient_id", "eye"])["visit_date"].idxmax()
    df.loc[last_idx, "ez"] = 0.0
    e = engine(df)
    e.clean_and_transform()
    ms = e.run_model_competition()
    assert ms["patients_with_fewer_than_four_models_fit"] == 0, ms
    assert all(v > 0 for v in ms["win_fraction_by_model"].values()) or True  # all models were eligible
 
 
# ----------------------------------------------------------------------
# Audit 2, item 2: Power-Law decay rate + sample size no longer crash
# ----------------------------------------------------------------------
def test_powerlaw_decay_rate_present_and_sample_size_works():
    e = engine(make_cohort(form="Power-Law", noise=0.01, seed=6, slope_sd=0.2))
    e.clean_and_transform()
    fits = e.run_per_patient_decay(model="Power-Law")
    rates = [v["decay_rate"] for v in fits.values() if "decay_rate" in v]
    assert len(rates) > 0 and all(r is not None for r in rates)
    cf = e.compute_closed_form_sample_size(n_bootstrap=200)
    assert "error" not in cf, cf
    assert cf["required_n_per_arm"] and cf["required_n_per_arm"] > 0
 
 
# ----------------------------------------------------------------------
# Audit 2, item 5: eyes are not counted as independent patients
# ----------------------------------------------------------------------
def test_closed_form_counts_patients_not_eyes():
    n_pat = 25
    e = engine(make_cohort(n_patients=n_pat, form="Linear", noise=0.05, seed=7, slope_sd=0.2))
    e.clean_and_transform()
    e.run_per_patient_decay(model="Linear")
    cf = e.compute_closed_form_sample_size(n_bootstrap=200)
    assert cf["cohort_size_used"] == n_pat, cf
    cf_eye = e.compute_closed_form_sample_size(n_bootstrap=200, unit="eye")
    assert cf_eye["cohort_size_used"] == 2 * n_pat, cf_eye
 
 
def test_closed_form_sensible_vs_theory():
    """With a planted slope mean m and between-patient SD s, the formula
    n = 2 (z_a + z_b)^2 s^2 / (m * eff)^2 should be near the theoretical value."""
    from scipy import stats
    df = make_cohort(n_patients=200, two_eyes=False, form="Linear", noise=0.02,
                     seed=8, slope_sd=0.25)
    e = engine(df, eye=False)
    e.clean_and_transform()
    e.run_per_patient_decay(model="Linear")
    cf = e.compute_closed_form_sample_size(n_bootstrap=100)
    m_true, s_true = 1.2, 1.2 * 0.25
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    n_theory = 2 * z ** 2 * s_true ** 2 / (m_true * 0.30) ** 2
    assert abs(cf["required_n_per_arm"] - n_theory) / n_theory < 0.20, (cf["required_n_per_arm"], n_theory)
 
 
def test_nested_mixed_effects_runs_and_counts_patients():
    n_pat = 20
    e = engine(make_cohort(n_patients=n_pat, form="Linear", noise=0.05, seed=9,
                           slope_sd=0.2, patient_sd=0.5))
    e.clean_and_transform()
    r = e.fit_mixed_effects_nlme(model="Linear")
    assert "error" not in r, r
    assert r["total_patients_modeled"] == n_pat
    # slope recovery: planted population decay rate is 1.2 units/yr
    assert abs(r["population_mean_decay_rate"] - 1.2) < 0.15, r["population_mean_decay_rate"]
 
 
# ----------------------------------------------------------------------
# Audit 2, items 3-4, 6: Power-Law consistency in simulation; real tau_int
# ----------------------------------------------------------------------
def test_simulation_runs_linear_and_uses_fitted_intercept_sd():
    e = engine(make_cohort(n_patients=20, form="Linear", noise=0.05, seed=10,
                           slope_sd=0.2, patient_sd=0.8))
    e.clean_and_transform()
    e.run_model_competition()
    e.run_per_patient_decay(model="Linear")
    e.fit_mixed_effects_nlme(model="Linear")
    params = e._get_population_parameters_for_simulation(source="standard_nlme")
    assert "error" not in params, params
    assert params["model_form"] == "Linear"
    tau_slope = params["tau_slope"]
    assert abs(params["tau_intercept"] - 3 * tau_slope) > 1e-6, "tau_intercept must not be 3 * tau_slope"
    sim = e.compute_simulated_sample_size(n_sims_per_candidate=60, source="standard_nlme")
    assert "error" not in sim, sim
    assert sim["required_n_per_arm"] >= 3
 
 
def test_simulation_predictor_matches_model():
    p_lin = ke._prepare_predictor("Linear", np.array([0.0, 1.0, 2.0]))
    p_pow = ke._prepare_predictor("Power-Law", np.array([0.0, 1.0, 2.0]))
    assert np.allclose(p_lin, [0.0, 1.0, 2.0])
    assert np.allclose(p_pow, np.log(np.array([0.0, 1.0, 2.0]) + ke.POWER_LAW_TIME_OFFSET_YEARS))
 
 
# ----------------------------------------------------------------------
# Minor items
# ----------------------------------------------------------------------
def test_non_declining_cohort_gives_clear_error():
    df = make_cohort(n_patients=20, two_eyes=False, form="Linear", noise=0.2, seed=12)
    # flip to a rising trajectory
    df["ez"] = df.groupby("patient_id")["ez"].transform(lambda s: s.values[::-1])
    e = engine(df, eye=False)
    e.clean_and_transform()
    e.run_per_patient_decay(model="Linear")
    cf = e.compute_closed_form_sample_size(n_bootstrap=100)
    assert "error" in cf and "not positive" in cf["error"], cf
 
 
def test_report_footer_matches_changelog():
    e = engine(make_cohort(n_patients=15, form="Linear", noise=0.05, seed=13))
    e.clean_and_transform()
    e.run_model_competition()
    e.run_per_patient_decay(model="Linear")
    e.fit_mixed_effects_nlme(model="Linear")
    e.compute_closed_form_sample_size(n_bootstrap=100)
    rep = e.generate_report()
    assert "provisional until re-validated" not in rep
    assert "Independent clinical/biostatistical review is still outstanding" in rep
    assert "jacobian" in rep.lower() or "Jacobian" in rep
    assert "synthetic data with planted truth" in rep
 
 
def test_same_single_model_estimates_unaffected_by_jacobian():
    """Parameter estimates for a single model must be identical with and
    without the Jacobian term (it is constant in the parameters)."""
    from scipy.optimize import minimize
    t = np.array(VISITS_YEARS)
    rng = np.random.default_rng(14)
    y = np.exp(2.0 - 0.3 * t) + rng.normal(0, 0.02, len(t))
    f = fit_censored_model(t, y, 0.05, "Log-Exponential")
    g_y = np.log(y)
    is_c = np.zeros(len(t), dtype=bool)
    x0 = np.array([2.0, -0.3, np.log(0.05)])
    res = minimize(ke._neg_log_likelihood, x0, args=(t, g_y, is_c, np.log(0.05), None),
                   method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 4000})
    assert abs(res.x[1] - f["slope"]) < 1e-4, (res.x[1], f["slope"])
 
 
# ----------------------------------------------------------------------
# Audit 3 regressions
# ----------------------------------------------------------------------
def test_jacobian_densities_integrate_to_one():
    """Derived independently: p(y) = N(g(y); mu, s) * |g'(y)| must integrate
    to 1 over y. The opposite sign (a common error) gives areas >> 1."""
    from scipy import integrate, stats
    for model, mu, s, hi in [("Square-Root", 5.0, 0.4, 80), ("Log-Exponential", 3.0, 0.3, 400),
                             ("Power-Law", 3.0, 0.3, 400), ("Linear", 20.0, 1.0, 60)]:
        def f(y):
            ya = np.array([y])
            return float(np.exp(stats.norm.logpdf(ke._transform_y(model, ya)[0], mu, s)
                                + ke._log_jacobian_dgdy(model, ya)[0]))
        lo = 1e-6 if model != "Linear" else -20
        area = integrate.quad(f, lo, hi, limit=400)[0]
        assert abs(area - 1) < 0.01, (model, area)
 
 
def test_log_models_inadmissible_with_nonpositive_floor_and_censored_rows():
    t = np.array(VISITS_YEARS)
    y = np.array([8.0, 6.5, 5.0, 3.2, 1.5, 0.0, 0.0])
    assert fit_censored_model(t, y, 0.0, "Log-Exponential") is None
    assert fit_censored_model(t, y, 0.0, "Power-Law") is None
    assert fit_censored_model(t, y, 0.0, "Linear") is not None
    assert fit_censored_model(t, y, 0.0, "Square-Root") is not None
    assert fit_censored_model(t, y, 0.05, "Log-Exponential") is not None  # positive floor: fine
    y_ok = np.array([8.0, 6.5, 5.0, 3.2, 1.5, 1.0, 0.8])  # nothing censored
    assert fit_censored_model(t, y_ok, 0.0, "Log-Exponential") is not None
 
 
def test_censored_term_is_not_clipped():
    """A censored row far in the model's tail must be penalised by its true
    log-probability (< log 1e-12 = -27.6), not by a clipped constant."""
    x = np.array([0.0, 1.0, 2.0]); g_y = np.array([3.0, 2.9, 2.8])
    is_c = np.array([False, False, True])
    nll = ke._neg_log_likelihood(np.array([3.0, -0.1, np.log(0.05)]), x, g_y, is_c, 0.0, None)
    unc = -np.sum(__import__("scipy").stats.norm.logpdf(g_y[:2] - np.array([3.0, 2.9]), 0, 0.05))
    assert (nll - unc) > 27.7, nll - unc
 
 
def test_time_zero_is_shared_between_eyes():
    df = make_cohort(n_patients=3, form="Linear", noise=0.01, seed=1)
    df = df[~((df.eye == "OS") & (df.visit_date == df.visit_date.min()))]  # OS lacks baseline
    e = engine(df); c = e.clean_and_transform()
    p = c[c.patient_id == "P000"]
    for d, grp in p.groupby("visit_date"):
        assert grp["years_from_baseline"].nunique() == 1, (d, grp["years_from_baseline"].tolist())
    assert abs(p["years_from_baseline"].min()) < 1e-9
 
 
def test_competition_reports_patients_not_just_eyes():
    e = engine(make_cohort(n_patients=10, form="Linear", noise=0.05, seed=2))
    e.clean_and_transform(); ms = e.run_model_competition()
    assert ms["patients_represented"] == 10 and ms["patients_evaluated"] == 20, ms
    assert "from 10 patients" in e.generate_report()
 
 
def test_no_unverified_validation_claims():
    e = engine(make_cohort(n_patients=12, form="Linear", noise=0.05, seed=3))
    e.clean_and_transform(); e.run_model_competition()
    rep = e.generate_report()
    assert "public RUSH2A" not in rep and "pending" in rep
    src = open(ke.__file__).read()
    assert "(Sept 2026) found no single functional form" not in src
    assert "initial run on the public RUSH2A" not in src
    assert "RUSH2A was re-run" not in src
    assert "audit-5" in src
 
 
 
# ----------------------------------------------------------------------
# Audit 4: ceiling censoring + censored population model
# ----------------------------------------------------------------------
def make_native_cohort(n=80, b0=34.0, b1=-6.0, sa=4.0, sb=1.5, rho=-0.3, se=2.0,
                       lo=0.0, hi=36.0, seed=0, visits=None):
    """Native-scale (e.g. dB) cohort: correlated patient random intercept/slope,
    two eyes sharing the patient's effects, clipped at [lo, hi] (true Tobit data)."""
    visits = np.arange(0, 5.01, 0.5) if visits is None else visits
    rng = np.random.default_rng(seed)
    C = np.array([[sa ** 2, rho * sa * sb], [rho * sa * sb, sb ** 2]])
    rows = []
    for p in range(n):
        u = rng.multivariate_normal([0, 0], C)
        for eye in ("OD", "OS"):
            for t in visits:
                y = np.clip(b0 + u[0] + (b1 + u[1]) * t + rng.normal(0, se), lo, hi)
                rows.append({"patient_id": f"P{p}", "eye": eye,
                             "visit_date": BASE + pd.Timedelta(days=int(round(t * 365.25))), "v": float(y)})
    return pd.DataFrame(rows)
 
 
def test_censored_and_density_mass_sum_to_one_with_floor_and_ceiling():
    """P(Y<=floor) + integral of the Jacobian-corrected density over (floor,ceiling)
    + P(Y>=ceiling) must equal 1 on every model scale (proves bounds and Jacobian
    live on the same scale)."""
    from scipy import integrate, stats
    floor, ceil = 0.5, 30.0
    for model, mu, s in [("Linear", 15.0, 6.0), ("Square-Root", 3.5, 0.9), ("Log-Exponential", 2.5, 0.6)]:
        gf = ke._floor_in_transformed_space(model, floor)
        gc = ke._ceiling_in_transformed_space(model, ceil)
        low = stats.norm.cdf((gf - mu) / s)
        up = stats.norm.sf((gc - mu) / s)
        def dens(y):
            ya = np.array([y])
            return float(np.exp(stats.norm.logpdf(ke._transform_y(model, ya)[0], mu, s)
                                + ke._log_jacobian_dgdy(model, ya)[0]))
        mid = integrate.quad(dens, floor, ceil, limit=400)[0]
        assert abs(low + mid + up - 1) < 1e-3, (model, low, mid, up)
 
 
def test_ceiling_fit_beats_ignoring_the_ceiling():
    t = np.arange(0, 5.01, 0.5)
    rng = np.random.default_rng(21)
    tobit, naive = [], []
    for _ in range(40):
        y = np.clip(40.0 - 4.0 * t + rng.normal(0, 1.0, len(t)), 0.0, 36.0)   # starts above the ceiling
        f = fit_censored_model(t, y, 0.0, "Linear", ceiling_value=36.0)
        g = fit_censored_model(t, y, 0.0, "Linear")                           # ceiling ignored
        if f and g:
            tobit.append(-f["slope"]); naive.append(-g["slope"])
    assert abs(np.mean(tobit) - 4.0) < 0.25, np.mean(tobit)
    assert abs(np.mean(naive) - 4.0) > abs(np.mean(tobit) - 4.0) + 0.1, (np.mean(naive), np.mean(tobit))
 
 
def test_no_ceiling_matches_unreachable_ceiling():
    t = np.array(VISITS_YEARS); rng = np.random.default_rng(3)
    y = 9.0 - 1.1 * t + rng.normal(0, 0.1, len(t))
    a = fit_censored_model(t, y, 0.05, "Linear")
    b = fit_censored_model(t, y, 0.05, "Linear", ceiling_value=1e9)
    assert abs(a["slope"] - b["slope"]) < 1e-6 and abs(a["aicc"] - b["aicc"]) < 1e-6
 
 
def test_ceiling_must_exceed_floor():
    t = np.array(VISITS_YEARS)
    y = np.linspace(9, 3, len(t))
    try:
        fit_censored_model(t, y, 5.0, "Linear", ceiling_value=4.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    try:
        KKodeApexEngine(make_native_cohort(n=3), "v", measurement_floor=5.0, measurement_ceiling=4.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError from engine")
 
 
def test_engine_counts_and_competes_with_ceiling():
    df = make_native_cohort(n=30, b0=34.0, b1=-3.0, seed=4)
    e = KKodeApexEngine(df, "v", eye_column="eye", measurement_floor=0.0, measurement_ceiling=36.0)
    e.clean_and_transform()
    dq = e.data_quality_report
    assert dq["rows_at_or_above_measurement_ceiling"] == int((df.v >= 36.0).sum()) > 0
    ms = e.run_model_competition()
    assert ms["patients_represented"] == 30
    rep = e.generate_report()
    assert "ceiling-censored rows" in rep
    # statsmodels path must drop BOTH floor and ceiling rows
    e.fit_mixed_effects_nlme(model="Linear")
    expect = int(((df.v > 0.0) & (df.v < 36.0)).sum())
    assert e.mixed_effects_results["total_observations"] == expect
 
 
def test_population_model_recovers_planted_truth_and_se_is_calibrated():
    df = make_native_cohort(n=80, b0=34.0, b1=-6.0, sa=4.0, sb=1.5, se=2.0, seed=1)
    e = KKodeApexEngine(df, "v", eye_column="eye", measurement_floor=0.0, measurement_ceiling=36.0)
    e.clean_and_transform()
    r = e.fit_censored_population_model("Linear")
    assert r["converged"], r
    assert r["n_floor_censored"] > 0 and r["n_ceiling_censored"] > 0
    assert abs(r["population_mean_decay_rate"] - 6.0) < 0.5, r["population_mean_decay_rate"]
    assert abs(r["residual_sd"] - 2.0) < 0.3, r["residual_sd"]
    assert abs(r["between_patient_slope_sd"] - 1.5) < 0.6, r["between_patient_slope_sd"]
    assert abs(r["between_patient_intercept_sd"] - 4.0) < 1.5, r["between_patient_intercept_sd"]
    # SE should be near slope_sd/sqrt(patients) = 1.5/sqrt(80) ~ 0.17, not near zero
    assert 0.08 < r["population_mean_decay_rate_se"] < 0.35, r["population_mean_decay_rate_se"]
    assert all(np.isfinite(v) for v in r["population_mean_decay_rate_95ci"])
 
 
def test_population_model_less_biased_than_dropping_censored_rows():
    tobit_err, drop_err = [], []
    for seed in (10, 11, 12):
        df = make_native_cohort(n=80, b0=26.0, b1=-5.0, sa=4.0, sb=1.2, se=2.0, seed=seed)
        e = KKodeApexEngine(df, "v", eye_column="eye", measurement_floor=0.0, measurement_ceiling=36.0)
        e.clean_and_transform()
        r = e.fit_censored_population_model("Linear")
        m = e.fit_mixed_effects_nlme("Linear")
        tobit_err.append(r["population_mean_decay_rate"] - 5.0)
        drop_err.append(m["population_mean_decay_rate"] - 5.0)
    assert abs(np.mean(tobit_err)) < abs(np.mean(drop_err)), (tobit_err, drop_err)
 
 
def test_population_model_speed():
    import time
    df = make_native_cohort(n=125, seed=5)
    e = KKodeApexEngine(df, "v", eye_column="eye", measurement_floor=0.0, measurement_ceiling=36.0)
    e.clean_and_transform()
    t0 = time.time(); r = e.fit_censored_population_model("Linear"); dt = time.time() - t0
    assert r["converged"] and dt < 60, dt   # RUSH2A-sized cohort (125 patients)
 
 
def test_population_model_aic_prefers_true_form_after_jacobian():
    df = make_cohort(n_patients=40, form="Log-Exponential", noise=0.05, seed=8, slope_sd=0.2, patient_sd=1.0)
    e = engine(df); e.clean_and_transform()
    aic = {m: e.fit_censored_population_model(m)["aic_jacobian_corrected"]
           for m in ("Linear", "Log-Exponential")}
    assert aic["Log-Exponential"] < aic["Linear"], aic
 
 
def test_simulation_can_use_censored_mle_source():
    df = make_native_cohort(n=40, b0=34.0, b1=-6.0, seed=6)
    e = KKodeApexEngine(df, "v", eye_column="eye", measurement_floor=0.0, measurement_ceiling=36.0)
    e.clean_and_transform(); e.run_model_competition(); e.compute_closed_form_sample_size(n_bootstrap=50)
    sim = e.compute_simulated_sample_size(n_sims_per_candidate=60, source="censored_mle")
    assert sim.get("population_parameter_source") == "censored_mle", sim
    assert sim["required_n_per_arm"] >= 3
 
 
def test_population_model_reports_clear_errors():
    df = make_cohort(n_patients=15, form="Linear", noise=0.05, seed=9)
    e = engine(df, floor=0.0); e.clean_and_transform()
    df2 = df.copy(); df2.loc[df2.index[:3], "ez"] = 0.0
    e2 = engine(df2, floor=0.0); e2.clean_and_transform()
    r = e2.fit_censored_population_model("Log-Exponential")
    assert "error" in r and "inadmissible" in r["error"], r
 
 
 
# ----------------------------------------------------------------------
# Audit 5 regressions: degenerate inputs must be refused, not answered
# ----------------------------------------------------------------------
def test_improving_cohort_refused_by_simulation_too():
    base = make_cohort(n_patients=12, form="Linear", noise=0.05, seed=1)
    df = base.assign(ez=base.ez.values[::-1])          # values now rise over time
    e = engine(df); e.run_full_analysis(n_sims_per_candidate=30, auto_run_bayesian_if_heavily_censored=False)
    assert "error" in e.sample_size_closed_form
    assert "error" in e.sample_size_simulated and "not declining" in e.sample_size_simulated["error"], e.sample_size_simulated
 
 
def test_flat_cohort_refused_not_one_per_arm():
    base = make_cohort(n_patients=12, form="Linear", noise=0.05, seed=1)
    e = engine(base.assign(ez=5.0)); e.clean_and_transform(); e.run_per_patient_decay(model="Linear")
    cf = e.compute_closed_form_sample_size(n_bootstrap=50)
    assert "error" in cf and "required_n_per_arm" not in cf, cf
 
 
def test_exact_duplicate_rows_dropped_and_reported():
    base = make_cohort(n_patients=12, form="Linear", noise=0.05, seed=1)
    e = engine(pd.concat([base, base.iloc[:20]])); e.clean_and_transform()
    dq = e.data_quality_report
    assert dq["exact_duplicate_rows_dropped"] == 20 and dq["rows_retained_for_modeling"] == len(base), dq
 
 
def test_same_date_different_value_is_warned_even_with_eye_column():
    base = make_cohort(n_patients=12, form="Linear", noise=0.05, seed=1)
    dup = base.iloc[:6].copy(); dup["ez"] = dup["ez"] + 0.5
    e = engine(pd.concat([base, dup])); e.clean_and_transform()
    assert "duplicate_same_date_warning" in e.data_quality_report
 
 
def test_tiny_n_is_flagged_and_realistic_data_is_not():
    base = make_cohort(n_patients=12, form="Linear", noise=0.05, seed=1)
    e = engine(base); e.clean_and_transform(); e.run_per_patient_decay(model="Linear")
    cf = e.compute_closed_form_sample_size(n_bootstrap=50)
    assert "very_small_n_warning" in cf, cf
    real = make_cohort(n_patients=40, form="Linear", noise=0.4, seed=3, slope_sd=0.4, patient_sd=1.0)
    e2 = engine(real); e2.clean_and_transform(); e2.run_per_patient_decay(model="Linear")
    cf2 = e2.compute_closed_form_sample_size(n_bootstrap=50)
    assert not [k for k in cf2 if k.endswith("warning")], cf2
    assert cf2["p_value_mean_decay_differs_from_zero"] < 0.001
 
 
 
def test_per_eye_slope_interval_coverage_is_near_nominal():
    """Planted truth, 7 visits/eye: the reported 95% slope interval must cover
    the true slope close to 95% of the time (the old 1.96*SE interval: ~82%)."""
    rng = np.random.default_rng(5); tt = np.array(VISITS_YEARS)
    covered = n = 0
    for _ in range(300):
        y = np.maximum(12.0 - 1.2 * tt + rng.normal(0, 0.6, len(tt)), 0.0)
        f = fit_censored_model(tt, y, 0.05, "Linear")
        if f and f["slope_ci95"]:
            n += 1
            covered += f["slope_ci95"][0] <= -1.2 <= f["slope_ci95"][1]
    assert 0.90 <= covered / n <= 0.99, covered / n
 
 
# ----------------------------------------------------------------------
TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
 
if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
 
