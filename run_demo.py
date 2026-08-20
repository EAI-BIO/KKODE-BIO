"""
run_demo.py — Built-in validation demo for the K-KODE Engine (v55.0)

Generates a 60-patient synthetic longitudinal cohort (120 eyes, 4 visits
each, matching a RUSH2A-style annual visit schedule) from a known
exponential decay process, runs it through the full K-KODE pipeline, and
prints the executive report.

This is a synthetic-data smoke test, not a validation against real
patient data — its purpose is to prove the pipeline installs and runs
correctly end to end on your machine before you point it at a real
clinical export.

Usage:
    python run_demo.py
"""

import numpy as np
import pandas as pd

from kkode_engine import KKodeApexEngine


def generate_rush2a_synthetic_cohort(n_patients: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generates a synthetic longitudinal perimetry-sensitivity dataset with
    exponential decay and a measurement floor, for demo/smoke-test purposes only."""
    rng = np.random.default_rng(seed)
    records = []

    for i in range(n_patients):
        pid = f"PATIENT_{i + 1:03d}"
        for eye in ["OD", "OS"]:
            y0 = rng.uniform(12.0, 24.0)
            decay_lambda = max(rng.normal(0.18, 0.05), 0.01)  # ~18% annual loss

            for yr in [0.0, 1.0, 2.0, 3.0]:
                visit_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(yr * 365.25))
                true_val = y0 * np.exp(-decay_lambda * yr) + rng.normal(0, 0.4)
                obs_val = max(0.05, float(true_val))  # measurement floor

                records.append({
                    "patient_id": pid,
                    "eye": eye,
                    "visit_date": visit_date.strftime("%Y-%m-%d"),
                    "static_perimetry_sensitivity_db": round(obs_val, 3),
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating 60-patient synthetic RUSH2A-style demo cohort...")
    df = generate_rush2a_synthetic_cohort(n_patients=60)
    print(f"  {len(df)} rows across {df['patient_id'].nunique()} patients x 2 eyes.")

    print("Initializing K-KODE Engine v55.0...")
    engine = KKodeApexEngine(
        data_source=df,
        endpoint_column="static_perimetry_sensitivity_db",
        eye_column="eye",
        measurement_floor=0.05,
        higher_is_better=True,
    )

    print("Running full analysis pipeline (model competition + Tobit MLE + "
          "mixed-effects NLME + Monte Carlo sample sizing)...")
    engine.run_full_analysis(
        target_power=0.80,
        alpha=0.05,
        therapeutic_efficacy=0.30,
        run_simulation=True,
        n_sims_per_candidate=100,
    )

    print("\n" + engine.generate_report())
    print("\nIf you see a report above with no errors, your installation is working "
          "correctly. Next step: point run_kkode.py or your own script at a real "
          "clinical CSV — see the README's 'Running on Your Own Data' section.")
