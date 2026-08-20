#!/usr/bin/env python3
"""
run_kkode.py — Command-line runner for the K-KODE Engine (v55.0)

Lets a team run a full biostatistical analysis on a longitudinal visit
dataset without writing any Python. Wraps KKodeApexEngine.run_full_analysis()
and generate_report().

USAGE
-----
    python run_kkode.py --data patient_data.csv --endpoint-column ez_width_mm

    python run_kkode.py \\
        --data patient_data.csv \\
        --endpoint-column static_perimetry_sensitivity_db \\
        --eye-column eye \\
        --measurement-floor 0.05 \\
        --target-power 0.80 \\
        --alpha 0.05 \\
        --therapeutic-efficacy 0.30 \\
        --n-sims 150 \\
        --output results/

INPUT DATA REQUIREMENTS
------------------------
Your CSV must contain, at minimum, these columns:
    patient_id    - any string or numeric patient identifier
    visit_date    - a date (any format pandas can parse, e.g. YYYY-MM-DD)
    <endpoint>    - the numeric biomarker column you pass to --endpoint-column
                    (e.g. ez_width_mm, ez_area_mm2, static_perimetry_sensitivity_db)

An optional `eye` column (OD/OS or left/right) can be passed via
--eye-column if your dataset tracks both eyes per patient.

OUTPUT
------
Writes two files into the --output directory (created if it doesn't exist):
    kkode_report.txt    - the plain-language executive summary
    kkode_results.json  - the full structured results (all fitted
                           parameters, model competition weights, sample
                           size estimates, and diagnostics)

If --output is omitted, the report is printed to the console and no
files are written.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kkode_engine import KKodeApexEngine


def _json_safe(obj):
    """Recursively convert numpy/pandas scalar types to native Python types
    so the results dict can be serialized with the standard json module."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the K-KODE Engine (v55.0) full analysis pipeline on a CSV of longitudinal visit data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", required=True, help="Path to the input CSV file.")
    parser.add_argument("--endpoint-column", required=True,
                         help="Name of the numeric biomarker column to model (e.g. ez_width_mm).")
    parser.add_argument("--eye-column", default=None,
                         help="Optional column name identifying eye (e.g. 'eye'). Omit if not tracked.")
    parser.add_argument("--measurement-floor", type=float, default=0.05,
                         help="Value at/below which a visit is treated as floor-censored. Default: 0.05")
    parser.add_argument("--lower-is-better", action="store_true",
                         help="Pass this flag if a LOWER endpoint value indicates disease progression "
                              "(e.g. some structural measures). Default assumes higher is better "
                              "(e.g. sensitivity, EZ area).")
    parser.add_argument("--target-power", type=float, default=0.80, help="Target statistical power. Default: 0.80")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold. Default: 0.05")
    parser.add_argument("--therapeutic-efficacy", type=float, default=0.30,
                         help="Assumed proportional treatment effect for sample sizing. Default: 0.30")
    parser.add_argument("--no-simulation", action="store_true",
                         help="Skip the Monte Carlo simulation-validated sample size check "
                              "(closed-form estimate only — faster, less rigorous).")
    parser.add_argument("--n-sims", type=int, default=150,
                         help="Simulated trials per candidate N, if simulation is run. Default: 150")
    parser.add_argument("--no-bayesian", action="store_true",
                         help="Never auto-run the hierarchical Bayesian censored NLME fit, even if "
                              "censoring exceeds the heavy-censoring threshold. Use this if pymc/arviz "
                              "are not installed, or to force a faster run.")
    parser.add_argument("--output", default=None,
                         help="Directory to write kkode_report.txt and kkode_results.json into. "
                              "If omitted, the report prints to the console and nothing is saved.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: input file not found: {data_path}", file=sys.stderr)
        return 1

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  {len(df)} rows, {df['patient_id'].nunique() if 'patient_id' in df.columns else '?'} unique patients.")

    try:
        engine = KKodeApexEngine(
            data_source=df,
            endpoint_column=args.endpoint_column,
            eye_column=args.eye_column,
            measurement_floor=args.measurement_floor,
            higher_is_better=not args.lower_is_better,
        )
    except (ValueError, TypeError) as e:
        print(f"ERROR initializing engine: {e}", file=sys.stderr)
        return 1

    print("Running full analysis pipeline (this may take under a minute; "
          "longer if the Bayesian NLME model auto-triggers on heavy censoring)...")
    try:
        results = engine.run_full_analysis(
            target_power=args.target_power,
            alpha=args.alpha,
            therapeutic_efficacy=args.therapeutic_efficacy,
            run_simulation=not args.no_simulation,
            n_sims_per_candidate=args.n_sims,
            auto_run_bayesian_if_heavily_censored=not args.no_bayesian,
        )
    except Exception as e:
        print(f"ERROR during analysis: {e}", file=sys.stderr)
        return 1

    report = engine.generate_report()

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        report_path = out_dir / "kkode_report.txt"
        report_path.write_text(report, encoding="utf-8")

        results_path = out_dir / "kkode_results.json"
        results_path.write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")

        print(f"\nDone. Wrote:\n  {report_path}\n  {results_path}")
    else:
        print("\n" + "=" * 72)
        print(report)
        print("=" * 72)
        print("\n(Tip: pass --output <directory> to save this report and the full "
              "structured results as files.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
