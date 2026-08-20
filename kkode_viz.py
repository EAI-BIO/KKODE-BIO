"""
kkode_viz.py — Visualization add-on for the K-KODE Engine (v55.0)

Optional module. The core engine (kkode_engine.py) has zero dependency on
this file or on matplotlib — this is a separate, opt-in layer so a standard
analysis run never has to import plotting libraries it doesn't need.

Install with:
    pip install -r requirements-viz.txt

Usage:
    from kkode_engine import KKodeApexEngine
    from kkode_viz import plot_patient_fit, plot_cohort_overview

    engine = KKodeApexEngine(...)
    engine.run_full_analysis()

    plot_patient_fit(engine, group_id="PATIENT_001__OD", save_path="patient_001.png")
    plot_cohort_overview(engine, save_path="cohort_overview.png")
"""

from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(
        "kkode_viz requires matplotlib. Install with: pip install -r requirements-viz.txt"
    ) from e


def _inverse_transform(model: str, pred: np.ndarray) -> np.ndarray:
    """Inverse of KKodeApexEngine._transform_y — maps fitted values back to
    the original endpoint scale for plotting."""
    if model == "Linear":
        return pred
    if model == "Square-Root":
        return np.clip(pred, a_min=0, a_max=None) ** 2
    if model in ("Log-Exponential", "Power-Law"):
        return np.exp(pred)
    raise ValueError(f"Unknown model: {model}")


def _predictor(model: str, t: np.ndarray) -> np.ndarray:
    """Mirrors KKodeApexEngine._prepare_predictor for curve generation."""
    if model == "Power-Law":
        offset = 1.0 / 365.25
        return np.log(t + offset)
    return t


def plot_patient_fit(engine, group_id: str, save_path: Optional[str] = None,
                      show: bool = False, figsize=(7, 5)):
    """
    Plots one patient's observed longitudinal visits against their fitted
    decay curve. Censored (at-or-below-floor) points are marked distinctly
    from uncensored points so the censoring handling is visually legible,
    not just a number in a report.
    """
    if engine.clean_df.empty:
        raise ValueError("Run engine.clean_and_transform() (or run_full_analysis()) first.")
    if group_id not in engine.per_patient_fits:
        raise ValueError(
            f"No fit found for group_id={group_id!r}. "
            f"Run engine.run_per_patient_decay() first, or check unique_group_ids()."
        )
    fit = engine.per_patient_fits[group_id]
    if "model" not in fit:
        raise ValueError(f"group_id={group_id!r} was not successfully fit: {fit.get('status')}")

    group = engine.clean_df[engine.clean_df["group_id"] == group_id].sort_values("years_from_baseline")
    t = group["years_from_baseline"].values
    y = group[engine.endpoint_column].values
    is_censored = y <= engine.measurement_floor

    model = fit["model"]
    a, b = fit["intercept"], fit["slope"]

    t_smooth = np.linspace(max(t.min(), 0.0), t.max(), 200)
    x_smooth = _predictor(model, t_smooth)
    pred_smooth = _inverse_transform(model, a + b * x_smooth)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_smooth, pred_smooth, color="#2b6cb0", linewidth=2,
            label=f"Fitted {model} decay (rate={fit.get('decay_rate'):.3f}/yr)"
            if fit.get("decay_rate") is not None else f"Fitted {model}")

    if np.any(~is_censored):
        ax.scatter(t[~is_censored], y[~is_censored], color="#2b6cb0", zorder=5,
                   s=60, label="Observed (uncensored)")
    if np.any(is_censored):
        ax.scatter(t[is_censored], y[is_censored], color="#e53e3e", marker="v",
                   zorder=5, s=70, label="At/below measurement floor (censored)")

    ax.axhline(engine.measurement_floor, color="#999999", linestyle="--", linewidth=1,
               label=f"Measurement floor ({engine.measurement_floor})")

    ax.set_xlabel("Years from baseline")
    ax.set_ylabel(engine.endpoint_column)
    ax.set_title(f"K-KODE Patient Trajectory — {group_id}")
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_cohort_overview(engine, save_path: Optional[str] = None,
                          show: bool = False, figsize=(8, 6), max_patients: int = 200):
    """
    Spaghetti plot of every patient's raw trajectory in the cohort, colored
    by censoring status, so heavy floor-censoring or outlier patients are
    visible at a glance rather than buried in summary statistics.
    """
    if engine.clean_df.empty:
        raise ValueError("Run engine.clean_and_transform() (or run_full_analysis()) first.")

    fig, ax = plt.subplots(figsize=figsize)
    group_ids = engine.clean_df["group_id"].unique()
    if len(group_ids) > max_patients:
        rng = np.random.default_rng(0)
        group_ids = rng.choice(group_ids, size=max_patients, replace=False)

    any_censored_label_used = False
    for gid in group_ids:
        group = engine.clean_df[engine.clean_df["group_id"] == gid].sort_values("years_from_baseline")
        t = group["years_from_baseline"].values
        y = group[engine.endpoint_column].values
        is_censored = y <= engine.measurement_floor
        ax.plot(t, y, color="#a0aec0", linewidth=0.8, alpha=0.6, zorder=1)
        if np.any(is_censored):
            label = "Censored visit" if not any_censored_label_used else None
            ax.scatter(t[is_censored], y[is_censored], color="#e53e3e", s=12,
                      zorder=3, label=label)
            any_censored_label_used = True

    ax.axhline(engine.measurement_floor, color="#333333", linestyle="--", linewidth=1,
               label=f"Measurement floor ({engine.measurement_floor})")

    n_shown = len(group_ids)
    n_total = engine.clean_df["group_id"].nunique()
    subtitle = f"{n_shown} of {n_total} patients shown" if n_shown < n_total else f"{n_total} patients"
    ax.set_xlabel("Years from baseline")
    ax.set_ylabel(engine.endpoint_column)
    ax.set_title(f"K-KODE Cohort Overview — {subtitle}")
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig
