udit report · PY
"""
audit_report.py -- optional helper that writes a Markdown summary from numbers you pass in.
 
This file is NOT used by the K-KODE engine, which prints its own report through
KKodeApexEngine.generate_report(). It only formats a dictionary of results that
YOU supply. It does not run any analysis and it checks nothing. Any value you do
not supply is shown as "not supplied" rather than being filled in with a default.
 
Research Use Only. Not an approved medical device. Not a clinical validation.
"""
 
 
def _show(value, fmt="{}", unit=""):
    """Format a value for the report, or say plainly that it was not supplied."""
    if value is None:
        return "not supplied"
    try:
        text = fmt.format(value)
    except (ValueError, TypeError):
        text = str(value)
    return f"{text} {unit}".rstrip()
 
 
def _pct(value):
    """Turn a fraction (0.8) into a percentage string (80%), or 'not supplied'."""
    if value is None:
        return "not supplied"
    try:
        return f"{float(value) * 100:.0f}%"
    except (ValueError, TypeError):
        return str(value)
 
 
def generate_audit_report(results: dict, output_filename="KKODE_Audit_Summary.md") -> str:
    """
    Writes a Markdown summary of results you pass in, and returns the file name.
 
    Recognised keys (all optional): endpoint_name, data_source, power, alpha,
    effect_size, n_naive, n_kkode, n_saved, n_patients, total_visits,
    censored_visits, censored_pct, floor_value, winning_model, rate_kkode, unit,
    n_sims.
    """
    r = results or {}
    floor_text = _show(r.get("floor_value"))
    censored = _show(r.get("censored_visits"))
    if r.get("censored_pct") is not None:
        censored = f"{censored} ({_show(r.get('censored_pct'), '{:.1f}')}%)"
 
    report_content = f"""# K-KODE RESULTS SUMMARY
**Engine:** K-KODE Engine (Research Use Only)
**Endpoint analyzed:** {_show(r.get('endpoint_name'))}
**Data source:** {_show(r.get('data_source'))}
 
> This summary only formats the numbers that were passed in. It is a planning aid
> for researchers and statisticians. The numbers shown are not a clinical
> validation and have not been independently reviewed, and this summary must not
> be used to make clinical decisions. See Section 4 for what has been tested.
 
---
 
## 1. Trial powering
- **Target statistical power:** {_pct(r.get('power'))}
- **Significance level (alpha):** {_show(r.get('alpha'))}
- **Assumed treatment effect:** {_pct(r.get('effect_size'))} slowing of the decay rate
- **Sample size, standard least-squares comparison:** {_show(r.get('n_naive'))} patients per arm
- **Sample size, K-KODE censored model:** {_show(r.get('n_kkode'))} patients per arm
- **Difference:** {_show(r.get('n_saved'))} fewer patients
 
---
 
## 2. Cohort, censoring and model selection
| Item | Value | Note |
| :--- | :--- | :--- |
| Patients | {_show(r.get('n_patients'))} | Patients, not eyes, are the unit for sample size |
| Visits | {_show(r.get('total_visits'))} | Longitudinal observations |
| Floor-censored visits (floor = {floor_text}) | {censored} | Kept and modeled with a censored likelihood |
| Best-supported decay form | {_show(r.get('winning_model'))} | Chosen by Jacobian-corrected AICc |
| Estimated decay rate | {_show(r.get('rate_kkode'), '{:.3f}', r.get('unit') or '')} | From the fitted model |
 
---
 
## 3. Methods notes
1. **Censoring.** Readings at or below the floor contribute a censored (Tobit-style) likelihood term, not a clamped or discarded value.
2. **Identifiability.** A fit with fewer than two uncensored points is not identified, so the engine refuses it rather than producing a number.
3. **Sample-size check.** The closed-form estimate is checked against Monte Carlo trial simulation ({_show(r.get('n_sims'))} simulated trials per candidate size). If the two disagree by more than 2x, the engine prints a warning.
 
## 4. Validation status
Validation to date is automated testing on synthetic data with planted ground truth, plus one exploratory end-to-end engineering run on the RUSH2A natural-history dataset. That run has not been independently reviewed, and no real-data results are reported here. See the README, section "Validation status & integrity".
 
---
*Generated from supplied values only. Research Use Only (RUO).*
"""
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    return output_filename
 
