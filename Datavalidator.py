import pandas as pd

def validate_clinical_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validates input datasets against clinical data schemas prior to running MLE/Bayesian fits.
    """
    required_cols = ['patient_id', 'visit_year', 'biomarker_value']
    
    # Column verification
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required dataset columns: {missing_cols}"
    
    # Non-negative visit checks
    if (df['visit_year'] < 0).any():
        return False, "Validation Error: Negative visit years detected."
        
    # Minimum visit checks per patient
    visit_counts = df.groupby('patient_id')['visit_year'].count()
    short_cohorts = visit_counts[visit_counts < 2]
    if not short_cohorts.empty:
        return False, f"Validation Error: {len(short_cohorts)} patients have < 2 visits. Identifiability requires >= 2 visits."
        
    return True, "Dataset schema passed validation."