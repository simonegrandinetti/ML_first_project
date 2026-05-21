"""Data cleaning and schema-enforcement utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_utils.constants import (
    ENGINEERED_FEATURE_DOCS,
    FEATURE_GROUPS,
    REGION_TO_MACROAREA
)


def normalize_text(text):
    if pd.isna(text):
        return ""
    return " ".join(str(text).lower().strip().split())


def get_feature_groups(df):
    """Return the available features grouped by semantic role."""
    return {
        group: [column for column in columns if column in df.columns]
        for group, columns in FEATURE_GROUPS.items()
    }


def feature_group_table(df):
    """Build a tidy table describing feature groups and their observation level."""
    known_columns = {
        column: group
        for group, columns in FEATURE_GROUPS.items()
        for column in columns
    }

    def infer_group(column):
        if column in known_columns:
            return known_columns[column]
        return "derived"

    def infer_level(column):
        if column in known_columns:
            return "artist" if known_columns[column] in {"artist_level", "geographic"} else "song"

        artist_prefixes = ("birth_date_", "active_start_", "active_end_", "artist_")
        artist_derived = {
            "artist_song_count",
            "artist_consistency_score",
            "artist_geographic_diversity",
        }

        if column.startswith(artist_prefixes) or column in artist_derived:
            return "artist"

        return "song"

    rows = []
    for column in df.columns:
        rows.append(
            {
                "feature": column,
                "group": infer_group(column),
                "level": infer_level(column),
            }
        )
    return pd.DataFrame(rows)


def clean_and_type_dataframe(
    df,
    column_types=None,
    datetime_columns=None,
    blank_placeholders_override=None,
):
    """Normalize placeholders and enforce target dtypes on matching columns."""
    if blank_placeholders_override is None:
        placeholders = ["[]", "none", "?", "nan", "<na>", ""]
    else:
        placeholders = blank_placeholders_override

    df = df.copy()
    df = df.replace(placeholders, np.nan)

    if column_types:
        for col, dtype in column_types.items():
            if col not in df.columns:
                continue
            try:
                df[col] = df[col].astype(dtype)
            except Exception:
                dtype_str = str(dtype).lower()
                if "int" in dtype_str:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    df[col] = numeric.astype("Int64")
                elif "float" in dtype_str:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    df[col] = numeric.astype("Float64")
                elif dtype_str in {"bool", "boolean"}:
                    bool_map = {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False,
                    }
                    normalized = (
                        df[col].astype("string").str.strip().str.lower().map(bool_map)
                    )
                    df[col] = normalized.astype("boolean")
                elif "datetime" in dtype_str:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(dtype)

    if datetime_columns:
        datetime_iter = (
            datetime_columns.keys()
            if isinstance(datetime_columns, dict)
            else datetime_columns
        )
        for col in datetime_iter:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def engineered_feature_table():
    """Return documentation for already engineered features in the source data."""
    return pd.DataFrame(ENGINEERED_FEATURE_DOCS)


def map_region_to_macroarea(region):
    """Map Italian regions to macro-areas used for regional comparisons."""
    key = normalize_text(region).replace("-", " ").replace("/", " ")
    key = " ".join(key.split())
    return REGION_TO_MACROAREA.get(key, "Missing")


def apply_domain_plausibility(
    df,
    numeric_rules=None,
    categorical_rules=None,
    consistency_rules=None,
    replace_invalid_with_nan=False,
):
    """Apply domain plausibility checks and return a quality report.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to validate.
    numeric_rules : dict | None
        Dict of {column: callable} where callable(series) returns boolean Series.
        If None, uses DOMAIN_PLAUSIBILITY_RULES from constants.
    categorical_rules : dict | None
        Dict of {column: set_of_valid_values} for membership checks.
        If None, uses DOMAIN_CATEGORICAL_RULES from constants.
    consistency_rules : dict | None
        Dict of {(col1, col2, ...): callable} for cross-column checks.
        If None, uses DOMAIN_CONSISTENCY_RULES from constants.
    replace_invalid_with_nan : bool, default False
        If True, replace any value flagged by numeric/categorical/consistency
        checks with missing values (pd.NA) in the returned dataframe.
    
    Returns
    -------
    tuple of (df, metadata)
        - df: validated dataframe (unchanged unless replace_invalid_with_nan=True)
        - metadata: dict with keys:
            - 'numeric_flags': boolean DataFrame (True = implausible)
            - 'categorical_flags': boolean DataFrame (True = not in valid set)
            - 'consistency_flags': boolean DataFrame (True = inconsistent)
            - 'summary': dict of implausible counts per column
    """
    from ml_utils.constants import (
        DOMAIN_PLAUSIBILITY_RULES,
        DOMAIN_CATEGORICAL_RULES,
        DOMAIN_CONSISTENCY_RULES,
    )
    
    if numeric_rules is None:
        numeric_rules = DOMAIN_PLAUSIBILITY_RULES
    if categorical_rules is None:
        categorical_rules = DOMAIN_CATEGORICAL_RULES
    if consistency_rules is None:
        consistency_rules = DOMAIN_CONSISTENCY_RULES
    
    numeric_flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    categorical_flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    consistency_flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    # Apply numeric plausibility rules
    for col, rule_fn in numeric_rules.items():
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            # A value is implausible if it's not NaN and the rule returns False
            try:
                is_plausible = rule_fn(numeric)
                numeric_flags[col] = numeric.notna() & ~is_plausible
            except Exception:
                pass
    
    # Apply categorical membership rules
    for col, valid_set in categorical_rules.items():
        if col in df.columns:
            if callable(valid_set):
                try:
                    is_valid = valid_set(df[col])
                    is_valid_series = pd.Series(np.asarray(is_valid), index=df.index)
                    categorical_flags[col] = (
                        df[col].notna() & ~is_valid_series.fillna(False).astype(bool)
                    )
                except Exception:
                    pass
            else:
                categorical_flags[col] = (
                    df[col].notna() & ~df[col].isin(valid_set)
                )
    
    # Apply cross-column consistency rules
    for cols, rule_fn in consistency_rules.items():
        try:
            is_consistent = rule_fn(*[df[c] if c in df.columns else pd.Series(True, index=df.index) for c in cols])
            for c in cols:
                if c in df.columns:
                    consistency_flags[c] = df[c].notna() & ~is_consistent
        except Exception:
            pass
    
    # Build summary
    summary = {}
    for col in numeric_flags.columns:
        numeric_count = int(numeric_flags[col].sum())
        categorical_count = int(categorical_flags[col].sum())
        consistency_count = int(consistency_flags[col].sum())
        if numeric_count > 0 or categorical_count > 0 or consistency_count > 0:
            summary[col] = {
                "numeric_implausible": numeric_count,
                "categorical_invalid": categorical_count,
                "consistency_violation": consistency_count,
            }
    
    if replace_invalid_with_nan:
        combined_flags = numeric_flags | categorical_flags | consistency_flags
        cols_to_update = [col for col in combined_flags.columns if combined_flags[col].any()]
        for col in cols_to_update:
            df.loc[combined_flags[col], col] = pd.NA

    return df, {
        "numeric_flags": numeric_flags,
        "categorical_flags": categorical_flags,
        "consistency_flags": consistency_flags,
        "summary": summary,
    }


def derive_date_fields(df, date_columns=None):
    """Extract year, month, day components from datetime columns.
    
    For each datetime column, creates _year, _month, _day fields with Int64 dtype.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    date_columns : list | None
        Datetime column names to decompose. If None, auto-detects datetime columns.
    
    Returns
    -------
    pandas.DataFrame
        Original dataframe with added derived date fields.
    """
    df = df.copy()
    
    if date_columns is None:
        date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    for col in date_columns:
        if col not in df.columns:
            continue
        
        # Skip if not datetime
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        df[f"{col}_year"] = df[col].dt.year.astype("Int64")
        df[f"{col}_month"] = df[col].dt.month.astype("Int64")
        df[f"{col}_day"] = df[col].dt.day.astype("Int64")
    
    return df


def compute_final_release_year(df, date_column="album_release_date", year_column="year"):
    """Compute canonical release year using multiple date sources.
    
    Priority: date_column (datetime) > year_column (numeric) > NaN
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    date_column : str
        Primary datetime column name (e.g., 'album_release_date').
    year_column : str
        Fallback numeric year column name (e.g., 'year').
    
    Returns
    -------
    pandas.DataFrame
        Dataframe with added 'final_release_year' column (Int64 dtype).
    """
    df = df.copy()
    
    final_year = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # Priority 1: Extract year from datetime column
    if date_column in df.columns:
        extracted_year = pd.to_datetime(df[date_column], errors="coerce").dt.year.astype("Int64")
        final_year = extracted_year

    # Priority 2: Use numeric year column as fallback only where the primary source is missing
    if year_column in df.columns:
        numeric_year = pd.to_numeric(df[year_column], errors="coerce").astype("Int64")
        final_year = final_year.fillna(numeric_year)

    df["final_release_year"] = final_year

    return df
