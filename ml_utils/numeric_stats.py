"""Numeric transformation and scoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_utils.preprocessing import normalize_text


def safe_divide(numerator, denominator):
    """Safely divide two pandas-compatible arrays and return NaN on zero division."""
    numerator = pd.Series(numerator)
    denominator = pd.Series(denominator).replace({0: np.nan})
    out = numerator / denominator
    return out.replace([np.inf, -np.inf], np.nan)


def zscore_series(series):
    """Standardize a numeric series while preserving missing values."""
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.where(numeric.notna(), 0.0, np.nan), index=numeric.index)
    return (numeric - numeric.mean()) / std


def build_weighted_zscore(df, columns, weights=None, min_non_missing=2):
    """Create a weighted average of z-scored columns."""
    if weights is None:
        weights = np.ones(len(columns), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(columns):
        raise ValueError("weights must have the same length as columns")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if weights.sum() == 0:
        raise ValueError("at least one weight must be positive")

    zscores = pd.DataFrame({column: zscore_series(df[column]) for column in columns})
    normalized_weights = weights / weights.sum()
    weighted = zscores.mul(normalized_weights, axis=1)
    available_weights = zscores.notna().mul(normalized_weights, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, min_count=1) / available_weights.where(available_weights > 0)
    score[zscores.notna().sum(axis=1) < min_non_missing] = np.nan
    return score


def iqr_bounds(series, whisker_width=1.5):
    """Return Tukey-style IQR bounds for a numeric series."""
    numeric = pd.to_numeric(series, errors="coerce").astype("float64").dropna()
    if numeric.empty:
        return np.nan, np.nan, np.nan, np.nan
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - whisker_width * iqr
    upper = q3 + whisker_width * iqr
    return q1, q3, lower, upper


def summarize_outliers(df, columns, domain_rules=None, whisker_width=1.5):
    """Summarize robust and domain-based outliers for selected numeric columns."""
    domain_rules = domain_rules or {}
    rows = []
    for column in columns:
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce").astype("float64")
        valid_mask = numeric.notna()
        valid = numeric[valid_mask]
        if valid.empty:
            continue

        q1, q3, lower, upper = iqr_bounds(numeric, whisker_width=whisker_width)
        iqr_mask = ((numeric < lower) | (numeric > upper)).fillna(False)

        if column in domain_rules:
            domain_mask = domain_rules[column](numeric)
            domain_mask = pd.Series(domain_mask, index=numeric.index).fillna(False).astype(bool)
        else:
            domain_mask = pd.Series(False, index=numeric.index)

        valid_count = int(valid.count())
        iqr_outliers = int(iqr_mask[valid_mask].sum())
        domain_outliers = int(domain_mask[valid_mask].sum())

        rows.append(
            {
                "feature": column,
                "valid_count": valid_count,
                "q1": q1,
                "q3": q3,
                "iqr_lower": lower,
                "iqr_upper": upper,
                "iqr_outliers": iqr_outliers,
                "iqr_outlier_pct": round(100 * iqr_outliers / valid_count, 2),
                "domain_outliers": domain_outliers,
                "domain_outlier_pct": round(100 * domain_outliers / valid_count, 2),
                "min": valid.min(),
                "max": valid.max(),
            }
        )
    return pd.DataFrame(rows)


def audio_signature_score(df):
    """Aggregate audio features into a normalized tonal profile score."""
    audio_features = [
        "centroid",
        "rolloff",
        "flux",
        "flatness",
        "loudness",
        "pitch",
        "rms",
        "zcr",
    ]
    subset = df[audio_features].copy()

    for col in audio_features:
        if col in subset.columns:
            numeric = pd.to_numeric(subset[col], errors="coerce")
            mean_val = numeric.mean()
            std_val = numeric.std()
            if pd.notna(mean_val) and std_val > 0:
                subset[col] = (numeric - mean_val) / std_val
            else:
                subset[col] = 0.0

    valid_mask = subset.notna().sum(axis=1) >= 3
    score = subset.mean(axis=1)
    score[~valid_mask] = np.nan
    return score


def artist_consistency_score(df, artist_col="id_author"):
    """Compute artist-level consistency based on variance of key metrics."""
    consistency = {}

    for artist_id in df[artist_col].unique():
        artist_tracks = df[df[artist_col] == artist_id]
        key_metrics = ["popularity", "duration_ms", "swear_density_total"]
        variances = []

        for metric in key_metrics:
            if metric in df.columns:
                values = pd.to_numeric(artist_tracks[metric], errors="coerce").dropna()
                if len(values) > 1:
                    variances.append(values.var())

        consistency[artist_id] = np.mean(variances) if variances else np.nan

    return df[artist_col].map(consistency)


def artist_geographic_diversity(df, artist_col="id_author", geo_col="region", missing_value=np.nan):
    """Compute artist geographic deviation from the dominant region."""
    if artist_col not in df.columns or geo_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    work = df[[artist_col, geo_col]].copy()
    work["_geo_norm"] = work[geo_col].map(normalize_text)
    work.loc[work["_geo_norm"] == "", "_geo_norm"] = np.nan

    # Keep one region per artist using the first non-missing value available.
    artist_region = (
        work.loc[work["_geo_norm"].notna(), [artist_col, "_geo_norm"]]
        .drop_duplicates(subset=[artist_col], keep="first")
        .copy()
    )

    all_artists = pd.Index(df[artist_col].drop_duplicates())
    by_artist = pd.Series(missing_value, index=all_artists)

    region_counts = artist_region["_geo_norm"].value_counts(dropna=True)
    if region_counts.empty:
        return df[artist_col].map(by_artist)

    dominant_count = float(region_counts.iloc[0])
    deviation_map = {
        region: 1.0 - (count / dominant_count)
        for region, count in region_counts.items()
    }

    artist_region["_geo_deviation"] = artist_region["_geo_norm"].map(deviation_map)
    artist_region["_geo_deviation"] = artist_region["_geo_deviation"].fillna(missing_value)

    by_artist.loc[artist_region[artist_col].values] = artist_region["_geo_deviation"].values
    return df[artist_col].map(by_artist)
