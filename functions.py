"""
Utility functions for ML project.
Import these functions into your notebook to keep cells clean.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

FEATURE_GROUPS = {
    "artist_level": [
        "id_author",
        "name",
        "gender",
        "birth_date",
        "description",
        "active_start",
        "active_end",
    ],
    "geographic": [
        "birth_place",
        "province",
        "region",
        "country",
        "nationality",
        "latitude",
        "longitude",
    ],
    "song_level": [
        "id_song",
        "id_artist",
        "name_artist",
        "full_title",
        "title",
        "featured_artists",
        "primary_artist",
        "language",
        "album",
        "album_name",
        "album_release_date",
        "album_type",
        "disc_number",
        "track_number",
        "duration_ms",
        "explicit",
        "popularity",
        "stats_pageviews",
        "album_image",
        "id_album",
        "lyrics",
        "modified_popularity",
        "year",
        "month",
        "day",
    ],
    "textual": [
        "swear_IT",
        "swear_EN",
        "swear_IT_words",
        "swear_EN_words",
        "n_sentences",
        "n_tokens",
        "tokens_per_sent",
        "char_per_tok",
        "lexical_density",
        "avg_token_per_clause",
    ],
    "audio": [
        "bpm",
        "centroid",
        "rolloff",
        "flux",
        "rms",
        "zcr",
        "flatness",
        "spectral_complexity",
        "pitch",
        "loudness",
    ],
}

ENGINEERED_FEATURE_DOCS = [
    {
        "feature": "swear_IT",
        "family": "textual",
        "role": "provided feature",
        "description": "Count of Italian swear words detected in the lyrics.",
    },
    {
        "feature": "swear_EN",
        "family": "textual",
        "role": "provided feature",
        "description": "Count of English swear words detected in the lyrics.",
    },
    {
        "feature": "swear_IT_words",
        "family": "textual",
        "role": "provided feature",
        "description": "List of matched Italian swear words found in the lyrics.",
    },
    {
        "feature": "swear_EN_words",
        "family": "textual",
        "role": "provided feature",
        "description": "List of matched English swear words found in the lyrics.",
    },
    {
        "feature": "n_sentences",
        "family": "textual",
        "role": "provided feature",
        "description": "Number of detected sentences in the lyrics.",
    },
    {
        "feature": "n_tokens",
        "family": "textual",
        "role": "provided feature",
        "description": "Number of detected tokens/words in the lyrics.",
    },
    {
        "feature": "tokens_per_sent",
        "family": "textual",
        "role": "provided feature",
        "description": "Average number of tokens per sentence.",
    },
    {
        "feature": "char_per_tok",
        "family": "textual",
        "role": "provided feature",
        "description": "Average number of characters per token.",
    },
    {
        "feature": "lexical_density",
        "family": "textual",
        "role": "provided feature",
        "description": "Ratio of lexical words over all tokens.",
    },
    {
        "feature": "avg_token_per_clause",
        "family": "textual",
        "role": "provided feature",
        "description": "Average number of tokens per clause.",
    },
    {
        "feature": "modified_popularity",
        "family": "song_level",
        "role": "provided feature",
        "description": "Alternative popularity-like score already included in the source data.",
    },
    {
        "feature": "centroid",
        "family": "audio",
        "role": "provided feature",
        "description": "Spectral centroid, a proxy for the brightness of the sound.",
    },
    {
        "feature": "rolloff",
        "family": "audio",
        "role": "provided feature",
        "description": "Spectral rolloff, the frequency below which most spectral energy is concentrated.",
    },
    {
        "feature": "flux",
        "family": "audio",
        "role": "provided feature",
        "description": "Spectral flux, measuring frame-to-frame spectral change.",
    },
    {
        "feature": "rms",
        "family": "audio",
        "role": "provided feature",
        "description": "Root mean square energy of the audio signal.",
    },
    {
        "feature": "zcr",
        "family": "audio",
        "role": "provided feature",
        "description": "Zero-crossing rate, often related to noisiness or percussiveness.",
    },
    {
        "feature": "flatness",
        "family": "audio",
        "role": "provided feature",
        "description": "Spectral flatness, indicating how tone-like versus noise-like the signal is.",
    },
    {
        "feature": "spectral_complexity",
        "family": "audio",
        "role": "provided feature",
        "description": "Count-based spectral complexity descriptor.",
    },
    {
        "feature": "pitch",
        "family": "audio",
        "role": "provided feature",
        "description": "Estimated pitch-related descriptor.",
    },
    {
        "feature": "loudness",
        "family": "audio",
        "role": "provided feature",
        "description": "Perceived loudness-related descriptor.",
    },
]

REGION_TO_MACROAREA = {
    "abruzzo": "South",
    "basilicata": "South",
    "calabria": "South",
    "campania": "South",
    "emilia romagna": "North",
    "friuli venezia giulia": "North",
    "lazio": "Center",
    "liguria": "North",
    "lombardia": "North",
    "marche": "Center",
    "molise": "South",
    "piemonte": "North",
    "puglia": "South",
    "sardegna": "Islands",
    "sicilia": "Islands",
    "toscana": "Center",
    "trentino alto adige": "North",
    "trentino alto adige/sudtirol": "North",
    "trentino alto adige sudtirol": "North",
    "umbria": "Center",
    "valle d aosta": "North",
    "valle d'aosta": "North",
    "veneto": "North",
}




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
    rows = []
    for group, columns in get_feature_groups(df).items():
        level = "artist" if group in {"artist_level", "geographic"} else "song"
        for column in columns:
            rows.append(
                {
                    "feature": column,
                    "group": group,
                    "level": level,
                }
            )
    return pd.DataFrame(rows)

def engineered_feature_table():
    """Return documentation for already engineered features in the source data."""
    return pd.DataFrame(ENGINEERED_FEATURE_DOCS)

def map_region_to_macroarea(region):
    """Map Italian regions to macro-areas used for regional comparisons."""
    key = normalize_text(region).replace("-", " ").replace("/", " ")
    key = " ".join(key.split())
    return REGION_TO_MACROAREA.get(key, "Missing")

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
    numeric = pd.to_numeric(series, errors="coerce").dropna()
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
        numeric = pd.to_numeric(df[column], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            continue

        q1, q3, lower, upper = iqr_bounds(numeric, whisker_width=whisker_width)
        iqr_mask = (numeric < lower) | (numeric > upper)

        if column in domain_rules:
            domain_mask = domain_rules[column](numeric)
            domain_mask = pd.Series(domain_mask, index=numeric.index).fillna(False).astype(bool)
        else:
            domain_mask = pd.Series(False, index=numeric.index)

        rows.append(
            {
                "feature": column,
                "valid_count": int(valid.count()),
                "q1": q1,
                "q3": q3,
                "iqr_lower": lower,
                "iqr_upper": upper,
                "iqr_outliers": int(iqr_mask.fillna(False).sum()),
                "iqr_outlier_pct": round(100 * iqr_mask.fillna(False).mean(), 2),
                "domain_outliers": int(domain_mask.sum()),
                "domain_outlier_pct": round(100 * domain_mask.mean(), 2),
                "min": valid.min(),
                "max": valid.max(),
            }
        )
    return pd.DataFrame(rows)

def compare_text_columns(df, col_a, col_b, method="token_set_ratio"):
    """Compare two unempty text columns in a DataFrame using fuzzy string matching.
    Parameters:
    - df: pandas DataFrame containing the columns to compare
    - col_a: name of the first text column
    - col_b: name of the second text column
    - method: fuzzy matching method to use (default: "token_set_ratio")"""
    # Drop rows where either column has NaN
    valid_idx = df[[col_a, col_b]].notna().all(axis=1)
    df_valid = df[valid_idx].copy()
    
    norm_a = df_valid[col_a].apply(normalize_text)
    norm_b = df_valid[col_b].apply(normalize_text)
    if method == "ratio":
        score = [fuzz.ratio(x, y) for x, y in zip(norm_a, norm_b)]
    elif method == "partial_ratio":
        score = [fuzz.partial_ratio(x, y) for x, y in zip(norm_a, norm_b)]
    elif method == "token_sort_ratio":
        score = [fuzz.token_sort_ratio(x, y) for x, y in zip(norm_a, norm_b)]
    elif method == "token_set_ratio":  # good default for text with extra words
        score = [fuzz.token_set_ratio(x, y) for x, y in zip(norm_a, norm_b)]
    else:
        raise ValueError(f"Unknown method: {method}")

    out = df_valid[[col_a, col_b]].copy()
    out["similarity_score"] = score
    avg_score = np.mean(score)
    return out, avg_score
