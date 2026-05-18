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

        # Percentages should be measured on valid numeric rows only.
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

#def summarize_outlier(df, columns, domain_rules=None, whisker_width=1.5):
    #"""Backward-compatible wrapper for summarize_outliers."""
    #return summarize_outliers(
    #    df,
    #    columns=columns,
    #    domain_rules=domain_rules,
    #    whisker_width=whisker_width,
    #)

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

def audio_signature_score(df):
    """Aggregate audio features into a normalized tonal profile score."""
    audio_features = ["centroid", "rolloff", "flux", "flatness", "loudness", "pitch", "rms", "zcr"]
    subset = df[audio_features].copy()
    
    # Standardize each feature and compute mean across available features
    for col in audio_features:
        if col in subset.columns:
            numeric = pd.to_numeric(subset[col], errors="coerce")
            mean_val = numeric.mean()
            std_val = numeric.std()
            if pd.notna(mean_val) and std_val > 0:
                subset[col] = (numeric - mean_val) / std_val
            else:
                subset[col] = 0.0
    
    # Return mean z-score across audio features (rows with at least 3 valid features)
    valid_mask = subset.notna().sum(axis=1) >= 3
    score = subset.mean(axis=1)
    score[~valid_mask] = np.nan
    return score

def artist_consistency_score(df, artist_col="id_author"):
    """Compute artist-level consistency based on variance of key metrics."""
    consistency = {}
    
    for artist_id in df[artist_col].unique():
        artist_tracks = df[df[artist_col] == artist_id]
        
        # Compute variance of key metrics for this artist
        key_metrics = ["popularity", "duration_ms", "swear_density_total"]
        variances = []
        
        for metric in key_metrics:
            if metric in df.columns:
                values = pd.to_numeric(artist_tracks[metric], errors="coerce").dropna()
                if len(values) > 1:
                    variances.append(values.var())
        
        # Mean variance across metrics; higher = more erratic, lower = more consistent
        consistency[artist_id] = np.mean(variances) if variances else np.nan
    
    # Map back to dataframe (lower scores = more consistent)
    return df[artist_col].map(consistency)

def artist_geographic_diversity(df, artist_col="id_author", geo_col="region", missing_value=np.nan):
    """Compute artist geographic deviation from the dominant region.

    The score is in [0, 1]:
    - 0.0 for artists in the most represented region
    - larger values for less represented regions
    - missing_value for artists with missing geographic origin
    """
    if artist_col not in df.columns or geo_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    artist_region = df[[artist_col, geo_col]].drop_duplicates(subset=[artist_col]).copy()
    artist_region["_geo_norm"] = artist_region[geo_col].map(normalize_text)
    artist_region.loc[artist_region["_geo_norm"] == "", "_geo_norm"] = np.nan

    region_counts = artist_region["_geo_norm"].value_counts(dropna=True)
    if region_counts.empty:
        return pd.Series(missing_value, index=df.index)

    dominant_count = float(region_counts.iloc[0])
    deviation_map = {
        region: 1.0 - (count / dominant_count)
        for region, count in region_counts.items()
    }

    artist_region["_geo_deviation"] = artist_region["_geo_norm"].map(deviation_map)
    artist_region["_geo_deviation"] = artist_region["_geo_deviation"].fillna(missing_value)

    by_artist = artist_region.set_index(artist_col)["_geo_deviation"]
    return df[artist_col].map(by_artist)


def build_correlation_groups(
    df,
    feature_columns,
    method="pearson",
    corr_threshold=0.85,
    min_periods=30,
):
    """Build redundancy groups from a correlation matrix using graph theory.

    Algorithm Overview (Plain English):
    -----------------------------------
    1. Compute correlations: Calculate pairwise Pearson or Spearman correlations
       between all candidate features, ignoring pairs with < min_periods overlaps.
    
    2. Build an adjacency graph: Two features are "connected" if their absolute
       correlation is >= corr_threshold. Think of features as nodes in a graph,
       and correlations as edges connecting them.
    
    3. Find connected components: Use depth-first search (DFS) to find all groups
       of features that are transitively related. For example:
       - If A correlates with B (r=0.90) AND B correlates with C (r=0.88),
         then A, B, C form one group even if A and C don't directly correlate.
    
    4. Sort and return: Larger groups come first; within ties, groups are sorted
       alphabetically by feature name for reproducibility.

    Why this works:
    - Captures indirect relationships: features connected through intermediaries
    - More powerful than simple threshold filtering on pairs
    - Groups represent actual redundancy (multicollinearity) in your data

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    feature_columns : list[str]
        Candidate features to analyze.
    method : str
        'pearson', 'spearman', or 'kendall' (passed to DataFrame.corr).
    corr_threshold : float
        Absolute correlation magnitude (0 to 1) above which features are
        considered "connected". Common: 0.85 or 0.90.
    min_periods : int
        Minimum overlapping non-null observations required to compute a
        correlation between two features.

    Returns
    -------
    corr : pandas.DataFrame
        Full correlation matrix (n_features × n_features).
    groups : list[list[str]]
        Each inner list is a redundancy group (sorted by keep_score later).
        Singleton groups are excluded (len >= 2 only).
    """
    available = [col for col in feature_columns if col in df.columns]
    if not available:
        return pd.DataFrame(), []

    numeric_df = df[available].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(method=method, min_periods=min_periods)

    adjacency = {col: set() for col in corr.columns}
    cols = list(corr.columns)
    for i, col_i in enumerate(cols):
        for col_j in cols[i + 1 :]:
            value = corr.at[col_i, col_j]
            if pd.notna(value) and abs(value) >= corr_threshold:
                adjacency[col_i].add(col_j)
                adjacency[col_j].add(col_i)

    visited = set()
    groups = []
    for node in cols:
        if node in visited:
            continue
        stack = [node]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)

        if len(component) > 1:
            groups.append(sorted(component))

    groups.sort(key=lambda g: (-len(g), g))
    return corr, groups


def explain_correlation_groups_readable(
    corr_matrix,
    redundancy_groups,
    corr_threshold=0.85,
):
    """Print a human-readable explanation of how correlation groups were formed.

    This function breaks down the grouping process in plain language, showing:
    - Which features are connected within each group
    - The actual correlation values linking them
    - Why they were grouped together
    """
    print("\n" + "=" * 80)
    print("CORRELATION-BASED FEATURE GROUPING: STEP-BY-STEP EXPLANATION")
    print("=" * 80)

    print(f"\nThreshold: Absolute correlation >= {corr_threshold} → features are 'connected'")
    print("\nHow it works:")
    print("  1. We compute all pairwise correlations")
    print("  2. Two features are linked if |correlation| >= threshold")
    print("  3. We find all groups where features are transitively linked")
    print("     (A links to B, B links to C → A, B, C are in the same group)")
    print("  4. Each group is a set of redundant/multicollinear features")

    print("\n" + "-" * 80)
    print("GROUPS FOUND:")
    print("-" * 80)

    if not redundancy_groups:
        print("No correlated groups found (all features are independent).")
        return

    for group_idx, group_features in enumerate(redundancy_groups, start=1):
        print(f"\n📊 GROUP {group_idx}: {len(group_features)} features")
        print(f"   Features: {', '.join(group_features)}")

        # Show internal correlations within this group
        print(f"\n   Internal correlations (within this group):")
        shown_pairs = set()
        for i, feat_a in enumerate(group_features):
            for feat_b in group_features[i + 1 :]:
                pair_key = tuple(sorted([feat_a, feat_b]))
                if pair_key in shown_pairs:
                    continue
                shown_pairs.add(pair_key)

                if feat_a in corr_matrix.index and feat_b in corr_matrix.columns:
                    corr_val = corr_matrix.at[feat_a, feat_b]
                    if pd.notna(corr_val):
                        marker = "✓" if abs(corr_val) >= corr_threshold else "○"
                        print(f"      {marker} {feat_a:25s} ↔ {feat_b:25s}  : {corr_val:7.3f}")

        # Explain why these features are grouped
        print(f"\n   Why grouped together:")
        print(f"   - At least one correlation within the group >= {corr_threshold}")
        print(f"   - Features may not all correlate directly, but are linked transitively")
        print(f"   - Example: if A-B correlate strongly and B-C correlate strongly,")
        print(f"     then A, B, C are grouped even if A-C might have lower correlation")

    print("\n" + "=" * 80)
    print("NEXT STEP: For each group, we rank features by 'keep_score' and retain")
    print("the highest-scoring anchor. Optionally keep a secondary feature if it")
    print("adds distinct signal (different semantic meaning or target relevance).")
    print("=" * 80 + "\n")


def _bounded_stability_score(series):
    """Return a bounded stability score in [0, 1] from coefficient of variation.

    Higher score means less dispersion relative to the mean.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0

    mean_abs = float(abs(numeric.mean()))
    std = float(numeric.std(ddof=0))
    if mean_abs == 0:
        return 1.0 if std == 0 else 0.0

    cv = std / mean_abs
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def compute_feature_keep_scores(
    df,
    feature_columns,
    target_column=None,
    semantic_group_map=None,
    interpretability_map=None,
    weights=None,
):
    """Compute a weighted score used to rank features inside correlated groups.

    The default scoring mirrors the criterion discussed in the notebook:
    score = 0.50 * target_relevance
          + 0.20 * low_missingness
          + 0.20 * stability
          + 0.10 * interpretability

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    feature_columns : list[str]
        Candidate features to score.
    target_column : str | None
        Optional target used for relevance as absolute Pearson correlation.
    semantic_group_map : dict[str, str] | None
        Optional semantic feature family labels.
    interpretability_map : dict[str, float] | None
        Optional interpretability priors in [0, 1]. Missing entries default to 0.5.
    weights : dict[str, float] | None
        Optional custom weights for keys:
        target_relevance, low_missingness, stability, interpretability.
    """
    semantic_group_map = semantic_group_map or {}
    interpretability_map = interpretability_map or {}
    default_weights = {
        "target_relevance": 0.50,
        "low_missingness": 0.20,
        "stability": 0.20,
        "interpretability": 0.10,
    }
    if weights:
        default_weights.update(weights)
    weight_sum = sum(default_weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")

    normalized_weights = {k: v / weight_sum for k, v in default_weights.items()}
    available = [col for col in feature_columns if col in df.columns]

    target_numeric = None
    if target_column and target_column in df.columns:
        target_numeric = pd.to_numeric(df[target_column], errors="coerce")

    rows = []
    for feature in available:
        series = pd.to_numeric(df[feature], errors="coerce")
        non_missing = float(series.notna().mean())
        low_missingness = non_missing

        if target_numeric is not None:
            valid = series.notna() & target_numeric.notna()
            if valid.sum() >= 3:
                target_relevance = float(abs(series[valid].corr(target_numeric[valid], method="pearson")))
            else:
                target_relevance = 0.0
        else:
            target_relevance = 0.0

        stability = _bounded_stability_score(series)
        interpretability = float(np.clip(interpretability_map.get(feature, 0.5), 0.0, 1.0))

        score = (
            normalized_weights["target_relevance"] * target_relevance
            + normalized_weights["low_missingness"] * low_missingness
            + normalized_weights["stability"] * stability
            + normalized_weights["interpretability"] * interpretability
        )

        rows.append(
            {
                "feature": feature,
                "semantic_group": semantic_group_map.get(feature, "unknown"),
                "target_relevance": target_relevance,
                "low_missingness": low_missingness,
                "stability": stability,
                "interpretability": interpretability,
                "keep_score": float(score),
            }
        )

    return pd.DataFrame(rows).sort_values("keep_score", ascending=False).reset_index(drop=True)


def select_redundant_features(
    corr_matrix,
    redundancy_groups,
    score_table,
    semantic_group_map=None,
    target_corr_delta=0.05,
    hard_drop_threshold=0.95,
    max_keep_per_group=2,
):
    """Select anchors and optional second features from correlated groups.

    Implements points 1-4 of the proposed strategy:
    1) use correlation groups,
    2) rank by keep_score,
    3) keep an anchor per group,
    4) optionally keep a second feature if it adds distinct value.

    A second feature is retained when it is highly correlated with the anchor
    but belongs to a different semantic group OR differs enough in target
    relevance (abs delta >= target_corr_delta).
    """
    semantic_group_map = semantic_group_map or {}
    if score_table.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    score_lookup = score_table.set_index("feature")
    selected = []
    decisions = []

    for group_id, group_features in enumerate(redundancy_groups, start=1):
        ranked = [f for f in score_lookup.index if f in group_features]
        if not ranked:
            continue
        ranked = sorted(ranked, key=lambda f: score_lookup.at[f, "keep_score"], reverse=True)

        anchor = ranked[0]
        selected.append(anchor)
        decisions.append(
            {
                "group_id": group_id,
                "feature": anchor,
                "decision": "keep_anchor",
                "reason": "highest_keep_score_in_group",
            }
        )

        kept_in_group = [anchor]
        for candidate in ranked[1:]:
            if len(kept_in_group) >= max_keep_per_group:
                decisions.append(
                    {
                        "group_id": group_id,
                        "feature": candidate,
                        "decision": "drop",
                        "reason": "max_keep_per_group_reached",
                    }
                )
                continue

            pair_corr = np.nan
            if (
                anchor in corr_matrix.index
                and candidate in corr_matrix.columns
                and anchor in corr_matrix.columns
                and candidate in corr_matrix.index
            ):
                pair_corr = corr_matrix.at[anchor, candidate]

            anchor_target = float(score_lookup.at[anchor, "target_relevance"])
            cand_target = float(score_lookup.at[candidate, "target_relevance"])
            target_gap_ok = abs(cand_target - anchor_target) >= target_corr_delta

            semantic_diff = (
                semantic_group_map.get(anchor, "unknown")
                != semantic_group_map.get(candidate, "unknown")
            )

            high_corr = pd.notna(pair_corr) and abs(pair_corr) >= hard_drop_threshold
            keep_second = (not high_corr) or semantic_diff or target_gap_ok

            if keep_second:
                kept_in_group.append(candidate)
                selected.append(candidate)
                reason_parts = []
                if not high_corr:
                    reason_parts.append("corr_below_hard_drop")
                if semantic_diff:
                    reason_parts.append("different_semantic_group")
                if target_gap_ok:
                    reason_parts.append("target_relevance_gap")
                decisions.append(
                    {
                        "group_id": group_id,
                        "feature": candidate,
                        "decision": "keep_secondary",
                        "reason": "|".join(reason_parts) if reason_parts else "rule_based_keep",
                    }
                )
            else:
                decisions.append(
                    {
                        "group_id": group_id,
                        "feature": candidate,
                        "decision": "drop",
                        "reason": "high_correlation_without_distinct_signal",
                    }
                )

    decision_table = pd.DataFrame(decisions)
    selected_unique = sorted(set(selected))
    selected_score_table = score_table[score_table["feature"].isin(selected_unique)].copy()
    selected_score_table = selected_score_table.sort_values("keep_score", ascending=False)
    return selected_score_table, decision_table, selected_unique
