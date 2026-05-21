"""Text comparison utilities."""

from __future__ import annotations

import numpy as np

from ml_utils.preprocessing import normalize_text


def compare_text_columns(df, col_a, col_b, method="token_set_ratio"):
    """Compare two non-empty text columns in a DataFrame using fuzzy matching."""
    try:
        from rapidfuzz import fuzz
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rapidfuzz is required for compare_text_columns. Install it with: conda install rapidfuzz"
        ) from exc

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
    elif method == "token_set_ratio":
        score = [fuzz.token_set_ratio(x, y) for x, y in zip(norm_a, norm_b)]
    else:
        raise ValueError(f"Unknown method: {method}")

    out = df_valid[[col_a, col_b]].copy()
    out["similarity_score"] = score
    avg_score = np.mean(score)
    return out, avg_score
