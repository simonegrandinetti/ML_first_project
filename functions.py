"""
Utility functions for ML project.
Import these functions into your notebook to keep cells clean.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from rapidfuzz import fuzz




def normalize_text(text):
    if pd.isna(text):
        return ""
    return " ".join(str(text).lower().strip().split())

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
