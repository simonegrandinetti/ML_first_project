"""Correlation grouping and feature selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_correlation_groups(
    df,
    feature_columns,
    method="pearson",
    corr_threshold=0.85,
    min_periods=30,
):
    """Build redundancy groups from a correlation matrix using graph theory."""
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
    """Print a human-readable explanation of how correlation groups were formed."""
    print("\n" + "=" * 80)
    print("CORRELATION-BASED FEATURE GROUPING: STEP-BY-STEP EXPLANATION")
    print("=" * 80)

    print(f"\nThreshold: Absolute correlation >= {corr_threshold} -> features are 'connected'")
    print("\nHow it works:")
    print("  1. We compute all pairwise correlations")
    print("  2. Two features are linked if |correlation| >= threshold")
    print("  3. We find all groups where features are transitively linked")
    print("     (A links to B, B links to C -> A, B, C are in the same group)")
    print("  4. Each group is a set of redundant/multicollinear features")

    print("\n" + "-" * 80)
    print("GROUPS FOUND:")
    print("-" * 80)

    if not redundancy_groups:
        print("No correlated groups found (all features are independent).")
        return

    for group_idx, group_features in enumerate(redundancy_groups, start=1):
        print(f"\nGROUP {group_idx}: {len(group_features)} features")
        print(f"   Features: {', '.join(group_features)}")

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
                        marker = "Y" if abs(corr_val) >= corr_threshold else "N"
                        print(f"      {marker} {feat_a:25s} <-> {feat_b:25s}  : {corr_val:7.3f}")

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
    """Return a bounded stability score in [0, 1] from coefficient of variation."""
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
    """Compute a weighted score used to rank features inside correlated groups."""
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
        low_missingness = float(series.notna().mean())

        if target_numeric is not None:
            valid = series.notna() & target_numeric.notna()
            if valid.sum() >= 3:
                target_relevance = float(
                    abs(series[valid].corr(target_numeric[valid], method="pearson"))
                )
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
    """Select anchors and optional secondary features from correlated groups."""
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
