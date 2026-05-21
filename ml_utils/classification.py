"""Classification utilities used by Task 4 notebook cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split

from ml_utils.models import (
    train_decision_tree_classifier,
    train_random_forest_classifier,
    train_svm_classifier,
)


TASK4_TARGET_ORDER = ["North", "Center", "Campania", "Lazio", "Sardegna", "South"]

TASK4_REGION_TO_MACRO = {
    "Valle d'Aosta": "North",
    "Piemonte": "North",
    "Liguria": "North",
    "Lombardia": "North",
    "Trentino-Alto Adige": "North",
    "Veneto": "North",
    "Friuli-Venezia Giulia": "North",
    "Emilia-Romagna": "North",
    "Toscana": "Center",
    "Umbria": "Center",
    "Marche": "Center",
    "Lazio": "Lazio",
    "Campania": "Campania",
    "Abruzzo": "South",
    "Molise": "South",
    "Puglia": "South",
    "Basilicata": "South",
    "Calabria": "South",
    "Sicilia": "South",
    "Sardegna": "Sardegna",
}


def build_task4_target(df, region_col="region"):
    """Build Giuseppe-style Task 4 target columns from the raw region field."""
    if region_col not in df.columns:
        raise KeyError(f"Missing region column: {region_col}")

    target = pd.Series(
        pd.Categorical(
            df[region_col].map(TASK4_REGION_TO_MACRO),
            categories=TASK4_TARGET_ORDER,
            ordered=True,
        ),
        index=df.index,
        name="target_macroarea",
    )
    target_codes = (
        pd.Series(target.cat.codes, index=df.index, name="target_macroarea_code")
        .replace(-1, pd.NA)
        .astype("Int64")
    )

    return pd.DataFrame(
        {
            "target_macroarea": target,
            "target_macroarea_code": target_codes,
        },
        index=df.index,
    )


def plot_target_distribution(df, target_col="target_macroarea", palette="viridis"):
    """Plot Task 4 target-class distribution without seaborn palette warnings."""
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    plot_df = df.dropna(subset=[target_col]).copy()
    order = plot_df[target_col].value_counts().index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=plot_df,
        x=target_col,
        hue=target_col,
        order=order,
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title("Distribution of Tracks by Macro-Region (Target Variable)")
    ax.set_xlabel("Macro-Region")
    ax.set_ylabel("Number of Tracks")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig


def run_task4_feature_usefulness_analysis(
    df,
    excluded_cols,
    target_col="target_macroarea",
    variance_threshold=0.01,
    corr_threshold=0.85,
    random_state=42,
):
    """Run Giuseppe-style Task 4 feature usefulness analysis."""
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    numeric_features = [
        col
        for col in df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    df_classification = df.dropna(subset=numeric_features).copy()

    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(df_classification[numeric_features])
    variance_supported = [
        numeric_features[i]
        for i in range(len(numeric_features))
        if selector.get_support()[i]
    ]
    dropped_by_variance = sorted(set(numeric_features) - set(variance_supported))

    if variance_supported:
        corr_matrix = df_classification[variance_supported].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        dropped_by_correlation = [
            column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)
        ]
    else:
        dropped_by_correlation = []

    final_features = [col for col in variance_supported if col not in dropped_by_correlation]

    X_clf = df_classification[final_features]
    y_clf = df_classification[target_col]

    mi_scores_raw = mutual_info_classif(X_clf, y_clf, random_state=random_state)
    mi_min = mi_scores_raw.min()
    mi_max = mi_scores_raw.max()
    if mi_max != mi_min:
        mi_scores_normalized = (mi_scores_raw - mi_min) / (mi_max - mi_min)
    else:
        mi_scores_normalized = mi_scores_raw
    mi_series = pd.Series(mi_scores_normalized, index=final_features).sort_values(ascending=False)

    f_values, _ = f_classif(X_clf, y_clf)
    anova_series = pd.Series(f_values, index=final_features).sort_values(ascending=False)

    rf_baseline = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_baseline.fit(X_clf, y_clf)
    rf_series = pd.Series(rf_baseline.feature_importances_, index=final_features).sort_values(
        ascending=False
    )

    return {
        "df_classification": df_classification,
        "numeric_features": numeric_features,
        "variance_supported": variance_supported,
        "dropped_by_variance": dropped_by_variance,
        "dropped_by_correlation": dropped_by_correlation,
        "final_features": final_features,
        "mi_series": mi_series,
        "anova_series": anova_series,
        "rf_series": rf_series,
    }


def plot_task4_feature_usefulness(mi_series, anova_series, rf_series):
    """Plot the three Task 4 feature-usefulness rankings."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    sns.barplot(
        x=mi_series.values,
        y=mi_series.index,
        hue=mi_series.index,
        palette="mako",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("1. Feature Usefulness via Normalized Mutual Information")
    axes[0].set_xlabel("Normalized Information Gain (0.0 to 1.0)")
    axes[0].set_ylabel("")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    sns.barplot(
        x=anova_series.values,
        y=anova_series.index,
        hue=anova_series.index,
        palette="crest",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("2. Feature Usefulness via ANOVA F-Value")
    axes[1].set_xlabel("F-Value (Higher = More Distinct Means)")
    axes[1].set_ylabel("")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    sns.barplot(
        x=rf_series.values,
        y=rf_series.index,
        hue=rf_series.index,
        palette="flare",
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title("3. Feature Importance via Random Forest (MDI)")
    axes[2].set_xlabel("Importance Score (Sum = 1.0)")
    axes[2].set_ylabel("")
    axes[2].grid(axis="x", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def make_task4_stratified_split(
    df,
    feature_columns,
    target_col="target_macroarea",
    test_size=0.20,
    random_state=42,
):
    """Create the Task 4 stratified train/test split artifacts."""
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    X_final = df[feature_columns]
    y_final = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y_final,
        test_size=test_size,
        stratify=y_final,
        random_state=random_state,
    )

    return {
        "X_final": X_final,
        "y_final": y_final,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def make_task4_feature_ablation_split(split_artifacts, drop_features):
    """Drop a selected feature subset from an existing Task 4 split."""
    X_final = split_artifacts["X_final"]
    missing = sorted(set(drop_features) - set(X_final.columns))
    if missing:
        raise KeyError(f"Requested ablation features are missing from X_final: {missing}")

    ablated_artifacts = dict(split_artifacts)
    remaining_features = [col for col in X_final.columns if col not in drop_features]

    for key in ("X_final", "X_train", "X_test"):
        ablated_artifacts[key] = split_artifacts[key].drop(columns=drop_features)

    ablated_artifacts["dropped_features"] = list(drop_features)
    ablated_artifacts["remaining_features"] = remaining_features
    ablated_artifacts["n_features_before"] = int(X_final.shape[1])
    ablated_artifacts["n_features_after"] = int(len(remaining_features))
    return ablated_artifacts


def train_task4_classifier_suite(X_train, y_train, X_test, y_test):
    """Train the three baseline Task 4 classifiers with fixed settings."""
    dt_result = train_decision_tree_classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        criterion="gini",
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
    )

    rf_result = train_random_forest_classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    svm_result = train_svm_classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=False,
        scale=True,
    )

    results_by_model = {
        "Decision Tree": dt_result,
        "Random Forest": rf_result,
        "SVM": svm_result,
    }

    comparison_df = (
        pd.DataFrame(
            [
                {
                    "model": model_name,
                    "accuracy": result["accuracy"],
                    "f1_macro": result["f1_macro"],
                }
                for model_name, result in results_by_model.items()
            ]
        )
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "dt_result": dt_result,
        "rf_result": rf_result,
        "svm_result": svm_result,
        "results_by_model": results_by_model,
        "comparison_df": comparison_df,
    }
