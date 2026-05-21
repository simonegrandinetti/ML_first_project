"""Clustering utilities used by Task 3 notebook cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score


def evaluate_kmeans_k_grid(
    X_scaled,
    k_values=range(2, 11),
    random_state=42,
    n_init=10,
    max_iter=300,
):
    """Evaluate K-Means across a fixed grid of cluster counts."""
    results = []
    for k in k_values:
        model = KMeans(
            n_clusters=int(k),
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels = model.fit_predict(X_scaled)
        results.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
            }
        )

    return pd.DataFrame(results)


def compute_cluster_profile_table(df, label_col, feature_columns):
    """Return mean feature profiles by cluster label."""
    required = [label_col, *feature_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing profiling columns: {missing}")

    return df.groupby(label_col)[feature_columns].mean()


def plot_cluster_boxplots(df, label_col, features_to_plot, title_prefix, palette="Set2"):
    """Plot a 2x2 boxplot grid for representative cluster features."""
    required = [label_col, *features_to_plot]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing boxplot columns: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        sns.boxplot(
            data=df,
            x=label_col,
            y=feature,
            hue=label_col,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"Distribution of {feature.upper()} by {title_prefix}")
        ax.set_xlabel(title_prefix)
        ax.set_ylabel(feature)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    return fig


def compute_cluster_projection_frame(X_scaled, random_state=42, tsne_n_jobs=-1):
    """Compute shared PCA and t-SNE 2D projections for clustering plots."""
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, random_state=random_state, n_jobs=tsne_n_jobs)
    X_tsne = tsne.fit_transform(X_scaled)

    return pd.DataFrame(
        {
            "PCA1": X_pca[:, 0],
            "PCA2": X_pca[:, 1],
            "tSNE1": X_tsne[:, 0],
            "tSNE2": X_tsne[:, 1],
        }
    )


def plot_cluster_projection_pairs(
    projection_df,
    labels,
    title_suffix="",
    palette=("black", "red", "yellow"),
):
    """Plot side-by-side PCA and t-SNE cluster projections."""
    if len(projection_df) != len(labels):
        raise ValueError("Projection rows and cluster labels must have the same length")

    plot_df = projection_df.copy()
    plot_df["Cluster"] = pd.Series(labels, index=projection_df.index)

    unique_clusters = sorted(plot_df["Cluster"].dropna().unique().tolist())
    if len(unique_clusters) > len(palette):
        raise ValueError("Not enough colors in the provided palette for the cluster labels")

    cluster_palette = list(palette)[: len(unique_clusters)]
    title_suffix = f" ({title_suffix})" if title_suffix else ""

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sns.scatterplot(
        data=plot_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette=cluster_palette,
        alpha=0.8,
        s=35,
        ax=axes[0],
    )
    axes[0].set_title(f"2D Cluster Projection via PCA{title_suffix}")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    sns.scatterplot(
        data=plot_df,
        x="tSNE1",
        y="tSNE2",
        hue="Cluster",
        palette=cluster_palette,
        alpha=0.8,
        s=35,
        ax=axes[1],
    )
    axes[1].set_title(f"2D Cluster Projection via t-SNE{title_suffix}")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
