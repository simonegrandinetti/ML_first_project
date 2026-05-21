"""Utility package for data cleaning, profiling, and feature engineering."""

from ml_utils.association import (
    find_frequent_itemsets,
    is_missing_category,
    row_to_transaction,
)
from ml_utils.classification import (
    build_task4_target,
    make_task4_feature_ablation_split,
    make_task4_stratified_split,
    plot_target_distribution,
    plot_task4_feature_usefulness,
    run_task4_feature_usefulness_analysis,
    train_task4_classifier_suite,
)
from ml_utils.clustering import (
    compute_cluster_profile_table,
    compute_cluster_projection_frame,
    evaluate_kmeans_k_grid,
    plot_cluster_boxplots,
    plot_cluster_projection_pairs,
)
from ml_utils.constants import (
    ARTISTS_DATETIME_COLUMNS,
    ARTISTS_DTYPE_MAP,
    BLANK_PLACEHOLDERS,
    DATE_COMPONENT_COLUMNS,
    DOMAIN_CATEGORICAL_RULES,
    DOMAIN_CONSISTENCY_RULES,
    DOMAIN_PLAUSIBILITY_RULES,
    ENGINEERED_FEATURE_DOCS,
    FEATURE_GROUPS,
    MERGED_DATETIME_COLUMNS,
    MERGED_DTYPE_MAP,
    REGION_TO_MACROAREA,
    TRACKS_DATETIME_COLUMNS,
    TRACKS_DTYPE_MAP,
)
from ml_utils.correlation_selection import (
    build_correlation_groups,
    compute_feature_keep_scores,
    explain_correlation_groups_readable,
    select_redundant_features,
)
from ml_utils.models import (
    split_for_unsupervised_downstream,
    train_decision_tree_classifier,
    train_random_forest_classifier,
    train_svm_classifier,
)
from ml_utils.numeric_stats import (
    artist_consistency_score,
    artist_geographic_diversity,
    audio_signature_score,
    build_weighted_zscore,
    compute_mer_mood_features,
    iqr_bounds,
    safe_divide,
    summarize_outliers,
    zscore_series,
)
from ml_utils.preprocessing import (
    apply_domain_plausibility,
    clean_and_type_dataframe,
    compute_final_release_year,
    derive_date_fields,
    engineered_feature_table,
    feature_group_table,
    get_feature_groups,
    map_region_to_macroarea,
    normalize_text,
)
from ml_utils.text_matching import compare_text_columns
