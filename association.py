"""Association-rule mining helper utilities."""

from __future__ import annotations

import pandas as pd


def find_frequent_itemsets(df, min_support):
    """Find frequent itemsets from a one-hot encoded dataframe."""
    df_bool = df.astype(bool)
    item_support = df_bool.sum(axis=0)
    frequent_singletons = [
        frozenset([item])
        for item, supp in item_support.items()
        if supp >= min_support
    ]

    itemsets = [
        (itemset, int(item_support[next(iter(itemset))]))
        for itemset in frequent_singletons
    ]
    current_l = frequent_singletons
    k = 2

    while current_l:
        candidates = []
        for i in range(len(current_l)):
            for j in range(i + 1, len(current_l)):
                candidate = current_l[i] | current_l[j]
                if len(candidate) == k:
                    candidates.append(candidate)
        candidates = list({candidate for candidate in candidates})

        next_l = []
        for candidate in candidates:
            support = int(df_bool[list(candidate)].all(axis=1).sum())
            if support >= min_support:
                next_l.append(candidate)
                itemsets.append((candidate, support))

        current_l = next_l
        k += 1

    return itemsets


def is_missing_category(value):
    """Return True for categories that should not become transaction items."""
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "<na>", "nan", "none", "missing"}


def row_to_transaction(row: pd.Series, categorical_columns: list) -> list:
    """Convert one row of discretized features into a list of item strings."""
    items = []
    for col in categorical_columns:
        value = row[col]
        if is_missing_category(value):
            continue
        feature = col.removeprefix("disc_")
        items.append(f"{feature}={value}")
    return items
