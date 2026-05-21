"""Model utilities for supervised downstream tasks built from unsupervised labels."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_for_unsupervised_downstream(
	data: pd.DataFrame,
	feature_columns: list[str],
	label_column: str,
	*,
	test_size: float = 0.2,
	random_state: int = 42,
	stratify: bool = True,
	drop_noise: bool = True,
	noise_label: int | str = -1,
	drop_na: bool = True,
	scale_for_svm: bool = True,
) -> dict[str, Any]:
	"""Split features and (pseudo) labels into train/test sets.

	This is designed for workflows where labels come from an unsupervised step
	(for example, clustering) and are then used to train supervised models
	such as Decision Tree, Random Forest, and SVM.
	"""

	required = set(feature_columns) | {label_column}
	missing = [col for col in required if col not in data.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	frame = data.loc[:, list(feature_columns) + [label_column]].copy()
	if drop_noise:
		frame = frame[frame[label_column] != noise_label]
	if drop_na:
		frame = frame.dropna(subset=list(feature_columns) + [label_column])

	if frame.empty:
		raise ValueError("No rows left after filtering. Check noise/NA settings.")

	X = frame[feature_columns]
	y = frame[label_column]

	can_stratify = y.nunique(dropna=False) > 1 and y.value_counts().min() > 1
	stratify_target = y if (stratify and can_stratify) else None

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify_target,
	)

	result: dict[str, Any] = {
		"X_train": X_train,
		"X_test": X_test,
		"y_train": y_train,
		"y_test": y_test,
		"used_stratify": stratify_target is not None,
	}

	if scale_for_svm:
		scaler = StandardScaler()
		result["X_train_svm"] = pd.DataFrame(
			scaler.fit_transform(X_train),
			index=X_train.index,
			columns=X_train.columns,
		)
		result["X_test_svm"] = pd.DataFrame(
			scaler.transform(X_test),
			index=X_test.index,
			columns=X_test.columns,
		)
		result["svm_scaler"] = scaler

	return result
