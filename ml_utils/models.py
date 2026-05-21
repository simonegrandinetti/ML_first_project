"""Model utilities for supervised downstream tasks built from unsupervised labels."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
	"""Split features and pseudo-labels for downstream supervised training.

	Args:
		data: Input dataframe containing both feature and label columns.
		feature_columns: Feature columns to include in ``X``.
		label_column: Column containing labels (often cluster-derived labels).
		test_size: Fraction of rows assigned to the test split.
		random_state: Random seed passed to ``train_test_split``.
		stratify: If ``True``, use label stratification when class counts allow it.
		drop_noise: If ``True``, remove rows whose label equals ``noise_label``.
		noise_label: Label value considered noise (e.g., ``-1`` for DBSCAN).
		drop_na: If ``True``, drop rows with missing values in selected columns.
		scale_for_svm: If ``True``, also compute standardized train/test features
			for SVM workflows.

	Returns:
		A dictionary with:
			- ``X_train``, ``X_test``, ``y_train``, ``y_test``
			- ``used_stratify`` (bool)
			- when ``scale_for_svm=True``: ``X_train_svm``, ``X_test_svm``,
			  and ``svm_scaler``.

	Raises:
		ValueError: If required columns are missing or filtering removes all rows.
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


def _evaluate_classifier(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
	"""Evaluate a fitted classifier on a held-out test set.

	Args:
		model: Fitted sklearn-compatible estimator exposing ``predict``.
		X_test: Test features.
		y_test: True test labels.

	Returns:
		A dictionary with model object, predictions, accuracy, macro-F1,
		classification report text, and confusion matrix.
	"""
	y_pred = model.predict(X_test)
	return {
		"model": model,
		"y_pred": y_pred,
		"accuracy": float(accuracy_score(y_test, y_pred)),
		"f1_macro": float(f1_score(y_test, y_pred, average="macro")),
		"classification_report": classification_report(y_test, y_pred),
		"confusion_matrix": confusion_matrix(y_test, y_pred),
	}


def train_decision_tree_classifier(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	*,
	criterion: Literal["gini", "entropy", "log_loss"] = "gini",
	max_depth: int | None = None,
	min_samples_leaf: int = 1,
	random_state: int = 42,
) -> dict[str, Any]:
	"""Train and evaluate a ``DecisionTreeClassifier``.

	Args:
		X_train: Training feature matrix.
		y_train: Training labels.
		X_test: Test feature matrix.
		y_test: Test labels.
		criterion: Split-quality metric (``gini``, ``entropy``, ``log_loss``).
		max_depth: Maximum tree depth; ``None`` grows full tree.
		min_samples_leaf: Minimum samples required in each leaf.
		random_state: Seed for reproducible tree construction.

	Returns:
		Evaluation dictionary from ``_evaluate_classifier``.
	"""
	model = DecisionTreeClassifier(
		criterion=criterion,
		max_depth=max_depth,
		min_samples_leaf=min_samples_leaf,
		random_state=random_state,
	)
	model.fit(X_train, y_train)
	return _evaluate_classifier(model, X_test, y_test)


def train_random_forest_classifier(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	*,
	n_estimators: int = 300,
	max_depth: int | None = None,
	min_samples_leaf: int = 1,
	random_state: int = 42,
	n_jobs: int = -1,
) -> dict[str, Any]:
	"""Train and evaluate a ``RandomForestClassifier``.

	Args:
		X_train: Training feature matrix.
		y_train: Training labels.
		X_test: Test feature matrix.
		y_test: Test labels.
		n_estimators: Number of trees in the forest.
		max_depth: Maximum depth per tree; ``None`` allows full depth.
		min_samples_leaf: Minimum samples required in each leaf.
		random_state: Seed for reproducible ensemble construction.
		n_jobs: Parallel jobs used by sklearn (``-1`` uses all cores).

	Returns:
		Evaluation dictionary from ``_evaluate_classifier``.
	"""
	model = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		min_samples_leaf=min_samples_leaf,
		random_state=random_state,
		n_jobs=n_jobs,
	)
	model.fit(X_train, y_train)
	return _evaluate_classifier(model, X_test, y_test)


def train_svm_classifier(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	*,
	C: float = 1.0,
	kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
	gamma: float | Literal["scale", "auto"] = "scale",
	probability: bool = False,
	scale: bool = True,
) -> dict[str, Any]:
	"""Train and evaluate an ``SVC`` model with optional feature scaling.

	Args:
		X_train: Training feature matrix.
		y_train: Training labels.
		X_test: Test feature matrix.
		y_test: Test labels.
		C: Regularization strength.
		kernel: SVM kernel type.
		gamma: Kernel coefficient for ``rbf``, ``poly``, and ``sigmoid``.
		probability: If ``True``, enable probability calibration in SVC.
		scale: If ``True``, apply ``StandardScaler`` before fitting/predicting.

	Returns:
		Evaluation dictionary from ``_evaluate_classifier`` plus ``scaler``
		(set to ``None`` when ``scale=False``).
	"""
	if scale:
		scaler = StandardScaler()
		X_train_fit = pd.DataFrame(
			scaler.fit_transform(X_train),
			index=X_train.index,
			columns=X_train.columns,
		)
		X_test_fit = pd.DataFrame(
			scaler.transform(X_test),
			index=X_test.index,
			columns=X_test.columns,
		)
	else:
		scaler = None
		X_train_fit = X_train
		X_test_fit = X_test

	model = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)
	model.fit(X_train_fit, y_train)

	result = _evaluate_classifier(model, X_test_fit, y_test)
	result["scaler"] = scaler
	return result
