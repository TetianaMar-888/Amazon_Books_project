"""Evaluation utilities: metrics, confusion matrices, and experiment tracking."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay
)

LABELS_ORDER = ["negative", "neutral", "positive"]


def evaluate_model(y_true, y_pred, model_name: str, verbose: bool = True) -> dict:
    """Compute standard classification metrics and optionally print a report."""
    if verbose:
        print(f"=== {model_name} ===")
        print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str = None, cmap="Blues"):
    """Plot and optionally save a confusion matrix for one model."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS_ORDER)
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def append_experiment(results_path: str, row: dict) -> pd.DataFrame:
    """Append a new row to the experiments tracking CSV, avoiding duplicates."""
    try:
        results_df = pd.read_csv(results_path)
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        results_df = pd.DataFrame([row])

    results_df = results_df.drop_duplicates(subset=["model", "macro_f1"], keep="last")
    results_df.to_csv(results_path, index=False)
    return results_df


def get_misclassified_examples(texts, y_true, y_pred, n=5, critical_only=True):
    """Return a DataFrame of misclassified examples, optionally only 'hard' errors
    (negative<->positive confusion)."""
    df = pd.DataFrame({"text": texts, "true_label": y_true, "predicted": y_pred})
    errors = df[df["true_label"] != df["predicted"]]

    if critical_only:
        errors = errors[
            ((errors["true_label"] == "negative") & (errors["predicted"] == "positive")) |
            ((errors["true_label"] == "positive") & (errors["predicted"] == "negative"))
        ]
    return errors.head(n)
