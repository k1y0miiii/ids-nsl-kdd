"""src/train.py — train models on NSL-KDD, evaluate, save figures + metrics."""

import os
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score

# Add project root to path so src.data is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_data

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save_confusion_matrix(
    cm,
    labels,
    title,
    filename,
):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[fig] Saved {path}")


def _save_f1_bar(
    per_class_f1,
    title,
    filename,
):
    classes = list(per_class_f1.keys())
    values = [per_class_f1[c] for c in classes]
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2), 4))
    bars = ax.bar(classes, values, color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[fig] Saved {path}")


def _save_feature_importances(
    importances,
    feature_names,
    filename,
    top_n=20,
):
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_names[::-1], top_vals[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"RandomForest — Top {top_n} feature importances")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[fig] Saved {path}")


def evaluate_binary(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    _save_confusion_matrix(
        cm, ["normal", "attack"],
        f"Binary CM — {model_name}",
        f"cm_binary_{model_name.lower().replace(' ', '_')}.png",
    )

    result = {
        "model": model_name,
        "task": "binary",
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
    }
    print(f"\n[binary] {model_name}")
    print(f"  CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    roc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    print(f"  Test accuracy: {acc:.4f}   F1-macro: {f1_macro:.4f}   ROC-AUC: {roc_str}")
    return result


def evaluate_multiclass(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
    feature_names,
    save_importances=False,
):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels_order = ["normal", "dos", "probe", "r2l", "u2r"]
    present_labels = [l for l in labels_order if l in y_test.values]

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0, labels=present_labels)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class F1
    f1_per = f1_score(y_test, y_pred, average=None, labels=present_labels, zero_division=0)
    per_class_f1 = {l: float(v) for l, v in zip(present_labels, f1_per)}

    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    _save_confusion_matrix(
        cm, present_labels,
        f"Multiclass CM — {model_name}",
        f"cm_multi_{model_name.lower().replace(' ', '_')}.png",
    )
    _save_f1_bar(
        per_class_f1,
        f"Per-class F1 — {model_name}",
        f"f1_bar_{model_name.lower().replace(' ', '_')}.png",
    )

    if save_importances and hasattr(model, "feature_importances_"):
        _save_feature_importances(
            model.feature_importances_,
            feature_names,
            "feature_importances_rf.png",
        )

    report = classification_report(
        y_test, y_pred, labels=present_labels, zero_division=0, output_dict=True
    )

    result = {
        "model": model_name,
        "task": "multiclass",
        "test_accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class_f1": per_class_f1,
        "classification_report": report,
    }
    print(f"\n[multi] {model_name}")
    print(f"  Test accuracy: {acc:.4f}   F1-macro: {f1_macro:.4f}")
    for cls, val in per_class_f1.items():
        print(f"    {cls:<10} F1: {val:.4f}")
    return result


def main():
    print("=" * 60)
    print("NSL-KDD Intrusion Detection -- Training & Evaluation")
    print("=" * 60)

    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test, feature_names = load_data()

    models_binary = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]

    models_multi = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]

    all_results = []

    print("\n--- BINARY CLASSIFICATION (normal vs attack) ---")
    for name, model in models_binary:
        r = evaluate_binary(model, X_train, X_test, y_bin_train, y_bin_test, name)
        all_results.append(r)

    print("\n--- MULTICLASS (normal / dos / probe / r2l / u2r) ---")
    for name, model in models_multi:
        save_imp = name == "RandomForest"
        r = evaluate_multiclass(
            model, X_train, X_test, y_multi_train, y_multi_test,
            name, feature_names, save_importances=save_imp,
        )
        all_results.append(r)

    # Save metrics
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[metrics] Saved to {metrics_path}")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE -- Binary")
    print(f"{'Model':<22} {'CV acc':>18} {'Test acc':>10} {'F1-macro':>10} {'ROC-AUC':>10}")
    print("-" * 72)
    for r in all_results:
        if r["task"] != "binary":
            continue
        roc = f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A"
        cv = f"{r['cv_accuracy_mean']:.4f}+/-{r['cv_accuracy_std']:.4f}"
        print(f"{r['model']:<22} {cv:>18} {r['test_accuracy']:>10.4f} {r['f1_macro']:>10.4f} {roc:>10}")

    print("\nSUMMARY TABLE -- Multiclass")
    print(f"{'Model':<22} {'Test acc':>10} {'F1-macro':>10} {'F1-weighted':>12}")
    print("-" * 56)
    for r in all_results:
        if r["task"] != "multiclass":
            continue
        print(f"{r['model']:<22} {r['test_accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r['f1_weighted']:>12.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
