"""Trains the target fraud classifier and fits the reference density/manifold
estimator used by Layer 2's macro signal.

Three artifacts are produced and must travel together at serving time:
  - scaler.joblib: a StandardScaler fit on the raw training features. V1-V28
    are already PCA components from the original release (roughly
    zero-mean), but Amount is on a wildly different scale (0 to tens of
    thousands) -- without scaling it would dominate both the classifier's
    splits and the reference density's distance metric.
  - target_classifier.joblib: a RandomForestClassifier trained on SCALED
    features. Fraud detection is a canonical severe-class-imbalance problem
    (0.17% positive here), so accuracy alone is close to meaningless -- a
    model that always predicts "not fraud" scores >99.8% accuracy while
    being useless. class_weight="balanced_subsample" and precision/recall/
    F1/ROC-AUC/PR-AUC reporting below are the industry-standard way to
    evaluate this instead.
  - reference_density.joblib: a GaussianMixture fit on SCALED features,
    used to estimate how far an incoming query sits from real transaction
    traffic (Layer 2's macro signal).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from modelvault.model.data_prep import generate_dataset
from modelvault.utils.config_loader import get_settings
from modelvault.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIR = PROJECT_ROOT / "artifacts" / "target"
REFERENCE_DIR = PROJECT_ROOT / "artifacts" / "reference"

GMM_CALIBRATION_SUBSAMPLE = 20_000


def train_target_model() -> None:
    settings = get_settings()
    X_train, X_test, y_train, y_test = generate_dataset(random_state=settings.model.random_state)

    logger.info("Training set: %d rows, %d fraud (%.3f%%)", len(y_train), y_train.sum(), 100 * y_train.mean())
    logger.info("Test set:     %d rows, %d fraud (%.3f%%)", len(y_test), y_test.sum(), 100 * y_test.mean())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced_subsample",
        random_state=settings.model.random_state,
        n_jobs=-1,  # fast parallel fit
    )
    clf.fit(X_train_scaled, y_train)

    # n_jobs is a plain attribute read fresh on every predict call, not baked
    # into the fitted trees -- so we can fit in parallel (fast) and then
    # serve single-row requests without joblib's parallel-dispatch overhead
    # (measured ~5x faster per call: ~41ms at n_jobs=-1 vs ~8.5ms at n_jobs=1
    # for a single-row predict_proba). A live gateway serves one row per
    # request, so single-row latency is what actually matters here, not
    # multi-row throughput.
    clf.n_jobs = 1

    test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    test_preds = clf.predict(X_test_scaled)
    metrics = {
        "precision": precision_score(y_test, test_preds),
        "recall": recall_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "pr_auc": average_precision_score(y_test, test_probs),
    }
    for name, value in metrics.items():
        logger.info("Test %s: %.4f", name, value)

    # Fit the reference density on a subsample -- GaussianMixture EM over the
    # full ~228k-row training set is unnecessarily slow to re-run and a
    # subsample of this size already estimates the manifold reliably.
    rng = np.random.default_rng(settings.model.random_state)
    subsample_size = min(GMM_CALIBRATION_SUBSAMPLE, len(X_train_scaled))
    subsample_idx = rng.choice(len(X_train_scaled), size=subsample_size, replace=False)

    reference = GaussianMixture(
        n_components=2, random_state=settings.model.random_state, covariance_type="full"
    )
    reference.fit(X_train_scaled[subsample_idx])

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, TARGET_DIR / "scaler.joblib")
    joblib.dump(clf, TARGET_DIR / "target_classifier.joblib")
    joblib.dump(reference, REFERENCE_DIR / "reference_density.joblib")

    logger.info("Saved scaler to %s", TARGET_DIR / "scaler.joblib")
    logger.info("Saved target classifier to %s", TARGET_DIR / "target_classifier.joblib")
    logger.info("Saved reference density model to %s", REFERENCE_DIR / "reference_density.joblib")

    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    train_target_model()
