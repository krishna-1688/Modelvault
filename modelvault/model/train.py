"""Trains the target classifier and fits the reference density/manifold estimator.

The reference density model is what Layer 2's feature_distortion signal measures
distance against — it approximates "what the training manifold looks like" so we
can score how far an incoming query sits from real usage.
"""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from modelvault.model.data_prep import generate_dataset
from modelvault.utils.config_loader import get_settings
from modelvault.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIR = PROJECT_ROOT / "artifacts" / "target"
REFERENCE_DIR = PROJECT_ROOT / "artifacts" / "reference"


def train_target_model() -> None:
    settings = get_settings()
    X_train, X_test, y_train, y_test = generate_dataset()

    clf = LogisticRegression(max_iter=1000, random_state=settings.model.random_state)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    logger.info("Target model train accuracy: %.4f, test accuracy: %.4f", train_acc, test_acc)

    reference = GaussianMixture(
        n_components=2, random_state=settings.model.random_state, covariance_type="full"
    )
    reference.fit(X_train)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, TARGET_DIR / "target_classifier.joblib")
    joblib.dump(reference, REFERENCE_DIR / "reference_density.joblib")

    logger.info("Saved target classifier to %s", TARGET_DIR / "target_classifier.joblib")
    logger.info("Saved reference density model to %s", REFERENCE_DIR / "reference_density.joblib")

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")


if __name__ == "__main__":
    train_target_model()
