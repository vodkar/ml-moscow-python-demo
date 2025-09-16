from __future__ import annotations

"""Model building utilities."""

from sklearn.ensemble import RandomForestClassifier

from .config import TrainConfig


def build_classifier(cfg: TrainConfig) -> RandomForestClassifier:
    """Build a RandomForestClassifier based on configuration.

    Args:
        cfg: Training configuration.

    Returns:
        RandomForestClassifier: Configured classifier instance.
    """

    clf: RandomForestClassifier = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    return clf
