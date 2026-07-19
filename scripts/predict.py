"""Этап 9: Генерация предсказаний и submission.csv."""
from __future__ import annotations

import pandas as pd

from .config import ProjectConfig


def generate_submission(
    model,
    X_tst,
    sample_submission: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    """Формирует submission с колонками id и target (0/1)."""
    y_proba = model.predict_proba(X_tst)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    sub = sample_submission.copy()
    sub[cfg.target_col] = y_pred
    return sub


def save_submission(sub: pd.DataFrame, cfg: ProjectConfig) -> None:
    sub.to_csv(cfg.submission_path, index=False)
    print(f"Submission сохранён: {cfg.submission_path}")