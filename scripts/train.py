"""Этапы 5-6: Обучение и подбор гиперпараметров (F1-метрика)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

from .config import ProjectConfig


def get_models(cfg: ProjectConfig):
    """Фабрика базовых моделей (этап 5 ноутбука)."""
    models = {}
    if "logreg" in cfg.model_list:
        models["logreg"] = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=cfg.random_state
        )
    if "multinomial_nb" in cfg.model_list:
        models["multinomial_nb"] = MultinomialNB()
    # XGBoost/LightGBM подключаются опционально (см. ниже)
    return models


def split_train_val(trn: pd.DataFrame, cfg: ProjectConfig):
    """Стратифицированный сплит 80/20 (этап 4 ноутбука)."""
    strat = trn[cfg.target_col] if cfg.stratify_on_target else None
    return train_test_split(
        trn, test_size=cfg.test_size, random_state=cfg.random_state, stratify=strat
    )


def tune_model(model, param_grid: dict, X, y, cfg: ProjectConfig) -> GridSearchCV:
    """GridSearchCV по F1 (этап 6)."""
    gs = GridSearchCV(model, param_grid, scoring=cfg.tune_metric, cv=3, n_jobs=-1)
    gs.fit(X, y)
    return gs


def train_all(X_trn, y_trn, cfg: ProjectConfig) -> dict:
    """Обучает все модели из cfg.model_list (без тюнинга)."""
    results = {}
    for name, model in get_models(cfg).items():
        model.fit(X_trn, y_trn)
        results[name] = model
    return results