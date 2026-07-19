"""Этап 4: Векторизация текста (TF-IDF + числовые признаки)."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .config import ProjectConfig


def build_preprocessor(cfg: ProjectConfig, numeric_cols: list[str]):
    """ColumnTransformer: TF-IDF для текста + StandardScaler для числовых."""
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(
                max_features=cfg.max_features,
                ngram_range=cfg.ngram_range,
                max_df=cfg.max_df,
                min_df=cfg.min_df,
            ), "clean_text"),
            ("num", StandardScaler(), numeric_cols),
        ]
    )


def vectorize(
    trn: pd.DataFrame,
    val: pd.DataFrame,
    cfg: ProjectConfig,
    numeric_cols: list[str],
):
    """Обучает векторизатор на trn и трансформирует val."""
    pre = build_preprocessor(cfg, numeric_cols)
    X_trn = pre.fit_transform(trn)
    X_val = pre.transform(val)
    return X_trn, X_val, pre