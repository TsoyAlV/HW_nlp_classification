"""Этап 1: Разведочный анализ данных (EDA)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .config import ProjectConfig


def class_balance(trn: pd.DataFrame, cfg: ProjectConfig) -> pd.Series:
    """Распределение целевой переменной."""
    return trn[cfg.target_col].value_counts()


def missing_ratio(df: pd.DataFrame, col: str) -> float:
    """Доля пропусков по колонке."""
    return round(df[col].isna().sum() / len(df), 4)


def text_length_features(trn: pd.DataFrame, tst: pd.DataFrame, cfg: ProjectConfig):
    """Добавляет длины keyword/location (как в ноутбуке)."""
    for df in (trn, tst):
        df[f"len_{cfg.keyword_col}"] = df[cfg.keyword_col].apply(lambda x: len(str(x)))
        df[f"len_{cfg.location_col}"] = df[cfg.location_col].apply(lambda x: len(str(x)))
    return trn, tst


def plot_keyword_length(trn: pd.DataFrame, tst: pd.DataFrame, cfg: ProjectConfig):
    """Гистограмма длины keyword для trn/tst."""
    col = f"len_{cfg.keyword_col}"
    plt.hist(trn[col].sort_values().reset_index(drop=True).values, bins=40, alpha=0.6)
    plt.hist(tst[col].sort_values().reset_index(drop=True).values, bins=40, alpha=0.6)
    plt.ylabel("Частота")
    plt.xlabel("длина текста")
    plt.title('длина текста колонки "keywords"')
    plt.legend(["Выборка TRN", "Выборка TST"])


def run_eda(trn: pd.DataFrame, tst: pd.DataFrame, cfg: ProjectConfig) -> dict:
    """Сводит основные метрики EDA в словарь."""
    trn, tst = text_length_features(trn, tst, cfg)
    info = {
        "class_balance": class_balance(trn, cfg),
        "missing_keyword_trn": missing_ratio(trn, cfg.keyword_col),
        "missing_keyword_tst": missing_ratio(tst, cfg.keyword_col),
        "missing_location_trn": missing_ratio(trn, cfg.location_col),
        "missing_location_tst": missing_ratio(tst, cfg.location_col),
        "keyword_cutoff_coverage": len(trn[trn[f"len_{cfg.keyword_col}"] >= cfg.keyword_len_cutoff]) / len(trn),
    }
    return info