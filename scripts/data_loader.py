"""Этап 0: Подготовка окружения и загрузка данных."""
from __future__ import annotations

import pandas as pd

from .config import ProjectConfig


def load_raw_data(cfg: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает trn, tst и sample_submission из data/.

    Returns:
        (trn, tst, sample_submission)
    """
    trn = pd.read_csv(cfg.trn_path)
    tst = pd.read_csv(cfg.tst_path)
    sample_submission = pd.read_csv(cfg.sample_submission_path)
    return trn, tst, sample_submission


def fill_missing_tokens(
    df: pd.DataFrame, cfg: ProjectConfig
) -> pd.DataFrame:
    """Заполняет пропуски в keyword/location токеном 'nones' (этап 0 ноутбука)."""
    df = df.copy()
    for col in (cfg.keyword_col, cfg.location_col):
        if col in df.columns:
            df[col] = df[col].fillna(cfg.missing_token)
    return df


def prepare_data(cfg: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Полный пайплайн этапа 0: загрузка + очистка пропусков."""
    trn, tst, sample_submission = load_raw_data(cfg)
    trn = fill_missing_tokens(trn, cfg)
    tst = fill_missing_tokens(tst, cfg)
    return trn, tst, sample_submission


if __name__ == "__main__":
    from .config import load_config

    c = load_config()
    trn, tst, sub = prepare_data(c)
    print("trn shape:", trn.shape)
    print("tst shape:", tst.shape)
    print("Пропуски keyword trn:", trn[c.keyword_col].isna().sum())