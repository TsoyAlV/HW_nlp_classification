"""Этап 3: Инженерия признаков (числовые фичи из текста)."""
from __future__ import annotations

import pandas as pd

from .config import ProjectConfig


def add_text_features(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Добавляет признаки длины, caps, пунктуации и наличия keyword."""
    df = df.copy()
    text = df[cfg.text_col].astype(str)

    df["n_words"] = text.apply(lambda s: len(s.split()))
    df["n_unique_words"] = text.apply(lambda s: len(set(s.lower().split())))
    df["n_upper"] = text.apply(lambda s: sum(1 for c in s if c.isupper()))
    df["n_exclam"] = text.apply(lambda s: s.count("!"))
    df["n_question"] = text.apply(lambda s: s.count("?"))
    df["has_keyword"] = (df[cfg.keyword_col] != cfg.missing_token).astype(int)

    return df