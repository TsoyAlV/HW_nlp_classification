"""Загрузка конфигурации проекта (этапы 0-10 плана из ноутбука)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    """Централизованная конфигурация эксперимента."""

    data_dir: str = "data"
    trn_file: str = "trn.csv"
    tst_file: str = "tst.csv"
    sample_submission_file: str = "sample_submission.csv"
    submission_file: str = "submission.csv"

    target_col: str = "target"
    id_col: str = "id"
    text_col: str = "text"
    keyword_col: str = "keyword"
    location_col: str = "location"

    missing_token: str = "nones"
    keyword_len_cutoff: int = 18

    test_size: float = 0.2
    random_state: int = 42
    stratify_on_target: bool = True

    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    max_df: float = 0.95
    min_df: int = 2

    model_list: list[str] = field(
        default_factory=lambda: ["logreg", "xgboost", "multinomial_nb"]
    )
    tune_metric: str = "f1"
    use_gridsearch: bool = True

    bert_model_name: str = "bert-base-uncased"
    bert_max_len: int = 64
    bert_batch_size: int = 2
    bert_lr: float = 2e-5
    bert_epochs: int = 1

    @property
    def trn_path(self) -> str:
        return os.path.join(self.data_dir, self.trn_file)

    @property
    def tst_path(self) -> str:
        return os.path.join(self.data_dir, self.tst_file)

    @property
    def sample_submission_path(self) -> str:
        return os.path.join(self.data_dir, self.sample_submission_file)

    @property
    def submission_path(self) -> str:
        return os.path.join(self.data_dir, self.submission_file)


def load_config(path: str | None = None) -> ProjectConfig:
    """Загружает YAML-конфиг поверх значений по умолчанию.

    Args:
        path: путь к configs/*.yaml. Если None — используются дефолты.
    """
    cfg = ProjectConfig()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            overrides: dict[str, Any] = yaml.safe_load(f) or {}
        for key, value in overrides.items():
            if hasattr(cfg, key):
                if key in ("ngram_range",) and isinstance(value, list):
                    value = tuple(value)
                setattr(cfg, key, value)
    return cfg