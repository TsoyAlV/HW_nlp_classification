"""Этап 10: Сводная таблица результатов всех моделей."""
from __future__ import annotations

import pandas as pd


def build_summary(metrics: dict[str, dict]) -> pd.DataFrame:
    """Собирает F1/Precision/Recall/ROC-AUC по моделям в DataFrame."""
    rows = []
    for name, m in metrics.items():
        rows.append({
            "model": name,
            "f1": m.get("f1"),
            "roc_auc": m.get("roc_auc"),
            "precision_1": m.get("report", {}).get("1", {}).get("precision"),
            "recall_1": m.get("report", {}).get("1", {}).get("recall"),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False)