"""Этапы 7-8: Оценка метрик и анализ ошибок."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate_model(model, X_val, y_val) -> dict:
    """Метрики: F1, precision, recall, ROC-AUC, confusion matrix."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    return {
        "f1": f1_score(y_val, y_pred),
        "report": classification_report(y_val, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }


def error_analysis(
    model, X_val_texts: pd.Series, y_val: pd.Series, n: int = 10
) -> pd.DataFrame:
    """Вывод примеров ошибок модели (этап 8)."""
    y_pred = model.predict(X_val_texts if hasattr(X_val_texts, "shape") else X_val_texts)
    mistakes = y_val != y_pred
    return pd.DataFrame({
        "text": X_val_texts[mistakes].head(n),
        "true": y_val[mistakes].head(n),
        "pred": y_pred[mistakes][:n],
    })