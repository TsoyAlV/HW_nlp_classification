"""Этап 2: Предобработка текста (очистка, токены, стемминг)."""
from __future__ import annotations

import re

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from .config import ProjectConfig

# Гарантируем наличие токенизатора (для PorterStemmer не нужны доп. ресурсы,
# но word_tokenize требует punkt).
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

STEMMER = PorterStemmer()

USER_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"http\S+|www\.\S+")
PUNCT_DIGIT_RE = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    """Заменяет @user и ссылки тегами, приводит к lowercase."""
    text = USER_RE.sub("USER_TAG", str(text))
    text = URL_RE.sub("URL_TAG", text)
    text = text.lower()
    return text


def tokenize_and_stem(text: str) -> list[str]:
    """Токенизация + стемминг (без стоп-слов на этом этапе — опц. в векторизаторе)."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    stems = [STEMMER.stem(t) for t in tokens if PUNCT_DIGIT_RE.sub("", t)]
    return stems


def preprocess_dataframe(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Добавляет колонку 'clean_text' со стеммированным текстом."""
    df = df.copy()
    df["clean_text"] = df[cfg.text_col].apply(
        lambda x: " ".join(tokenize_and_stem(x))
    )
    return df