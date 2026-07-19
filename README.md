# 🐦 Классификация твитов: реальная катастрофа или нет

Проект по NLP (Kaggle «Getting Started — Natural Language Processing with Disaster Tweets»).
Цель — по тексту твита предсказать бинарную метку `target`:

- `1` — твит о **реальной** катастрофе / ЧП
- `0` — твит **не** о реальной катастрофе (шутка, метафора, новости о фильмах и т.п.)

Весь исследовательский процесс задокументирован в тетрадке
[`notebooks/tweet_classification.ipynb`](notebooks/tweet_classification.ipynb), а вся воспроизводимая
логика вынесена в модули `scripts/` и параметризуется через `configs/default.yaml`.

---

## 📚 С чего начать: ноутбук

Основная точка входа в проект — тетрадка **`notebooks/tweet_classification.ipynb`**.
Она реализует план из 10 этапов и последовательно вызывает модули из `scripts/`.

> 💡 Из ноутбука используется `os.chdir('..')`, чтобы подняться из `notebooks/` в корень
> и импортировать `scripts` как пакет. Запускайте тетрадку из папки `notebooks/`.

Краткое содержание тетрадки:

| Этап | Что делается | Модуль |
|------|--------------|--------|
| 0 | Окружение, загрузка данных | `scripts.config`, `scripts.data_loader` |
| 1 | EDA (баланс классов, длины, пропуски) | `scripts.eda` |
| 2 | Предобработка текста (очистка, лемматизация) | `scripts.preprocess` |
| 3 | Инженерия признаков (n_words, upper, !, ? и т.д.) | `scripts.features` |
| 4–6 | Сплит, векторизация (TF-IDF + числовые), обучение | `scripts.train`, `scripts.vectorize` |
| 7–8 | Оценка (F1) и анализ ошибок | `scripts.evaluate` |
| 9 | Генерация и сохранение submission | `scripts.predict` |
| 10 | Сводная таблица метрик | `scripts.report` |

В конце тетрадки также находятся **экспериментальные наброски** с трансформерами
(BGE, DistilBERT, fine-tuning BERT) — они опциональны и требуют `torch` + `transformers`.

---

## 🗂 Структура репозитория

```text
.
├── README.md                      # этот файл
├── requirements.txt               # зависимости
├── configs/
│   └── default.yaml               # центральная конфигурация эксперимента
├── notebooks/
│   └── tweet_classification.ipynb # основной EDA + pipeline (10 этапов)
│   └── test.ipynb                 # вспомогательная тетрадка
├── scripts/                       # переиспользуемые модули
│   ├── config.py                  # ProjectConfig + load_config()
│   ├── data_loader.py             # загрузка и базовая подготовка данных
│   ├── eda.py                     # разведочный анализ и графики
│   ├── preprocess.py              # очистка и стемминг текста
│   ├── features.py                # числовые признаки из текста
│   ├── vectorize.py               # TF-IDF + ColumnTransformer
│   ├── train.py                   # сплит, обучение, grid-search
│   ├── evaluate.py                # метрики и анализ ошибок
│   ├── predict.py                 # генерация submission
│   └── report.py                  # сводная таблица
├── models/
│   └── bert_classifier/           # сохранённая модель BERT (safetensors + tokenizer)
└── data/                          # сюда кладутся trn.csv, tst.csv, sample_submission.csv
```

---

## ⚙️ Конфигурация

Все ключевые параметры — в [`configs/default.yaml`](configs/default.yaml) и дублируются
в датаклассе `ProjectConfig` (`scripts/config.py`). Основные группы:

- **Данные**: `data_dir`, `trn_file`, `tst_file`, `sample_submission_file`,
  `target_col`, `text_col`, `keyword_col`, `location_col`, `missing_token`.
- **Сплит**: `test_size: 0.2`, `random_state: 42`, стратификация по `target`.
- **Векторизация**: `max_features: 5000`, `ngram_range: [1, 2]`, `max_df`, `min_df`.
- **Модели**: `model_list: [logreg, xgboost]` (опционально `multinomial_nb`),
  `use_gridsearch: true`, метрика `tune_metric: f1`.
- **BERT (опционально)**: `bert_model_name: bert-base-uncased`, `bert_max_len: 64`,
  `bert_batch_size`, `bert_lr`, `bert_epochs`.

---

## 🚀 Быстрый старт

### 1. Клонирование и окружение

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Данные

Скачайте датасет с Kaggle (`train.csv`, `test.csv`, `sample_submission.csv`)
и положите их в папку `data/` с именами из конфига:
`trn.csv`, `tst.csv`, `sample_submission.csv`.

### 3. Запуск пайплайна

Откройте и выполните тетрадку:

```bash
jupyter lab notebooks/tweet_classification.ipynb
```

Или используйте модули напрямую из Python (из корня проекта):

```python
from scripts.config import load_config
from scripts.data_loader import prepare_data
from scripts.preprocess import preprocess_dataframe
from scripts.features import add_text_features
from scripts.vectorize import vectorize
from scripts.train import split_train_val, train_all
from scripts.evaluate import evaluate_model
from scripts.predict import generate_submission, save_submission

cfg = load_config("configs/default.yaml")
trn, tst, sample_submission = prepare_data(cfg)

trn = preprocess_dataframe(trn, cfg)
tst = preprocess_dataframe(tst, cfg)
trn = add_text_features(trn, cfg)
tst = add_text_features(tst, cfg)

numeric_cols = ["n_words", "n_unique_words", "n_upper",
                "n_exclam", "n_question", "has_keyword"]
trn_split, val = split_train_val(trn, cfg)
X_trn, X_val, pre = vectorize(trn_split, val, cfg, numeric_cols)
models = train_all(X_trn, trn_split[cfg.target_col], cfg)
```

Результат (`submission.csv`) сохранится в папку `data/` согласно конфигу.

---

## 🧠 Модели и эксперименты

### Классический пайплайн (этапы 0–10)
- **Logistic Regression** и **XGBoost** поверх TF-IDF (1–2 n-граммы) + числовых признаков.
- При `use_gridsearch: true` проводится подбор гиперпараметров по F1.
- Лучшая модель по F1 на валидации используется для финального submission.

### Эксперименты с трансформерами (в ноутбуке, опционально)
Требуют раскомментировать `torch` / `transformers` в `requirements.txt`:

1. **BGE embeddings** (`BAAI/bge-large-en-v1.5`) + линейная головая на PyTorch.
2. **DistilBERT embeddings** + `EmbedClassifier` (дообучаемый классификатор).
3. **Fine-tuning BERT** (`bert-base-uncased`) с разморозкой последних N слоёв.
4. Сохранённая модель fine-tune BERT лежит в `models/bert_classifier/`.

> Вывод из ноутбука: на данном датасете классический TF-IDF + LogReg/XGBoost
> показал качество не хуже (а местами лучше) тяжёлых трансформерных моделей
> при гораздо меньших вычислительных затратах.

---

## 📊 Метрики и оценка

- Основная метрика — **F1-score** (задана в `tune_metric`).
- `scripts.evaluate.evaluate_model` считает метрики на валидации.
- `scripts.evaluate.error_analysis` выводит примеры ошибочных предсказаний.
- `scripts.report.build_summary` собирает сводную таблицу по всем моделям.
---

## 🔧 Зависимости

Базовый набор (этапы 0–10):

```
pandas, numpy, pyyaml, nltk, scikit-learn, matplotlib
xgboost   # для градиентного бустинга
```

Опционально (трансформеры, закомментировано в `requirements.txt`):

```
torch>=2.0
transformers>=4.30
tqdm
```

---

## 📝 Заметки

- Текст очищается в `scripts/preprocess.clean_text` (удаление URL, упоминаний,
  стемминг через `nltk`).
- Пропущенные `keyword` / `location` заменяются на `missing_token` (`nones`).
- Ноутбук рассчитан на запуск из папки `notebooks/` (или с `sys.path.append('..')`).


## Результаты работы различных моделей (F1 score):

0.76082 - LogisticRegression
0.78446 BERT 
0.75777 DistilBERT
0.76923 BERT Fine Tuning

Модели выдают примерно одни и те же результаты. Лучшая метрика у базового трансформера BERT. Однако есть вероятность что модели просто не были дообучены в нужной степени или выборка Val не репрезентативна. 