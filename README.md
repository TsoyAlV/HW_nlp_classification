"# HW_nlp_classification

Классификация твитов: реальная катастрофа или нет (NLP Getting Started).
Целевая переменная `target` ∈ {0, 1} — 1 = реальная катастрофа.

## Структура

```
configs/default.yaml        # все гиперпараметры и пути
data/                       # trn.csv, tst.csv, sample_submission.csv (положить вручную)
notebooks/
  tweet_classification.ipynb  # пошаговый pipeline по этапам 0-10
scripts/                    # модули, импортируемые в ноутбуке
  config.py                 # ProjectConfig + load_config()
  data_loader.py            # этап 0: загрузка + пропуски -> 'nones'
  eda.py                    # этап 1: баланс классов, длины, гистограммы
  preprocess.py             # этап 2: lowercase, USER_TAG/URL_TAG, PorterStemmer
  features.py               # этап 3: числовые признаки из текста
  vectorize.py              # этап 4: TfidfVectorizer + StandardScaler
  train.py                  # этапы 5/6: LogReg, MultinomialNB, GridSearch (F1)
  evaluate.py               # этапы 7/8: метрики, confusion matrix, error analysis
  predict.py                # этап 9: генерация submission.csv
  report.py                 # этап 10: сводная таблица моделей
```

## Этапы (по ноутбуку)

0. Загрузка данных и заполнение пропусков `keyword`/`location` токеном `nones`
1. EDA: баланс классов, длина текстов, пропуски
2. Предобработка: очистка, теги `@user`→`USER_TAG`, ссылки→`URL_TAG`, стемминг
3. Инженерия признаков: длина, CAPS, знаки препинания, наличие keyword
4. Векторизация: TF-IDF (ngram 1-2) + StandardScaler через ColumnTransformer
5. Модели: LogisticRegression (balanced), MultinomialNB (опц. XGBoost)
6. Тюнинг: GridSearchCV по F1-score
7. Оценка: confusion matrix, Precision/Recall/F1, ROC-AUC
8. Анализ ошибок: вывод примеров неверных предсказаний
9. Генерация `data/submission.csv` (id, target)
10. Сводная таблица всех моделей

## Быстрый старт

```bash
pip install -r requirements.txt
# положите trn.csv, tst.csv, sample_submission.csv в папку data/
jupyter lab notebooks/tweet_classification.ipynb
```

Запускайте ячейки по порядку. `random_state=42` зафиксирован во всех этапах.

## Пути улучшения

- Добавить XGBoost/LightGBM (раскомментировать в `requirements.txt` и `config.yaml`)
- Использовать BERT/RuBERT (заготовка в ноутбуке, этап 2)
- Сентимент-анализ (VADER/TextBlob) как доп. признак
"