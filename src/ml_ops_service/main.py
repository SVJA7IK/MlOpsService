import catboost as cb
import pandas as pd
import shap
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Загрузка данных
df = pd.read_csv("train.csv", index_col="client_id")
submit = pd.read_csv("test.csv", index_col="client_id")
sample = pd.read_csv("sample_submission.csv", index_col="client_id")

# Оценка пропусков в данных
data_info = pd.DataFrame(df.dtypes, columns=["Тип колонки"])
data_info["Пропусков"] = df.isnull().sum()
data_info["Пропусков %"] = (df.isnull().sum() / len(df) * 100).round(2)
data_info[data_info["Пропусков"] > 0].sort_values(by="Пропусков %", ascending=False)

# Оценка количества дубликатов строк
# %%
df.duplicated().value_counts(1)

# Баланс классов
df["binary_target"].value_counts(1)

# Замечания о данных:
# - `mrg_` константа.
# - Дисбаланс целевой переменной.
# - Много пропусков, строки с пропусками образуют дубликаты.
# - Большинство признаков вносит минимальный вклад, за исключением `секретный_скор`, `on_net`, `частота_пополнения`.

# Подготовка фичей
target = "binary_target"
cat_features = ["регион", "использование", "mrg_", "pack"]

# Быстрое заполнение пропусков в категориальных признаках
df[cat_features] = df[cat_features].fillna("Пропуск")

X = df.drop(target, axis=1)
y = df[target]


def plot_shap(model, X_test):
    # Инициализация объекта SHAP Explainer
    explainer = shap.TreeExplainer(model)

    # Вычисление SHAP значений для обучающего набора данных
    shap_values = explainer.shap_values(X_test)

    # Визуализация важности признаков
    shap.summary_plot(shap_values, X_test, plot_type="bar")


# Прогон на 0.6478 – лучший скор на публичном лидерборде

# Подбор гиперпараметров с помощью Optuna, конретный запуск Optuna не сохранился.
col = df.columns.drop(target).to_list()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X[col], y, test_size=0.2, random_state=42)

# Выбор категориальных признаков
cat_features = X[col].dtypes[X[col].dtypes == "object"].index.to_list()
print(f"Selected cols: {col}\n")

# Создание объекта Pool для работы с категориальными признаками
train_pool = cb.Pool(X_train, y_train, cat_features=cat_features)
test_pool = cb.Pool(X_test, y_test, cat_features=cat_features)

# Параметры модели
params = {
    "iterations": 100,
    "learning_rate": 0.03258020430505722,
    "depth": 10,
    "subsample": 0.05939807129645091,
    "colsample_bylevel": 0.9815947638133953,
    "min_data_in_leaf": 33,
    "eval_metric": "F1",
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": 200,
    "auto_class_weights": "SqrtBalanced",
}

# Обучение модели
model = cb.CatBoostClassifier(**params)
model.fit(train_pool, eval_set=test_pool)

# Предсказание
predictions = model.predict(X_test)
predictions_proba = model.predict_proba(X_test)[:, 1]

# Вычисление метрики F1
print(classification_report(y_test, predictions, digits=4))
print(f"F1 score: {f1_score(y_test, predictions)}")

plot_shap(model, X_test)

# Сохранение сабмита
submit[cat_features] = submit[cat_features].fillna("пропуск")
submit_pool = cb.Pool(submit[col], cat_features=cat_features)

sample["preds"] = model.predict(submit_pool)
sample.to_csv("catboost_test_fullcols_optuna.csv.csv")

# Прогон на 0.6458 – на трех фичах

col = ["секретный_скор", "on_net", "частота_пополнения"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X[col], y, test_size=0.2, random_state=42)

# Выбор категориальных признаков
cat_features = X[col].dtypes[X[col].dtypes == "object"].index.to_list()
print(f"Selected cols: {col}\n")

# Создание объекта Pool для работы с категориальными признаками
train_pool = cb.Pool(X_train, y_train, cat_features=cat_features)
test_pool = cb.Pool(X_test, y_test, cat_features=cat_features)

# Параметры модели
params = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 5,
    "eval_metric": "F1",
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": 200,
    "auto_class_weights": "SqrtBalanced",
    "early_stopping_rounds": 50,
}

# Обучение модели
model = cb.CatBoostClassifier(**params)
model.fit(train_pool, eval_set=test_pool)

# Предсказание
predictions = model.predict(X_test)
predictions_proba = model.predict_proba(X_test)[:, 1]

# Вычисление метрики F1
print(classification_report(y_test, predictions, digits=4))
print(f"F1 score: {f1_score(y_test, predictions)}")
# Сохранение сабмита

submit[cat_features] = submit[cat_features].fillna("пропуск")
submit_pool = cb.Pool(submit[col], cat_features=cat_features)

sample["preds"] = model.predict(submit_pool)
sample.to_csv("catboost_test_shortcols_SqrtBalanced_depth5.csv")
