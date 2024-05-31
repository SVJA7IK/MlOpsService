from io import BytesIO

import catboost as cb
import pandas as pd

model = cb.CatBoostClassifier()
model.load_model("./models/model.cbm")


def make_prediction(model: cb.CatBoostClassifier, input_file: BytesIO) -> BytesIO:
    # Получение датафрейма
    input_file_data = pd.read_csv(input_file, index_col="client_id")

    # Фичи
    cat_features = ["регион", "использование", "mrg_", "pack"]

    # Сохранение сабмита
    input_file_data[cat_features] = input_file_data[cat_features].fillna("пропуск")
    submit_pool = cb.Pool(input_file, cat_features=cat_features)

    sample = pd.DataFrame(index=input_file_data.index)

    sample["preds"] = model.predict(submit_pool)

    # Создание буфера и запись датафрейма
    buffer = BytesIO()
    sample.to_csv(buffer)
    buffer.seek(0)
    return buffer
