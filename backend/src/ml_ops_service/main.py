from io import BytesIO
from typing import Annotated, Final

import catboost as cb
import pandas as pd
import uvicorn
from fastapi import FastAPI, File
from fastapi.responses import StreamingResponse

from ml_ops_service.config import config_loader

SAMPLE_SUBMISSION_FILE_NAME: Final = "sample_submission.csv"


config = config_loader()

model = cb.CatBoostClassifier()
model.load_model(config.path_to_model)


def make_prediction(model: cb.CatBoostClassifier, input_file: BytesIO) -> BytesIO:
    # Получение датафрейма
    input_file_data = pd.read_csv(input_file, index_col="client_id")

    # Фичи
    cat_features = ["регион", "использование", "mrg_", "pack"]

    # Сохранение сабмита
    input_file_data[cat_features] = input_file_data[cat_features].fillna("пропуск")
    submit_pool = cb.Pool(input_file_data, cat_features=cat_features)

    sample = pd.DataFrame(index=input_file_data.index)

    sample["preds"] = model.predict(submit_pool)

    # Создание буфера и запись датафрейма
    buffer = BytesIO()
    sample.to_csv(buffer)
    buffer.seek(0)
    return buffer


app = FastAPI(title="Ml Ops Service Backend")


@app.post("/uploadfile/")
async def create_upload_file(file: Annotated[bytes, File()]) -> StreamingResponse:
    return StreamingResponse(
        make_prediction(model, BytesIO(file)),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={SAMPLE_SUBMISSION_FILE_NAME}",
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", port=5000)
