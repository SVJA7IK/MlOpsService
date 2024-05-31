from io import BytesIO
from typing import Final

import aiohttp
import uvicorn
from config import Config
from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

SAMPLE_SUBMISSION_FILE_NAME: Final = "sample_submission.csv"


app = FastAPI(title="Ml Ops Service Frontend")

config = Config()


@app.post("/uploadfile")
async def upload(file: UploadFile):
    try:
        contents = await file.read()
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="There was an error uploading the file",
        ) from err
    finally:
        await file.close()
    # fmt: off
    async with (aiohttp.ClientSession() as session,
                session.post(f"{config.backend_url}/uploadfile", data={"file": contents}) as response):
        if response.ok:
            return StreamingResponse(BytesIO(await response.read()), headers={
            "Content-Disposition": f"attachment; filename={SAMPLE_SUBMISSION_FILE_NAME}",
        })
        raise HTTPException(
            status_code=response.status,
            detail=f"Backend error: {await response.text()}",
        )
    # fmt: on


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", port=3000)
