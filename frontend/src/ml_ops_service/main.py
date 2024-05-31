from os import environ
from typing import Final

import aiohttp
from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

BACKEND_URL: Final = environ.get("BACKEND_URL")

app = FastAPI(title="Ml Ops Service Frontend")


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
                session.post(f"{BACKEND_URL}/uploadfile", data=contents) as response):
        if response.ok:
            return StreamingResponse(response.content)
        raise HTTPException(
            status_code=response.status,
            detail=f"Backend error: {await response.text()}",
        )
    # fmt: on


@app.get("/")
async def main():
    content = """
    <body>
    <form action='/uploadfile' enctype='multipart/form-data' method='post'>
    <input name='file' type='file'>
    <input type='submit'>
    </form>
    </body>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run("main:app", port=3000)