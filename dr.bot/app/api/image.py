from fastapi import APIRouter, UploadFile, File
import os
import shutil

from app.rag.pipeline import multimodal_pipeline

router = APIRouter()

UPLOAD_FOLDER = "data/query_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/image-query")
async def image_query(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    answer = multimodal_pipeline(image_path=file_path)

    return {"answer": answer}