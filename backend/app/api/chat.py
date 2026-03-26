# app/api/chat.py

from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
import os
import shutil

from app.rag.pipeline import multimodal_pipeline

router = APIRouter()

UPLOAD_FOLDER = "data/query_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/ask")
async def ask(
    question: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    # validation
    if not question and not file:
        return {"error": "Please provide a question, an image, or both."}

    image_path = None

    # save uploaded image if provided
    # OLD image.py was saving to data/query_images — kept same folder
    if file:
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # OLD chat.py: multimodal_pipeline(question=query.question)
    # OLD image.py: multimodal_pipeline(image_path=file_path)
    # NEW: both passed together — whichever is None gets ignored in pipeline
    answer = multimodal_pipeline(question=question, image_path=image_path)

    return {
        "question": question,
        "answer": answer
    }