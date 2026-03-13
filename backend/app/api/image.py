from fastapi import APIRouter
from pydantic import BaseModel

from app.rag.pipeline import multimodal_pipeline


router = APIRouter()


class Query(BaseModel):

    question: str


@router.post("/chat")

def chat(query: Query):

    answer = multimodal_pipeline(question=query.question)

    return {"answer": answer}