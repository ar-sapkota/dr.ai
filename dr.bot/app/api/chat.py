# app/api/chat.py

from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
import os
import shutil
from app.memory.chat_memory import load_session
from app.rag.pipeline import multimodal_pipeline
from app.memory.chat_memory import create_session, list_sessions, delete_session

router = APIRouter()

UPLOAD_FOLDER = "data/query_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/ask")
async def ask(
    question: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)     # NEW: which conversation this belongs to
):
    if not question and not file:
        return {"error": "Please provide a question, an image, or both."}

    # if no session_id provided, create a new one
    if not session_id:
        session_id = create_session()

    image_path = None
    if file:
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    answer = multimodal_pipeline(
        question=question,
        image_path=image_path,
        session_id=session_id        # NEW: pass session into pipeline
    )

    return {
        "session_id": session_id,    # NEW: frontend needs this to track the session
        "question": question,
        "answer": answer
    }


@router.get("/sessions")
def get_sessions():
    """Return all past conversations for sidebar."""
    return list_sessions()


@router.post("/sessions/new")
def new_session():
    """Create and return a fresh session."""
    session_id = create_session()
    return {"session_id": session_id}


@router.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    """Delete a conversation."""
    delete_session(session_id)
    return {"deleted": session_id}


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Load full message history for a session."""
    session = load_session(session_id)
    if not session:
        return {"error": "Session not found"}
    return session