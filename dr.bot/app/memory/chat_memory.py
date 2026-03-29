# app/memory/chat_history.py

import json
import os
import uuid
from datetime import datetime

HISTORY_DIR = "memory/chats"
os.makedirs(HISTORY_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{session_id}.json")


def create_session() -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())[:8]   # short readable ID e.g. "a3f9c12b"
    session = {
        "id": session_id,
        "title": "New conversation",
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    with open(_session_path(session_id), "w") as f:
        json.dump(session, f, indent=2)
    return session_id


def load_session(session_id: str) -> dict:
    """Load a session by ID."""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_message(session_id: str, question: str, answer: str):
    """Append a Q&A turn to a session."""
    session = load_session(session_id)
    if not session:
        return

    session["messages"].append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    })

    # auto-title: use first question as the conversation title
    if len(session["messages"]) == 1:
        # truncate to 40 chars for sidebar display
        session["title"] = question[:40] + ("..." if len(question) > 40 else "")

    with open(_session_path(session_id), "w") as f:
        json.dump(session, f, indent=2)


def list_sessions() -> list:
    """Return all sessions sorted by most recent first."""
    sessions = []
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(HISTORY_DIR, filename), "r") as f:
                session = json.load(f)
                sessions.append({
                    "id": session["id"],
                    "title": session["title"],
                    "created_at": session["created_at"],
                    "message_count": len(session["messages"])
                })
    # most recent first
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return sessions


def delete_session(session_id: str):
    """Delete a session file."""
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)


def get_history_as_text(session_id: str, last_n: int = 6) -> str:
    """Get last N turns formatted for prompt injection."""
    session = load_session(session_id)
    if not session or not session["messages"]:
        return "No previous conversation."
    # only use last N turns to avoid bloating the prompt
    recent = session["messages"][-last_n:]
    lines = []
    for msg in recent:
        lines.append(f"Patient: {msg['question']}")
        lines.append(f"Dr Sahab: {msg['answer']}")
    return "\n".join(lines)