import streamlit as st
from PIL import Image
import os
import shutil
from backend.app.rag.pipeline import multimodal_pipeline
from backend.app.memory.chat_memory import (
    create_session, list_sessions,
    delete_session, load_session
)

API_URL = "http://127.0.0.1:8000"  # no longer needed

UPLOAD_FOLDER = "data/query_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="Dr Sahab AI", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def load_session_into_state(session_id: str):
    session = load_session(session_id)
    if session:
        st.session_state.session_id = session_id
        st.session_state.messages = session.get("messages", [])


# ── sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dr Sahab AI")

    if st.button("+ New conversation", use_container_width=True):
        new_id = create_session()
        st.session_state.session_id = new_id
        st.session_state.messages = []
        st.rerun()

    st.divider()

    sessions = list_sessions()
    if not sessions:
        st.caption("No conversations yet.")
    else:
        for s in sessions:
            col_title, col_del = st.columns([5, 1])
            is_active = s["id"] == st.session_state.session_id
            with col_title:
                label = f"**{s['title']}**" if is_active else s["title"]
                if st.button(label, key=f"sess_{s['id']}", use_container_width=True):
                    load_session_into_state(s["id"])
                    st.rerun()
            with col_del:
                if st.button("✕", key=f"del_{s['id']}"):
                    delete_session(s["id"])
                    if st.session_state.session_id == s["id"]:
                        st.session_state.session_id = None
                        st.session_state.messages = []
                    st.rerun()

    st.divider()
    st.caption("Patient profile")
    try:
        import json
        with open("memory/patient_profile.json", "r") as f:
            profile = json.load(f)
        st.write(f"**Name:** {profile.get('name') or '—'}")
        st.write(f"**Age:** {profile.get('age') or '—'}")
        conditions = ", ".join(profile.get("conditions", [])) or "—"
        st.write(f"**Conditions:** {conditions}")
    except FileNotFoundError:
        st.caption("No profile yet.")


# ── main chat area ───────────────────────────────────────────────
st.title("🩺 Dr Sahab")

if not st.session_state.session_id:
    st.info("Start a new conversation or select one from the sidebar.")
else:
    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])

    uploaded_image = st.file_uploader(
        "Attach a medical image (optional)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    if uploaded_image:
        st.image(Image.open(uploaded_image), width=200)

    user_question = st.chat_input("Ask Dr Sahab...")

    if user_question or uploaded_image:
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)

        image_path = None
        if uploaded_image:
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_image.name)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(uploaded_image, f)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = multimodal_pipeline(
                    question=user_question,
                    image_path=image_path,
                    session_id=st.session_state.session_id
                )
            st.write(answer)
            st.session_state.messages.append({
                "question": user_question or "Image query",
                "answer": answer
            })