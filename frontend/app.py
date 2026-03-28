import streamlit as st
import requests
from PIL import Image
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Dr Sahab AI", layout="wide")

# ── session state init ───────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def load_session_into_state(session_id: str):
    """Fetch full message history for a session and load into state."""
    res = requests.get(f"{API_URL}/sessions/{session_id}")
    if res.status_code == 200:
        data = res.json()
        st.session_state.session_id = session_id
        st.session_state.messages = data.get("messages", [])


# ── sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dr Sahab AI")

    # new chat button
    if st.button("+ New conversation", use_container_width=True):
        res = requests.post(f"{API_URL}/sessions/new")
        if res.status_code == 200:
            new_id = res.json()["session_id"]
            st.session_state.session_id = new_id
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # list past conversations
    sessions_res = requests.get(f"{API_URL}/sessions")
    if sessions_res.status_code == 200:
        sessions = sessions_res.json()

        if not sessions:
            st.caption("No conversations yet.")
        else:
            for s in sessions:
                col_title, col_del = st.columns([5, 1])
                is_active = s["id"] == st.session_state.session_id

                with col_title:
                    label = f"**{s['title']}**" if is_active else s["title"]
                    if st.button(
                        label,
                        key=f"sess_{s['id']}",
                        use_container_width=True
                    ):
                        load_session_into_state(s["id"])
                        st.rerun()

                with col_del:
                    if st.button("✕", key=f"del_{s['id']}"):
                        requests.delete(f"{API_URL}/sessions/{s['id']}")
                        # if deleted session was active, clear state
                        if st.session_state.session_id == s["id"]:
                            st.session_state.session_id = None
                            st.session_state.messages = []
                        st.rerun()

    st.divider()

    # patient profile
    st.caption("Patient profile")
    try:
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
    # render existing messages
    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])

    # image uploader above input
    uploaded_image = st.file_uploader(
        "Attach a medical image (optional)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    if uploaded_image:
        st.image(Image.open(uploaded_image), width=200)

    # chat input at the bottom
    user_question = st.chat_input("Ask Dr Sahab...")

    if user_question or uploaded_image:
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)

        data = {"session_id": st.session_state.session_id}
        files = {}

        if user_question:
            data["question"] = user_question
        if uploaded_image:
            files["file"] = (
                uploaded_image.name,
                uploaded_image.getvalue(),
                uploaded_image.type
            )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    data=data,
                    files=files if files else None
                )

            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                st.write(answer)

                # update local state so new message shows immediately
                st.session_state.messages.append({
                    "question": user_question or "Image query",
                    "answer": answer
                })
            else:
                st.error(f"Error {response.status_code}: {response.text}")