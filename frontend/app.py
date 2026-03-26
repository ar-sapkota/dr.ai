import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Dr Sahab AI", layout="wide")

st.title("🩺 Dr Sahab - Multimodal Medical AI")
st.write("Ask a medical question using **text, image, or both**.")

user_question = st.text_input("Enter your question")

uploaded_image = st.file_uploader(
    "Upload medical image (optional)",
    type=["png", "jpg", "jpeg"]
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Ask Doctor AI"):

        # validation — at least one input required
        if not user_question and not uploaded_image:
            st.warning("Please enter a question or upload an image.")
            st.stop()

        # OLD: three separate if-elif blocks calling three different endpoints
        # if uploaded_image and user_question:
        #     response = requests.post(f"{API_URL}/multimodal-query", ...)
        # elif uploaded_image:
        #     response = requests.post(f"{API_URL}/image-query", ...)
        # elif user_question:
        #     response = requests.post(f"{API_URL}/chat", json={"question": ...})

        # NEW: always call /ask with multipart/form-data
        # both fields are optional — backend handles whichever is provided
        data = {}
        files = {}

        if user_question:
            data["question"] = user_question

        if uploaded_image:
            files["file"] = (
                uploaded_image.name,
                uploaded_image.getvalue(),
                uploaded_image.type       # preserves image/png or image/jpeg
            )

        with st.spinner("Dr Sahab is analyzing..."):
            response = requests.post(
                f"{API_URL}/ask",         # single unified endpoint
                data=data,
                files=files if files else None
            )

        if response.status_code == 200:
            result = response.json()
            st.success("Dr Sahab's Response")
            st.write(result["answer"])
        else:
            st.error(f"Backend error: {response.status_code} — {response.text}")

with col2:
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)