import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Dr Sahab AI", layout="wide")

st.title("🩺 Dr Sahab - Multimodal Medical AI")

st.write("Ask a medical question using **text, image, or both**.")

# Text input
user_question = st.text_input("Enter your question")

# Image upload
uploaded_image = st.file_uploader(
    "Upload medical image (optional)",
    type=["png", "jpg", "jpeg"]
)

col1, col2 = st.columns([1,1])

with col1:

    if st.button("Ask Doctor AI"):

        if uploaded_image and user_question:

            st.info("Sending text + image query...")

            files = {"file": uploaded_image.getvalue()}

            data = {"question": user_question}

            response = requests.post(
                f"{API_URL}/multimodal-query",
                files={"file": uploaded_image},
                data=data
            )

        elif uploaded_image:

            st.info("Sending image query...")

            response = requests.post(
                f"{API_URL}/image-query",
                files={"file": uploaded_image}
            )

        elif user_question:

            st.info("Sending text query...")

            response = requests.post(
                f"{API_URL}/chat",
                json={"question": user_question}
            )

        else:
            st.warning("Please enter a question or upload an image.")
            st.stop()

        if response.status_code == 200:
            result = response.json()
            st.success("Doctor AI Response")
            st.write(result["answer"])

        else:
            st.error("Backend error")


with col2:

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)