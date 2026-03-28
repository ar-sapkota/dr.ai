# app/rag/rag_chain.py

from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")


def generate_answer(
    query: str,
    context: str,
    image_path: str = None,
    history_text: str = None,       # ← ADD THIS
):
    history_section = ""
    if history_text and history_text != "No previous conversation.":
        history_section = f"\nConversation history:\n{history_text}\n"

    system_prompt = f"""You are Dr Sahab, a polite and knowledgeable medical AI assistant.
Always remind users to consult a qualified doctor for actual diagnosis.
{history_section}
Relevant context from medical literature:
{context}
"""
    user_message = f"Patient question: {query}"

    if image_path:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        mime_type = "image/png" if image_path.endswith(".png") else "image/jpeg"

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_text(text=system_prompt),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part.from_text(text=user_message)
            ]
        )
    else:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_text(text=system_prompt),
                types.Part.from_text(text=user_message)
            ]
        )

    return response.text