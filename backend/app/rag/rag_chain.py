import google.generativeai as genai
from PIL import Image
from app.config import settings


genai.configure(api_key=settings.GOOGLE_API_KEY)

model = genai.GenerativeModel(settings.MODEL_NAME)

def generate_answer(query: str, context: str, image_path: str = None):
    system_prompt = f"""You are Dr Sahab, a polite and knowledgeable medical AI assistant.
Always remind users to consult a qualified doctor for actual diagnosis.

Relevant context from medical literature:
{context}
"""
    user_message = f"Patient question: {query}"

    # FIX: build a multimodal content list when image is provided
    if image_path:
        image = Image.open(image_path)
        content = [system_prompt, image, user_message]  # ← image passed as PIL object
    else:
        content = [system_prompt, user_message]

    response = model.generate_content(content)
    return response.text