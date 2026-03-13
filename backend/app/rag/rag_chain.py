import google.generativeai as genai

from app.config import settings


genai.configure(api_key=settings.GOOGLE_API_KEY)

model = genai.GenerativeModel(settings.MODEL_NAME)


def generate_answer(query, context):

    prompt = f"""
You are Dr Sahab, a medical AI assistant.

Context:
{context}

User Question:
{query}

Provide a helpful medical explanation.
"""

    response = model.generate_content(prompt)

    return response.text