from app.rag.retriever import search_text, search_image
from app.rag.rag_chain import generate_answer


def multimodal_pipeline(question=None, image_path=None):

    context = ""

    if question:

        docs = search_text(question)

        for doc in docs:

            if doc["type"] == "text":
                context += doc["content"] + "\n"

    if image_path:

        images = search_image(image_path)

        for img in images:

            if img["type"] == "image":
                context += f"Related medical image: {img['path']}\n"

    answer = generate_answer(question, context)

    return answer