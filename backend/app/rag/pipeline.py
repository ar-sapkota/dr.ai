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

        docs = search_image(image_path)

        for doc in docs:

            if doc["type"] == "text":
                context += doc["content"] + "\n"

            if doc["type"] == "image":
                context += f"Related medical image: {doc['path']}\n"


    query = question if question else "Explain this medical image."

    answer = generate_answer(query, context, image_path=image_path)

    return answer