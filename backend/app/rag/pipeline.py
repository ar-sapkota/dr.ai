# app/rag/pipeline.py

from app.rag.retriever import search_text, search_image
from app.rag.rag_chain import generate_answer
from app.memory.chat_memory import (
    get_history_as_text,
    save_message,
    create_session
)
from app.memory.chat_memory import (
    get_history_as_text,
    # load_patient_profile,
    # extract_and_update_profile
)


def multimodal_pipeline(question=None, image_path=None, session_id=None):

    # create session if not provided
    if not session_id:
        session_id = create_session()

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
                context += f"[Retrieved similar medical image: {doc['path']}]\n"

    query = question if question else "Describe what you see in this medical image."

    # load memory
    history_text = get_history_as_text(session_id, last_n=6)
    # profile_text = get_profile_as_text(load_patient_profile())

    answer = generate_answer(
        query,
        context,
        image_path=image_path,
        history_text=history_text,
        # profile_text=profile_text
    )

    # save this turn to the session file
    save_message(session_id, query, answer)

    # update long-term patient profile
    # extract_and_update_profile(query, answer)

    return answer