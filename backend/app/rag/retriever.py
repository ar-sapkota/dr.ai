import faiss
import numpy as np

from app.config import settings
from models.clip import CLIPEmbedder


embedder = CLIPEmbedder()

index = faiss.read_index(settings.VECTOR_DB_PATH)

metadata = np.load(
    settings.METADATA_PATH,
    allow_pickle=True
)


def search_text(query, k=5):

    vector = embedder.embed_text(query)

    D, I = index.search(
        np.array([vector]).astype("float32"),
        k
    )

    results = []

    for idx in I[0]:
        results.append(metadata[idx])

    return results


def search_image(image_path, k=5):

    vector = embedder.embed_image(image_path)

    D, I = index.search(
        np.array([vector]).astype("float32"),
        k
    )

    results = []

    for idx in I[0]:
        results.append(metadata[idx])

    return results