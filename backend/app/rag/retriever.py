
import numpy as np

from app.config import settings
from models.clip import CLIPEmbedder
import getpass
import os
import fitz
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai 
from google.genai import types
import numpy as np

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

DIMENSION = 3072
MODEL_ID = "gemini-embedding-2-preview"


class GeminiEmbedding_2:
    def embed_text(self, text:str):
        result = client.models.embed_content(
            model = MODEL_ID,
            contents = text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return result.embeddings[0].values
    
    def embed_image(self, image_path:str):
        with open(image_path, "rb") as f:
            image_bytes = f.read()

            mime_type = "image/png"if image_path.endswith(".png") else "image/jpeg"

        result = client.models.embed_content(
            model=MODEL_ID,
            contents = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
            ]
        )
        return result.embeddings[0].values
    
embedder = GeminiEmbedding_2()

index = faiss.read_index(settings.VECTOR_DB_PATH)


metadata = np.load(
    settings.METADATA_PATH,
    allow_pickle=True
).tolist()


def perform_search(vector: np.ndarray, k=5):
    query_vector = np.array([vector]).astype("float32") ###
    faiss.normalize_L2(query_vector)

    

    #search
    distances, indices = index.search(query_vector, k)
    results = []

    for i,idx in enumerate(indices[0]):
        if idx != -1:
            item = metadata[idx].copy()
            item["score"] = float(distances[0][i])
            results.append(item)

    return results

def search_text(query, k=5):
    vector = embedder.embed_text(query)###
    return perform_search(vector, k)

def search_image(image_path, k=5):
    vector = embedder.embed_image(image_path)
    return perform_search(vector, k)
