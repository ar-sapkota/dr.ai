import getpass
import os
import fitz
import faiss
import time                                          # NEW: needed for time.sleep
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai 
from google.genai import types
import numpy as np
from dotenv import load_dotenv

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

DIMENSION = 3072
MODEL_ID = "gemini-embedding-2-preview"


class GeminiEmbedding_2:
    def embed_text(self, text: str):
        result = client.models.embed_content(
            model=MODEL_ID,
            contents=text
        )
        return result.embeddings[0].values

    def embed_image(self, image_path: str):
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # FIXED: removed trailing space from "image/png "
        # OLD: mime_type = "image/png "if image_path.endswith(".png") else "image/jpeg"
        mime_type = "image/png" if image_path.endswith(".png") else "image/jpeg"

        result = client.models.embed_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
            ]
        )
        return result.embeddings[0].values


embedder = GeminiEmbedding_2()

VECTORSTORE_PATH = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_PATH, "faiss.index")
METADATA_PATH = os.path.join(VECTORSTORE_PATH, "metadata.npy")
PROCESSED_FILE = os.path.join(VECTORSTORE_PATH, "processed_pdfs.txt")

# Load existing index and metadata (if exists)
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()
else:
    index = faiss.IndexFlatIP(DIMENSION)
    metadata = []

# Load already processed files
if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r") as f:
        processed_pdfs = set(f.read().splitlines())
else:
    processed_pdfs = set()


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_paths = []
    os.makedirs("data/images", exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_path = f"data/images/{pdf_name}_page{page_index}_{img_index}.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(image_path)

    doc.close()
    return image_paths


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)


# ─────────────────────────────────────────────────────────────────
# NEW: rate limiting wrapper
# Without this, all chunks were embedded back-to-back with no delay,
# instantly hitting Google's free tier RPM limit.
# This function batches items and sleeps between batches.
# It is generic — works for both text chunks and images.
# ─────────────────────────────────────────────────────────────────
def embed_with_rate_limit(items, embed_fn, batch_size=5, delay=1.5):
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        for item in batch:
            results.append(embed_fn(item))
        print(f"  Embedded {min(i + batch_size, len(items))}/{len(items)}")
        if i + batch_size < len(items):
            time.sleep(delay)   # pause between batches to respect RPM limit
    return results


# ─────────────────────────────────────────────────────────────────
# NEW: dedicated embed function for text chunks during ingestion
# task_type="RETRIEVAL_DOCUMENT" tells Gemini these are documents
# to be stored and searched — not queries.
# OLD code had no task_type at all, which misaligned stored vectors
# with query vectors and caused poor retrieval quality.
# ─────────────────────────────────────────────────────────────────
def embed_chunk(chunk: str):
    # OLD: vector = embedder.embed_text(chunk)
    #      (no task_type — misaligned with RETRIEVAL_QUERY used in retriever)
    result = client.models.embed_content(
        model=MODEL_ID,
        contents=chunk,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return result.embeddings[0].values


def ingest_pdf(pdf_path):
    filename = os.path.basename(pdf_path)

    # === Duplicate Check ===
    if filename in processed_pdfs:
        print(f"Skipping (already processed): {filename}")
        return

    print(f"Processing: {filename}")

    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    # ── TEXT EMBEDDING ──────────────────────────────────────────
    # OLD (no rate limiting, no task_type):
    # for chunk in chunks:
    #     vector = embedder.embed_text(chunk)
    #     vector_np = np.array([vector]).astype("float32")
    #     faiss.normalize_L2(vector_np)
    #     index.add(vector_np)
    #     metadata.append({"type": "text", "content": chunk, "source": filename})
    #
    # NEW: embed all chunks with rate limiting and correct task_type
    print(f"  Embedding {len(chunks)} text chunks...")
    text_vectors = embed_with_rate_limit(
        items=chunks,
        embed_fn=embed_chunk,       # uses RETRIEVAL_DOCUMENT task_type
        batch_size=5,               # 5 chunks per batch
        delay=12                  # 1.5s pause between batches
    )

    for chunk, vector in zip(chunks, text_vectors):
        vector_np = np.array([vector]).astype("float32")
        faiss.normalize_L2(vector_np)
        index.add(vector_np)
        metadata.append({"type": "text", "content": chunk, "source": filename})

    # ── IMAGE EMBEDDING ─────────────────────────────────────────
    # OLD (no rate limiting):
    # images = extract_images(pdf_path)
    # for path in images:
    #     vector = embedder.embed_image(path)
    #     vector_np = np.array([vector]).astype("float32")
    #     faiss.normalize_L2(vector_np)
    #     index.add(vector_np)
    #     metadata.append({"type": "image", "path": path, "source": filename})
    #
    # NEW: embed images with rate limiting
    # batch_size=3 and delay=2.0 because images are larger payloads
    # and more likely to hit limits than small text chunks
    images = extract_images(pdf_path)
    print(f"  Embedding {len(images)} images...")
    image_vectors = embed_with_rate_limit(
        items=images,
        embed_fn=embedder.embed_image,  # raw image bytes, no task_type needed
        batch_size=3,                   # smaller batch — images are heavy
        delay=2.0                       # longer pause between batches
    )

    for path, vector in zip(images, image_vectors):
        vector_np = np.array([vector]).astype("float32")
        faiss.normalize_L2(vector_np)
        index.add(vector_np)
        metadata.append({"type": "image", "path": path, "source": filename})

    # === Mark as processed ===
    processed_pdfs.add(filename)
    with open(PROCESSED_FILE, "a+", encoding="utf-8") as f:
        f.write(filename + "\n")

    # === Save updated index and metadata ===
    faiss.write_index(index, INDEX_PATH)
    np.save(METADATA_PATH, metadata)

    print(f"✅ Done: {filename}")


if __name__ == "__main__":
    print("Ingestion module loaded. Use run_ingestion.py to start.")