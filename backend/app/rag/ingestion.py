import os
import fitz
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.clip import CLIPEmbedder
from dotenv import load_dotenv
load_dotenv()
os.makedirs("vectorstore", exist_ok=True)

embedder = CLIPEmbedder()
dimension = 512

# Paths
VECTORSTORE_PATH = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_PATH, "faiss.index")
METADATA_PATH = os.path.join(VECTORSTORE_PATH, "metadata.npy")
PROCESSED_FILE = os.path.join(VECTORSTORE_PATH, "processed_pdfs.txt")

# Load existing index and metadata (if exists)
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()
else:
    index = faiss.IndexFlatL2(dimension)
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

    # Get clean PDF name
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60,length_function=len)
    return splitter.split_text(text)


def ingest_pdf(pdf_path):
    filename = os.path.basename(pdf_path)

    # === Duplicate Check ===
    if filename in processed_pdfs:
        print(f"⏭️ Skipping (already processed): {filename}")
        return

    print(f"Processing: {filename}")

    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    # Add text chunks
    for chunk in chunks:
        vector = embedder.embed_text(chunk)
        index.add(np.array([vector]).astype("float32"))
        metadata.append({"type": "text", "content": chunk, "source": filename})

    # Add images
    images = extract_images(pdf_path)
    for path in images:
        vector = embedder.embed_image(path)
        index.add(np.array([vector]).astype("float32"))
        metadata.append({"type": "image", "path": path, "source": filename})

    # Mark as processed
    processed_pdfs.add(filename)
    with open(PROCESSED_FILE, "a+", encoding="utf-8") as f:
        f.write(filename + "\n")

    # Save updated index and metadata
    faiss.write_index(index, INDEX_PATH)
    np.save(METADATA_PATH, metadata)

    print(f"✅ Added: {filename}")


if __name__ == "__main__":
    print("Ingestion module loaded. Use run_ingestion.py to start.")