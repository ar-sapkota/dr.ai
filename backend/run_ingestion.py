import os
from app.rag.ingestion import ingest_pdf   # Updated import

DATA_FOLDER = "data/medical_docs"

print("🚀 Starting ingestion of all PDFs...\n")

for file in os.listdir(DATA_FOLDER):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(DATA_FOLDER, file)
        ingest_pdf(pdf_path)

print("\n🎉 All PDFs processed successfully!")