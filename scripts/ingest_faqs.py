"""
FAQ Ingestion Script.
Reads vehicle insurance FAQs from JSON and loads them into ChromaDB vector store
using sentence-transformers/all-MiniLM-L6-v2 embeddings.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path to enable imports
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.services.retrieval_service import retrieval_service

def main():
    print("=" * 60)
    print("Starting FAQ Ingestion into ChromaDB...")
    print(f"Data file: {settings.FAQ_DATA_PATH}")
    print(f"Chroma DB Path: {settings.CHROMA_PERSIST_DIRECTORY}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    print("=" * 60)

    try:
        count = retrieval_service.ingest_faqs(force_reindex=True)
        print(f"\n[SUCCESS] Successfully ingested {count} FAQs into ChromaDB!")
        print("ChromaDB is now primed and ready for semantic search queries.")
    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

