import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.embedding_service import embedding_service
from app.utils.preprocessing import clean_query_for_search

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Manages vector storage in ChromaDB and semantic search retrieval for vehicle insurance FAQs.
    """

    def __init__(
        self,
        persist_directory: Path = settings.CHROMA_PERSIST_DIRECTORY,
        collection_name: str = settings.CHROMA_COLLECTION_NAME,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Return the number of stored FAQ items in the collection."""
        return self.collection.count()

    def ingest_faqs(self, faq_file_path: Optional[Path] = None, force_reindex: bool = False) -> int:
        """
        Loads FAQs from JSON dataset, computes embeddings, and indexes them into ChromaDB.
        """
        if faq_file_path is None:
            faq_file_path = settings.FAQ_DATA_PATH

        if not faq_file_path.exists():
            raise FileNotFoundError(f"FAQ dataset file not found at: {faq_file_path}")

        current_count = self.count()
        if current_count > 0 and not force_reindex:
            logger.info(f"ChromaDB collection already contains {current_count} documents. Skipping re-indexing.")
            return current_count

        with open(faq_file_path, "r", encoding="utf-8") as f:
            faqs: List[Dict[str, Any]] = json.load(f)

        if not faqs:
            logger.warning("No FAQs found in dataset.")
            return 0

        # If reindexing, clear existing collection items
        if current_count > 0 and force_reindex:
            logger.info("Force reindex requested. Resetting collection.")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        # Texts to embed: combine question, keywords, and answer for high-accuracy retrieval
        texts_to_embed: List[str] = []

        for faq in faqs:
            faq_id = faq["id"]
            question = faq["question"]
            answer = faq["answer"]
            category = faq.get("category", "general_faq")
            keywords = ", ".join(faq.get("keywords", []))
            source = faq.get("source", "Vehicle Insurance FAQ")

            # Document content stored in ChromaDB
            stored_doc = f"Question: {question}\nAnswer: {answer}"
            # Embed question + keywords + answer
            embed_text = f"Question: {question} Keywords: {keywords} Answer: {answer}"

            ids.append(faq_id)
            documents.append(stored_doc)
            metadatas.append({
                "faq_id": faq_id,
                "question": question,
                "answer": answer,
                "category": category,
                "source": source
            })
            texts_to_embed.append(embed_text)

        logger.info(f"Generating embeddings for {len(texts_to_embed)} FAQs using all-MiniLM-L6-v2...")
        embeddings = embedding_service.embed_texts(texts_to_embed)

        logger.info("Adding FAQs into ChromaDB collection...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        total_indexed = self.count()
        logger.info(f"Successfully indexed {total_indexed} FAQs into ChromaDB.")
        return total_indexed

    def retrieve(
        self,
        query: str,
        top_k: int = settings.TOP_K_RESULTS,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k relevant FAQs for a user query.
        Returns empty list if top match does not meet the similarity threshold.
        """
        if self.count() == 0:
            logger.warning("ChromaDB is empty! Ingesting FAQs now...")
            self.ingest_faqs()

        cleaned_query = clean_query_for_search(query)
        if not cleaned_query:
            return []

        query_embedding = embedding_service.embed_text(cleaned_query)

        where_filter = {"category": category} if category else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_items: List[Dict[str, Any]] = []

        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            return []

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)

        for i in range(len(ids)):
            # Cosine distance to similarity: similarity = 1 - distance
            distance = distances[i]
            similarity_score = round(max(0.0, min(1.0, 1.0 - distance)), 3)

            retrieved_items.append({
                "id": ids[i],
                "question": metadatas[i].get("question", ""),
                "answer": metadatas[i].get("answer", ""),
                "category": metadatas[i].get("category", "general_faq"),
                "source": metadatas[i].get("source", "Vehicle Insurance FAQ"),
                "similarity_score": similarity_score,
                "raw_document": documents[i]
            })

        # Filter against threshold:
        # Check if the top result meets the similarity threshold
        if retrieved_items and retrieved_items[0]["similarity_score"] < self.similarity_threshold:
            logger.info(
                f"Top match '{retrieved_items[0]['id']}' score {retrieved_items[0]['similarity_score']} "
                f"below threshold {self.similarity_threshold}. Discarding results."
            )
            return []

        return retrieved_items

    def get_all_faqs(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Reads FAQs directly from the JSON dataset for listing with optional category filter.
        """
        if not settings.FAQ_DATA_PATH.exists():
            return []

        with open(settings.FAQ_DATA_PATH, "r", encoding="utf-8") as f:
            faqs: List[Dict[str, Any]] = json.load(f)

        if category:
            faqs = [f for f in faqs if f.get("category", "").lower() == category.lower()]

        return faqs

# Singleton instance
retrieval_service = RetrievalService()

