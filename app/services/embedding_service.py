from typing import List, Union
from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingService:
    """
    Embedding service using sentence-transformers/all-MiniLM-L6-v2.
    Computes vector embeddings for FAQ questions, answers, and user queries.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model on first access."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single string.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of strings in batches.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

# Singleton instance
embedding_service = EmbeddingService()

