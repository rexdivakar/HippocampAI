"""Embedding service for generating text embeddings using sentence-transformers."""

import hashlib
import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings with caching support."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformer model to use
            cache_size: Maximum number of embeddings to cache (0 to disable)

        Raises:
            RuntimeError: If model fails to load
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self._cache: dict = {}
        self._model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence-transformer model."""
        try:
            logger.info(f"Loading model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model '{self.model_name}' loaded successfully")
            logger.info(f"Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise RuntimeError(f"Could not load embedding model '{self.model_name}'") from e

    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.

        Args:
            text: Input text

        Returns:
            MD5 hash of the text
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _add_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """
        Add an embedding to the cache.

        Args:
            text: Input text
            embedding: Generated embedding vector
        """
        if self.cache_size == 0:
            return

        cache_key = self._get_cache_key(text)

        # Evict oldest entry if cache is full (simple FIFO)
        if len(self._cache) >= self.cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug("Cache full, evicted oldest entry")

        self._cache[cache_key] = embedding.copy()

    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve an embedding from cache if available.

        Args:
            text: Input text

        Returns:
            Cached embedding or None if not found
        """
        if self.cache_size == 0:
            return None

        cache_key = self._get_cache_key(text)
        return self._cache.get(cache_key)

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Numpy array containing the embedding vector

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If embedding generation fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        # Check cache first
        cached_embedding = self._get_from_cache(text)
        if cached_embedding is not None:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return cached_embedding

        try:
            # Generate embedding
            embedding = self._model.encode(text, convert_to_numpy=True)

            # Add to cache
            self._add_to_cache(text, embedding)

            logger.debug(f"Generated embedding for text: {text[:50]}...")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError("Embedding generation failed") from e

    def generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts to embed

        Returns:
            List of numpy arrays containing embedding vectors

        Raises:
            ValueError: If texts list is empty or contains invalid entries
            RuntimeError: If batch embedding generation fails
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of strings")

        if not all(isinstance(t, str) and t for t in texts):
            raise ValueError("All items in the list must be non-empty strings")

        embeddings = []
        texts_to_encode = []
        text_indices = []

        # Check cache for each text
        for idx, text in enumerate(texts):
            cached_embedding = self._get_from_cache(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                logger.debug(f"Cache hit for batch item {idx}")
            else:
                embeddings.append(None)  # Placeholder
                texts_to_encode.append(text)
                text_indices.append(idx)

        # Generate embeddings for uncached texts
        if texts_to_encode:
            try:
                logger.debug(f"Generating embeddings for {len(texts_to_encode)} uncached texts")
                new_embeddings = self._model.encode(texts_to_encode, convert_to_numpy=True)

                # Handle single vs batch output
                if len(texts_to_encode) == 1:
                    new_embeddings = [new_embeddings]

                # Update cache and results
                for text, embedding, idx in zip(texts_to_encode, new_embeddings, text_indices):
                    self._add_to_cache(text, embedding)
                    embeddings[idx] = embedding

            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                raise RuntimeError("Batch embedding generation failed") from e

        logger.info(
            f"Generated embeddings for {len(texts)} texts ({len(texts_to_encode)} new, {len(texts) - len(texts_to_encode)} cached)"
        )
        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension size
        """
        return self._model.get_sentence_embedding_dimension()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> dict:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
        }

    def switch_model(self, model_name: str) -> None:
        """
        Switch to a different embedding model.

        Args:
            model_name: Name of the new model to load

        Raises:
            RuntimeError: If new model fails to load
        """
        logger.info(f"Switching from '{self.model_name}' to '{model_name}'")
        old_model = self.model_name

        try:
            self.model_name = model_name
            self._load_model()
            # Clear cache since embeddings are from different model
            self.clear_cache()
            logger.info(f"Successfully switched to model '{model_name}'")
        except Exception:
            # Revert to old model on failure
            logger.error(f"Failed to switch model, reverting to '{old_model}'")
            self.model_name = old_model
            self._load_model()
            raise
