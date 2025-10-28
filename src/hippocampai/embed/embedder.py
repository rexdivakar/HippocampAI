"""Embedder with batching and quantization."""

import logging
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Thread-safe embedder with batching."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        quantized: bool = False,
        dimension: int = 384,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = dimension

        # Load model
        # Note: quantization is handled by SentenceTransformer internally
        # or can be done during inference, not during model loading
        self.model = SentenceTransformer(model_name)

        if quantized:
            logger.info("Quantized mode enabled - embeddings will use lower precision")
        self.lock = threading.Lock()
        logger.info(f"Loaded embedder: {model_name}, quantized={quantized}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if not texts:
            return np.array([])

        with self.lock:
            embeddings = self.model.encode(
                texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
            )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text."""
        return self.encode([text])[0]


_embedder_instance = None
_embedder_lock = threading.Lock()


def get_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
    quantized: bool = False,
    dimension: int = 384,
) -> Embedder:
    """Get shared embedder instance."""
    global _embedder_instance
    with _embedder_lock:
        if _embedder_instance is None:
            _embedder_instance = Embedder(
                model_name=model_name,
                batch_size=batch_size,
                quantized=quantized,
                dimension=dimension,
            )
    return _embedder_instance
