"""Dense + sparse retrieval over authoritative NBR documents (paper Sec. 3.3.4)."""
from .embedder import Embedder, HashingEmbedder, SentenceTransformerEmbedder, load_embedder
from .index import DocumentIndex
from .retriever import HybridRetriever, RetrievedChunk

__all__ = [
    "Embedder", "HashingEmbedder", "SentenceTransformerEmbedder", "load_embedder",
    "DocumentIndex", "HybridRetriever", "RetrievedChunk",
]
