from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source: str
    metadata: dict


class ChromaVectorStore:
    def __init__(self, collection: str) -> None:
        self.collection = collection

    def upsert(self, chunks: Iterable[DocumentChunk]) -> None:
        for _chunk in chunks:
            continue

    def query(self, query_embedding: List[float], top_k: int) -> List[RetrievalResult]:
        return []


class Neo4jKnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self.password = password

    def upsert_entities(self, chunks: Iterable[DocumentChunk]) -> None:
        for _chunk in chunks:
            continue

    def query(self, query: str, top_k: int) -> List[RetrievalResult]:
        return []


class EmbeddingModel:
    def embed_text(self, text: str) -> List[float]:
        return [0.0]


class Ranker:
    def rank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        return sorted(results, key=lambda item: item.score, reverse=True)


class LLMClient:
    def generate(self, prompt: str) -> str:
        return "Generated answer placeholder."

