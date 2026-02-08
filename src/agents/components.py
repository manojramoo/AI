from dataclasses import dataclass
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import chromadb
from neo4j import GraphDatabase


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
    def __init__(self, collection: str, url: str = "http://localhost:8000") -> None:
        self.collection = collection
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8000
        ssl = parsed.scheme == "https"
        self.client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
        self._collection = self.client.get_or_create_collection(collection)

    def upsert(
        self, chunks: Iterable[DocumentChunk], embeddings: Optional[List[List[float]]] = None
    ) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return
        if embeddings is None:
            embeddings = [[0.0] for _ in chunk_list]
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunk_list],
            documents=[chunk.text for chunk in chunk_list],
            metadatas=[chunk.metadata for chunk in chunk_list],
            embeddings=embeddings,
        )

    def query(self, query_embedding: List[float], top_k: int) -> List[RetrievalResult]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"],
        )
        hits: List[RetrievalResult] = []
        for idx, chunk_id in enumerate(results.get("ids", [[]])[0]):
            text = results.get("documents", [[]])[0][idx]
            metadata = results.get("metadatas", [[]])[0][idx] or {}
            distance = results.get("distances", [[]])[0][idx]
            score = 1 / (1 + distance) if distance is not None else 0.0
            hits.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    source=metadata.get("source", "chroma"),
                    metadata=metadata,
                )
            )
        return hits


class Neo4jKnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self.uri = self._normalize_uri(uri)
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def _normalize_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme in {"http", "https"}:
            host = parsed.hostname or "localhost"
            port = 7687 if parsed.port == 7474 or parsed.port is None else parsed.port
            return f"neo4j://{host}:{port}"
        return uri

    def upsert_entities(self, chunks: Iterable[DocumentChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return
        query = (
            "UNWIND $rows AS row "
            "MERGE (d:Document {id: row.id}) "
            "SET d.text = row.text, d.source = row.source, d.metadata = row.metadata"
        )
        rows = [
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.metadata.get("source", "unknown"),
                "metadata": chunk.metadata,
            }
            for chunk in chunk_list
        ]
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, rows=rows))

    def query(self, query: str, top_k: int) -> List[RetrievalResult]:
        cypher = (
            "MATCH (d:Document) "
            "WHERE d.text CONTAINS $query "
            "RETURN d.id AS id, d.text AS text, d.source AS source, d.metadata AS metadata "
            "LIMIT $limit"
        )
        with self.driver.session(database=self.database) as session:
            records = session.run(cypher, query=query, limit=top_k)
            return [
                RetrievalResult(
                    chunk_id=record["id"],
                    text=record["text"],
                    score=1.0,
                    source=record.get("source") or "neo4j",
                    metadata=record.get("metadata") or {},
                )
                for record in records
            ]

    def close(self) -> None:
        self.driver.close()


class EmbeddingModel:
    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return [0.0 for _ in range(self.dimensions)]
        vector = [0.0 for _ in range(self.dimensions)]
        for idx, value in enumerate(text.encode("utf-8")):
            vector[idx % self.dimensions] += (value % 31) / 31.0
        norm = sum(component * component for component in vector) ** 0.5
        if norm == 0:
            return vector
        return [component / norm for component in vector]


class Ranker:
    def rank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        return sorted(results, key=lambda item: item.score, reverse=True)


class LLMClient:
    def generate(self, prompt: str) -> str:
        return "Generated answer placeholder."
