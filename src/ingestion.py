import csv
import json
from pathlib import Path
from typing import Iterable, List

from agents.components import ChromaVectorStore, DocumentChunk, EmbeddingModel, Neo4jKnowledgeGraph


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def clean_rows(rows: Iterable[dict]) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for row in rows:
        text = (row.get("question", "") + "\n" + row.get("answer", "")).strip()
        if not text:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=row.get("id", text[:32]),
                text=text,
                metadata={"source": row.get("source", "unknown")},
            )
        )
    return chunks


def ingest(
    rows: Iterable[dict],
    vector_store: ChromaVectorStore,
    knowledge_graph: Neo4jKnowledgeGraph,
    embedder: EmbeddingModel,
) -> None:
    chunks = clean_rows(rows)
    embeddings = [embedder.embed_text(chunk.text) for chunk in chunks]
    vector_store.upsert(chunks, embeddings=embeddings)
    knowledge_graph.upsert_entities(chunks)


def main() -> None:
    data_path = Path("data/sample.json")
    if data_path.suffix == ".json":
        rows = load_json(data_path)
    else:
        rows = load_csv(data_path)

    vector_store = ChromaVectorStore(collection="qa", url="http://localhost:8000")
    knowledge_graph = Neo4jKnowledgeGraph(
        uri="http://localhost:7474",
        user="neo4j",
        password="password123",
    )
    embedder = EmbeddingModel(dimensions=128)
    ingest(rows, vector_store, knowledge_graph, embedder)
    knowledge_graph.close()


if __name__ == "__main__":
    main()
