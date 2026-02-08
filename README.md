# Agentic RAG (GraphLang + Neo4j + Chroma)
This repository contains a reference implementation for an agentic RAG workflow that mirrors the provided architecture. The workflow is declared in GraphLang and backed by a Neo4j knowledge graph plus a Chroma vector store.

## Architecture
The pipeline follows these stages:
1. **Data ingestion & processing**: load Q&A data, clean it, and chunk it.
2. **Embeddings & storage**: create embeddings and persist them into Chroma and Neo4j.
3. **Agentic workflow**:
   - Text Processing Agent
   - Retriever Agent
   - Ranker Agent
   - LLM Agent
   - Critic Agent
   - Recommendation Agent

The GraphLang diagram that encodes this architecture lives at `graph/agent.graphlang`.

## Repository layout
- `graph/agent.graphlang` — GraphLang definition of the workflow.
- `src/agents/` — core agent components and pipeline logic.
- `src/ingestion.py` — data ingestion entrypoint.

## Usage
```bash
python -m src.ingestion
```

## Notes
- `ChromaVectorStore` and `Neo4jKnowledgeGraph` are lightweight stubs meant to be wired to actual services.
- Update connection details in `src/ingestion.py` for your Neo4j instance.
