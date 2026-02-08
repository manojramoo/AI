from dataclasses import dataclass
from typing import List

from agents.components import (
    ChromaVectorStore,
    EmbeddingModel,
    LLMClient,
    Neo4jKnowledgeGraph,
    Ranker,
    RetrievalResult,
)


@dataclass
class AgentResponse:
    answer: str
    suggestions: List[str]
    context: List[RetrievalResult]


class AgenticRAGPipeline:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        knowledge_graph: Neo4jKnowledgeGraph,
        embedder: EmbeddingModel,
        ranker: Ranker,
        llm: LLMClient,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.embedder = embedder
        self.ranker = ranker
        self.llm = llm

    def run(self, user_question: str, top_k: int = 5) -> AgentResponse:
        query_embedding = self.embedder.embed_text(user_question)
        vector_hits = self.vector_store.query(query_embedding, top_k=top_k)
        graph_hits = self.knowledge_graph.query(user_question, top_k=top_k)
        ranked = self.ranker.rank(vector_hits + graph_hits)
        answer = self.llm.generate(self._build_prompt(user_question, ranked))
        suggestions = self._suggest_followups(user_question)
        return AgentResponse(answer=answer, suggestions=suggestions, context=ranked)

    def _build_prompt(self, question: str, context: List[RetrievalResult]) -> str:
        context_block = "\n".join(f"- {item.text}" for item in context)
        return (
            "Answer the question using the context below.\n\n"
            f"Question: {question}\n\n"
            "Context:\n"
            f"{context_block}\n"
        )

    def _suggest_followups(self, question: str) -> List[str]:
        return [
            f"Can you clarify the scope of '{question}'?",
            f"What are common alternatives to {question}?",
        ]

