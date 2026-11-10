
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.indexing_service import indexing_service
from app.services.embedding_service import generate_embeddings
from app.services.hybrid_search import HybridRetriever
from app.services.reranking_service import Reranker
from app.services.llm_service import generate_answer


router = APIRouter(
    prefix="/chat",
    tags=["AI Chat"]
)

hybrid_retriever = HybridRetriever(
    indexing_service.vector_store,
    indexing_service.bm25_retriever
)

reranker = Reranker()


class ChatRequest(BaseModel):
    question: str
    document_id: str | None = None


@router.post("/ask")
def ask_question(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:

        # 1. Generate question embedding
        query_embedding = generate_embeddings([
            request.question
        ])[0]

        # 2. Retrieve candidates using hybrid search
        retrieved_chunks = hybrid_retriever.search(
            query=request.question,
            query_embedding=query_embedding,
            top_k=10,
            document_id=request.document_id
        )

        if not retrieved_chunks:
            return {
                "question": request.question,
                "answer": "No relevant information found in the uploaded documents.",
                "sources": []
            }

        # 3. Rerank candidates
        ranked_chunks = reranker.rerank(
            query=request.question,
            retrieved_chunks=retrieved_chunks,
            top_k=3
        )

        # 4. Prepare document context
        context = "\n\n".join(
            result["metadata"]["text"]
            for result in ranked_chunks
        )

        # 5. Generate answer using Ollama
        answer = generate_answer(
            question=request.question,
            context=context
        )

        # 6. Prepare source citations
        sources = []

        for result in ranked_chunks:

            metadata = result["metadata"]

            sources.append({
                "filename": metadata["filename"],
                "page_number": metadata["page_number"],
                "chunk_id": metadata["chunk_id"],
                "relevance_score": result["score"]
            })

        return {
            "question": request.question,
            "answer": answer,
            "retrieved_chunks": len(ranked_chunks),
            "sources": sources
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {str(error)}"
        )
