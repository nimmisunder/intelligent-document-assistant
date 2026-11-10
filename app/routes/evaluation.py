
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.indexing_service import indexing_service
from app.services.embedding_service import generate_embeddings
from app.services.hybrid_search import HybridRetriever
from app.services.reranking_service import Reranker
from app.services.evaluation_service import calculate_metrics


router = APIRouter(
    prefix="/evaluation",
    tags=["RAG Evaluation"]
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "evaluation" / "evaluation_dataset.json"


@router.get("/run")
def run_evaluation():

    if not DATASET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation dataset not found"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise HTTPException(
            status_code=400,
            detail="Evaluation dataset is empty"
        )

    hybrid = HybridRetriever(
        indexing_service.vector_store,
        indexing_service.bm25_retriever
    )

    reranker = Reranker()

    all_metrics = []
    question_results = []

    for item in dataset:

        question = item["question"]
        document_id = item["document_id"]
        relevant_ids = item["relevant_chunk_ids"]

        query_embedding = generate_embeddings([question])[0]

        retrieved = hybrid.search(
            query=question,
            query_embedding=query_embedding,
            top_k=10,
            document_id=document_id
        )

        reranked = reranker.rerank(
            query=question,
            retrieved_chunks=retrieved,
            top_k=5
        )

        retrieved_ids = [
            result["metadata"]["chunk_id"]
            for result in reranked
        ]

        metrics = calculate_metrics(
            retrieved_ids,
            relevant_ids,
            k=5
        )

        all_metrics.append(metrics)

        question_results.append({
            "question": question,
            "retrieved_chunks": retrieved_ids,
            "metrics": metrics
        })

    metric_names = all_metrics[0].keys()

    average_metrics = {
        metric: round(
            sum(result[metric] for result in all_metrics)
            / len(all_metrics),
            4
        )
        for metric in metric_names
    }

    return {
        "total_questions": len(dataset),
        "average_metrics": average_metrics,
        "question_results": question_results
    }
