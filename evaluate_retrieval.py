
import json
from pathlib import Path

from app.services.indexing_service import indexing_service
from app.services.embedding_service import generate_embeddings
from app.services.hybrid_search import HybridRetriever
from app.services.reranking_service import Reranker
from app.services.evaluation_service import calculate_metrics


# Project root directory
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "evaluation" / "evaluation_dataset.json"
RESULTS_PATH = BASE_DIR / "evaluation" / "evaluation_results.json"


# Load evaluation dataset
with open(DATASET_PATH, "r", encoding="utf-8") as file:
    dataset = json.load(file)


if not dataset:
    raise ValueError("Evaluation dataset is empty")


# Initialize retrieval components
hybrid = HybridRetriever(
    indexing_service.vector_store,
    indexing_service.bm25_retriever
)

reranker = Reranker()

all_metrics = []
question_results = []


# Evaluate each question
for item in dataset:

    question = item["question"]
    document_id = item["document_id"]
    relevant_ids = item["relevant_chunk_ids"]

    query_embedding = generate_embeddings([question])[0]

    # Retrieve candidates
    retrieved = hybrid.search(
        query=question,
        query_embedding=query_embedding,
        top_k=10,
        document_id=document_id
    )

    # Rerank candidates
    reranked = reranker.rerank(
        query=question,
        retrieved_chunks=retrieved,
        top_k=5
    )

    retrieved_ids = [
        result["metadata"]["chunk_id"]
        for result in reranked
    ]

    # Calculate metrics
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

    print("\nQuestion:", question)
    print("Retrieved chunks:", retrieved_ids)
    print("Metrics:", metrics)


# Calculate average metrics
metric_names = all_metrics[0].keys()

average_metrics = {
    metric: round(
        sum(result[metric] for result in all_metrics)
        / len(all_metrics),
        4
    )
    for metric in metric_names
}


# Prepare final evaluation report
evaluation_results = {
    "total_questions": len(dataset),
    "average_metrics": average_metrics,
    "question_results": question_results
}


# Save results
RESULTS_PATH.parent.mkdir(exist_ok=True)

with open(RESULTS_PATH, "w", encoding="utf-8") as file:

    json.dump(
        evaluation_results,
        file,
        ensure_ascii=False,
        indent=4
    )


print("\n========== FINAL EVALUATION ==========")

for metric, value in average_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nResults saved to:", RESULTS_PATH)
