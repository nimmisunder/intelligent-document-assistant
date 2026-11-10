
def calculate_metrics(retrieved_ids, relevant_ids, k=5):

    retrieved = retrieved_ids[:k]

    relevant_set = set(relevant_ids)

    if not relevant_set:
        return {
            "precision_at_k": 0,
            "recall_at_k": 0,
            "mrr": 0,
            "hit_rate_at_k": 0
        }

    relevant_retrieved = [
        chunk_id
        for chunk_id in retrieved
        if chunk_id in relevant_set
    ]

    precision = len(relevant_retrieved) / k

    recall = (
        len(set(relevant_retrieved)) / len(relevant_set)
    )

    reciprocal_rank = 0

    for rank, chunk_id in enumerate(retrieved, start=1):

        if chunk_id in relevant_set:
            reciprocal_rank = 1 / rank
            break

    hit_rate = 1 if relevant_retrieved else 0

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": reciprocal_rank,
        "hit_rate_at_k": hit_rate
    }
