from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, retrieved_chunks, top_k=3):

        if not retrieved_chunks:
            return []

        # Prepare query and chunk pairs
        pairs = [
            (query, result["metadata"]["text"])
            for result in retrieved_chunks
        ]

        # Calculate relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to results
        reranked_results = []

        for result, score in zip(retrieved_chunks, scores):
            reranked_results.append({
                "score": float(score),
                "metadata": result["metadata"]
            })

        # Sort by relevance score
        reranked_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return reranked_results[:top_k]