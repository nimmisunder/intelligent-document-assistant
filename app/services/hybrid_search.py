class HybridRetriever:

    def __init__(self, vector_store, bm25_retriever):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def search(
        self,
        query,
        query_embedding,
        top_k=5,
        document_id=None,
        rrf_k=60
    ):

        # Retrieve results from FAISS
        vector_results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            document_id=document_id
        )

        # Retrieve results from BM25
        keyword_results = self.bm25_retriever.search(
            query,
            top_k=top_k,
            document_id=document_id
        )

        # Combine rankings
        fused_scores = {}
        chunk_metadata = {}

        result_lists = [vector_results, keyword_results]

        for results in result_lists:

            for rank, result in enumerate(results, start=1):

                metadata = result["metadata"]
                chunk_id = metadata["chunk_id"]

                if chunk_id not in fused_scores:
                    fused_scores[chunk_id] = 0
                    chunk_metadata[chunk_id] = metadata

                fused_scores[chunk_id] += 1 / (rrf_k + rank)

        # Sort by combined RRF score
        ranked_chunks = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            {
                "score": score,
                "metadata": chunk_metadata[chunk_id]
            }
            for chunk_id, score in ranked_chunks[:top_k]
        ]