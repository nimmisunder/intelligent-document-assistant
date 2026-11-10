import re
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        self.chunks = []
        self.bm25 = None

    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def build_index(self, chunks):
        self.chunks = chunks

        tokenized_corpus = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query, top_k=3, document_id=None):

        if self.bm25 is None or top_k <= 0:
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Apply document filtering before selecting top results
        candidate_positions = [
            i for i, chunk in enumerate(self.chunks)
            if document_id is None
            or chunk["document_id"] == document_id
        ]

        ranked_positions = sorted(
            candidate_positions,
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for position in ranked_positions:
            score = float(scores[position])

            # Ignore chunks with no keyword match
            if score <= 0:
                continue

            results.append({
                "score": score,
                "metadata": self.chunks[position]
            })

            if len(results) >= top_k:
                break

        return results