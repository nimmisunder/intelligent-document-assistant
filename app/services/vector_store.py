
import faiss
import numpy as np
import json

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent

VECTOR_DIR = BASE_DIR / "vector_store"

VECTOR_DIR.mkdir(exist_ok=True)

INDEX_PATH = VECTOR_DIR / "faiss.index"

METADATA_PATH = VECTOR_DIR / "metadata.json"


class VectorStore:

    def __init__(self, dimension=384):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.metadata = []

    def add_documents(self, chunks, embeddings):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings count mismatch")

        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch")

        if not chunks:
            return

        self.index.add(embeddings)

        self.metadata.extend(chunks)

  
    def search(self, query_embedding, top_k=5, document_id=None):

        if self.index.ntotal == 0 or top_k <= 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        ).reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension mismatch")

        # Select eligible chunks
        matching_positions = [
            i for i, item in enumerate(self.metadata)
            if document_id is None
            or item["document_id"] == document_id
        ]

        if not matching_positions:
            return []

        # Build temporary index for filtered search
        filtered_vectors = np.array(
            [
                self.index.reconstruct(i)
                for i in matching_positions
            ],
            dtype="float32"
        )

        filtered_index = faiss.IndexFlatIP(self.dimension)
        filtered_index.add(filtered_vectors)

        scores, indices = filtered_index.search(
            query_embedding,
            min(top_k, len(matching_positions))
        )

        results = []

        for score, filtered_idx in zip(scores[0], indices[0]):

            if filtered_idx == -1:
                continue

            original_idx = matching_positions[filtered_idx]

            results.append({
                "score": float(score),
                "metadata": self.metadata[original_idx]
            })

        return results

    def save(self):

        faiss.write_index(
            self.index,
            str(INDEX_PATH)
        )

        with open(METADATA_PATH, "w", encoding="utf-8") as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=4
            )

    def load(self):

        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            return

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        with open(METADATA_PATH, "r", encoding="utf-8") as file:

            self.metadata = json.load(file)

        if self.index.d != self.dimension:
            raise ValueError("Stored index dimension mismatch")

        if self.index.ntotal != len(self.metadata):
            raise ValueError("Index and metadata are inconsistent")
