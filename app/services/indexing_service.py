from app.services.chunking_service import create_chunks
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import VectorStore
from app.services.bm25_service import BM25Retriever

import faiss
from app.services.embedding_service import generate_embeddings

class IndexingService:

    def __init__(self):

        self.vector_store = VectorStore()
        self.vector_store.load()

        self.bm25_retriever = BM25Retriever()

        # Rebuild BM25 from persisted FAISS metadata
        self.bm25_retriever.build_index(
            self.vector_store.metadata
        )

    def index_document(
        self,
        document_id,
        filename,
        pages
    ):

        # 1. Convert pages into chunks
        chunks = create_chunks(
            pages=pages,
            document_id=document_id,
            filename=filename
        )

        if not chunks:
            return 0

        # 2. Generate embeddings
        embeddings = generate_embeddings([
            chunk["text"] for chunk in chunks
        ])

        # 3. Add chunks to FAISS
        self.vector_store.add_documents(
            chunks,
            embeddings
        )

        # 4. Save FAISS index and metadata
        self.vector_store.save()

        # 5. Rebuild BM25 with all indexed chunks
        self.bm25_retriever.build_index(
            self.vector_store.metadata
        )

        return len(chunks)
    
    def delete_document(self, document_id):

        # Keep only chunks belonging to other documents
        remaining_chunks = [
            chunk
            for chunk in self.vector_store.metadata
            if chunk["document_id"] != document_id
        ]

        # Rebuild the FAISS index
        self.vector_store.index = faiss.IndexFlatIP(
            self.vector_store.dimension
        )
        self.vector_store.metadata = []

        if remaining_chunks:

            embeddings = generate_embeddings([
                chunk["text"]
                for chunk in remaining_chunks
            ])

            self.vector_store.add_documents(
                remaining_chunks,
                embeddings
            )

        # Persist the rebuilt index
        self.vector_store.save()

        # Rebuild BM25
        self.bm25_retriever.build_index(
            remaining_chunks
        )
# Shared indexing service instance
indexing_service = IndexingService()