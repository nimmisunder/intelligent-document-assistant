
import uuid
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.pdf_service import extract_text_from_pdf
from app.services.indexing_service import IndexingService
from app.database import get_connection


router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

# Create one shared indexing service
from app.services.indexing_service import indexing_service

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="File is empty"
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 10 MB"
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file"
        )

    document_id = str(uuid.uuid4())
    stored_filename = f"{document_id}.pdf"
    file_path = UPLOAD_DIR / stored_filename

    try:

        # 1. Extract PDF text
        extracted_pages = extract_text_from_pdf(content)

        if not any(page["text"].strip() for page in extracted_pages):
            raise ValueError("PDF contains no extractable text")

        # 2. Save PDF
        file_path.write_bytes(content)

        # 3. Save document metadata
        with get_connection() as connection:

            connection.execute("""
                INSERT INTO documents (
                    id,
                    original_filename,
                    stored_filename,
                    file_size,
                    total_pages,
                    uploaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                document_id,
                file.filename,
                stored_filename,
                len(content),
                len(extracted_pages),
                datetime.now(timezone.utc).isoformat()
            ))

        # 4. Index document for RAG
        chunk_count = indexing_service.index_document(
            document_id=document_id,
            filename=file.filename,
            pages=extracted_pages
        )

        return {
            "message": "Document uploaded and indexed successfully",
            "document_id": document_id,
            "filename": file.filename,
            "total_pages": len(extracted_pages),
            "total_chunks": chunk_count,
            "file_size": len(content)
        }

    except Exception as error:

        file_path.unlink(missing_ok=True)

        # Roll back document metadata if indexing fails
        with get_connection() as connection:
            connection.execute(
                "DELETE FROM documents WHERE id = ?",
                (document_id,)
            )

        raise HTTPException(
            status_code=500,
            detail=f"Unable to process or index PDF: {str(error)}"
        )


@router.get("/")
def list_documents():

    with get_connection() as connection:

        documents = connection.execute("""
            SELECT
                id,
                original_filename,
                file_size,
                total_pages,
                uploaded_at
            FROM documents
            ORDER BY uploaded_at DESC
        """).fetchall()

    return {
        "total_documents": len(documents),
        "documents": [dict(doc) for doc in documents]
    }


@router.delete("/{document_id}")
def delete_document(document_id: str):

    with get_connection() as connection:

        document = connection.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,)
        ).fetchone()

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

    try:

        # Remove chunks and vectors from both indexes
        indexing_service.delete_document(document_id)

        # Delete database record
        with get_connection() as connection:
            connection.execute(
                "DELETE FROM documents WHERE id = ?",
                (document_id,)
            )

        # Delete physical PDF
        file_path = UPLOAD_DIR / document["stored_filename"]
        file_path.unlink(missing_ok=True)

        return {
            "message": "Document and its index deleted successfully",
            "document_id": document_id
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to delete document: {str(error)}"
        )
