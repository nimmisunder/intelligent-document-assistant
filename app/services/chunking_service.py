
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    pages,
    document_id,
    filename,
    chunk_size=500,
    chunk_overlap=100
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    chunk_index = 0

    for page in pages:

        page_text = page["text"]
        page_number = page["page_number"]

        if not page_text.strip():
            continue

        chunks = splitter.split_text(page_text)

        for chunk in chunks:

            chunk_index += 1

            all_chunks.append({
                "chunk_id": f"{document_id}_{chunk_index}",
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk
            })

    return all_chunks

