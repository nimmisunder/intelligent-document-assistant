import fitz


def extract_text_from_pdf(pdf_content: bytes):

    pdf = fitz.open(
        stream=pdf_content,
        filetype="pdf"
    )

    extracted_pages = []

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text("text")

        extracted_pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    pdf.close()

    return extracted_pages