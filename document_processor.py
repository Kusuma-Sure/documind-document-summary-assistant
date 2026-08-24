import io
import fitz
import pytesseract
from PIL import Image


def extract_pdf_text(file_bytes):

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    all_text = []
    total_pages = len(document)

    extraction_methods = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        # Try normal PDF text extraction
        page_text = page.get_text(
            "text"
        ).strip()

        if page_text:

            extraction_methods.append(
                "PDF Text"
            )

            all_text.append(
                {
                    "page": page_number,
                    "text": page_text
                }
            )

        else:

            # OCR fallback for scanned pages
            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    2,
                    2
                )
            )

            image_bytes = pix.tobytes(
                "png"
            )

            image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )

            ocr_text = pytesseract.image_to_string(
                image
            ).strip()

            extraction_methods.append(
                "PDF OCR"
            )

            if ocr_text:

                all_text.append(
                    {
                        "page": page_number,
                        "text": ocr_text
                    }
                )

    document.close()

    # Decide extraction method
    if all(
        method == "PDF Text"
        for method in extraction_methods
    ):

        extraction_method = "PDF Text"

    elif any(
        method == "PDF OCR"
        for method in extraction_methods
    ):

        extraction_method = "PDF OCR"

    else:

        extraction_method = "PDF"

    # Full text for summarization/statistics
    full_text = "\n\n".join(
        item["text"]
        for item in all_text
    )

    return (
        full_text,
        total_pages,
        extraction_method,
        all_text
    )