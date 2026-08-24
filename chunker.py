import re


def chunk_text(
    pages,
    chunk_size=500,
    overlap=100
):
    """
    Create chunks while preserving
    the original PDF page number.
    """

    if not pages:

        return []

    chunks = []

    for page_data in pages:

        page_number = page_data.get(
            "page",
            "Unknown"
        )

        text = page_data.get(
            "text",
            ""
        )

        if not text.strip():

            continue

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        words = text.split()

        if not words:

            continue

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk_words = words[
                start:end
            ]

            chunk = " ".join(
                chunk_words
            )

            chunks.append(
                {
                    "text": chunk,
                    "page": page_number
                }
            )

            if end >= len(words):

                break

            start = end - overlap

    return chunks