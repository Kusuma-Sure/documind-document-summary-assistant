from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_embeddings(chunks):

    if not chunks:
        return []

    texts = []

    for chunk in chunks:

        if isinstance(
            chunk,
            dict
        ):

            texts.append(
                chunk.get(
                    "text",
                    ""
                )
            )

        else:

            texts.append(
                str(chunk)
            )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings