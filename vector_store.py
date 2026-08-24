import faiss
import numpy as np


def build_index(embeddings):

    if embeddings is None:
        raise ValueError(
            "Embeddings cannot be None."
        )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    if embeddings.size == 0:
        raise ValueError(
            "Embeddings cannot be empty."
        )

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(
            1,
            -1
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    faiss.normalize_L2(
        embeddings
    )

    index.add(
        embeddings
    )

    return index


def search_index(
    index,
    query_embedding,
    chunks,
    top_k=5,
    similarity_threshold=0.35
):

    if index is None:
        return []

    if chunks is None or len(chunks) == 0:
        return []

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(
            1,
            -1
        )

    faiss.normalize_L2(
        query_embedding
    )

    number_to_search = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        number_to_search
    )

    results = []

    seen_pages = set()

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(chunks):
            continue

        score = float(score)

        # Ignore weak matches
        if score < similarity_threshold:
            continue

        chunk = chunks[idx]

        page = chunk.get(
            "page",
            "Unknown"
        )

        # Avoid repeatedly showing the same page
        if page in seen_pages:
            continue

        seen_pages.add(page)

        results.append(
            {
                "score": score,
                "chunk": chunk
            }
        )

    return results