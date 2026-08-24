import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load the embedding model once
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def split_text(text, max_words=120):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    chunks = []

    current_chunk = []
    current_words = 0

    for sentence in sentences:

        words = sentence.split()

        if not words:
            continue

        if current_words + len(words) <= max_words:

            current_chunk.append(sentence)
            current_words += len(words)

        else:

            if current_chunk:

                chunks.append(
                    " ".join(current_chunk)
                )

            current_chunk = [sentence]
            current_words = len(words)


    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )


    return chunks


def answer_question(text, question):

    if not text or not question:

        return (
            "Please provide a document "
            "and a question."
        )


    # ----------------------------------------------
    # Create document chunks
    # ----------------------------------------------

    chunks = split_text(text)


    if not chunks:

        return (
            "Unable to find readable content "
            "in the document."
        )


    # ----------------------------------------------
    # Create embeddings
    # ----------------------------------------------

    chunk_embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )


    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )


    # ----------------------------------------------
    # Calculate semantic similarity
    # ----------------------------------------------

    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )[0]


    # ----------------------------------------------
    # Find best matching chunk
    # ----------------------------------------------

    best_index = similarities.argmax()

    best_score = similarities[best_index]


    # ----------------------------------------------
    # Confidence check
    # ----------------------------------------------

    if best_score < 0.25:

        return (
            "I could not find enough relevant "
            "information in the document to answer "
            "this question."
        )


    answer = chunks[best_index]


    return answer