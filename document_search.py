from chunker import split_into_chunks
from embeddings import create_embeddings
from vector_store import VectorStore


class DocumentSearch:

    def __init__(self):
        self.store = None
        self.chunks = []


    def build_index(self, text):

        # Split document into chunks
        self.chunks = split_into_chunks(
            text,
            chunk_size=500,
            overlap=100
        )

        if not self.chunks:
            return False

        # Get text from each chunk
        chunk_texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        # Create embeddings
        embeddings = create_embeddings(
            chunk_texts
        )

        # Create FAISS vector store
        self.store = VectorStore(
            dimension=embeddings.shape[1]
        )

        # Add chunks to FAISS
        self.store.add_chunks(
            self.chunks,
            embeddings
        )

        return True


    def search(self, question, top_k=3):

        if self.store is None:
            return []

        # Convert question into embedding
        question_embedding = create_embeddings(
            [question]
        )

        # Search FAISS
        results = self.store.search(
            question_embedding,
            top_k=top_k
        )

        return results


# ==========================================
# TEST DOCUMENT SEARCH
# ==========================================

if __name__ == "__main__":

    sample_document = """
    Artificial intelligence is a field of computer science.

    Machine learning is a subset of artificial intelligence.

    Machine learning allows computers to learn from data.

    Deep learning uses neural networks to learn complex patterns.

    Natural language processing allows computers to
    understand human language.
    """

    print("\nBuilding document index...")

    search_engine = DocumentSearch()

    success = search_engine.build_index(
        sample_document
    )

    if success:

        print(
            "Document indexed successfully!"
        )

        question = (
            "How do computers learn from data?"
        )

        print(
            "\nQuestion:",
            question
        )

        results = search_engine.search(
            question,
            top_k=3
        )

        print(
            "\nRelevant Information:\n"
        )

        for result in results:

            print(
                "Similarity:",
                round(
                    result["score"],
                    3
                )
            )

            print(
                "Text:",
                result["chunk"]["text"]
            )

            print(
                "-" * 60
            )

    else:

        print(
            "Unable to build document index."
        )