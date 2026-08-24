import streamlit as st

from document_processor import extract_pdf_text
from ocr_processor import extract_image_text
from summarizer import summarize
from utils import (
    count_words,
    count_characters,
    reading_time
)

from text_cleaner import clean_text

from chunker import chunk_text
from embeddings import create_embeddings

from vector_store import (
    build_index,
    search_index
)

from rag import generate_answer


st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide"
)


st.title("📄 DocuMind")

st.subheader(
    "Intelligent Document Summary Assistant"
)

st.write(
    "Upload a PDF or image and generate an "
    "intelligent summary, key points, document "
    "insights, and AI-powered answers with "
    "page citations."
)


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg"
    ]
)


# ==========================================================
# SUMMARY LENGTH
# ==========================================================

summary_length = st.selectbox(
    "Summary Length",
    [
        "Short",
        "Medium",
        "Long"
    ]
)


# ==========================================================
# PROCESS DOCUMENT
# ==========================================================

if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    try:

        with st.status(
            "Processing document...",
            expanded=True
        ) as status:

            # --------------------------------------------------
            # STEP 1: READ DOCUMENT
            # --------------------------------------------------

            st.write(
                "📄 Reading document..."
            )

            if uploaded_file.name.lower().endswith(
                ".pdf"
            ):

                st.write(
                    "🔍 Extracting PDF text..."
                )

                (
                    text,
                    pages,
                    extraction_method,
                    page_data
                ) = extract_pdf_text(
                    file_bytes
                )

                document_type = "PDF"

            else:

                st.write(
                    "🔍 Running OCR on image..."
                )

                image_text = extract_image_text(
                    file_bytes
                )

                text = image_text

                pages = 1

                extraction_method = (
                    "Image OCR"
                )

                document_type = (
                    "Image / OCR"
                )

                page_data = [
                    {
                        "page": 1,
                        "text": image_text
                    }
                ]

            # --------------------------------------------------
            # STEP 2: CHECK EXTRACTED TEXT
            # --------------------------------------------------

            if not text or not text.strip():

                st.error(
                    "Unable to extract readable "
                    "text from this document."
                )

                st.stop()

            # --------------------------------------------------
            # STEP 3: CLEAN FULL TEXT
            # --------------------------------------------------

            st.write(
                "🧹 Cleaning extracted text..."
            )

            text = clean_text(
                text
            )

            # --------------------------------------------------
            # STEP 4: GENERATE SUMMARY
            # --------------------------------------------------

            st.write(
                "🧠 Generating intelligent summary..."
            )

            summary, key_points = summarize(
                text,
                summary_length.lower()
            )

            # --------------------------------------------------
            # STEP 5: CLEAN PAGE DATA
            # --------------------------------------------------

            cleaned_page_data = []

            for item in page_data:

                page_number = item.get(
                    "page",
                    "Unknown"
                )

                page_text = item.get(
                    "text",
                    ""
                )

                cleaned_page_text = clean_text(
                    page_text
                )

                if cleaned_page_text:

                    cleaned_page_data.append(
                        {
                            "page": page_number,
                            "text": cleaned_page_text
                        }
                    )

            # --------------------------------------------------
            # STEP 6: CHUNK DOCUMENT
            # --------------------------------------------------

            st.write(
                "✂️ Splitting document into "
                "page-aware chunks..."
            )

            chunks = chunk_text(
                cleaned_page_data
            )

            if not chunks:

                st.error(
                    "Unable to create document chunks."
                )

                st.stop()

            # --------------------------------------------------
            # STEP 7: CREATE EMBEDDINGS
            # --------------------------------------------------

            st.write(
                "🔢 Creating document embeddings..."
            )

            embeddings = create_embeddings(
                chunks
            )

            # --------------------------------------------------
            # STEP 8: BUILD FAISS INDEX
            # --------------------------------------------------

            st.write(
                "🔎 Building semantic search index..."
            )

            index = build_index(
                embeddings
            )

            status.update(
                label=(
                    "✅ Document processed successfully!"
                ),
                state="complete",
                expanded=False
            )

        st.success(
            "Document processed successfully!"
        )

        # ======================================================
        # DOCUMENT INSIGHTS
        # ======================================================

        st.divider()

        st.header(
            "📊 Document Insights"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Pages",
            pages
        )

        col2.metric(
            "Words",
            count_words(text)
        )

        col3.metric(
            "Characters",
            count_characters(text)
        )

        col4.metric(
            "Reading Time",
            f"{reading_time(text)} min"
        )

        # ======================================================
        # SUMMARY
        # ======================================================

        st.divider()

        st.header(
            "📝 Summary"
        )

        st.write(
            summary
        )

        # ======================================================
        # KEY POINTS
        # ======================================================

        st.divider()

        st.header(
            "🔑 Key Points"
        )

        for point in key_points:

            st.markdown(
                f"- {point}"
            )

        # ======================================================
        # ASK YOUR DOCUMENT
        # ======================================================

        st.divider()

        st.header(
            "💬 Ask Your Document"
        )

        st.write(
            "Ask a question based on the uploaded document."
        )

        question = st.text_input(
            "Enter your question"
        )

        ask_button = st.button(
            "🔍 Ask Document"
        )

        if ask_button:

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                # --------------------------------------------------
                # QUESTION EMBEDDING
                # --------------------------------------------------

                with st.spinner(
                    "Searching the document..."
                ):

                    question_embedding = (
                        create_embeddings(
                            [question]
                        )
                    )

                    # --------------------------------------------------
                    # RETRIEVAL WITH FILTERING
                    # --------------------------------------------------

                    search_results = search_index(
                        index,
                        question_embedding,
                        chunks,
                        top_k=5,
                        similarity_threshold=0.35
                    )

                # --------------------------------------------------
                # NO RELEVANT RESULTS
                # --------------------------------------------------

                if not search_results:

                    st.warning(
                        "I could not find enough relevant "
                        "information in the document to "
                        "answer this question."
                    )

                else:

                    # --------------------------------------------------
                    # GENERATE RAG ANSWER
                    # --------------------------------------------------

                    with st.spinner(
                        "Generating answer..."
                    ):

                        answer = generate_answer(
                            question,
                            search_results
                        )

                    # ==================================================
                    # ANSWER
                    # ==================================================

                    st.subheader(
                        "🤖 Answer"
                    )

                    st.write(
                        answer
                    )

                    # ==================================================
                    # SOURCES
                    # ==================================================

                    st.subheader(
                        "📚 Sources"
                    )

                    st.write(
                        "The following pages contain "
                        "information relevant to your question:"
                    )

                    for result in search_results:

                        score = result.get(
                            "score",
                            0
                        )

                        chunk = result.get(
                            "chunk",
                            {}
                        )

                        if isinstance(
                            chunk,
                            dict
                        ):

                            source_text = chunk.get(
                                "text",
                                ""
                            )

                            page_number = chunk.get(
                                "page",
                                "Unknown"
                            )

                        else:

                            source_text = str(
                                chunk
                            )

                            page_number = "Unknown"

                        with st.expander(
                            f"📄 Page {page_number} "
                            f"• Similarity: {score:.3f}"
                        ):

                            st.write(
                                f"**Source:** Page "
                                f"{page_number}"
                            )

                            st.write(
                                source_text
                            )

        # ======================================================
        # PAGE-WISE CHUNK INFORMATION
        # ======================================================

        st.divider()

        st.subheader(
            "📚 View Page-wise Chunk Information"
        )

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            page_number = chunk.get(
                "page",
                "Unknown"
            )

            chunk_text_value = chunk.get(
                "text",
                ""
            )

            with st.expander(
                f"Chunk {i} • Page {page_number}"
            ):

                st.write(
                    chunk_text_value
                )

        # ======================================================
        # DOCUMENT INFORMATION
        # ======================================================

        st.divider()

        st.header(
            "📄 Document Information"
        )

        st.write(
            f"**File:** {uploaded_file.name}"
        )

        st.write(
            f"**Type:** {document_type}"
        )

        st.write(
            f"**Summary Length:** {summary_length}"
        )

        st.write(
            f"**Extraction Method:** "
            f"{extraction_method}"
        )

        st.write(
            f"**Number of Chunks:** "
            f"{len(chunks)}"
        )

        st.write(
            f"**Pages Available for Citations:** "
            f"{pages}"
        )

        # ======================================================
        # DOWNLOAD SUMMARY
        # ======================================================

        st.divider()

        download_text = (
            "DOCUMIND - DOCUMENT SUMMARY\n\n"
            "SUMMARY\n"
            f"{summary}\n\n"
            "KEY POINTS\n"
            +
            "\n".join(
                f"- {point}"
                for point in key_points
            )
        )

        st.download_button(
            "⬇️ Download Summary",
            download_text,
            file_name="document_summary.txt",
            mime="text/plain"
        )

    # ==========================================================
    # ERROR HANDLING
    # ==========================================================

    except Exception as e:

        st.error(
            "Something went wrong while "
            "processing the document."
        )

        st.exception(e)
