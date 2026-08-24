
import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def clean_text(text):
    """Clean OCR text before sending it to the LLM."""

    if not text:
        return ""

    text = str(text)

    # Remove common OCR/control problems
    text = text.replace("\x00", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Fix repeated characters/words caused by OCR
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

    # Fix repeated fragments such as:
    # "distinct distinctive"
    text = re.sub(
        r"\b(\w+)\s+\w+\s+\1\b",
        r"\1",
        text,
        flags=re.IGNORECASE
    )

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_answer(text):
    """Clean unwanted repetition from Ollama's answer."""

    if not text:
        return ""

    text = text.strip()

    # Remove repeated complete words:
    # "feature feature" -> "feature"
    text = re.sub(
        r"\b(\w+)(\s+\1\b)+",
        r"\1",
        text,
        flags=re.IGNORECASE
    )

    # Remove obvious OCR-style repeated fragments
    text = re.sub(
        r"\b(\w+)\s+\w+\s+\1\b",
        r"\1",
        text,
        flags=re.IGNORECASE
    )

    # Fix spacing before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def generate_answer(question, search_results):
    """
    Generate an answer using the retrieved document chunks.
    """

    if not search_results:
        return "I could not find relevant information in the document to answer this question."

    context_parts = []

    for result in search_results:
        if not isinstance(result, dict):
            continue

        chunk = result.get("chunk", result)

        if isinstance(chunk, str):
            text = chunk
            page = "Unknown"
        elif isinstance(chunk, dict):
            text = chunk.get("text", "")
            page = chunk.get("page", "Unknown")
        else:
            continue

        text = clean_text(text)

        if text:
            context_parts.append(
                f"[Page {page}]\n{text}"
            )

    if not context_parts:
        return "I could not find relevant information in the document to answer this question."

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information provided in the document context.

Important rules:
1. Give a clear and natural answer.
2. Do not invent information.
3. Do not repeat words.
4. Do not copy OCR errors.
5. Do not use phrases like "According to the document" unless necessary.
6. Keep the answer concise.
7. If the answer is not present in the context, say:
   "I could not find enough relevant information in the document to answer this question."
8. Give the page number at the end when the answer comes from a specific page.

User question:
{question}

Document context:
{context}

Answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            return "I could not generate an answer from the document."

        answer = clean_answer(answer)

        return answer

    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        )

    except Exception as e:
        return f"Error generating answer: {e}"

