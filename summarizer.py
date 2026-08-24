import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"


# =========================================================
# CLEAN OCR TEXT
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", str(text)).strip()

    replacements = {
        "distinctiv distinctive": "distinctive",
        "descrip description": "description",
        "descri description": "description",
        "match matching": "matching",
        "key keypoints": "keypoints",
        "fea feature": "feature",
        "stru structure": "structure",
        "orien orientation": "orientation",
        "histo histograms": "histograms",
        "analy analysis": "analysis",
        "extrac extraction": "extraction",
        "detectio detection": "detection",
        "derivative derivatives": "derivatives",
        "calculati calculating": "calculating",
        "cal calculating": "calculating",
        "calc calculating": "calculating",
        "box-filte box-filter": "box-filter",
        "effi efficient": "efficient",
        "eff efficient": "efficient",
    }

    for old, new in replacements.items():

        text = re.sub(
            r"\b" + re.escape(old) + r"\b",
            new,
            text,
            flags=re.IGNORECASE
        )

    words = text.split()
    cleaned = []

    for word in words:

        current = re.sub(
            r"[^\w-]",
            "",
            word.lower()
        )

        if cleaned:

            previous = re.sub(
                r"[^\w-]",
                "",
                cleaned[-1].lower()
            )

            if current and current == previous:
                continue

        cleaned.append(word)

    return " ".join(cleaned).strip()


# =========================================================
# CLEAN AI OUTPUT
# =========================================================

def clean_ai_text(text):

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"[#*_`]",
        "",
        text
    )

    text = re.sub(
        r"\b(\w+)(\s+\1\b)+",
        r"\1",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CALL OLLAMA
# =========================================================

def call_ollama(prompt):

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=180
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            ""
        ).strip()

    except Exception as e:

        return f"ERROR: {e}"


# =========================================================
# GENERATE SUMMARY
# =========================================================

def generate_summary(
    text,
    length="short"
):

    text = clean_text(text)

    if not text:
        return "No document text available."

    # -----------------------------------------------------
    # SUMMARY LENGTH
    # -----------------------------------------------------

    if length == "short":

        sentence_count = 5
        detail_level = "very concise"

    elif length == "medium":

        sentence_count = 8
        detail_level = "moderately detailed"

    else:

        sentence_count = 12
        detail_level = "detailed and comprehensive"

    # -----------------------------------------------------
    # SUMMARY PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are an expert document summarizer.

Create a {detail_level} summary of the document.

Write exactly {sentence_count} clear sentences.

Short summaries should contain only the most essential ideas.
Medium summaries should explain the major concepts with some detail.
Long summaries should explain the major concepts and important
supporting details.

IMPORTANT TOPICS:

If these topics are present in the document, make sure the
summary gives meaningful coverage to them:

- SIFT
- SURF
- HOG (Histogram of Oriented Gradients)
- Image Pyramids
- Scale-Space
- Keypoint Localization

Do not focus almost entirely on SIFT.

Give balanced coverage to the major topics.

STRICT RULES:

1. Use ONLY information from the document.
2. Do NOT invent facts.
3. Do NOT use outside knowledge.
4. Do NOT repeat information.
5. Correct obvious OCR mistakes.
6. Use simple natural English.
7. Do NOT use bullet points.
8. Do NOT add an introduction.
9. Return ONLY the summary.
10. Make every sentence meaningful.

DOCUMENT:

{text}
"""

    answer = call_ollama(prompt)

    if answer.startswith("ERROR:"):

        return answer

    answer = clean_ai_text(answer)

    # -----------------------------------------------------
    # SPLIT INTO SENTENCES
    # -----------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 15
    ]

    # -----------------------------------------------------
    # REMOVE DUPLICATE SENTENCES
    # -----------------------------------------------------

    final_sentences = []

    for sentence in sentences:

        normalized = re.sub(
            r"\W+",
            " ",
            sentence.lower()
        ).strip()

        duplicate = False

        for existing in final_sentences:

            existing_normalized = re.sub(
                r"\W+",
                " ",
                existing.lower()
            ).strip()

            if normalized == existing_normalized:

                duplicate = True
                break

        if not duplicate:

            final_sentences.append(sentence)

    # -----------------------------------------------------
    # LIMIT SENTENCE COUNT
    # -----------------------------------------------------

    final_sentences = final_sentences[
        :sentence_count
    ]

    return " ".join(final_sentences)


# =========================================================
# GENERATE KEY POINTS
# =========================================================

def generate_key_points(
    text,
    number_of_points=5
):

    text = clean_text(text)

    if not text:

        return []

    prompt = f"""
Read the document and extract exactly
{number_of_points} important key points.

Try to cover different major topics.

If present in the document, consider:

SIFT
SURF
HOG
Image Pyramids
Scale-Space
Keypoint Localization

STRICT RULES:

- One idea per point.
- Each point must be a separate line.
- Do not combine ideas.
- Do not repeat ideas.
- Do not invent facts.
- Use only information from the document.
- Correct obvious OCR mistakes.
- Do not write an introduction.

FORMAT:

POINT 1: sentence
POINT 2: sentence
POINT 3: sentence
POINT 4: sentence
POINT 5: sentence

DOCUMENT:

{text}
"""

    answer = call_ollama(prompt)

    if answer.startswith("ERROR:"):

        return []

    answer = clean_ai_text(answer)

    # -----------------------------------------------------
    # FIND POINTS
    # -----------------------------------------------------

    matches = re.findall(
        r"POINT\s*\d+\s*:\s*(.*?)(?=\s*POINT\s*\d+\s*:|$)",
        answer,
        flags=re.IGNORECASE
    )

    points = []

    if matches:

        for point in matches:

            point = point.strip()

            if len(point) < 20:

                continue

            points.append(
                clean_ai_text(point)
            )

    else:

        lines = re.split(
            r"[\n•]",
            answer
        )

        for line in lines:

            line = re.sub(
                r"^\s*[\-\*\d.)]+\s*",
                "",
                line
            ).strip()

            if len(line) >= 20:

                points.append(
                    clean_ai_text(line)
                )

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    final_points = []

    for point in points:

        normalized = re.sub(
            r"\W+",
            " ",
            point.lower()
        ).strip()

        duplicate = False

        for existing in final_points:

            existing_normalized = re.sub(
                r"\W+",
                " ",
                existing.lower()
            ).strip()

            if normalized == existing_normalized:

                duplicate = True
                break

        if not duplicate:

            final_points.append(point)

    return final_points[
        :number_of_points
    ]


# =========================================================
# MAIN FUNCTION
# =========================================================

def summarize(
    text,
    length="short",
    number_of_points=5
):

    summary = generate_summary(
        text,
        length
    )

    key_points = generate_key_points(
        text,
        number_of_points
    )

    return summary, key_points
