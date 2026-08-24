import re
from difflib import SequenceMatcher


# =========================================================
# Basic OCR cleanup
# =========================================================

def basic_cleanup(text):

    if not text:
        return ""

    # Remove terminal/control characters
    text = re.sub(
        r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])',
        '',
        text
    )

    text = text.replace("\x08", "")
    text = text.replace("\x1b", "")

    # Normalize unusual whitespace
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove spaces before punctuation
    text = re.sub(
        r'\s+([,.!?;:])',
        r'\1',
        text
    )

    return text.strip()


# =========================================================
# Fix OCR fragments
# =========================================================

def fix_fragment_pair(first, second):

    f = first.lower()
    s = second.lower()

    # -----------------------------------------------------
    # Exact duplicate
    # -----------------------------------------------------

    if f == s:
        return second

    # -----------------------------------------------------
    # First word is a shortened version of second word
    #
    # distinctiv distinctive
    # match matching
    # key keypoints
    # incre increasing
    # -----------------------------------------------------

    if len(f) >= 3 and s.startswith(f):

        if len(s) - len(f) <= 6:
            return second

    # -----------------------------------------------------
    # Second word is a shortened version of first
    #
    # derivative derivatives
    # feature features
    # -----------------------------------------------------

    if len(s) >= 3 and f.startswith(s):

        if len(f) - len(s) <= 6:
            return first

    # -----------------------------------------------------
    # Similar OCR words
    # -----------------------------------------------------

    if len(f) >= 5 and len(s) >= 5:

        ratio = SequenceMatcher(
            None,
            f,
            s
        ).ratio()

        if ratio >= 0.88:

            # Prefer the longer word
            if len(first) >= len(second):
                return first

            return second

    return None


# =========================================================
# Fix adjacent OCR fragments
# =========================================================

def fix_adjacent_fragments(text):

    words = text.split()

    if not words:
        return ""

    cleaned = []

    i = 0

    while i < len(words):

        current = words[i]

        # -------------------------------------------------
        # Handle current word + next word
        # -------------------------------------------------

        if i + 1 < len(words):

            next_word = words[i + 1]

            # Remove punctuation temporarily
            current_clean = current.strip(".,!?;:")
            next_clean = next_word.strip(".,!?;:")

            result = fix_fragment_pair(
                current_clean,
                next_clean
            )

            if result is not None:

                # Preserve punctuation from second word
                punctuation = ""

                if next_word.endswith((".", ",", "!", "?", ";", ":")):
                    punctuation = next_word[-1]

                cleaned.append(
                    result + punctuation
                )

                i += 2
                continue

        cleaned.append(current)

        i += 1

    return " ".join(cleaned)


# =========================================================
# Fix repeated phrases / words
# =========================================================

def remove_repeated_words(text):

    # Example:
    # "SIFT SIFT is useful"
    # -> "SIFT is useful"

    text = re.sub(
        r'\b([A-Za-z][A-Za-z-]*)\s+\1\b',
        r'\1',
        text,
        flags=re.IGNORECASE
    )

    # Example:
    # "feature feature descriptors"
    # -> "feature descriptors"

    return text


# =========================================================
# Fix common OCR constructions
# =========================================================

def fix_common_patterns(text):

    patterns = [

        # Fragment + full word
        (r'\bdistinctiv\s+distinctive\b', 'distinctive'),
        (r'\bmatch\s+matching\b', 'matching'),
        (r'\bkey\s+keypoints\b', 'keypoints'),
        (r'\bkeypo\s+keypoints\b', 'keypoints'),
        (r'\bpoi\s+points\b', 'points'),

        (r'\bcalc\s+calculate\b', 'calculate'),
        (r'\bcalc\s+calculating\b', 'calculating'),

        (r'\bincre\s+increasing\b', 'increasing'),
        (r'\bimp\s+improve\b', 'improve'),

        (r'\bdescrip\s+description\b', 'description'),
        (r'\bdescri\s+description\b', 'description'),
        (r'\bdesc\s+describe\b', 'describe'),

        (r'\bextrac\s+extraction\b', 'extraction'),

        (r'\bstru\s+structure\b', 'structure'),

        (r'\borie\s+orientation\b', 'orientation'),
        (r'\bhisto\s+histograms\b', 'histograms'),

        (r'\banaly\s+analysis\b', 'analysis'),

        (r'\bdetectio\s+detection\b', 'detection'),

        (r'\bderivative\s+derivatives\b', 'derivatives'),

        # Hyphenated OCR
        (r'\bbox-filte\s+box-filter\b', 'box-filter'),

        # Single-letter OCR fragments
        (r'\bi\s+in\b', 'in'),
        (r'\ba\s+and\b', 'and'),
        (r'\bb\s+be\b', 'be'),
        (r'\bf\s+feature\b', 'feature'),
        (r'\bt\s+that\b', 'that'),
        (r'\bo\s+of\b', 'of'),
        (r'\bs\s+scale\b', 'scale'),

        # Repeated technical terms
        (r'\bSIFT\s*,\s*SIFT\b', 'SIFT'),
        (r'\bSURF\s*,\s*SURF\b', 'SURF'),
        (r'\bHOG\s*,\s*HOG\b', 'HOG'),
        (r'\bDoG\s*,\s*DoG\b', 'DoG')
    ]

    for pattern, replacement in patterns:

        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


# =========================================================
# Remove unnecessary OCR symbols
# =========================================================

def remove_ocr_noise(text):

    # Remove isolated meaningless symbols
    text = re.sub(
        r'(?<!\w)[=><~`^|]+(?!\w)',
        ' ',
        text
    )

    # Remove excessive punctuation
    text = re.sub(
        r'([,.!?;:])\1+',
        r'\1',
        text
    )

    # Normalize spaces again
    text = re.sub(
        r'[ \t]+',
        ' ',
        text
    )

    return text.strip()


# =========================================================
# Main cleaning function
# =========================================================

def clean_text(text):

    if not text:
        return ""

    # Step 1
    text = basic_cleanup(text)

    # Step 2
    text = fix_common_patterns(text)

    # Step 3
    text = fix_adjacent_fragments(text)

    # Step 4
    text = remove_repeated_words(text)

    # Step 5
    text = fix_common_patterns(text)

    # Step 6
    text = remove_ocr_noise(text)

    # Final whitespace cleanup
    text = re.sub(
        r'\n\s*\n\s*\n+',
        '\n\n',
        text
    )

    return text.strip()


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    sample = """
    SIFT is a feature detection and desc describe technique
    that finds distinctiv distinctive keypoints and creates
    feature descriptors for match matching between images.

    SIFT SIFT uses Difference of Gaussian to find key keypoints
    at each s scale.

    SURF, SURF is a faster alternative using box-filte
    box-filter approximations.

    HOG describes the shape and stru structure of objects.
    """

    print("\n================ BEFORE ================\n")
    print(sample)

    print("\n================ AFTER =================\n")
    print(clean_text(sample))