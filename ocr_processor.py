import pytesseract
from PIL import Image
import io


def extract_image_text(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image)

    return text.strip()