from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import pytesseract
import easyocr
import re
from spellchecker import SpellChecker
from unidecode import unidecode
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Initialize EasyOCR reader and spell checker
reader = easyocr.Reader(["en"])
spell = SpellChecker()

def preprocess_image(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to improve OCR accuracy
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian Blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply adaptive thresholding
    processed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    return processed

def extract_text_tesseract(image):
    config = "--psm 6 -l eng"
    return pytesseract.image_to_string(image, config=config).strip()

def extract_text_easyocr(image):
    text = reader.readtext(image, detail=0)
    return " ".join(text)

def extract_text_advanced(image):
    img = Image.fromarray(image)
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(img, config=config).strip()

def clean_and_correct_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s,.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    corrected_words = [word for word in corrected_words if len(word) > 1 or word in {'a', 'I'}]

    return " ".join(corrected_words).strip()

def select_best_text(tess_text, easyocr_text, advanced_text):
    # Use the most complete and readable version
    base_text = advanced_text if len(advanced_text) > len(easyocr_text) else easyocr_text
    base_text = base_text if len(base_text) > len(tess_text) else tess_text
    
    # Fix typos and structure
    return clean_and_correct_text(base_text)

@app.post("/extract_text/")
async def extract_text(file: UploadFile = File(...)):
    # Read image from uploaded file
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)))

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Extract text from different methods
    tess_text = extract_text_tesseract(processed_image)
    easyocr_text = extract_text_easyocr(processed_image)
    advanced_text = extract_text_advanced(processed_image)

    # Select the best output
    best_text = select_best_text(tess_text, easyocr_text, advanced_text)

    return {"extracted_text": best_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
