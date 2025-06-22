# Install necessary packages first:
# pip install fastapi uvicorn python-multipart google-generativeai pillow python-dotenv

import os
import io
import uvicorn # <-- إضافة مهمة لاستيراد Uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # اسمح لكل المواقع
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the OCR and Translation API!"}

@app.post("/ocr-translate")
async def ocr_and_translate(file: UploadFile = File(...)):
    try:
        # Step 1: Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Step 2: OCR extraction
        ocr_response = model.generate_content(
            [image, "Extract *all visible text* from this image *exactly as it appears*, without summarizing or interpreting. Do not translate."],
            generation_config={"temperature": 0.2}
        )
        # Check if response has text, handle potential empty or invalid responses
        if not ocr_response.parts:
             english_text = "" # Or handle as an error/empty case
        else:
             english_text = ocr_response.text.strip()


        # Step 3: Translation (only if english_text is not empty)
        if english_text:
            translation_response = model.generate_content(
                [f"Translate this English text to Germany dont say anything else just the translation:\n\n{english_text}"],
                generation_config={"temperature": 0.3}
            )
            # Check if translation response has text
            if not translation_response.parts:
                translated_text = "" # Or handle as an error/empty case
            else:
                translated_text = translation_response.text.strip()
        else:
             translated_text = "" # No text to translate


        return JSONResponse(content={
            "english_text": english_text,
            "translated_text": translated_text
        })

    except Exception as e:
        # It's good practice to log the actual error for debugging
        print(f"Error processing request: {e}")
        return JSONResponse(content={"error": "An internal server error occurred."}, status_code=500)


# --- الجزء المضاف لتشغيل السيرفر ---
if __name__ == "__main__":
    # حاول تقرأ البورت من متغير البيئة PORT، لو مش موجود استخدم 8080
    port = int(os.environ.get("PORT", 8080))
    # شغل السيرفر باستخدام uvicorn
    # host="0.0.0.0" مهم جداً عشان Railway يقدر يوصل للتطبيق
    uvicorn.run(app, host="0.0.0.0", port=port)
# --- نهاية الجزء المضاف ---
