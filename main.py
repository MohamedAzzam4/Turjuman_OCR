# main.py

# Install necessary packages first:
# pip install fastapi uvicorn python-multipart pillow python-dotenv openai

import os
import io
import uvicorn
import base64
import openai # <-- استيراد مكتبة OpenAI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# --- 1. تهيئة الـ API والنموذج ---

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

# استخدام مفتاح requesty.ai
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY")
if not ROUTER_API_KEY:
    raise ValueError("ROUTER_API_KEY environment variable is missing.")

# تحديد اسم النموذج الذي سيستخدمه requesty

llm = 'google/gemini-2.5-flash-lite-preview-06-17'

# تهيئة العميل (Client) للاتصال بـ requesty.ai
# هذه هي نفس الطريقة المستخدمة في الكود السابق
client = openai.OpenAI(
    api_key=ROUTER_API_KEY,
    base_url="https://router.requesty.ai/v1",
    default_headers={"Authorization": f"Bearer {ROUTER_API_KEY}"}
)

# --- 2. إعداد تطبيق FastAPI ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the OCR and Translation API (using Requesty)!"}


# --- 3. تعديل نقطة النهاية (Endpoint) ---

@app.post("/ocr-translate")
async def ocr_and_translate(file: UploadFile = File(...)):
    try:
        # الخطوة 1: قراءة الصورة وتحويلها إلى Base64
        image_bytes = await file.read()
        
        # استخدام Pillow لتحديد نوع الصورة (jpeg, png, etc.)
        image = Image.open(io.BytesIO(image_bytes))
        image_format = image.format or 'JPEG' # الافتراضي هو JPEG
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:image/{image_format.lower()};base64,{base64_image}"

        # الخطوة 2: استخلاص النص (OCR) باستخدام requesty
        ocr_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract *all visible text* from this image *exactly as it appears*, without summarizing or interpreting. Do not translate."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }
                ]
            }
        ]
        
        ocr_response = client.chat.completions.create(
            model=llm,
            messages=ocr_messages,
            temperature=0.2
        )
        english_text = ocr_response.choices[0].message.content.strip()

        # الخطوة 3: الترجمة (فقط إذا كان هناك نص)
        translated_text = ""
        if english_text:
            translation_messages = [
                {
                    "role": "user", 
                    "content": f"Translate this English text to Arabic, don't say anything else, just the translation:\n\n{english_text}"
                }
            ]
            translation_response = client.chat.completions.create(
                model=llm,
                messages=translation_messages,
                temperature=0.3
            )
            translated_text = translation_response.choices[0].message.content.strip()

        return JSONResponse(content={
            "english_text": english_text,
            "translated_text": translated_text
        })

    except Exception as e:
        print(f"Error processing request")
        return JSONResponse(content={"error": "An internal server error occurred."}, status_code=500)


# --- 4. تشغيل السيرفر ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
