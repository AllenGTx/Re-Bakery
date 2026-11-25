import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import base64
from inference import (
    ReBakeryInference,
    UPLOAD_CONFIDENCE_THRESHOLD,
    CAMERA_CONFIDENCE_THRESHOLD,
)

app = FastAPI(
    title="Re:Bakery AI API",
    description="API Pendeteksi 8 Jenis Bakery berbasis ConvNeXt-Tiny ONNX",
    version="1.1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi engine inferensi (lazy-load agar cold start Vercel tetap aman)
model_engine = None

def get_engine():
    global model_engine
    if model_engine is None:
        model_engine = ReBakeryInference()
        print("✅ ONNX Model Re:Bakery berhasil dimuat ke memori!")
    return model_engine

@app.on_event("startup")
def load_model():
    try:
        get_engine()
    except Exception as e:
        print(f"⚠️ Peringatan: {e}")
        print("💡 Pastikan file rebakery.onnx dan labels.json sudah ditaruh di backend/models/")

# Mount static folder untuk web frontend
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "Sample")
if os.path.exists(sample_dir):
    app.mount("/samples", StaticFiles(directory=sample_dir), name="samples")

@app.get("/")
def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "app": "Re:Bakery API",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/api/health")
def health_check():
    try:
        engine = get_engine()
    except Exception:
        engine = None
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "architecture": engine.architecture if engine else None,
        "upload_threshold": UPLOAD_CONFIDENCE_THRESHOLD,
        "camera_threshold": CAMERA_CONFIDENCE_THRESHOLD,
        "classes": engine.classes if engine else []
    }

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """Mode 1: Prediksi dari upload file gambar"""
    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model belum dimuat: {e}")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar.")

    try:
        contents = await file.read()
        result = engine.predict(contents, threshold=UPLOAD_CONFIDENCE_THRESHOLD)
        return {
            "success": True,
            "filename": file.filename,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {str(e)}")

class Base64FrameRequest(BaseModel):
    image_base64: str

@app.post("/api/predict-frame")
async def predict_frame(payload: Base64FrameRequest):
    """Mode 2: Prediksi dari frame webcam realtime (base64 string)"""
    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model belum dimuat: {e}")

    try:
        data = payload.image_base64
        if "," in data:
            data = data.split(",")[1]
        
        image_bytes = base64.b64decode(data)
        result = engine.predict(image_bytes, threshold=CAMERA_CONFIDENCE_THRESHOLD)
        return {
            "success": True,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses frame: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
