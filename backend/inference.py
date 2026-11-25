import os
import json
import io
import time
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

UPLOAD_CONFIDENCE_THRESHOLD = 32.0
CAMERA_CONFIDENCE_THRESHOLD = 40.0


class ReBakeryInference:
    def __init__(
        self,
        model_path: str = "models/rebakery.int8.onnx",
        labels_path: str = "models/labels.json",
        confidence_threshold: float = UPLOAD_CONFIDENCE_THRESHOLD,
    ):
        # Cari file model & label
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, model_path)
        self.labels_path = os.path.join(base_dir, labels_path)
        if not os.path.exists(self.model_path):
            fp32_fallback = os.path.join(base_dir, "models", "rebakery.onnx")
            if os.path.exists(fp32_fallback):
                self.model_path = fp32_fallback
        self.confidence_threshold = confidence_threshold
        self.architecture = "ConvNeXt-Tiny"

        # Load metadata label
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.classes = data.get("classes", [])
                self.emoji_map = data.get("emoji", {})
                model_info = data.get("model_info", {})
                self.architecture = model_info.get("architecture", self.architecture)
        else:
            # Fallback jika labels.json belum dipindah
            self.classes = [
                "Baguette", "Croissant", "Cromboloni",
                "Donut", "Hole_Donut", "Pie_Susu",
                "Pretzel", "Sliced_Bread"
            ]
            self.emoji_map = {
                "Baguette": "🥖", "Croissant": "🥐", "Cromboloni": "🌀",
                "Donut": "🍩", "Hole_Donut": "⭕", "Pie_Susu": "🥧",
                "Pretzel": "🥨", "Sliced_Bread": "🍞"
            }

        # Load ONNX Session
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model ONNX tidak ditemukan di: {self.model_path}. Silakan salin rebakery.int8.onnx atau rebakery.onnx ke folder models/")

        # Gunakan CPUExecutionProvider (bisa CUDAExecutionProvider jika GPU server tersedia)
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.image_size = 224

        # ImageNet constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Decode image bytes, fix EXIF orientation, resize 224x224, normalize ke NCHW Float32"""
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        
        # Konversi ke array [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC
        arr = np.transpose(arr, (2, 0, 1))             # CHW
        arr = np.expand_dims(arr, axis=0)              # NCHW: (1, 3, 224, 224)
        
        # Normalisasi ImageNet
        arr = (arr - self.mean) / self.std
        return arr.astype(np.float32)

    def predict(self, image_bytes: bytes, threshold: float | None = None) -> dict:
        start_time = time.time()
        active_threshold = self.confidence_threshold if threshold is None else threshold
        
        # Preprocessing
        input_tensor = self.preprocess(image_bytes)

        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})[0] # shape (1, 8)
        logits = outputs[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Ambil Top-1
        top_idx = int(np.argmax(probs))
        top_class = self.classes[top_idx] if top_idx < len(self.classes) else f"Class_{top_idx}"
        top_conf = float(probs[top_idx])

        # Ambil Top-3
        sorted_indices = np.argsort(probs)[::-1][:3]
        top3 = []
        for idx in sorted_indices:
            cls_name = self.classes[idx] if idx < len(self.classes) else f"Class_{idx}"
            top3.append({
                "class": cls_name,
                "confidence": round(float(probs[idx]) * 100, 2),
                "emoji": self.emoji_map.get(cls_name, "🍞")
            })

        duration_ms = round((time.time() - start_time) * 1000, 2)
        top_conf_pct = round(top_conf * 100, 2)
        is_recognized = top_conf_pct >= active_threshold

        return {
            "is_recognized": is_recognized,
            "status": "recognized" if is_recognized else "unclassified",
            "threshold": active_threshold,
            "message": (
                None
                if is_recognized
                else "Tidak terklasifikasi sebagai salah satu dari 8 jenis roti yang dikenali."
            ),
            "prediction": {
                "class": top_class,
                "confidence": top_conf_pct,
                "emoji": self.emoji_map.get(top_class, "🍞")
            },
            "top3": top3,
            "inference_time_ms": duration_ms
        }
