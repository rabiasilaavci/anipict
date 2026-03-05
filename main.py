from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
from PIL import Image
import json
import os

app = FastAPI()

# 🧠 YOLO modelini yükle
model = YOLO(r"C:\Users\rabia\OneDrive - Muğla Sıtkı Koçman Üniversitesi\Masaüstü\bitki project\project\models\bitki.pt")

# 📂 JSON dosyasını yükle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "tehlike_seviyeleri.json")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    tehlike_bilgileri = json.load(f)

# 🔤 Harf duyarsız karşılaştırma için (önerilir)
tehlike_bilgileri_lower = {k.lower(): v for k, v in tehlike_bilgileri.items()}


@app.get("/")
def root():
    return {"message": "Kamera tabanlı tespit API'si aktif!"}


@app.get("/capture")
def capture_and_predict():
    # 📸 Kamerayı başlat
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return JSONResponse(content={"error": "Kamera açılamadı!"}, status_code=500)

    # Tek kare al
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JSONResponse(content={"error": "Görüntü alınamadı!"}, status_code=500)

    # 🎨 Görüntüyü RGB'ye çevir
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 🧠 Modelle tahmin yap
    results = model.predict(image)

    # 🎯 Sonuçları işle
    boxes = results[0].boxes
    predictions = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        # 💡 Tehlike seviyesini bul (harf duyarsız)
        tehlike = tehlike_bilgileri_lower.get(label.lower(), "Bilinmiyor")

        predictions.append({
            "label": label,
            "confidence": round(conf, 2),
            "tehlike_seviyesi": tehlike
        })

    return {"predictions": predictions}
