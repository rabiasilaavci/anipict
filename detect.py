import cv2
import random
import numpy as np
from ultralytics import YOLO

# Model yolu
model = YOLO(r"C:\Users\rabia\OneDrive - Muğla Sıtkı Koçman Üniversitesi\Masaüstü\sığır her şey\sığır project\models/best.pt")

# Fotoğraf yolu
IMAGE_PATH = r"C:\Users\rabia\OneDrive - Muğla Sıtkı Koçman Üniversitesi\Masaüstü\1b0c2757-482a-11ee-8f8d-089df4df72d9.jpg"

labels = model.names
colors = [[random.randint(0,255) for _ in range(3)] for _ in labels]

confidence_score = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

# Fotoğrafı oku
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ Fotoğraf bulunamadı.")
    exit()

# ✅ Fotoğrafı küçült (orijinalin %70'i)
scale_percent = 70
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# Tahmin yap
results = model(img, verbose=False)[0]
boxes = np.array(results.boxes.data.tolist())

# Kutuları ve büyütülmüş yazıları çiz
for box in boxes:
    x1, y1, x2, y2, score, class_id = box
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    if score > confidence_score:
        color = colors[int(class_id)]

        # Kutu kalın
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # ✅ Daha büyük yazı
        label = f"{results.names[int(class_id)]} {score*100:.1f}%"
        font_scale = 1.3
        thickness = 3

        # Yazı boyutu
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Siyah arka plan
        cv2.rectangle(
            img,
            (x1, y1 - text_h - 18),
            (x1 + text_w + 6, y1),
            (0, 0, 0),
            -1
        )

        # Yazı
        cv2.putText(
            img,
            label,
            (x1 + 3, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

# Pencere ayarları
window_name = "YOLO Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 500, 400)

cv2.imshow(window_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
