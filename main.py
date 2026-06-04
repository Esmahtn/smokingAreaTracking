"""
main.py — UYARI: Bu dosya eski/legacy bir test scriptidir.
============================================================
Production kullanımı için app.py + smoking_analyzer.py kullanın.

Bu dosyada EKSİK olan özellikler:
  - İzleme bölgesi (zone) kontrolü yok
  - Veritabanı kaydı yok
  - RTSP yeniden bağlantı (reconnect) yok
  - Saat başı sıfırlama yok
  - Re-ID galeri kalıcılığı yok

Sadece hızlı yerel test amacıyla kullanılabilir.
"""
import cv2
import time
import numpy as np
from ultralytics import YOLO
from reid import FeatureExtractor
from scipy.spatial.distance import cosine


def get_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def main():
    print("=" * 60)
    print("UYARI: Bu script legacy/test amaçlıdır.")
    print("Production için: python app.py kullanın.")
    print("=" * 60)

    print("YOLOv8 Modeli yükleniyor...")
    model = YOLO("yolov8s.pt")

    print("Öznitelik Çıkarıcı (Re-ID) yükleniyor...")
    feature_extractor = FeatureExtractor()

    # RTSP URL'yi .env'den veya doğrudan girin (şifreyi koda yazmayın)
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        RTSP_URL = os.getenv("RTSP_URL", "")
    except ImportError:
        RTSP_URL = ""

    if not RTSP_URL:
        print("HATA: RTSP_URL .env dosyasında tanımlı değil.")
        print("Lütfen .env dosyasını oluşturun: RTSP_URL=rtsp://...")
        return

    print(f"Bağlanılıyor: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print(f"HATA: Kamera açılamadı: {RTSP_URL}")
        return

    # Re-ID veritabanı
    database = {}    # real_id -> {"embedding": ..., "entry_time": ...}
    id_map = {}      # yolo_id -> real_id
    sessions = {}    # real_id -> {"entry_time": float, "violation_reported": bool}
    next_real_id = 0
    TIME_LIMIT = 60  # saniye

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kamera okunamadı veya bağlantı koptu.")
            break

        results = model.track(image, classes=[0], conf=0.35, persist=True,
                              tracker="bytetrack.yaml", verbose=False)
        current_time = time.time()
        annotated_frame = image.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids   = results[0].boxes.id.cpu().numpy().astype(int)

            for box, yolo_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                # Boyut filtresi — çok küçük veya yatay kutuları atla
                if w < 30 or h < 60:
                    continue
                if w > h * 1.5:
                    continue

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

                # Re-ID: Yeni yolo_id için gerçek kişiyle eşleştir
                if yolo_id not in id_map:
                    crop = image[y1:y2, x1:x2]
                    embedding = feature_extractor.extract(crop)

                    if embedding is not None:
                        best_match_id = None
                        best_similarity = -1

                        for real_id, data in database.items():
                            sim = get_cosine_similarity(embedding, data["embedding"])
                            if sim > best_similarity:
                                best_similarity = sim
                                best_match_id = real_id

                        if best_match_id is not None and best_similarity > 0.70:
                            id_map[yolo_id] = best_match_id
                            # EMA ile embedding güncelle
                            old_emb = database[best_match_id]["embedding"]
                            new_emb = 0.8 * old_emb + 0.2 * embedding
                            database[best_match_id]["embedding"] = new_emb / np.linalg.norm(new_emb)
                        else:
                            # Yeni kişi
                            id_map[yolo_id] = next_real_id
                            database[next_real_id] = {"embedding": embedding}
                            next_real_id += 1

                if yolo_id in id_map:
                    real_id = id_map[yolo_id]

                    # Oturum başlat (yoksa)
                    if real_id not in sessions:
                        sessions[real_id] = {
                            "entry_time": current_time,
                            "violation_reported": False  # ✅ DÜZELTME: her frame ihlal basma önlendi
                        }

                    session = sessions[real_id]
                    time_spent = current_time - session["entry_time"]

                    if time_spent > TIME_LIMIT:
                        color = (0, 0, 255)
                        text = f"ID:{real_id} IHLAL! ({int(time_spent)}s)"

                        # ✅ DÜZELTME: Aynı ziyarette sadece bir kez logla
                        if not session["violation_reported"]:
                            session["violation_reported"] = True
                            print(f"[İHLAL] ID:{real_id} — {int(time_spent)}s bölgede kaldı.")
                    else:
                        color = (0, 255, 0)
                        text = f"ID:{real_id} Sure: {int(time_spent)}s"

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        resized_frame = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow('Sigara Alani Takip (LEGACY)', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
