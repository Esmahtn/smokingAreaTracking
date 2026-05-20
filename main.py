import cv2
import time
import numpy as np
from ultralytics import YOLO
from reid import FeatureExtractor
from scipy.spatial.distance import cosine

def get_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def main():
    print("YOLOv8 Modeli yükleniyor...")
    model = YOLO("yolov8s.pt")
    
    print("Öznitelik Çıkarıcı (Re-ID) yükleniyor...")
    feature_extractor = FeatureExtractor()

    RTSP_URL = "rtsp://admin:Yt2240cn@192.168.12.71:554/cam/realmonitor?channel=1&subtype=0"
    print(f"{RTSP_URL} adresine bağlanılıyor...")
    cap = cv2.VideoCapture(RTSP_URL)
    
    database = {}
    id_map = {}
    next_real_id = 0
    TIME_LIMIT = 60 

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kamera okunamadı veya bağlantı koptu.")
            break

        # 1. İnsan Olmayan Şeyleri Yanlış Algılamamak İçin:
        # conf=0.25'ten 0.35'e çıkardık ki sahte tespitler (false positive) azalsın.
        results = model.track(image, classes=[0], conf=0.35, persist=True, tracker="bytetrack.yaml", verbose=False)
        current_time = time.time()
        annotated_frame = image.copy()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, yolo_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                
                # 2. İnsan Olmayan Şeyleri Yanlış Algılamamak İçin Boyut Kontrolü:
                w = x2 - x1
                h = y2 - y1
                
                # Aşırı küçük şeyleri veya yatay (araba, masa vs) dikdörtgenleri insan sayma
                if w < 30 or h < 60:
                    continue
                if w > h * 1.5: 
                    continue
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                
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
                        
                        # 3. ID Kaybetmemesi İçin Re-ID İyileştirmesi:
                        # Eşiği 0.75'ten 0.65'e düşürdük, kişi açısını değiştirse bile tanısın.
                        if best_match_id is not None and best_similarity > 0.65:
                            id_map[yolo_id] = best_match_id
                            
                            # Kişi yürürken vektörünü dinamik olarak güncelle (Eski görünümle harmanla)
                            old_emb = database[best_match_id]["embedding"]
                            new_emb = 0.8 * old_emb + 0.2 * embedding
                            database[best_match_id]["embedding"] = new_emb / np.linalg.norm(new_emb)
                        else:
                            id_map[yolo_id] = next_real_id
                            database[next_real_id] = {
                                "entry_time": current_time,
                                "embedding": embedding
                            }
                            next_real_id += 1
                
                if yolo_id in id_map:
                    real_id = id_map[yolo_id]
                    entry_time = database[real_id]["entry_time"]
                    time_spent = current_time - entry_time
                    
                    if time_spent > TIME_LIMIT:
                        color = (0, 0, 255)
                        text = f"ID:{real_id} IHLAL! ({int(time_spent)}s)"
                    else:
                        color = (0, 255, 0)
                        text = f"ID:{real_id} Sure: {int(time_spent)}s"
                        
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        resized_frame = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow('Sigara Alani Takip', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
