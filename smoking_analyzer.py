import cv2
import time
import numpy as np
import os
import queue
import threading
import logging
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
from reid import FeatureExtractor
from scipy.spatial.distance import cosine
import db_manager

db_manager.init_db()

logger = logging.getLogger("SmokingAnalyzer")

# Renkler (BGR formatı)
COLOR_ZONE = (0, 0, 255)       # Neon Kırmızı İzleme Bölgesi
COLOR_ACTIVE = (0, 255, 0)     # Yeşil (Normal Süre)
COLOR_VIOLATION = (0, 0, 255)  # Kırmızı (İhlal)
COLOR_OUTSIDE = (120, 120, 120)# Gri (İzleme Bölgesi Dışında)

class SmokingAnalyzer:
    def __init__(
        self,
        source: str,
        model_path: str = "yolov8s.pt",
        zone_coords: list = [0.0, 0.0, 1.0, 1.0], # [x1, y1, x2, y2]
        conf: float = 0.35,
        time_limit: int = 60
    ) -> None:
        self.source = source
        self.model_path = model_path
        self.zone_coords = zone_coords # Normalize edilmiş [x1, y1, x2, y2]
        self.conf = conf
        self.time_limit = time_limit

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self._active_count = 0
        self._violation_count = 0
        self._fps = 0.0
        self._status = "stopped"
        self._error_msg = ""
        self._reset_flag = False

        self.frame_queue = queue.Queue(maxsize=2)
        
        # İhlali bildirilen ID'ler (Tekrar tekrar resim çekilmesini önlemek için)
        self.notified_violations = set()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "active": self._active_count,
                "violation": self._violation_count,
                "in": self._active_count,        # AICarCounter uyumluluğu için
                "out": self._violation_count,    # AICarCounter uyumluluğu için
                "fps": round(self._fps, 1),
                "status": self._status,
                "error": self._error_msg,
                "zone": self.zone_coords,
                "time_limit": self.time_limit
            }

    @property
    def is_running(self) -> bool:
        return self._status == "running"

    def start(self) -> bool:
        if self._status == "running": 
            return False
        self._stop_event.clear()
        with self._lock:
            self._active_count = 0
            self._violation_count = 0
            self._status = "running"
            self.notified_violations.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread: 
            self._thread.join(timeout=2)
        self._status = "stopped"

    def reset_counts(self) -> None:
        with self._lock:
            self._active_count = 0
            self._violation_count = 0
            self._reset_flag = True
            self.notified_violations.clear()

    def update_zone(self, coords: list):
        with self._lock:
            self.zone_coords = coords

    def update_time_limit(self, limit: int):
        with self._lock:
            self.time_limit = limit

    def _get_cosine_similarity(self, vec1, vec2):
        return 1 - cosine(vec1, vec2)

    def _run_loop(self) -> None:
        try:
            logger.info("YOLOv8 Modeli yükleniyor...")
            model = YOLO(self.model_path)
            
            logger.info("Öznitelik Çıkarıcı (Re-ID) yükleniyor...")
            feature_extractor = FeatureExtractor()

            is_rtsp = isinstance(self.source, str) and self.source.startswith(("rtsp://", "rtmp://", "http://"))
            
            def get_capture(src):
                if is_rtsp:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    c = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return c
                return cv2.VideoCapture(src)

            cap = get_capture(self.source)
            if not cap.isOpened(): 
                raise RuntimeError(f"Kaynak açilamadi: {self.source}")

            target_w, target_h = 1280, 720
            
            # Re-ID takip veritabanı
            database = {}
            id_map = {}
            next_real_id = 0
            last_seen = {}  # Kişinin en son görüldüğü zamanı tutar
            
            fps_buf = []
            last_db_log_time = time.time()
            last_hour = datetime.now().hour

            # violations klasörünün varlığından emin ol
            os.makedirs("static/violations", exist_ok=True)

            while not self._stop_event.is_set():
                t0 = time.time()
                now = datetime.now()

                # Saat başı otomatik sıfırlama
                if now.hour != last_hour:
                    logger.info(f"Yeni saat ({now.hour}) başladı, veriler sıfırlanıyor.")
                    self.reset_counts()
                    last_hour = now.hour

                # Veritabanına yoğunluk raporunu kaydet (Her 60 saniyede bir)
                if t0 - last_db_log_time > 60:
                    with self._lock:
                        cur_active = self._active_count
                        cur_viol = self._violation_count
                    db_manager.add_log(cur_active, cur_viol)
                    last_db_log_time = t0

                ret, frame = cap.read()
                if not ret:
                    if not is_rtsp:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.warning("Canlı yayın koptu, 5 saniye içinde yeniden bağlanılıyor...")
                        cap.release()
                        time.sleep(5)
                        cap = get_capture(self.source)
                        continue

                with self._lock:
                    curr_zone = self.zone_coords
                    do_reset = self._reset_flag
                    curr_time_limit = self.time_limit
                    if do_reset:
                        self._reset_flag = False
                        database.clear()
                        id_map.clear()
                        next_real_id = 0

                # Çözünürlüğü standardize et (hız ve kararlılık için)
                frame_small = cv2.resize(frame, (target_w, target_h))
                annotated_frame = frame_small.copy()

                # İzleme Bölgesi koordinatlarını hesapla
                zx1 = int(curr_zone[0] * target_w)
                zy1 = int(curr_zone[1] * target_h)
                zx2 = int(curr_zone[2] * target_w)
                zy2 = int(curr_zone[3] * target_h)
                
                # Çizim sınırları için min/max düzenleme
                x_start, x_end = min(zx1, zx2), max(zx1, zx2)
                y_start, y_end = min(zy1, zy2), max(zy1, zy2)

                # İzleme Bölgesini Çiz (Neon Kırmızı Kesikli Çizgi simülasyonu)
                cv2.rectangle(annotated_frame, (x_start, y_start), (x_end, y_end), COLOR_ZONE, 2)
                cv2.putText(annotated_frame, "IZLEME ALANI", (x_start + 10, y_start + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ZONE, 2)

                # YOLOv8 Takip İşlemi
                results = model.track(frame_small, classes=[0], conf=self.conf, persist=True, tracker="bytetrack.yaml", verbose=False)
                current_time = time.time()
                
                frame_active_count = 0
                frame_violation_count = 0

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)

                    for box, yolo_id in zip(boxes, ids):
                        bx1, by1, bx2, by2 = box
                        
                        # İnsan dışı nesneleri elemek için boyut kontrolü
                        w = bx2 - bx1
                        h = by2 - by1
                        if w < 30 or h < 60:
                            continue
                        if w > h * 1.5:
                            continue

                        # Sınır taşmalarını engelle
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(target_w, bx2), min(target_h, by2)

                        # Bounding box alt-orta noktasını kontrol noktası seçelim
                        bc_x = int((bx1 + bx2) / 2.0)
                        bc_y = bx2 # Y ekseninde ayak hizası en doğrusudur

                        # Kişi izleme bölgesinin içinde mi?
                        is_inside = (x_start <= bc_x <= x_end) and (y_start <= bc_y <= y_end)

                        if is_inside:
                            # Re-ID eşleştirme
                            if yolo_id not in id_map:
                                crop = frame_small[by1:by2, bx1:bx2]
                                embedding = feature_extractor.extract(crop)

                                if embedding is not None:
                                    best_match_id = None
                                    best_similarity = -1

                                    for real_id, data in database.items():
                                        sim = self._get_cosine_similarity(embedding, data["embedding"])
                                        if sim > best_similarity:
                                            best_similarity = sim
                                            best_match_id = real_id

                                    # Eşik değeri aşılırsa eski ID ile eşleştir
                                    if best_match_id is not None and best_similarity > 0.88:
                                        id_map[yolo_id] = best_match_id
                                        # embedding güncelle
                                        old_emb = database[best_match_id]["embedding"]
                                        new_emb = 0.8 * old_emb + 0.2 * embedding
                                        database[best_match_id]["embedding"] = new_emb / np.linalg.norm(new_emb)
                                    else:
                                        # Yeni kişi kaydı
                                        id_map[yolo_id] = next_real_id
                                        database[next_real_id] = {
                                            "entry_time": current_time,
                                            "embedding": embedding
                                        }
                                        next_real_id += 1

                            if yolo_id in id_map:
                                real_id = id_map[yolo_id]
                                last_seen[real_id] = current_time
                                entry_time = database[real_id]["entry_time"]
                                time_spent = current_time - entry_time
                                frame_active_count += 1

                                if time_spent > curr_time_limit:
                                    frame_violation_count += 1
                                    color = COLOR_VIOLATION
                                    text = f"ID:{real_id} IHLAL! ({int(time_spent)}s)"

                                    # İHLAL ANININ YAKIN PLAN FOTOĞRAFINI ÇEK (Tek Seferlik)
                                    if real_id not in self.notified_violations:
                                        self.notified_violations.add(real_id)
                                        
                                        # Vücut kırpıntısını al (biraz genişletilmiş sınırlarla)
                                        pad_x = int(w * 0.1)
                                        pad_y = int(h * 0.1)
                                        sy1 = max(0, by1 - pad_y)
                                        sy2 = min(target_h, by2 + pad_y)
                                        sx1 = max(0, bx1 - pad_x)
                                        sx2 = min(target_w, bx2 + pad_x)
                                        
                                        crop_img = frame_small[sy1:sy2, sx1:sx2]
                                        if crop_img.size > 0:
                                            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            filename = f"violation_{real_id}_{ts_str}.jpg"
                                            rel_path = f"static/violations/{filename}"
                                            
                                            # Kaydet ve veritabanına ekle
                                            cv2.imwrite(rel_path, crop_img)
                                            db_manager.add_violation(real_id, int(time_spent), rel_path)
                                            logger.info(f"İhlal Fotoğrafı Kaydedildi: {rel_path}")
                                else:
                                    color = COLOR_ACTIVE
                                    text = f"ID:{real_id} {int(time_spent)}s"

                                # Çizim yap
                                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                                cv2.putText(annotated_frame, text, (bx1, by1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            # İzleme bölgesi dışında kalanlar için gri kutu çiz
                            color = COLOR_OUTSIDE
                            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 1)
                            cv2.putText(annotated_frame, f"ID:{yolo_id} (Disarida)", (bx1, by1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Pasiflik Temizliği: 15 saniyeden uzun süredir görünmeyen kişileri sil
                    for rid in list(database.keys()):
                        if rid in last_seen and current_time - last_seen[rid] > 15.0:
                            database.pop(rid, None)
                            last_seen.pop(rid, None)
                            with self._lock:
                                self.notified_violations.discard(rid)
                            # id_map içindeki bu rid'ye eşleşen tüm yolo_id'leri sil
                            for yid in list(id_map.keys()):
                                if id_map[yid] == rid:
                                    id_map.pop(yid, None)

                # İstatistikleri güncelle
                with self._lock:
                    self._active_count = frame_active_count
                    self._violation_count = frame_violation_count

                # FPS Hesabı
                fps_buf.append(time.time() - t0)
                if len(fps_buf) > 30: 
                    fps_buf.pop(0)
                self._fps = 1.0 / (sum(fps_buf) / len(fps_buf))

                # Video Yayını için Kareyi Sıkıştır ve Sıraya Ekle
                ok, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    try:
                        self.frame_queue.put_nowait(buf.tobytes())
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                        self.frame_queue.put(buf.tobytes())

            cap.release()
            logger.info("Analiz döngüsü sonlandırıldı.")
        except Exception as e:
            logger.exception("Analiz motorunda kritik hata oluştu:")
            self._status = "error"
            self._error_msg = str(e)
