import cv2
import time
import numpy as np
import os
import queue
import threading
import logging
import pickle
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
        time_limit: int = 10,
        max_fps: int = 0,
        frame_skip: int = 0,
        jpeg_quality: int = 75,
        dynamic_conf: bool = True,
        adaptive_skip: bool = True,
        use_yolov8m: bool = False
    ) -> None:
        self.source = source
        # YOLOv8m seçiliyse ve dosya yoksa, otomatik olarak yolov8s kullan
        if use_yolov8m and not os.path.exists("yolov8m.pt"):
            logger.warning("yolov8m.pt bulunamadı, yolov8s.pt kullanılacak")
            use_yolov8m = False
        self.model_path = "yolov8m.pt" if use_yolov8m else model_path
        self.zone_coords = zone_coords # Normalize edilmiş [x1, y1, x2, y2]
        self.conf = conf
        self.base_conf = conf
        self.time_limit = time_limit
        self.max_fps = max_fps
        self.frame_skip = frame_skip
        self.jpeg_quality = jpeg_quality
        self.dynamic_conf = dynamic_conf
        self.adaptive_skip = adaptive_skip
        # Re-ID temporal smoothing için son N embedding sakla
        self.reid_smoothing_window = 5  # Son 5 embedding'i ortamala (artırıldı)
        self.reid_threshold = 0.70  # Re-ID threshold (0.78'den düşürüldü)
        # Initialize cooldown (seconds) and daily tracking
        self.COOLDOWN_SECONDS = 0  # Disable long cooldown so repeated violations are counted by time_limit
        self.last_violation_time = {}  # real_id -> timestamp of last counted violation
        self.current_day = datetime.now().date()
        self.daily_violation_count = 0
        self.frame_counter = 0  # processed frame count for skipping

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self._active_count = 0
        self._violation_count = 0
        self._fps = 0.0
        self._status = "stopped"
        self._error_msg = ""
        self._reset_flag = False
        self._full_reset_flag = False

        self.frame_queue = queue.Queue(maxsize=2)
        
        # İhlali bildirilen ID'ler (Tekrar tekrar resim çekilmesini önlemek için)
        self.notified_violations = set()
        # Hiç bir noktada ihlal eden tüm ID'leri takip et (kümülatif sayım için)
        self.all_violations_ever = set()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "active": self._active_count,
                "violation": self._violation_count,
                "daily_violation": self.daily_violation_count,
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
        # Preserve Re-ID gallery across restarts. Existing IDs are retained.
        if os.path.exists("reid_gallery.pkl"):
            logger.info("Re-ID gallery retained across restart.")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        with self._lock:
            self._status = "stopped"

    def reset_counts(self, full_reset: bool = False) -> None:
        """Sayaçları sıfırla.
        full_reset=True ise Re-ID galerisini de temizler."""
        with self._lock:
            self._active_count = 0
            self._violation_count = 0
            self._reset_flag = True
            if full_reset:
                self._full_reset_flag = True
            self.notified_violations.clear()
            self.all_violations_ever.clear()

    def update_zone(self, coords: list):
        with self._lock:
            self.zone_coords = coords

    def update_time_limit(self, limit: int):
        with self._lock:
            self.time_limit = limit

    def _get_cosine_similarity(self, vec1, vec2):
        return 1 - cosine(vec1, vec2)

    def _calculate_dynamic_conf(self, frame):
        """Kare parlaklığına göre dinamik confidence threshold hesapla"""
        if not self.dynamic_conf:
            return self.base_conf
        
        # Kare parlaklığını hesapla (0-255 arası)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        
        # Parlaklık 0-255 arası, normalize et (0.0-1.0)
        norm_brightness = brightness / 255.0
        
        # Karanlıkta daha düşük threshold (daha fazla detection)
        # Aydınlıkta daha yüksek threshold (daha az false positive)
        if norm_brightness < 0.3:
            # Çok karanlık
            return max(0.20, self.base_conf - 0.10)
        elif norm_brightness < 0.5:
            # Karanlık
            return max(0.25, self.base_conf - 0.05)
        elif norm_brightness < 0.7:
            # Normal
            return self.base_conf
        else:
            # Çok aydınlık
            return min(0.50, self.base_conf + 0.10)

    def _get_smoothed_embedding(self, embedding, history):
        """Temporal smoothing - son N embedding'in ortalamasını al"""
        if not history or len(history) == 0:
            return embedding
        
        # History'ye yeni embedding'i ekle
        history.append(embedding)
        
        # Sadece son N embedding'i tut
        if len(history) > self.reid_smoothing_window:
            history = history[-self.reid_smoothing_window:]
        
        # Ortalamayı al
        smoothed = np.mean(history, axis=0)
        
        # L2 normalize et
        smoothed = smoothed / (np.linalg.norm(smoothed) + 1e-8)
        
        return smoothed, history

    def _compute_iou(self, b1, b2) -> float:
        """İki bounding box arasındaki IoU (Intersection over Union) değerini hesapla."""
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _filter_duplicate_boxes(self, boxes, ids, iou_threshold: float = 0.65):
        """Ayna/yansıma veya yanlış duplikat tespitlerini IoU tabanlı NMS ile eleyin.
        
        Eşik 0.65: Sadece neredeyse tamamen üst üste binen kutuları (gerçek yansımalar)
        eler. Kalabalık sahnelerde komşu insanların kutularını yanlışlıkla silmez.
        """
        n = len(boxes)
        suppressed = set()
        areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in range(n)]

        for i in range(n):
            if i in suppressed:
                continue
            for j in range(i + 1, n):
                if j in suppressed:
                    continue
                iou = self._compute_iou(boxes[i], boxes[j])
                if iou > iou_threshold:
                    # Küçük alanı at — yansıma genelde küçük görünür
                    suppressed.add(j if areas[i] >= areas[j] else i)

        kept_boxes = [boxes[k] for k in range(n) if k not in suppressed]
        kept_ids   = [ids[k]   for k in range(n) if k not in suppressed]
        if suppressed:
            logger.debug(f"Duplikat filtresi: {len(suppressed)} kutu elendi (ayna/yansıma koruma).")
        return kept_boxes, kept_ids

    def _save_gallery(self, database):
        try:
            with open("reid_gallery.pkl", "wb") as f:
                pickle.dump(database, f)
        except Exception as e:
            logger.error(f"Re-ID galerisi kaydedilirken hata: {e}")

    def _load_gallery(self) -> dict:
        try:
            if os.path.exists("reid_gallery.pkl"):
                with open("reid_gallery.pkl", "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Re-ID galerisi yuklenirken hata: {e}")
        return {}

    def _run_loop(self) -> None:
        try:
            logger.info("YOLOv8 Modeli yükleniyor...")
            model = YOLO(self.model_path)
            
            logger.info("Öznitelik Çıkarıcı (Re-ID) yükleniyor...")
            feature_extractor = FeatureExtractor()

            is_rtsp = isinstance(self.source, str) and self.source.startswith(("rtsp://", "rtmp://", "http://"))
            
            # RTSP bağlantı sayaçları
            reconnect_attempts = 0
            max_reconnect_attempts = 10
            reconnect_delay = 2  # İlk deneme 2 saniye
            
            def get_capture(src):
                if is_rtsp:
                    # RTSP optimizasyonları
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
                    c = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
                    c.set(cv2.CAP_PROP_FPS, 30)
                    # Network timeout (5 saniye)
                    c.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    c.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                    return c
                return cv2.VideoCapture(src)

            cap = get_capture(self.source)
            if not cap.isOpened(): 
                raise RuntimeError(f"Kaynak açilamadi: {self.source}")

            target_w, target_h = 960, 540
            
            # Re-ID takip veritabanı — her kayıtta 'embedding' ve 'total_seconds' bulunur
            database = self._load_gallery()
            
            # Model değişimini kontrol et (embedding boyut uyuşmazlığı)
            if database:
                dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
                dummy_emb = feature_extractor.extract(dummy_crop)
                if dummy_emb is not None:
                    first_key = list(database.keys())[0]
                    first_emb = database[first_key].get("embedding")
                    if first_emb is not None and first_emb.shape != dummy_emb.shape:
                        logger.warning(f"Re-ID model değişimi algılandı (Eski: {first_emb.shape}, Yeni: {dummy_emb.shape}). Eski galeri temizleniyor.")
                        database = {}
                        if os.path.exists("reid_gallery.pkl"):
                            try:
                                os.remove("reid_gallery.pkl")
                            except Exception as e:
                                logger.error(f"Eski galeri dosyası silinemedi: {e}")
            
            # Eski kayıtlarda total_seconds yoksa 0 ile başlat
            for _rid, _data in database.items():
                if "total_seconds" not in _data:
                    _data["total_seconds"] = 0.0
                if "embedding_history" not in _data:
                    _data["embedding_history"] = []
            id_map = {}          # yolo_id -> real_id
            yolo_seen = {}       # yolo_id -> last frame index where it was seen
            next_real_id = max(database.keys()) + 1 if database else 0
            last_seen = {}       # real_id -> time.time() kişinin son görüldüğü zaman
            active_sessions = {} # real_id -> oturum bilgileri

            fps_buf = []
            last_db_log_time = time.time()
            last_gallery_save = time.time()  # Galeriyi her 60 saniyede bir kaydet
            last_hour = datetime.now().hour
            processed_frame_idx = 0  # işlenen kare sayacı
            last_health_check = time.time()
            health_check_interval = 10  # Her 10 saniyede bir health check

            # violations klasörünün varlığından emin ol
            os.makedirs("static/violations", exist_ok=True)

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if not is_rtsp:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        reconnect_attempts += 1
                        if reconnect_attempts > max_reconnect_attempts:
                            logger.error(f"Maksimum yeniden bağlantı denemesi ({max_reconnect_attempts}) aşıldı. Durduruluyor.")
                            break
                        
                        # Exponential backoff: 2, 4, 8, 16 saniye
                        current_delay = min(reconnect_delay * (2 ** (reconnect_attempts - 1)), 30)
                        logger.warning(f"Canlı yayın koptu (Deneme {reconnect_attempts}/{max_reconnect_attempts}). {current_delay} saniye içinde yeniden bağlanılıyor...")
                        cap.release()
                        time.sleep(current_delay)
                        
                        cap = get_capture(self.source)
                        if cap.isOpened():
                            logger.info("Yeniden bağlantı başarılı!")
                            reconnect_attempts = 0  # Sayaçları sıfırla
                            self.frame_counter = 0
                        else:
                            logger.error("Yeniden bağlantı başarısız. Tekrar deneniyor...")
                        continue

                frame_small = cv2.resize(frame, (target_w, target_h))
                self.frame_counter += 1
                
                # Health check - her 10 saniyede bir bağlantı durumunu kontrol et
                current_time = time.time()
                if is_rtsp and (current_time - last_health_check > health_check_interval):
                    last_health_check = current_time
                    # FPS çok düşükse bağlantı sorunu olabilir
                    if self._fps < 1.0 and reconnect_attempts == 0:
                        logger.warning(f"FPS çok düşük ({self._fps:.1f}), bağlantı kontrol ediliyor...")
                        # Bir frame okumayı dene
                        test_ret, test_frame = cap.read()
                        if not test_ret or test_frame is None:
                            logger.warning("Health check başarısız, yeniden bağlanılıyor...")
                            ret = False  # Ana döngüde yeniden bağlanmayı tetikle
                        else:
                            # Test frame'i okundu, normal devam et
                            pass
                
                # Adaptive frame skipping: Zone'da aktivite yoksa daha fazla skip
                current_skip = self.frame_skip
                if self.adaptive_skip and self._active_count == 0:
                    current_skip = min(current_skip + 1, 4)  # Max 4 frame skip when no activity
                elif self.adaptive_skip and self._active_count > 0:
                    current_skip = max(current_skip - 1, 0)  # Reduce skip when activity detected
                
                # If frame_skip is set, process only every (frame_skip+1)th frame
                if current_skip > 0 and (self.frame_counter % (current_skip + 1)) != 0:
                    # Skip heavy detection, just encode and enqueue the resized frame
                    annotated_frame = frame_small.copy()
                    ok, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    if ok:
                        try:
                            self.frame_queue.put_nowait(buf.tobytes())
                        except queue.Full:
                            try:
                                self.frame_queue.get_nowait()
                            except:
                                pass
                            self.frame_queue.put(buf.tobytes())
                    continue
                t0 = time.time()
                processed_frame_idx += 1
                now = datetime.now()
                # Gün değişimi kontrolü – günlük ihlal sayısını sıfırla
                today_date = now.date()
                if today_date != self.current_day:
                    self.current_day = today_date
                    self.daily_violation_count = 0
                    # Not needed to clear last_violation_time (keep for cooldown)

                # Saat başı otomatik sıfırlama — galeri korunur, sadece sayaçlar ve oturumlar sıfırlanır
                if now.hour != last_hour:
                    logger.info(f"Yeni saat ({now.hour}) başladı, sayaçlar sıfırlanıyor (galeri korunuyor).")
                    self.reset_counts()
                    last_hour = now.hour

                # Veritabanına yoğunluk raporunu kaydet (Her 60 saniyede bir)
                if t0 - last_db_log_time > 60:
                    with self._lock:
                        cur_active = self._active_count
                        cur_viol = self._violation_count
                    db_manager.add_log(cur_active, cur_viol)
                    last_db_log_time = t0



                with self._lock:
                    curr_zone = self.zone_coords
                    do_reset = self._reset_flag
                    do_full_reset = getattr(self, "_full_reset_flag", False)
                    curr_time_limit = self.time_limit
                    if do_reset:
                        self._reset_flag = False
                        id_map.clear()
                        last_seen.clear()
                        active_sessions.clear()
                        if do_full_reset:
                            self._full_reset_flag = False
                            database.clear()
                            next_real_id = 0
                            if os.path.exists("reid_gallery.pkl"):
                                try:
                                    os.remove("reid_gallery.pkl")
                                    logger.info("Kalıcı Re-ID galerisi temizlendi.")
                                except:
                                    pass
                            logger.info("Sayaçlar ve Re-ID galerisi tamamen sıfırlandı.")
                        else:
                            logger.info("Sayaçlar sıfırlandı. Re-ID galerisi ve kümülatif süreler korundu.")

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

                # YOLOv8 Takip İşlemi (Geliştirilmiş custom_tracker.yaml kullanılıyor)
                # Dinamik confidence threshold hesapla
                current_conf = self._calculate_dynamic_conf(frame_small)
                results = model.track(frame_small, classes=[0], conf=current_conf, persist=True, tracker="custom_tracker.yaml", verbose=False)
                current_time = time.time()
                
                frame_active_count = 0
                violators = set()
                assigned_real_ids_this_frame = set()

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)

                    # 🔴 Ayna/Yansıma Koruma: Yüksek IoU'lu duplikat kutuları eleyin
                    boxes, ids = self._filter_duplicate_boxes(list(boxes), list(ids))

                    for box, yolo_id in zip(boxes, ids):
                        bx1, by1, bx2, by2 = box
                        
                        # İnsan dışı çok küçük gürültüleri elemek için boyut kontrolü
                        w = bx2 - bx1
                        h = by2 - by1
                        if w < 15 or h < 30:
                            continue

                        # Sınır taşmalarını engelle
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(target_w, bx2), min(target_h, by2)

                        # Bounding box alt-orta noktasını kontrol noktası seçelim
                        bc_x = int((bx1 + bx2) / 2.0)
                        bc_y = by2

                        # Kişi izleme bölgesinin içinde mi? (Geliştirilmiş overlap hesabı)
                        ix1 = max(bx1, x_start)
                        iy1 = max(by1, y_start)
                        ix2 = min(bx2, x_end)
                        iy2 = min(by2, y_end)
                        iw = max(0, ix2 - ix1)
                        ih = max(0, iy2 - iy1)
                        intersection_area = iw * ih
                        box_area = w * h
                        overlap_ratio = intersection_area / box_area if box_area > 0 else 0.0
                        
                        # Ayrıca merkez noktasının bölge içinde olup olmadığını kontrol et
                        center_inside = (x_start <= bc_x <= x_end) and (y_start <= bc_y <= y_end)
                        
                        # Geliştirilmiş kriter: Ya %50 overlap YA DA merkez noktası içinde + %30 overlap
                        is_inside = (overlap_ratio >= 0.50) or (center_inside and overlap_ratio >= 0.30)

                        if is_inside:
                            # Eşleştirme işlemi — her frame'de çalıştır (daha sık Re-ID)
                            gallery_dirty = False
                            
                            # Eğer yolo_id zaten eşlenmişse ancak o real_id bu karede başka bir kutuya atandıysa,
                            # çakışmayı önlemek için yolo_id'yi yeniden değerlendir
                            if yolo_id in id_map:
                                if id_map[yolo_id] in assigned_real_ids_this_frame:
                                    logger.warning(f"Cakisma algilandi: real_id {id_map[yolo_id]} bu karede zaten atandi. yolo_id {yolo_id} yeniden degerlendiriliyor.")
                                    id_map.pop(yolo_id, None)

                            is_new_yolo = yolo_id not in id_map
                            
                            # Sadece yeni bir yolo_id gördüğümüzde Re-ID kontrolü yap
                            if is_new_yolo:
                                # 1. ÖNCE KONUM VE KUTU ÖRTÜŞME TABANLI HIZLI EŞLEŞTİRME (Postür değişimleri ve kısa süreli kayıplar için)
                                fallback_id = None
                                fallback_score = -1.0
                                for rid, session in active_sessions.items():
                                    # Bu karede zaten başka birine atanmış ID'leri atla
                                    if rid in assigned_real_ids_this_frame:
                                        continue
                                    if rid in last_seen and current_time - last_seen[rid] < 5.0:
                                        sbx1, sby1, sbx2, sby2 = session.get("last_box", (0, 0, 0, 0))
                                        # Kutu kesişim alanı (Overlap) hesabı
                                        ix1 = max(bx1, sbx1)
                                        iy1 = max(by1, sby1)
                                        ix2 = min(bx2, sbx2)
                                        iy2 = min(by2, sby2)
                                        iw = max(0, ix2 - ix1)
                                        ih = max(0, iy2 - iy1)
                                        intersection = iw * ih
                                        
                                        overlap = 0.0
                                        if intersection > 0:
                                            area1 = (bx2 - bx1) * (by2 - by1)
                                            area2 = (sbx2 - sbx1) * (sby2 - sby1)
                                            min_area = min(area1, area2)
                                            overlap = intersection / min_area if min_area > 0 else 0.0
                                        
                                        cx, cy = session.get("last_center", (None, None))
                                        dist = np.hypot(bc_x - cx, bc_y - cy) if cx is not None else 999.0
                                        
                                        # Kısa süreli görünmeyenlerle örtüşme/yakınlık kontrolü
                                        # Kalabalık sahneler için mesafe eşiği kutu yüksekliğiyle orantılı
                                        box_h = by2 - by1
                                        max_dist = max(60, box_h * 0.6)  # maks 60px veya kutunun %60'ı
                                        if overlap > 0.10 or dist < max_dist:
                                            # Örtüşme öncelikli skorlama, yoksa yakınlık tabanlı
                                            score = overlap if overlap > 0.10 else (1.0 / (dist + 1.0))
                                            if score > fallback_score:
                                                fallback_score = score
                                                fallback_id = rid

                                if fallback_id is not None:
                                    id_map[yolo_id] = fallback_id
                                else:
                                    # 2. YAKINDA AKTİF BİRİ YOKSA, Re-ID İLE VERİTABANINDAN SORGULA
                                    crop = frame_small[by1:by2, bx1:bx2]
                                    embedding = feature_extractor.extract(crop)
                                    
                                    # Yüz tanıma ekle - vücut embedding'i ile birlikte yüz embedding'i de al
                                    face_embedding = feature_extractor.extract_face(crop)

                                    if embedding is not None:
                                        # Temporal smoothing uygula
                                        history = []
                                        best_match_id = None
                                        best_similarity = -1
                                        best_face_similarity = -1

                                        for real_id, data in database.items():
                                            if real_id in assigned_real_ids_this_frame:
                                                continue
                                            person_history = data.get("embedding_history", [])
                                            if person_history:
                                                smoothed_emb, _ = self._get_smoothed_embedding(embedding.copy(), person_history.copy())
                                            else:
                                                smoothed_emb = embedding
                                            
                                            # Vücut similarity
                                            body_sim = self._get_cosine_similarity(smoothed_emb, data["embedding"])
                                            
                                            # Yüz similarity - eğer yüz embedding'i varsa
                                            face_sim = -1
                                            if face_embedding is not None and "face_embedding" in data:
                                                face_sim = self._get_cosine_similarity(face_embedding, data["face_embedding"])
                                            
                                            # Combined score: yüz varsa %60 yüz + %40 vücut, yoksa %100 vücut
                                            if face_sim > 0:
                                                combined_sim = 0.6 * face_sim + 0.4 * body_sim
                                            else:
                                                combined_sim = body_sim
                                            
                                            if combined_sim > best_similarity:
                                                best_similarity = combined_sim
                                                best_match_id = real_id
                                                best_face_similarity = face_sim

                                        # Threshold: yüz tanıma varsa daha düşük threshold (0.60), yoksa 0.70
                                        threshold = 0.60 if best_face_similarity > 0 else self.reid_threshold
                                        
                                        if best_match_id is not None and best_similarity > threshold:
                                            id_map[yolo_id] = best_match_id
                                            old_emb = database[best_match_id]["embedding"]
                                            new_emb = 0.7 * old_emb + 0.3 * embedding
                                            database[best_match_id]["embedding"] = new_emb / np.linalg.norm(new_emb)
                                            database[best_match_id]["embedding_history"] = database[best_match_id].get("embedding_history", [])
                                            database[best_match_id]["embedding_history"].append(embedding)
                                            if len(database[best_match_id]["embedding_history"]) > self.reid_smoothing_window:
                                                database[best_match_id]["embedding_history"] = database[best_match_id]["embedding_history"][-self.reid_smoothing_window:]
                                            
                                            # Yüz embedding'ini de güncelle
                                            if face_embedding is not None:
                                                if "face_embedding" in database[best_match_id]:
                                                    old_face = database[best_match_id]["face_embedding"]
                                                    new_face = 0.7 * old_face + 0.3 * face_embedding
                                                    database[best_match_id]["face_embedding"] = new_face / np.linalg.norm(new_face)
                                                else:
                                                    database[best_match_id]["face_embedding"] = face_embedding
                                            gallery_dirty = True
                                        else:
                                            # Tamamen yeni bir kişi
                                            id_map[yolo_id] = next_real_id
                                            person_data = {
                                                "embedding": embedding,
                                                "total_seconds": 0.0,
                                                "embedding_history": [embedding]
                                            }
                                            # Yüz embedding'i de ekle
                                            if face_embedding is not None:
                                                person_data["face_embedding"] = face_embedding
                                            database[next_real_id] = person_data
                                            next_real_id += 1
                                            gallery_dirty = True
                                    else:
                                        # ⚠️ Embedding çıkmadı (kötü kırpma, kapanma vb.)
                                        # Kişiyi tamamen atlamak yerine yeni ID ile kaydet
                                        # Böylece aktif sayım ve ihlal takibi doğru çalışır
                                        logger.debug(f"yolo_id={yolo_id} için embedding çıkarılamadı, yeni ID atanıyor.")
                                        id_map[yolo_id] = next_real_id
                                        database[next_real_id] = {
                                            "embedding": np.zeros(576),  # boş placeholder embedding
                                            "total_seconds": 0.0,
                                            "embedding_history": []
                                        }
                                        next_real_id += 1
                                        gallery_dirty = True

                            # Galeriyi diske yaz — sık sık yazmak FPS'i öldürür
                            if gallery_dirty and (current_time - last_gallery_save > 60.0):
                                self._save_gallery(database)
                                last_gallery_save = current_time
                            
                            if yolo_id in id_map:
                                real_id = id_map[yolo_id]
                                assigned_real_ids_this_frame.add(real_id)
                                last_seen[real_id] = current_time
                                
                                # Eğer bu kişi aktif bir ziyarette değilse yeni oturum başlat
                                if real_id not in active_sessions:
                                    active_sessions[real_id] = {
                                        "entry_time": current_time,
                                        "last_center": (bc_x, bc_y),
                                        "last_box": (bx1, by1, bx2, by2),
                                        "next_violation_time": current_time + curr_time_limit,
                                        "last_ss_time": 0  # Son SS zamanı
                                    }
                                    with self._lock:
                                        self.notified_violations.discard(real_id)
                                else:
                                    active_sessions[real_id]["last_center"] = (bc_x, bc_y)
                                    active_sessions[real_id]["last_box"] = (bx1, by1, bx2, by2)
                                    if "next_violation_time" not in active_sessions[real_id]:
                                        active_sessions[real_id]["next_violation_time"] = active_sessions[real_id]["entry_time"] + curr_time_limit
                                    if "last_ss_time" not in active_sessions[real_id]:
                                        active_sessions[real_id]["last_ss_time"] = 0

                                entry_time = active_sessions[real_id]["entry_time"]
                                time_spent = current_time - entry_time
                                frame_active_count += 1

                                session = active_sessions[real_id]
                                is_violating = current_time >= session["next_violation_time"]
                                if is_violating:
                                    color = COLOR_VIOLATION
                                    text = f"ID:{real_id} IHLAL! ({int(time_spent)}s)"

                                    # Her ziyaret için ayrı SS al - violation_reported flag'i kaldırıldı
                                    # Her 5 saniyede bir yeni SS al (flood önlemek için)
                                    last_ss_time = session.get("last_ss_time", 0)
                                    if current_time - last_ss_time > 5.0:
                                        session["last_ss_time"] = current_time
                                        violators.add(real_id)
                                        # 🟠 Race Condition Düzeltme: Tüm paylaşılan değişkenler
                                        # tek bir lock bloğunda güncelleniyor
                                        with self._lock:
                                            self.last_violation_time[real_id] = current_time
                                            self.daily_violation_count += 1
                                            self.all_violations_ever.add(real_id)
                                            self._violation_count += 1

                                        # Gün değişimi kontrolü
                                        today = datetime.now().date()
                                        if today != self.current_day:
                                            self.current_day = today
                                            self.daily_violation_count = 0

                                        # Vücut fotoğrafını kaydet (her 5 saniyede bir)
                                        # Padding artırıldı - daha büyük ve daha net fotoğraf için
                                        pad_x = int(w * 0.25)  # %10'dan %25'e çıkarıldı
                                        pad_y = int(h * 0.25)  # %10'dan %25'e çıkarıldı
                                        sy1 = max(0, by1 - pad_y)
                                        sy2 = min(target_h, by2 + pad_y)
                                        sx1 = max(0, bx1 - pad_x)
                                        sx2 = min(target_w, bx2 + pad_x)

                                        crop_img = frame_small[sy1:sy2, sx1:sx2]
                                        # Crop boyut kontrolü - daha güvenli
                                        if crop_img is not None and crop_img.size > 0 and crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                                            # Sharpening filter ekle - daha net fotoğraf için
                                            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                                            sharpened = cv2.filter2D(crop_img, -1, kernel)
                                            
                                            # Daha yüksek kalite ile kaydet
                                            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            ms = int(time.time() * 1000) % 1000
                                            filename = f"violation_{real_id}_{ts_str}_{ms:03d}.jpg"
                                            rel_path = f"static/violations/{filename}"
                                            saved = cv2.imwrite(rel_path, sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Kalite 95'e çıkarıldı
                                            if saved:
                                                db_manager.add_violation(real_id, int(time_spent), rel_path)
                                                logger.info(f"İhlal Fotoğrafı Kaydedildi: {rel_path}")
                                            else:
                                                logger.error(f"İhlal fotoğrafı kaydedilemedi: {rel_path}")
                                                # Fotoğraf kaydedilemese bile veritabanına kaydet
                                                db_manager.add_violation(real_id, int(time_spent), rel_path)
                                        else:
                                            # Crop başarısız olsa bile veritabanına kaydet
                                            logger.warning(f"Crop başarısız, sadece veritabanına kaydediliyor: ID {real_id}")
                                            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            ms = int(time.time() * 1000) % 1000
                                            filename = f"violation_{real_id}_{ts_str}_{ms:03d}.jpg"
                                            rel_path = f"static/violations/{filename}"
                                            db_manager.add_violation(real_id, int(time_spent), rel_path)
                                        
                                        # Loglama başarılı olduğunu onayla
                                        logger.info(f"İhlal loglandı: ID {real_id}, Süre: {int(time_spent)}s")
                                else:
                                    color = COLOR_ACTIVE
                                    text = f"ID:{real_id} {int(time_spent)}s"

                                # Kümülatif toplam süreyi etikete ekle
                                total_s = database.get(real_id, {}).get("total_seconds", 0.0)
                                total_this_visit = total_s + time_spent  # şu anki ziyaret henüz eklenmedi
                                total_min = int(total_this_visit // 60)
                                total_sec = int(total_this_visit % 60)
                                if total_min > 0:
                                    total_label = f" [Top:{total_min}dk{total_sec}s]"
                                else:
                                    total_label = f" [Top:{total_sec}s]"
                                text = text + total_label

                                # Çizim yap
                                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                                cv2.putText(annotated_frame, text, (bx1, by1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        else:
                            # İzleme bölgesi dışında kalanlar için gri kutu çiz
                            color = COLOR_OUTSIDE
                            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 1)
                            cv2.putText(annotated_frame, f"ID:{yolo_id} (Disarida)", (bx1, by1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Pasiflik Temizliği: 15 saniyeden uzun süredir görünmeyenlerin aktif oturumunu sonlandır
                    # Re-ID galerisi ve kümülatif süreler ASLA silinmez!
                    for rid in list(active_sessions.keys()):
                        if rid in last_seen and current_time - last_seen[rid] > 15.0:
                            session_data = active_sessions.pop(rid, {})
                            # Bu ziyaretin süresini kümülatif toplama ekle
                            visit_duration = last_seen.get(rid, current_time) - session_data.get("entry_time", current_time)
                            if visit_duration > 0 and rid in database:
                                database[rid]["total_seconds"] = database[rid].get("total_seconds", 0.0) + visit_duration
                                # Galeriyi kaydet (kümülatif süre güncellemesi)
                                self._save_gallery(database)
                                last_gallery_save = current_time
                                logger.info(f"ID:{rid} alandan ayrıldı. Ziyaret: {int(visit_duration)}s | Toplam: {int(database[rid]['total_seconds'])}s")
                            last_seen.pop(rid, None)
                            with self._lock:
                                self.notified_violations.discard(rid)
                            # id_map içindeki bu rid'ye eşleşen tüm yolo_id'leri sil
                            for yid in list(id_map.keys()):
                                if id_map[yid] == rid:
                                    id_map.pop(yid, None)

                frame_violation_count = len(violators)

                # İstatistikleri güncelle
                with self._lock:
                    self._active_count = frame_active_count
                    # Toplam ihlal sayısı zaten her ihlal olayında artırılıyor

                # FPS Hesabı
                fps_buf.append(time.time() - t0)
                if len(fps_buf) > 30: 
                    fps_buf.pop(0)
                # Enforce max FPS if set
                elapsed = time.time() - t0
                if self.max_fps > 0:
                    target_frame_time = 1.0 / self.max_fps
                    if elapsed < target_frame_time:
                        time.sleep(target_frame_time - elapsed)
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
        finally:
            if 'database' in locals() and database:
                self._save_gallery(database)
                logger.info("Re-ID galerisi durdurulurken diske kaydedildi.")
