"""
app.py — Sigara Alanı Takip Sistemi Web Sunucusu
"""
import os
import time
import json
import logging
import shutil
from pathlib import Path
from flask import Flask, Response, jsonify, render_template, request, send_file
from smoking_analyzer import SmokingAnalyzer
import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__)

# Küresel analizör nesnesi
analyzer: SmokingAnalyzer | None = None

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

# Canlı yayın başlamadan önce ekranda gösterilecek şık yer tutucu görsel (Placeholder)
def _placeholder() -> bytes:
    import cv2
    import numpy as np
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Tasarımla uyumlu koyu arka plan (15, 15, 25)
    f[:] = (25, 15, 15)
    cv2.putText(f, "Kamera Bekleniyor...", (420, 320), cv2.FONT_HERSHEY_DUPLEX, 1.3, (80, 80, 220), 2, cv2.LINE_AA)
    cv2.putText(f, "Izleme alanini cizin ve Analizi Baslatin", (380, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 150), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()

PH = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    global PH
    if PH is None: 
        PH = _placeholder()
    def generate():
        while True:
            a = analyzer
            if a and a.is_running:
                try: 
                    data = a.frame_queue.get(timeout=2.0)
                except: 
                    data = PH
            else:
                data = PH
                time.sleep(0.1)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stream")
def api_stream():
    def sse():
        while True:
            a = analyzer
            d = a.stats if a else {
                "active": 0, 
                "violation": 0, 
                "in": 0, 
                "out": 0, 
                "fps": 0, 
                "status": "stopped", 
                "error": "", 
                "zone": [0,0,1,1],
                "time_limit": 60
            }
            yield f"data: {json.dumps(d)}\n\n"
            time.sleep(0.5)
    return Response(sse(), mimetype="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/start", methods=["POST"])
def api_start():
    global analyzer
    b = request.get_json(silent=True) or {}
    if analyzer and analyzer.is_running: 
        analyzer.stop()
    
    # Parametreleri al
    source = b.get("source", "rtsp://admin:Yt2240cn@192.168.12.71:554/cam/realmonitor?channel=1&subtype=0").strip()
    zone_coords = b.get("zone_coords", [0.0, 0.0, 1.0, 1.0])
    conf = float(b.get("conf", 0.35))
    time_limit = int(b.get("time_limit", 60))

    analyzer = SmokingAnalyzer(
        source=source,
        zone_coords=zone_coords,
        conf=conf,
        time_limit=time_limit
    )
    ok = analyzer.start()
    return jsonify({"ok": ok})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    if analyzer: 
        analyzer.stop()
    return jsonify({"ok": True})

@app.route("/api/update_zone", methods=["POST"])
def api_update_zone():
    global analyzer
    b = request.get_json(silent=True) or {}
    coords = b.get("zone_coords")
    if analyzer and coords:
        analyzer.update_zone(coords)
        return jsonify({"ok": True})
    return jsonify({"ok": False})

@app.route("/api/update_time_limit", methods=["POST"])
def api_update_time_limit():
    global analyzer
    b = request.get_json(silent=True) or {}
    limit = b.get("time_limit")
    if analyzer and limit is not None:
        analyzer.update_time_limit(int(limit))
        return jsonify({"ok": True})
    return jsonify({"ok": False})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    if analyzer: 
        analyzer.reset_counts()
    db_manager.reset_db()
    
    # static/violations dizinindeki resimleri temizle
    violations_dir = Path("static/violations")
    if violations_dir.exists():
        for file in violations_dir.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    logging.error(f"Görsel silinirken hata: {e}")
                    
    return jsonify({"ok": True})

@app.route("/api/reports")
def api_reports():
    violations = db_manager.get_violations(50)
    logs = db_manager.get_logs(50)
    return jsonify({
        "violations": violations,
        "logs": logs
    })

@app.route("/api/hourly_report")
def api_hourly_report():
    rows = db_manager.get_hourly_logs()
    return jsonify(rows)

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/violations", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, threaded=True)
