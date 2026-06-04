import sqlite3
import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from flask import Flask
import threading

db_lock = threading.Lock()

DB_PATH = "smoking.db"

# Configure rotating file logger
log_file = os.path.join(os.path.dirname(__file__), "smoking_app.log")
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = Flask(__name__)

def init_db():
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS smoking_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  active_count INTEGER,
                  violation_count INTEGER)''')
                  
    c.execute('''CREATE TABLE IF NOT EXISTS violations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  person_id INTEGER,
                  duration INTEGER,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

def add_log(active_count, violation_count):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO smoking_logs (timestamp, active_count, violation_count) VALUES (?, ?, ?)",
              (ts, active_count, violation_count))
    conn.commit()
    conn.close()
    return {"timestamp": ts, "active": active_count, "violation": violation_count}

def add_violation(person_id, duration, image_path):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    # Milisaniye hassasiyeti ekle - aynı saniyedeki ihlaller için
    import time
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
    c.execute("INSERT INTO violations (timestamp, person_id, duration, image_path) VALUES (?, ?, ?, ?)",
              (ts, person_id, duration, image_path))
    conn.commit()
    conn.close()
    return {"timestamp": ts, "person_id": person_id, "duration": duration, "image_path": image_path}

def get_logs(limit=50):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM smoking_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_violations(limit=50):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # Timestamp'e göre sırala - milisaniye hassasiyeti ile
    c.execute("SELECT * FROM violations ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_daily_person_stats():
    """Kişilerin o günkü toplam sürelerini getir"""
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # Bugünün tarihini al
    today = datetime.now().strftime("%Y-%m-%d")
    # Her kişi için bugünkü ihfal sürelerini topla
    query = """
        SELECT person_id, 
               SUM(duration) as total_duration,
               COUNT(*) as violation_count
        FROM violations 
        WHERE date(timestamp) = ?
        GROUP BY person_id
        ORDER BY total_duration DESC
    """
    c.execute(query, (today,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def delete_violation(violation_id):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    c.execute("SELECT image_path FROM violations WHERE id = ?", (violation_id,))
    row = c.fetchone()
    if row and row[0]:
        try:
            os.remove(row[0])
        except Exception:
            pass
    c.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
    conn.commit()
    conn.close()

def clear_violations():
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    c.execute("DELETE FROM violations")
    conn.commit()
    conn.close()

def delete_violations_by_person(person_id):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    c.execute("SELECT image_path FROM violations WHERE person_id = ?", (person_id,))
    rows = c.fetchall()
    for row in rows:
        if row[0]:
            try:
                os.remove(row[0])
            except Exception:
                pass
    c.execute("DELETE FROM violations WHERE person_id = ?", (person_id,))
    conn.commit()
    conn.close()


def get_hourly_logs():
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # Her saatin en son/maksimum aktif kişi ve ihlal değerlerini alıyoruz
    query = """
        SELECT strftime('%Y-%m-%d %H:00:00', timestamp) as hour_start,
               MAX(active_count) as active_total,
               MAX(violation_count) as violation_total
        FROM smoking_logs
        GROUP BY hour_start
        ORDER BY hour_start DESC
    """
    c.execute(query)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def reset_db():
    with db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    c.execute("DELETE FROM smoking_logs")
    c.execute("DELETE FROM violations")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
