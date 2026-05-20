import sqlite3
import os
from datetime import datetime

DB_PATH = "smoking.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO smoking_logs (timestamp, active_count, violation_count) VALUES (?, ?, ?)",
              (ts, active_count, violation_count))
    conn.commit()
    conn.close()
    return {"timestamp": ts, "active": active_count, "violation": violation_count}

def add_violation(person_id, duration, image_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO violations (timestamp, person_id, duration, image_path) VALUES (?, ?, ?, ?)",
              (ts, person_id, duration, image_path))
    conn.commit()
    conn.close()
    return {"timestamp": ts, "person_id": person_id, "duration": duration, "image_path": image_path}

def get_logs(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM smoking_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_violations(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM violations ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_hourly_logs():
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM smoking_logs")
    c.execute("DELETE FROM violations")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
