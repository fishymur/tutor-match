import os
import json
import sqlite3
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("TM_DB", str(BASE_DIR / "tutormatch.db"))
SEED_FILE = os.environ.get("TM_TUTORS", str(BASE_DIR / "data" / "tutors.json"))

ROLES = {"student", "tutor", "hybrid"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name          TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('student','tutor','hybrid')),
    bio           TEXT    DEFAULT '',
    specialties   TEXT    DEFAULT '',   -- comma-separated subjects (tutor/hybrid); embedded for matching
    learning      TEXT    DEFAULT '',   -- comma-separated topics currently being learned (student/hybrid)
    education     TEXT    DEFAULT '',
    languages     TEXT    DEFAULT '',
    rate          REAL,                 -- hourly $ (tutor/hybrid)
    tz            INTEGER,              -- UTC offset in hours
    avail_start   INTEGER,              -- local availability window start hour (0-23)
    avail_end     INTEGER,              -- local availability window end hour (0-23)
    rating        REAL,                 -- NULL until a tutor has reviews
    created_at    TEXT NOT NULL
);
"""

EDITABLE = {"name", "role", "bio", "specialties", "learning", "education",
            "languages", "rate", "tz", "avail_start", "avail_end"}

PUBLIC = ("id", "name", "role", "bio", "specialties", "learning",
          "education", "languages", "rate", "tz", "avail_start",
          "avail_end", "rating")

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init(path=None, seed=True):
    global DB_PATH
    if path:
        DB_PATH = path
    conn = connect()
    conn.executescript(SCHEMA)
    conn.commit()
    if seed:
        _seed_from_json(conn)
    conn.close()

# Load the tutors in if empty
def _seed_from_json(conn):
    if conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"] > 0:
        return
    try:
        with open(SEED_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return

    from .auth import hash_password
    placeholder = hash_password(os.urandom(12).hex())  # seed accounts aren't meant to be logged into
    now = datetime.datetime.utcnow().isoformat()
    for t in data:
        email = f"{t['name'].lower().replace(' ', '')}@seed.tutormatch.local"
        h0, h1 = t["hours"]
        conn.execute(
            "INSERT OR IGNORE INTO users "
            "(email, password_hash, name, role, specialties, rate, tz, avail_start, avail_end, rating, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (email, placeholder, t["name"], "tutor", t["expertise"],
             t["rate"], t["tz"], h0, h1, t["rating"], now),
        )
    conn.commit()

def create_user(email, password_hash, name, role):
    conn = connect()
    now = datetime.datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT INTO users (email, password_hash, name, role, created_at) VALUES (?,?,?,?,?)",
        (email.lower().strip(), password_hash, name.strip(), role, now),
    )
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return uid

def get_user(uid):
    conn = connect()
    row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_email(email):
    conn = connect()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email.lower().strip(),)).fetchone()
    conn.close()
    return dict(row) if row else None

def update_profile(uid, fields):
    sets = {k: v for k, v in fields.items() if k in EDITABLE and v is not None}
    if not sets:
        return
    cols = ", ".join(f"{k}=?" for k in sets)
    conn = connect()
    conn.execute(f"UPDATE users SET {cols} WHERE id=?", (*sets.values(), uid))
    conn.commit()
    conn.close()

def public_view(user):
    return {k: user.get(k) for k in PUBLIC}

def list_matchable_tutors():
    conn = connect()
    rows = conn.execute(
        "SELECT id, name, specialties, rating, rate, tz, avail_start, avail_end FROM users "
        "WHERE role IN ('tutor','hybrid') "
        "AND specialties IS NOT NULL AND specialties != '' "
        "AND rate IS NOT NULL AND tz IS NOT NULL "
        "AND avail_start IS NOT NULL AND avail_end IS NOT NULL"
    ).fetchall()
    conn.close()
    return [{
        "id": r["id"], "name": r["name"], "expertise": r["specialties"],
        "rating": r["rating"], "rate": r["rate"], "tz": r["tz"],
        "hours": [r["avail_start"], r["avail_end"]],
    } for r in rows]