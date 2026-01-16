import sqlite3
from pathlib import Path
import pandas as pd
import hashlib

DB_PATH = Path("app/app.db")

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Updated to store password_hash instead of plain text
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL 
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        rating INTEGER NOT NULL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP, 
        UNIQUE(user_id, movie_id)
    )
    """)



    # Start custom movie IDs at 300,000 to prevent conflict with CSV or User IDs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS custom_movies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        genres TEXT,
        overview TEXT
    )
    """)
    # Set User ID sequence to start at 200,000
    cur.execute("INSERT OR IGNORE INTO sqlite_sequence (name, seq) VALUES ('users', 200000)")

    #Set Custom Movie ID sequence to start at 300,000
    cur.execute("INSERT OR IGNORE INTO sqlite_sequence (name, seq) VALUES ('custom_movies', 300000)")

    conn.commit()
    conn.close()

def create_user(username: str, password: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users(username, password_hash) VALUES(?,?)",
                    (username, _hash(password)))
        conn.commit()
        conn.close()
        return True, ""
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, str(e)

def auth_user(username: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    uid, pw_hash = row
    return int(uid) if pw_hash == _hash(password) else None

def set_rating(user_id: int, movie_id: int, rating: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ratings(user_id, movie_id, rating)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating, ts=CURRENT_TIMESTAMP
    """, (int(user_id), int(movie_id), int(rating)))
    conn.commit()
    conn.close()

def get_user_ratings(user_id: int) -> dict[int, int]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT movie_id, rating FROM ratings WHERE user_id=?", (int(user_id),))
    rows = cur.fetchall()
    conn.close()
    return {int(mid): int(r) for mid, r in rows}

def get_all_web_ratings_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()
    return df


def get_username(user_id: int) -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE id=?", (int(user_id),))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else "User"


def add_custom_movie_to_db(title, genres, overview):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # CHECK: Does this movie already exist?
    cur.execute("SELECT id FROM custom_movies WHERE title = ?", (title,))
    exists = cur.fetchone()

    if exists:
        conn.close()
        return exists[0]

    cur.execute("INSERT INTO custom_movies (title, genres, overview) VALUES (?, ?, ?)",
                (title, genres, overview))
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id
# ADDED NEW MOVUES---
def get_custom_movies_df():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT id as movieId, title, genres, overview FROM custom_movies", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df