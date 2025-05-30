import sqlite3

DB_PATH = "signups.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signups (
                name TEXT,
                email TEXT PRIMARY KEY,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_signup(name, email):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR IGNORE INTO signups (name, email) VALUES (?, ?)", (name, email))
        conn.commit()

def get_signups():
    conn = sqlite3.connect("signups.db")
    c = conn.cursor()
    c.execute("SELECT * FROM signups")
    data = c.fetchall()
    conn.close()
    return data
