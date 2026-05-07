import sqlite3
import hashlib

def init_db():
    conn = sqlite3.connect('career_guidance.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Consent table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            agreed INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nqf_level INTEGER,
            duration INTEGER,
            skills_count INTEGER,
            field TEXT,
            institution TEXT,
            prediction TEXT,
            confidence TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("Database ready!")

def register_user(name, email, password):
    conn = sqlite3.connect('career_guidance.db')
    cursor = conn.cursor()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', (name, email, hashed))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False

def login_user(email, password):
    conn = sqlite3.connect('career_guidance.db')
    cursor = conn.cursor()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('SELECT * FROM users WHERE email=? AND password=?', (email, hashed))
    user = cursor.fetchone()
    conn.close()
    return user

def save_consent(name):
    conn = sqlite3.connect('career_guidance.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO consent (name, agreed) VALUES (?, ?)', (name, 1))
    conn.commit()
    conn.close()

def save_prediction(nqf_level, duration, skills_count, field, institution, prediction, confidence):
    conn = sqlite3.connect('career_guidance.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions
        (nqf_level, duration, skills_count, field, institution, prediction, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (nqf_level, duration, skills_count, field, institution, prediction, confidence))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect('career_guidance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows