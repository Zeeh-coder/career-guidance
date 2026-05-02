import sqlite3

def init_db():
    conn = sqlite3.connect('career_guidance.db')
    cursor = conn.cursor()

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
    print("✓ Database ready!")

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