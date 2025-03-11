import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Le charte visite 🐱 - Voir les leads", layout="centered")
st.title("Le charte visite 🐱 - Voir les leads")

@st.cache_resource
def get_db_connection():
    # On active check_same_thread=False pour éviter des problèmes de threads
    return sqlite3.connect("leads.db", check_same_thread=False)

conn = get_db_connection()
cursor = conn.cursor()

# Création de la table si elle n'existe pas
try:
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ocr_text TEXT,
        nom TEXT,
        prenom TEXT,
        telephone TEXT,
        mail TEXT,
        agent1 TEXT,
        agent2 TEXT,
        agent3 TEXT,
        qualification TEXT,
        note TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
except Exception as e:
    st.error("Erreur lors de la création de la table : " + str(e))

# Récupération des données
try:
    cursor.execute("SELECT id, ocr_text, nom, prenom, telephone, mail, agent1, agent2, agent3, qualification, note, timestamp FROM leads ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    if rows:
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        st.dataframe(df)
    else:
        st.info("Aucun lead n'a été enregistré pour le moment.")
except Exception as e:
    st.error("Erreur lors de la récupération des leads : " + str(e))

conn.close()
