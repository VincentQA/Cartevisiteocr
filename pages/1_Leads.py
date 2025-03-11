import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Le charte visite 🐱 - Voir les leads", layout="centered")
st.title("Le charte visite 🐱 - Voir les leads")

# Connexion à la base de données et création de la table si nécessaire
conn = sqlite3.connect("leads.db")
cursor = conn.cursor()
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

cursor.execute("SELECT id, ocr_text, nom, prenom, telephone, mail, agent1, agent2, agent3, qualification, note, timestamp FROM leads ORDER BY timestamp DESC")
rows = cursor.fetchall()

if rows:
    columns = [description[0] for description in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    st.dataframe(df)
else:
    st.info("Aucun lead n'a été enregistré pour le moment.")

conn.close()
