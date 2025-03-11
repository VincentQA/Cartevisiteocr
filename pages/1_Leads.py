import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Le charte visite 🐱 - Voir les leads", layout="centered")
st.title("Le charte visite 🐱 - Voir les leads")

# Connexion à la base de données
conn = sqlite3.connect("leads.db")
cursor = conn.cursor()
cursor.execute("SELECT id, ocr_text, agent1, agent2, agent3, qualification, note, timestamp FROM leads ORDER BY timestamp DESC")
rows = cursor.fetchall()

if rows:
    columns = [description[0] for description in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    st.dataframe(df)
else:
    st.info("Aucun lead n'a été enregistré pour le moment.")

conn.close()
