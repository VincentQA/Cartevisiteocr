import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Le charte visite 🐱 - Voir les leads", layout="centered")
st.title("Le charte visite 🐱 - Voir les leads")

# Connexion à la base de données
conn = sqlite3.connect("leads.db", check_same_thread=False)
cursor = conn.cursor()

# Fonction pour ajouter une colonne si elle n'existe pas
def add_column_if_missing(cursor, table, column, col_type):
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()

# Migration du schéma : ajouter les colonnes manquantes
add_column_if_missing(cursor, "leads", "ocr_text", "TEXT")
add_column_if_missing(cursor, "leads", "nom", "TEXT")
add_column_if_missing(cursor, "leads", "prenom", "TEXT")
add_column_if_missing(cursor, "leads", "telephone", "TEXT")
add_column_if_missing(cursor, "leads", "mail", "TEXT")
add_column_if_missing(cursor, "leads", "agent1", "TEXT")
add_column_if_missing(cursor, "leads", "agent2", "TEXT")
add_column_if_missing(cursor, "leads", "agent3", "TEXT")
add_column_if_missing(cursor, "leads", "qualification", "TEXT")
add_column_if_missing(cursor, "leads", "note", "TEXT")
add_column_if_missing(cursor, "leads", "timestamp", "DATETIME DEFAULT CURRENT_TIMESTAMP")

# Récupération des données
try:
    cursor.execute("""
        SELECT id, ocr_text, nom, prenom, telephone, mail, 
               agent1, agent2, agent3, qualification, note, timestamp 
        FROM leads ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    
    if rows:
        columns = [description[0] for description in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        st.dataframe(df)
    else:
        st.info("Aucun lead n'a été enregistré pour le moment.")
except Exception as e:
    st.error("Erreur lors de la récupération des leads : " + str(e))

conn.close()
