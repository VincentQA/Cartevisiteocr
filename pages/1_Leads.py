import streamlit as st
import sqlite3
import pandas as pd

# Configuration de la page Streamlit
st.set_page_config(page_title="Le charte visite 🐱 - Voir les leads", layout="centered")
st.title("Le charte visite 🐱 - Voir les leads")

# Connexion à la base de données SQLite
conn = sqlite3.connect("leads.db", check_same_thread=False)
cursor = conn.cursor()

# Création de la table 'leads' si elle n'existe pas déjà
cursor.execute("""
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT
    )
""")
conn.commit()

# Fonction pour ajouter une colonne à une table si elle n'existe pas déjà
def add_column_if_missing(cursor, table, column, col_type):
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()

# Migration du schéma : ajout des colonnes manquantes dans la table 'leads'
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

# Bouton pour ajouter une ligne fictive
if st.button("Ajouter une ligne fictive"):
    dummy_data = (
        "Ceci est un OCR fictif",
        "Doe",
        "John",
        "0123456789",
        "john.doe@example.com",
        "Réponse fictive agent1",
        "Réponse fictive agent2",
        "Réponse fictive agent3",
        "Smart Talk",
        "Ceci est une note fictive"
    )
    cursor.execute(
        "INSERT INTO leads (ocr_text, nom, prenom, telephone, mail, agent1, agent2, agent3, qualification, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        dummy_data
    )
    conn.commit()
    st.success("Ligne fictive ajoutée.")

# Bouton pour réinitialiser la base de données (supprime toutes les lignes)
if st.button("Reset la base de données"):
    cursor.execute("DELETE FROM leads")
    conn.commit()
    st.success("La base de données a été réinitialisée.")

# Récupération et affichage des données dans un DataFrame
try:
    cursor.execute("""
        SELECT id, ocr_text, nom, prenom, telephone, mail, 
               agent1, agent2, agent3, qualification, note, timestamp 
        FROM leads 
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    
    if rows:
        # Récupération du nom des colonnes
        columns = [description[0] for description in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        st.dataframe(df)
    else:
        st.info("Aucun lead n'a été enregistré pour le moment.")
except Exception as e:
    st.error("Erreur lors de la récupération des leads : " + str(e))

# Fermeture de la connexion à la base de données
conn.close()
