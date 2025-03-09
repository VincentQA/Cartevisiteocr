import streamlit as st
import os
import base64
import json
from mistralai import Mistral
from tavily import TavilyClient

# Récupération des clés API
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not MISTRAL_API_KEY or not TAVILY_API_KEY:
    st.error("Veuillez définir MISTRAL_API_KEY et TAVILY_API_KEY dans vos variables d'environnement.")
    st.stop()

# Initialisation des clients Mistral et Tavily
client_mistral = Mistral(api_key=MISTRAL_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

##############################################
# Configuration de l'assistant
##############################################

# Le prompt est modifié pour demander à l'assistant de renvoyer, en plus des informations extraites, 
# une clé "call_tavily_search" contenant la requête si une recherche complémentaire est nécessaire.
assistant_prompt_instruction = """
Vous êtes Chat IA, un assistant expert en analyse de cartes de visite.
Votre tâche est la suivante :
1. Extraire le nom, le prénom et le nom de l'entreprise à partir du texte OCR fourni.
2. Si des informations complémentaires sont nécessaires, ajoutez dans votre réponse une clé "call_tavily_search" avec la requête à effectuer.
Répondez uniquement sous forme d'un objet JSON contenant obligatoirement les clés "nom", "prenom", "entreprise" et "infos_en_ligne". 
Si une recherche complémentaire est nécessaire, incluez également la clé "call_tavily_search" avec la requête correspondante.
"""

##############################################
# Fonctions utilitaires
##############################################

def tavily_search(query):
    # Effectue une recherche en ligne via Tavily
    search_result = tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)
    return search_result

def extract_text_from_ocr_response(ocr_response):
    """
    Parcourt les pages de la réponse OCR et extrait le texte en supprimant les lignes contenant des images (commençant par "![").
    """
    extracted_text = ""
    if hasattr(ocr_response, "pages"):
        pages = ocr_response.pages
    elif isinstance(ocr_response, list):
        pages = ocr_response
    else:
        pages = []
    for page in pages:
        if hasattr(page, "markdown") and page.markdown:
            lines = page.markdown.split("\n")
            filtered_lines = [line.strip() for line in lines if not line.startswith("![")]
            if filtered_lines:
                extracted_text += "\n".join(filtered_lines) + "\n"
    return extracted_text.strip()

##############################################
# Interface Streamlit : Nouvelle Version
##############################################

st.set_page_config(page_title="Le charte visite 🐱", layout="centered")
st.title("Le charte visite 🐱")

# Organisation en colonnes pour afficher côte à côte la capture d'image et les autres inputs
col1, col2 = st.columns(2)
with col1:
    # Capture de l'image de la carte de visite
    image_file = st.camera_input("Prenez une photo des cartes de visite")
with col2:
    # Sélection du niveau de discussion
    niveau_discussion = st.selectbox(
        "Sélectionnez le niveau de discussion :",
        options=[
            "Smart Talk à creuser",
            "Incubation collective",
            "Incubation individuelle",
            "Renvoyer vers transformation numérique"
        ]
    )
    # Saisie d'une note complémentaire
    note_utilisateur = st.text_area("Ajoutez une note (facultatif) :", placeholder="Saisissez ici votre note...")

if image_file is not None:
    st.image(image_file, caption="Carte de visite capturée", use_column_width=True)
    st.write("Niveau de discussion choisi :", niveau_discussion)
    st.write("Note saisie :", note_utilisateur)
    
    # Conversion de l'image en base64 pour l'envoi à l'API OCR
    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"
    
    try:
        # Appel à l'API OCR de Mistral
        ocr_response = client_mistral.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "image_url", "image_url": image_data_uri}
        )
        st.subheader("Résultat brut de l'OCR :")
        st.write(ocr_response)
        
        # Extraction du texte exploitable
        ocr_text = extract_text_from_ocr_response(ocr_response)
        if not ocr_text:
            st.warning("Aucun texte exploitable extrait par l'OCR.")
        else:
            st.subheader("Texte OCR extrait :")
            st.text(ocr_text)
            
            # Ajout du contexte utilisateur (niveau et note)
            contexte_utilisateur = f"Niveau de discussion : {niveau_discussion}\nNote : {note_utilisateur}"
            
            # Préparation des messages pour l'assistant
            messages = [
                {"role": "system", "content": assistant_prompt_instruction},
                {"role": "user", "content": f"{contexte_utilisateur}\nVoici le texte OCR extrait :\n{ocr_text}"}
            ]
            
            # Appel initial à l'assistant (sans paramètres "functions")
            response = client_mistral.chat.complete(
                model="mistral-small-latest",
                messages=messages
            )
            
            # On tente de parser la réponse de l'assistant en JSON
            try:
                response_content = response.get("message", {}).get("content", "")
                response_json = json.loads(response_content)
            except Exception as e:
                st.error(f"Erreur de parsing JSON de la réponse : {e}")
                response_json = {}
            
            # Si l'assistant demande une recherche complémentaire via "call_tavily_search"
            if "call_tavily_search" in response_json:
                query = response_json["call_tavily_search"]
                search_output = tavily_search(query)
                # Ajout de la réponse de la fonction dans le contexte
                messages.append({
                    "role": "tool",
                    "name": "tavily_search",
                    "content": search_output
                })
                # Relance de l'assistant avec le contexte mis à jour
                final_response = client_mistral.chat.complete(
                    model="mistral-small-latest",
                    messages=messages
                )
            else:
                final_response = response
            
            # Extraction et affichage du message final
            final_output = final_response.get("message", {}).get("content", "")
            st.subheader("Résultat final de l'assistant :")
            try:
                parsed_json = json.loads(final_output)
                st.json(parsed_json)
            except Exception as e:
                st.text("Erreur lors du parsing JSON, voici le résultat brut :")
                st.text(final_output)
                
    except Exception as e:
        st.error(f"Erreur lors du traitement OCR ou de l'analyse par l'assistant : {e}")
