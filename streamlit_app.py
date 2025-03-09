import streamlit as st
import os
import base64
import json
import time
from mistralai import Mistral
from tavily import TavilyClient

##############################################
# Fonction utilitaire pour l'exponential backoff
##############################################
def api_call_with_backoff(api_func, *args, max_attempts=5, initial_delay=1, **kwargs):
    delay = initial_delay
    for attempt in range(max_attempts):
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                st.warning(f"Rate limit atteint, nouvelle tentative dans {delay} secondes...")
                time.sleep(delay)
                delay *= 2  # Délai exponentiel
            else:
                raise
    raise Exception("Nombre maximal de tentatives dépassé à cause du rate limiting.")

##############################################
# Récupération des clés API et initialisation des clients
##############################################
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not MISTRAL_API_KEY or not TAVILY_API_KEY:
    st.error("Veuillez définir MISTRAL_API_KEY et TAVILY_API_KEY dans vos variables d'environnement.")
    st.stop()

client_mistral = Mistral(api_key=MISTRAL_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

##############################################
# Configuration de l'assistant
##############################################
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
    # Appel via backoff pour gérer les erreurs de rate limiting
    search_result = api_call_with_backoff(
        tavily_client.get_search_context,
        query,
        search_depth="advanced",
        max_tokens=8000
    )
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
# Interface Streamlit – Nouvelle Version
##############################################
st.set_page_config(page_title="Le charte visite 🐱", layout="centered")
st.title("Le charte visite 🐱")

# Organisation en colonnes : image à gauche, niveau & note à droite
col1, col2 = st.columns(2)
with col1:
    image_file = st.camera_input("Prenez une photo des cartes de visite")
with col2:
    niveau_discussion = st.selectbox(
        "Sélectionnez le niveau de discussion :",
        options=[
            "Smart Talk à creuser",
            "Incubation collective",
            "Incubation individuelle",
            "Renvoyer vers transformation numérique"
        ]
    )
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
        # Appel à l'API OCR avec backoff
        ocr_response = api_call_with_backoff(
            client_mistral.ocr.process,
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
            
            # Appel initial à l'assistant via backoff
            response = api_call_with_backoff(
                client_mistral.chat.complete,
                model="mistral-small-latest",
                messages=messages
            )
            
            # Accès via attributs à la réponse de l'assistant
            try:
                response_content = response.message.content
                response_json = json.loads(response_content)
            except Exception as e:
                st.error(f"Erreur de parsing JSON de la réponse : {e}")
                response_json = {}
            
            # Si l'assistant demande une recherche complémentaire via "call_tavily_search"
            if "call_tavily_search" in response_json:
                query = response_json["call_tavily_search"]
                search_output = tavily_search(query)
                messages.append({
                    "role": "tool",
                    "name": "tavily_search",
                    "content": search_output
                })
                # Relance de l'assistant avec le contexte mis à jour
                final_response = api_call_with_backoff(
                    client_mistral.chat.complete,
                    model="mistral-small-latest",
                    messages=messages
                )
            else:
                final_response = response
            
            # Extraction et affichage du résultat final
            final_output = final_response.message.content
            st.subheader("Résultat final de l'assistant :")
            try:
                parsed_json = json.loads(final_output)
                st.json(parsed_json)
            except Exception as e:
                st.text("Erreur lors du parsing JSON, voici le résultat brut :")
                st.text(final_output)
                
    except Exception as e:
        st.error(f"Erreur lors du traitement OCR ou de l'analyse par l'assistant : {e}")
