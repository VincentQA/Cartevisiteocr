import streamlit as st
import os
import base64
import json
import time
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
# Configuration de l'assistant Mistral avec fonction calling
##############################################

assistant_prompt_instruction = """
Vous êtes Chat IA, un assistant expert en analyse de cartes de visite.
Votre tâche est la suivante:
1. Extraire le nom, le prénom et le nom de l'entreprise à partir du texte OCR fourni.
2. Utiliser la fonction tavily_search pour effectuer une recherche en ligne et fournir un maximum d'informations sur l'intervenant et son entreprise.
Répondez uniquement sous forme d'un objet JSON avec les clés "nom", "prenom", "entreprise" et "infos_en_ligne".
"""

# Définition de la fonction tavily_search pour le function calling
tavily_search_function = {
    "name": "tavily_search",
    "description": "Recherche en ligne pour obtenir des informations sur une personne ou une entreprise.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "La requête de recherche, par exemple 'John Doe, PDG de Example Corp'."
            }
        },
        "required": ["query"]
    }
}

##############################################
# Fonctions utilitaires
##############################################

def tavily_search(query):
    # Effectue une recherche en ligne via Tavily
    search_result = tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)
    return search_result

def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client_mistral.chat.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status in ['completed', 'failed', 'requires_action']:
            return run

def extract_text_from_ocr_response(ocr_response):
    """
    Parcourt les pages de la réponse OCR et extrait le texte en supprimant les lignes contenant l'image (commençant par "![").
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
# Interface Streamlit
##############################################

st.set_page_config(page_title="Le charte visite 🐱", layout="centered")
st.title("Le charte visite 🐱")

# Capture de l'image via la caméra
image_file = st.camera_input("Prenez une photo des cartes de visite")

if image_file is not None:
    st.image(image_file, caption="Carte de visite capturée", use_column_width=True)
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
            
            # Préparation des messages pour l'agent Mistral
            messages = [
                {"role": "system", "content": assistant_prompt_instruction},
                {"role": "user", "content": f"Voici le texte OCR extrait :\n{ocr_text}\nExtrais les informations demandées et, si nécessaire, appelez la fonction tavily_search pour obtenir des infos en ligne."}
            ]
            
            # Appel initial à l'agent Mistral avec function calling
            response = client_mistral.chat.complete(
                model="mistral-small-latest",
                messages=messages,
                functions=[tavily_search_function],
                function_call="auto"
            )
            
            # Si l'agent a appelé la fonction tavily_search, on récupère l'appel et on l'exécute
            if response.get("function_call"):
                func_call = response["function_call"]
                if func_call["name"] == "tavily_search":
                    try:
                        args = json.loads(func_call["arguments"])
                        query = args["query"]
                        search_output = tavily_search(query)
                        # Ajout de la réponse de l'outil dans la conversation
                        messages.append({
                            "role": "tool",
                            "name": "tavily_search",
                            "content": search_output
                        })
                        # Relance de l'agent avec le contexte mis à jour
                        final_response = client_mistral.chat.complete(
                            model="mistral-small-latest",
                            messages=messages
                        )
                    except Exception as e:
                        final_response = {"error": str(e)}
                else:
                    final_response = response
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
        st.error(f"Erreur lors du traitement OCR ou de l'analyse par l'assistant Mistral : {e}")
