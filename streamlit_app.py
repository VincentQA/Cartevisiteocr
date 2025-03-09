import streamlit as st
import os
import base64
import json
import time
from mistralai import Mistral
from tavily import TavilyClient

##############################################
# Récupération et initialisation des clients #
##############################################

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not MISTRAL_API_KEY or not TAVILY_API_KEY:
    st.error("Veuillez définir MISTRAL_API_KEY et TAVILY_API_KEY dans vos variables d'environnement.")
    st.stop()

client_mistral = Mistral(api_key=MISTRAL_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

##############################################
# Configuration de l'assistant Mistral avec function calling
##############################################

assistant_prompt_instruction = """
Vous êtes Chat IA, un assistant expert en analyse de cartes de visite.
Votre tâche est la suivante:
1. Extraire le nom, le prénom et le nom de l'entreprise à partir du texte OCR fourni.
2. Utiliser la fonction tavily_search pour effectuer une recherche en ligne et fournir un maximum d'informations sur l'intervenant et son entreprise.
Répondez uniquement sous forme d'un objet JSON avec les clés "nom", "prenom", "entreprise" et "infos_en_ligne".
"""

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

assistant = client_mistral.chat.assistants.create(
    instructions=assistant_prompt_instruction,
    model="mistral-small-latest",
    functions=[tavily_search_function]
)
assistant_id = assistant.id

##############################################
# Fonctions utilitaires
##############################################

def tavily_search(query):
    return tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)

def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client_mistral.chat.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status in ['completed', 'failed', 'requires_action']:
            return run

def submit_tool_outputs(thread_id, run_id, tools_to_call):
    tool_output_array = []
    for tool in tools_to_call:
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments
        if function_name == "tavily_search":
            query = json.loads(function_args)["query"]
            output = tavily_search(query)
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})
    return client_mistral.chat.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )

def get_final_assistant_message(thread_id):
    messages = client_mistral.chat.messages.list(thread_id=thread_id)
    assistant_messages = [msg for msg in messages if msg.role == "assistant"]
    if assistant_messages:
        final_msg = assistant_messages[-1]
        text_val = ""
        for content_item in final_msg.content:
            if isinstance(content_item, dict):
                text_val += content_item.get("text", "")
            else:
                text_val += str(content_item)
        return text_val.strip()
    return None

def extract_text_from_ocr_response(ocr_response):
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

def format_assistant_output(raw_output):
    prefix = "json\n"
    cleaned = raw_output.strip()
    if cleaned.lower().startswith(prefix):
        cleaned = cleaned[len(prefix):].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None

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
            
            # Création d'un thread pour la conversation avec l'assistant Mistral
            thread = client_mistral.chat.threads.create()
            
            # Envoi du message utilisateur contenant le texte OCR et les instructions
            user_message = (
                f"Voici le texte OCR extrait :\n{ocr_text}\n\n"
                "Extrais les informations suivantes : nom, prenom, entreprise. "
                "Ensuite, effectue une recherche en ligne pour obtenir un maximum d'informations sur l'intervenant et son entreprise."
            )
            client_mistral.chat.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )
            
            # Lancer un run pour que l'assistant traite le message
            run = client_mistral.chat.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            run = wait_for_run_completion(thread.id, run.id)
            if run.status == 'failed':
                st.error(run.error)
            elif run.status == 'requires_action':
                run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
                run = wait_for_run_completion(thread.id, run.id)
            
            raw_output = get_final_assistant_message(thread.id)
            st.subheader("Résultat final de l'assistant :")
            if raw_output:
                parsed_output = format_assistant_output(raw_output)
                if parsed_output:
                    st.json(parsed_output)
                else:
                    st.text("Erreur lors du parsing JSON, voici le résultat brut :")
                    st.text(raw_output)
            else:
                st.warning("Aucun message de l'assistant n'a été trouvé.")
            
            ##############################################
            # Section Sujets abordés et commentaires
            ##############################################
            st.markdown("---")
            st.subheader("Sujets abordés")
            sujet1 = st.checkbox("Rencontre courtoise mais potentiel de faire plus ensemble")
            sujet2 = st.checkbox("Relancer sur le programme incubation collective")
            sujet3 = st.checkbox("Relancer sur le programme d'incubation individuelle")
            sujet4 = st.checkbox("Mettre en relation avec la transformation numérique car on ne peut pas l'accompagner")
            commentaires = st.text_area("Commentaires additionnels", placeholder="Ajouter des commentaires ici...")
            
            if st.button("Valider les sujets"):
                sujets_selectionnes = []
                if sujet1:
                    sujets_selectionnes.append("Rencontre courtoise mais potentiel de faire plus ensemble")
                if sujet2:
                    sujets_selectionnes.append("Relancer sur le programme incubation collective")
                if sujet3:
                    sujets_selectionnes.append("Relancer sur le programme d'incubation individuelle")
                if sujet4:
                    sujets_selectionnes.append("Mettre en relation avec la transformation numérique car on ne peut pas l'accompagner")
                st.subheader("Sujets sélectionnés")
                st.write(sujets_selectionnes)
                st.subheader("Commentaires additionnels")
                st.write(commentaires)
                
    except Exception as e:
        st.error(f"Erreur lors du traitement OCR ou de l'analyse par l'assistant Mistral : {e}")
