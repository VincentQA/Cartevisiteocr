import streamlit as st
import os
import base64
import json
import time
from openai import OpenAI
from mistralai import Mistral
from tavily import TavilyClient

# Récupération des clés API depuis les variables d'environnement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY or not MISTRAL_API_KEY or not TAVILY_API_KEY:
    st.error("Vérifiez que OPENAI_API_KEY, MISTRAL_API_KEY et TAVILY_API_KEY sont définies dans vos variables d'environnement.")
    st.stop()

# Initialisation des clients
client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_mistral = Mistral(api_key=MISTRAL_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

########################################
# Création de l'assistant avec outils  #
########################################

assistant_prompt_instruction = """
Vous êtes Chat IA, un assistant expert en analyse de cartes de visite.
Votre tâche est la suivante :
1. Extraire le nom, le prénom et le nom de l'entreprise à partir du texte OCR fourni.
2. Utiliser la fonction tavily_search pour effectuer une recherche en ligne et fournir un maximum d'informations sur l'intervenant ainsi que son entreprise.
Répondez uniquement sous forme d'un objet JSON avec les clés "nom", "prenom", "entreprise" et "infos_en_ligne".
"""

assistant = client_openai.beta.assistants.create(
    instructions=assistant_prompt_instruction,
    model="gpt-4-1106-preview",
    tools=[{
        "type": "function",
        "function": {
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
    }]
)
assistant_id = assistant.id

##############################################
# Fonctions utilitaires pour l'assistant   #
##############################################

def tavily_search(query):
    # Effectue une recherche avec Tavily
    search_result = tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)
    return search_result

def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client_openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
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
    return client_openai.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )

def print_messages_from_thread(thread_id):
    messages = client_openai.beta.threads.messages.list(thread_id=thread_id)
    for msg in messages:
        role = msg.role
        text_val = "".join([c.get("text", "") for c in msg.content])
        st.write(f"{role}: {text_val}")

##############################################
# Fonction d'extraction du texte OCR         #
##############################################

def extract_text_from_ocr_response(ocr_response):
    """
    Parcourt les pages de la réponse OCR et extrait le texte en supprimant la ligne image.
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
# Interface Streamlit                        #
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
            
            # Création d'un thread pour communiquer avec l'assistant
            thread = client_openai.beta.threads.create()
            
            # Envoi d'un message incluant le texte OCR et les instructions
            user_message = (
                f"Voici le texte OCR extrait :\n{ocr_text}\n\n"
                "Extrais les informations suivantes : nom, prenom, entreprise. "
                "Ensuite, effectue une recherche en ligne pour obtenir un maximum d'informations sur l'intervenant et son entreprise."
            )
            client_openai.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )
            
            # Lancer une exécution (run) pour que l'assistant traite le message
            run = client_openai.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            run = wait_for_run_completion(thread.id, run.id)
            if run.status == 'failed':
                st.error(run.error)
            elif run.status == 'requires_action':
                run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
                run = wait_for_run_completion(thread.id, run.id)
            
            st.subheader("Résultat final de l'assistant :")
            print_messages_from_thread(thread.id)
            
    except Exception as e:
        st.error(f"Erreur lors du traitement OCR ou de l'analyse par l'assistant : {e}")


