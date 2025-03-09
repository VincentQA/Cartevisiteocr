import streamlit as st
import os
import base64
import json
import time
from openai import OpenAI
from mistralai import Mistral
from tavily import TavilyClient

# Récupération des clés API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY or not MISTRAL_API_KEY or not TAVILY_API_KEY:
    st.error("Veuillez définir les variables OPENAI_API_KEY, MISTRAL_API_KEY et TAVILY_API_KEY dans votre environnement.")
    st.stop()

# Initialisation des clients
client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_mistral = Mistral(api_key=MISTRAL_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

###########################################
# Assistant 1 : Extraction & recherche    #
###########################################

assistant_prompt_instruction = """
Vous êtes Chat IA, un assistant expert en analyse de cartes de visite.
Votre tâche est la suivante :
1. Extraire le nom, le prénom et le nom de l'entreprise à partir du texte OCR fourni.
2. Utiliser la fonction tavily_search pour effectuer une recherche en ligne et fournir un maximum d'informations sur l'intervenant ainsi que son entreprise.
L'objectif est d'obtenir des informations clés sur l'intervenant et sa structure pour faciliter la prise de contact.
Tu cherches donc des informations sur l'entreprise ainsi que les derniers posts sur les réseaux sociaux.
Répondez uniquement sous forme de texte structuré avec des catégories pour chaque partie.
"""

assistant = client_openai.beta.assistants.create(
    instructions=assistant_prompt_instruction,
    model="gpt-4o",
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

###########################################
# Assistant 2 : Description des produits  #
###########################################

product_assistant_instruction = """
Vous êtes Chat IA, un expert en description de produits.
Votre tâche est de rédiger une description détaillée de nos produits à partir des informations sur l'entreprise ainsi que des résultats de qualification du lead et de la note associée.
Structurez votre réponse en sections claires en mettant en avant les points forts, l'utilité et l'innovation de nos produits.
Répondez sous forme d'un texte structuré.
"""
product_assistant = client_openai.beta.assistants.create(
    instructions=product_assistant_instruction,
    model="gpt-4o"
)
product_assistant_id = product_assistant.id

#####################################################
# Fonctions utilitaires pour assistants & Tavily     #
#####################################################

def tavily_search(query):
    # Effectue une recherche en ligne via Tavily
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
        text_val = ""
        for content_item in msg.content:
            if isinstance(content_item, dict):
                text_val += content_item.get("text", "")
            else:
                text_val += str(content_item)
        st.write(f"{role}: {text_val}")

def get_final_assistant_message(thread_id):
    """
    Récupère le dernier message de l'assistant dans un thread.
    """
    messages = client_openai.beta.threads.messages.list(thread_id=thread_id)
    final_msg = ""
    for msg in messages:
        if msg.role == "assistant":
            msg_text = ""
            for content_item in msg.content:
                if isinstance(content_item, dict):
                    msg_text += content_item.get("text", "")
                else:
                    msg_text += str(content_item)
            final_msg = msg_text
    return final_msg

def extract_text_from_ocr_response(ocr_response):
    """
    Parcourt les pages de la réponse OCR et extrait le texte en supprimant la ligne contenant l'image.
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

###########################################
# Interface Streamlit                     #
###########################################

st.set_page_config(page_title="Le charte visite 🐱", layout="centered")
st.title("Le charte visite 🐱")

# Capture de l'image via la caméra
image_file = st.camera_input("Prenez une photo des cartes de visite")

# Sélection de la qualification du lead et saisie d'une note
qualification = st.selectbox(
    "Qualification du leads",
    ["Smart Talk", "Incubation collective", "Incubation individuelle", "Transformation numérique"]
)

note = st.text_area("Ajouter une note", placeholder="Entrez votre note ici...")

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
            
            ##########################################
            # Appel du premier assistant (extraction)#
            ##########################################
            
            # Création d'un thread pour la conversation avec le premier assistant
            thread = client_openai.beta.threads.create()
            
            # Envoi du message utilisateur incluant la qualification et la note
            user_message = (
                f"Qualification: {qualification}\n"
                f"Note: {note}\n\n"
                f"Voici le texte OCR extrait :\n{ocr_text}\n\n"
                "Extrais les informations suivantes : nom, prenom, entreprise. "
                "Ensuite, effectue une recherche en ligne pour obtenir un maximum d'informations sur l'intervenant et son entreprise."
            )
            client_openai.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )
            
            # Création d'un run pour que l'assistant traite le message
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
            
            st.subheader("Résultat du premier assistant :")
            print_messages_from_thread(thread.id)
            
            # Récupération du message final du premier assistant
            entreprise_infos = get_final_assistant_message(thread.id)
            
            ###############################################
            # Appel du second assistant (description produit) #
            ###############################################
            
            # Création d'un nouveau thread pour le second assistant
            product_thread = client_openai.beta.threads.create()
            
            # Message utilisateur pour l'assistant produit
            user_message_product = (
                f"Voici les informations sur l'entreprise extraites précédemment :\n{entreprise_infos}\n\n"
                f"Qualification: {qualification}\n"
                f"Note: {note}\n\n"
                "En vous basant sur ces informations, rédigez une description détaillée de nos produits, "
                "en mettant en avant leurs points forts, leur utilité et leur innovation."
            )
            client_openai.beta.threads.messages.create(
                thread_id=product_thread.id,
                role="user",
                content=user_message_product
            )
            
            # Création d'un run pour que l'assistant produit traite le message
            run_product = client_openai.beta.threads.runs.create(
                thread_id=product_thread.id,
                assistant_id=product_assistant_id
            )
            run_product = wait_for_run_completion(product_thread.id, run_product.id)
            if run_product.status == 'failed':
                st.error(run_product.error)
            elif run_product.status == 'requires_action':
                run_product = submit_tool_outputs(product_thread.id, run_product.id, run_product.required_action.submit_tool_outputs.tool_calls)
                run_product = wait_for_run_completion(product_thread.id, run_product.id)
            
            st.subheader("Description des produits par le second assistant :")
            print_messages_from_thread(product_thread.id)
            
    except Exception as e:
        st.error(f"Erreur lors du traitement OCR ou de l'analyse par les assistants : {e}")
