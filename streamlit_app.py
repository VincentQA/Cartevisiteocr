import streamlit as st
import os
import base64
import json
import time
import re
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
# Fonction de nettoyage des réponses      #
###########################################

def clean_response(response):
    """
    Extrait le texte contenu dans un éventuel TextContentBlock, supprime tout tag HTML
    et remplace les séquences '\\n' par des retours à la ligne réels.
    """
    match = re.search(r'value="(.*?)"\)', response, re.DOTALL)
    if match:
        cleaned = match.group(1)
    else:
        cleaned = response
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = cleaned.replace("\\n", "\n")
    return cleaned.strip()

###########################################
# Assistant 1 : Extraction & recherche    #
###########################################

assistant_prompt_instruction = """
Vous êtes Chat IA, un expert en analyse de cartes de visite.
Votre tâche est la suivante :
1. Extraire le nom, le prénom et le nom de l'entreprise à partir du texte OCR fourni.
2. Compléter ces informations par une recherche en ligne (via la fonction tavily_search) pour obtenir des données complémentaires sur l'intervenant et son entreprise (ex. derniers posts sur les réseaux sociaux).
Répondez uniquement sous forme de texte structuré avec des catégories claires.
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
                        "description": "Par exemple : 'John Doe, PDG de Example Corp'."
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
Vous êtes Chat IA, expert en commerce et analyse de besoins.
Votre tâche est de réaliser un matching entre nos produits et les besoins du client, à partir des informations sur l'entreprise et des données du lead.
Voici nos produits :
- Incubation collective : 3 mois de cours collectif intensif pour préparer des projets innovants (marché, business plan, pitch deck, communication, etc.).
- Incubation individuelle : Pour projets matures, accompagnement individuel avec un comité pour préparer des dossiers de financement.
- Transformation numérique : Accompagnement digital (site web, réseaux sociaux, ads, etc.), proposé par nos collègues de la CCI.
Répondez sous forme d'un texte structuré en mettant en avant les points forts et l'utilité de nos offres.
"""
product_assistant = client_openai.beta.assistants.create(
    instructions=product_assistant_instruction,
    model="gpt-4o"
)
product_assistant_id = product_assistant.id

###########################################
# Assistant 3 : Rédaction du mail          #
###########################################

email_assistant_instruction = """
Vous êtes Chat IA, expert en rédaction de mails de relance et assistant de Rach de Quai Alpha.
Vos mails commencent toujours par "Bonjour [prénom]" et se terminent par "Cordialement Rach Startup manager et Program Manager à Quai Alpha".
Votre tâche est de rédiger un mail de relance percutant pour convertir le lead, en tenant compte :
- des informations extraites (Assistant 1),
- du matching de notre offre (Assistant 2),
- de la qualification et des notes du lead.
Veillez à intégrer les notes de l'utilisateur pour instaurer une relation de proximité.
Répondez sous forme d'un texte structuré (salutation, introduction, corps, conclusion).
"""
email_assistant = client_openai.beta.assistants.create(
    instructions=email_assistant_instruction,
    model="gpt-4o"
)
email_assistant_id = email_assistant.id

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

def get_final_assistant_message(thread_id):
    """Récupère le dernier message de l'assistant dans un thread."""
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
    return final_msg.strip()

def extract_text_from_ocr_response(ocr_response):
    """Extrait le texte de la réponse OCR en ignorant les balises image."""
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

# Affichage de la zone de capture via caméra
st.subheader("Capture de la carte de visite")
image_file = st.camera_input("Prenez une photo des cartes de visite")

# Espace / Séparation avec possibilité d'upload
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>OU</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Uploader la carte", type=["jpg", "jpeg", "png"])

# Forcer l'utilisateur à choisir la qualification et saisir une note
qualification = st.selectbox(
    "Qualification du lead",
    ["Smart Talk", "Incubation collective", "Incubation individuelle", "Transformation numérique"]
)
note = st.text_area("Ajouter une note", placeholder="Entrez votre note ici...")

if note.strip() == "":
    st.error("Veuillez saisir une note avant de continuer.")
    st.stop()

# Récupération de l'image : soit via la caméra, soit via l'upload
image_data_uri = None
if image_file is not None:
    st.image(image_file, caption="Carte de visite capturée", use_column_width=True)
    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"
elif uploaded_file is not None:
    st.image(uploaded_file, caption="Carte uploadée", use_column_width=True)
    image_bytes = uploaded_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"
else:
    st.info("Veuillez capturer ou uploader une photo de la carte.")

if image_data_uri is not None:
    try:
        # Appel à l'API OCR de Mistral et affichage du texte extrait
        ocr_response = client_mistral.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "image_url", "image_url": image_data_uri}
        )
        ocr_text = extract_text_from_ocr_response(ocr_response)
        if not ocr_text:
            st.warning("Aucun texte exploitable n'a été extrait.")
        else:
            st.subheader("Texte OCR extrait :")
            st.text(ocr_text)

            ##################################################
            # Assistant 1 : Extraction & recherche           #
            ##################################################
            thread1 = client_openai.beta.threads.create()
            user_message_agent1 = (
                f"Données extraites de la carte :\n"
                f"Qualification : {qualification}\n"
                f"Note : {note}\n"
                f"Texte : {ocr_text}\n\n"
                "Veuillez extraire les informations clés (nom, prénom, entreprise) "
                "et compléter par une recherche en ligne pour obtenir des informations complémentaires."
            )
            client_openai.beta.threads.messages.create(
                thread_id=thread1.id,
                role="user",
                content=user_message_agent1
            )
            run1 = client_openai.beta.threads.runs.create(
                thread_id=thread1.id,
                assistant_id=assistant_id
            )
            run1 = wait_for_run_completion(thread1.id, run1.id)
            if run1.status == 'requires_action':
                run1 = submit_tool_outputs(thread1.id, run1.id, run1.required_action.submit_tool_outputs.tool_calls)
                run1 = wait_for_run_completion(thread1.id, run1.id)
            response_agent1 = get_final_assistant_message(thread1.id)
            cleaned_response_agent1 = clean_response(response_agent1)
            st.subheader("Réponse agent 1 :")
            st.markdown(cleaned_response_agent1)

            ##################################################
            # Assistant 2 : Description des produits         #
            ##################################################
            thread2 = client_openai.beta.threads.create()
            user_message_agent2 = (
                f"Informations sur l'entreprise extraites :\n{cleaned_response_agent1}\n\n"
                f"Qualification : {qualification}\n"
                f"Note : {note}\n\n"
                "Veuillez rédiger un matching entre nos produits et les besoins du client. "
                "Présentez clairement les avantages et l'utilité de nos offres (Incubation collective, Incubation individuelle, Transformation numérique)."
            )
            client_openai.beta.threads.messages.create(
                thread_id=thread2.id,
                role="user",
                content=user_message_agent2
            )
            run2 = client_openai.beta.threads.runs.create(
                thread_id=thread2.id,
                assistant_id=product_assistant_id
            )
            run2 = wait_for_run_completion(thread2.id, run2.id)
            if run2.status == 'requires_action':
                run2 = submit_tool_outputs(thread2.id, run2.id, run2.required_action.submit_tool_outputs.tool_calls)
                run2 = wait_for_run_completion(thread2.id, run2.id)
            response_agent2 = get_final_assistant_message(thread2.id)
            cleaned_response_agent2 = clean_response(response_agent2)
            st.subheader("Réponse agent 2 :")
            st.markdown(cleaned_response_agent2)

            ##################################################
            # Assistant 3 : Rédaction du mail                #
            ##################################################
            thread3 = client_openai.beta.threads.create()
            user_message_agent3 = (
                f"Informations sur l'intervenant et son entreprise :\n{cleaned_response_agent1}\n\n"
                f"Matching de notre offre :\n{cleaned_response_agent2}\n\n"
                f"Qualification : {qualification}\n"
                f"Note : {note}\n\n"
                "Veuillez rédiger un mail de relance percutant pour convertir ce lead. "
                "Le mail doit commencer par 'Bonjour [prénom]' et se terminer par 'Cordialement Rach Startup manager et Program Manager à Quai Alpha'."
            )
            client_openai.beta.threads.messages.create(
                thread_id=thread3.id,
                role="user",
                content=user_message_agent3
            )
            run3 = client_openai.beta.threads.runs.create(
                thread_id=thread3.id,
                assistant_id=email_assistant_id
            )
            run3 = wait_for_run_completion(thread3.id, run3.id)
            if run3.status == 'requires_action':
                run3 = submit_tool_outputs(thread3.id, run3.id, run3.required_action.submit_tool_outputs.tool_calls)
                run3 = wait_for_run_completion(thread3.id, run3.id)
            response_agent3 = get_final_assistant_message(thread3.id)
            cleaned_response_agent3 = clean_response(response_agent3)
            st.subheader("Réponse agent 3 :")
            st.markdown(cleaned_response_agent3)

    except Exception as e:
        st.error(f"Erreur lors du traitement OCR ou de l'analyse par les assistants : {e}")
