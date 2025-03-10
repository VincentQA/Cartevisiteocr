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
Vous êtes Chat IA, un expert commerce et en analyse de besoins.
Ta tâche est de rédiger un matching de proposition de nos produits avec les besoins du clients à partir des informations sur l'entreprise ainsi que des résultats de qualification du lead et de la note associée.
Structurez votre réponse en sections claires en mettant en avant les points forts, l'utilité de nos produits.
Voici nos produits : 
Incubation collective : On propose 3 mois de cours collectif intensif afin de préparer les jeunes projets innovants à se confronter au marché, des besoins du marché, au business plan, au pitch deck tout en passant par la communicationt tout y passe
Incubation individuelle : Uniquement à destination des projets matures, proposer de passer par un comité pour rentrer directement en incubation individuelle pour préparer les potentiels dossier de financement mais surtout pour profiter d'un accompagnement individuel et spécialisé avec l'un de nos 3 chargés d'affaires
Transformation numérique : Cette section est chargée par nos collègues de la CCI, proposer lorsque l'utilisateur précise dans la qualification du leads. Ils sont spécialisés dans l'accompagnement digital des entreprises vosgiennes, du site web, gestion des réseaux sociaux,ads, tout y passe.
Répondez sous forme d'un texte structuré.
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
Vous êtes Chat IA, un expert en rédaction de mails de relance assistant de Rach de Quai Alpha.
Comme toujours tes mails par bonjour "prénom"
Et fini toujours tes mails par : Cordialement Rach Startup manager et Program Manager à Quai Alpha

Ta tâche est de rédiger le meilleur mail de relance possible afin de convertir ce lead.
Pour ce faire, tenez compte des informations extraites par le premier assistant sur l'intervenant et son entreprise,
du matching de l'offre par le second assistant, ainsi que de la qualification du lead et des notes de l'utilisateur.
Ta priorité est de bien consulter les notes de l'utilisateur afin de donner un sentiment de proximité
Répondez sous forme d'un texte structuré comprenant une salutation, une introduction, le corps du mail et une conclusion.
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
            
            ##################################################
            # Appel du premier assistant (extraction & recherche)
            ##################################################
            
            thread = client_openai.beta.threads.create()
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
            
            ##################################################
            # Appel du second assistant (description des produits)
            ##################################################
            
            product_thread = client_openai.beta.threads.create()
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
            
            # Récupération du message final du second assistant
            product_description = get_final_assistant_message(product_thread.id)
            
            ##################################################
            # Appel du troisième assistant (rédaction du mail)
            ##################################################
            
            email_thread = client_openai.beta.threads.create()
            user_message_email = (
                f"Informations de l'intervenant et de son entreprise (premier assistant) :\n{entreprise_infos}\n\n"
                f"Description de notre offre (second assistant) :\n{product_description}\n\n"
                f"Qualification du lead: {qualification}\n"
                f"Note de l'utilisateur: {note}\n\n"
                "En vous basant sur ces informations, rédigez le meilleur mail de relance possible pour convertir ce lead. "
                "Incluez une salutation, une introduction, le corps du mail et une conclusion."
            )
            client_openai.beta.threads.messages.create(
                thread_id=email_thread.id,
                role="user",
                content=user_message_email
            )
            
            run_email = client_openai.beta.threads.runs.create(
                thread_id=email_thread.id,
                assistant_id=email_assistant_id
            )
            run_email = wait_for_run_completion(email_thread.id, run_email.id)
            if run_email.status == 'failed':
                st.error(run_email.error)
            elif run_email.status == 'requires_action':
                run_email = submit_tool_outputs(email_thread.id, run_email.id, run_email.required_action.submit_tool_outputs.tool_calls)
                run_email = wait_for_run_completion(email_thread.id, run_email.id)
            
            st.subheader("Mail de relance généré par le troisième assistant :")
            print_messages_from_thread(email_thread.id)
            
    except Exception as e:
        st.error(f"Erreur lors du traitement OCR ou de l'analyse par les assistants : {e}")
