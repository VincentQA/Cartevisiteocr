import streamlit as st
import os
import base64
import json
import time
import re
import sqlite3
import pandas as pd
from openai import OpenAI
from mistralai import Mistral
from tavily import TavilyClient

##############################
# Configuration de la page   #
##############################
st.set_page_config(page_title="Charte de visite 🐱", layout="centered")
st.title("Charte de visite 🐱")

##############################
# Clés API & initialisation  #
##############################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY or not MISTRAL_API_KEY or not TAVILY_API_KEY:
    st.error("Veuillez définir les variables OPENAI_API_KEY, MISTRAL_API_KEY et TAVILY_API_KEY dans votre environnement.")
    st.stop()

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_mistral = Mistral(api_key=MISTRAL_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

##############################
# Connexion à la base SQLite  #
##############################
conn = sqlite3.connect("leads.db", check_same_thread=False)
cursor = conn.cursor()
# On suppose que la création de la table est gérée sur la page "Voir les leads"

##############################
# Fonctions utilitaires      #
##############################
def clean_response(response):
    """Nettoie la réponse en supprimant les tags HTML et convertit '\\n' en retours à la ligne."""
    match = re.search(r'value="(.*?)"\)', response, re.DOTALL)
    cleaned = match.group(1) if match else response
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    return cleaned.replace("\\n", "\n").strip()

def extract_text_from_ocr_response(ocr_response):
    """Extrait le texte OCR en ignorant les balises image."""
    extracted_text = ""
    pages = ocr_response.pages if hasattr(ocr_response, "pages") else (ocr_response if isinstance(ocr_response, list) else [])
    for page in pages:
        if hasattr(page, "markdown") and page.markdown:
            lines = page.markdown.split("\n")
            filtered = [line.strip() for line in lines if not line.startswith("![")]
            if filtered:
                extracted_text += "\n".join(filtered) + "\n"
    return extracted_text.strip()

def tavily_search(query):
    """Effectue une recherche en ligne via Tavily."""
    return tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)

def wait_for_run_completion(thread_id, run_id):
    """Attend la fin d'un run d'assistant."""
    while True:
        time.sleep(1)
        run = client_openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status in ['completed', 'failed', 'requires_action']:
            return run

def submit_tool_outputs(thread_id, run_id, tools_to_call):
    """Soumet les sorties d'outils si nécessaire."""
    tool_output_array = []
    for tool in tools_to_call:
        if tool.function.name == "tavily_search":
            query = json.loads(tool.function.arguments)["query"]
            output = tavily_search(query)
            tool_output_array.append({"tool_call_id": tool.id, "output": output})
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
            for content in msg.content:
                final_msg += content.get("text", "") if isinstance(content, dict) else str(content)
    return final_msg.strip()

def parse_agent1_response(text):
    """
    Extrait Nom, Prénom, Téléphone et Mail à partir de la réponse de l'assistant 1.
    La réponse doit contenir des lignes telles que :
      Nom: Doe
      Prénom: John
      Téléphone: 0123456789
      Mail: john.doe@example.com
    """
    data = {"nom": "", "prenom": "", "telephone": "", "mail": ""}
    nom = re.search(r"Nom\s*:\s*(.+)", text)
    prenom = re.search(r"Pr[ée]nom\s*:\s*(.+)", text)
    tel = re.search(r"T[eé]l[eé]phone?\s*:\s*(.+)", text, re.IGNORECASE)
    mail = re.search(r"Mail\s*:\s*(.+)", text, re.IGNORECASE)
    if nom:
        data["nom"] = nom.group(1).strip()
    if prenom:
        data["prenom"] = prenom.group(1).strip()
    if tel:
        data["telephone"] = tel.group(1).strip()
    if mail:
        data["mail"] = mail.group(1).strip()
    return data

##############################
# Définition des assistants  #
##############################
# Assistant 1 : Extraction & recherche
assistant_prompt_instruction = """
Vous êtes Chat IA, expert en analyse de cartes de visite.
Votre tâche est d'extraire les informations suivantes du texte OCR fourni :
    - Nom
    - Prénom
    - Téléphone
    - Mail
Et de compléter ces informations par une recherche en ligne.
Répondez sous forme de texte structuré, par exemple :
Nom: Doe
Prénom: John
Téléphone: 0123456789
Mail: john.doe@example.com
Entreprise: Example Corp
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
                    "query": {"type": "string", "description": "Par exemple : 'John Doe, PDG de Example Corp'."}
                },
                "required": ["query"]
            }
        }
    }]
)
assistant_id = assistant.id

# Assistant 2 : Description des produits
product_assistant_instruction = """
Tu es un responsable commerciale.
Ta tâche est de réaliser en fonction des informations sur le client ainsi que des notes de l’utilisateur un matching entre nos produits et les besoins du client.

Voici la présentation de ce que Nin-IA propose : 

**Propulsez Votre Expertise en IA avec NIN-IA : Formations, Modules et Audits, la Triade du Succès !**

L'Intelligence Artificielle est la clé du futur, et NIN-IA vous offre la boîte à outils complète pour la maîtriser. Nos **formations de pointe** sont au cœur de notre offre, vous dotant des compétences essentielles. Pour une flexibilité maximale et des besoins spécifiques, découvrez nos **modules IA à la carte**. Et pour assurer le succès de vos projets, nos **audits IA experts** sont votre filet de sécurité.

**Notre priorité : Votre montée en compétences grâce à nos formations !**

- **Formations de Pointe : Devenez un Expert en IA Générative** : Nos formations vous plongent au cœur des algorithmes et des outils d'IA les plus performants. Adaptées à tous les niveaux, elles vous permettent de créer du contenu innovant, d'optimiser vos processus et de surpasser vos concurrents. **Ne vous contentez pas de suivre la vague, surfez sur elle !**
- **Modules IA : Apprentissage Personnalisé, Impact Immédiat** : Pour compléter votre formation ou répondre à des besoins précis, explorez nos modules IA à la carte. Concentrés sur des compétences spécifiques, ils vous offrent un apprentissage ciblé et une mise en œuvre rapide. **La flexibilité au service de votre expertise !**
- **Audits IA : Sécurisez Votre Investissement, Maximisez Votre ROI** : Avant d'investir massivement dans l'IA, assurez-vous que votre stratégie est solide. Nos audits IA identifient les points faibles de votre projet, optimisent vos ressources et évitent les erreurs coûteuses. **L'assurance d'un succès durable !**

**Détails de Notre Offre :**

- **Formations Structurées :**
    - **IA Générative 101 : Les Fondamentaux (Débutant) :** Apprenez les bases et explorez les premières applications concrètes.
    - **Création de Contenu Révolutionnaire avec ChatGPT (Intermédiaire) :** Maîtrisez ChatGPT pour générer des textes percutants.
    - **Deep Learning pour l'IA Générative : Devenez un Expert (Avancé) :** Plongez au cœur des réseaux neuronaux et débloquez le plein potentiel de l'IA.
    - **IA Générative pour le Marketing Digital (Spécial Marketing) :** Multipliez vos leads et convertissez vos prospects grâce à l'IA.
    - **Intégration de l'IA Générative dans Votre Entreprise (Spécial Entreprise) :** Intégrez l'IA dans vos processus et créez de nouvelles opportunités.
- **Modules IA à la Carte (Nouveauté !) :**
    - **[Exemple] : "Module : Optimisation des Prompts pour ChatGPT" :** Maîtrisez l'art de formuler des requêtes efficaces pour obtenir des résultats exceptionnels avec ChatGPT. **Transformez vos instructions en or !**
    - **[Exemple] : "Module : Analyse de Sentiments avec l'IA" :** Comprenez les émotions de vos clients et adaptez votre communication en conséquence. **Transformez les données en insights précieux !**
    - **[Exemple] : "Module : Génération d'Images avec Stable Diffusion" :** Créez des visuels époustouflants en quelques clics grâce à la puissance de l'IA. **Donnez vie à vos idées les plus folles !**
- **Audits IA Experts :**
    - Analyse approfondie de votre projet IA.
    - Identification des risques et des opportunités.
    - Recommandations personnalisées pour optimiser votre ROI.
    - Garantie de conformité réglementaire.

**Pourquoi choisir NIN-IA ?**

- **Expertise Reconnue :** Des formateurs passionnés et des experts en IA à votre service.
- **Approche Pédagogique Innovante :** Apprentissage pratique et mises en situation réelles.
- **Offre Complète :** Formations, modules et audits pour répondre à tous vos besoins.
- **Accompagnement Personnalisé :** Nous sommes à vos côtés à chaque étape de votre parcours.
"""
product_assistant = client_openai.beta.assistants.create(
    instructions=product_assistant_instruction,
    model="gpt-4o"
)
product_assistant_id = product_assistant.id

# Assistant 3 : Rédaction du mail
email_assistant_instruction = """
Tu es un expert en rédaction de mails de relance et assistant d’Emeline de Nin-IA.
Vos mails commencent toujours par "Bonjour [prénom]" et se terminent par "Cordialement Emeline Boulange, Co-dirigeante de Nin-IA.

TA tâche est de rédiger un mail de relance percutant pour convertir le lead, en tenant compte :

- des informations extraites (Assistant 1),
- du matching de notre offre (Assistant 2),
- de la qualification et des notes du lead.
Veillez à intégrer les notes de l'utilisateur pour instaurer une relation de proximité.
Et surtout bien mettre en place le contexte de la rencontre si cela est précisé 
Répondez sous forme d'un texte structuré (salutation, introduction, corps, conclusion).
"""
email_assistant = client_openai.beta.assistants.create(
    instructions=email_assistant_instruction,
    model="gpt-4o"
)
email_assistant_id = email_assistant.id

##############################
# Interface utilisateur      #
##############################
st.subheader("Capture / Upload de la carte de visite")

# Option de capture ou upload
image_file = st.camera_input("Prenez une photo des cartes de visite")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>OU</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Uploader la carte", type=["jpg", "jpeg", "png"])

qualification = st.selectbox("Qualification du lead", 
                               ["Smart Talk", "Mise en avant de la formation", "Mise en avant des audits", "Mise en avant des modules IA"])
note = st.text_area("Ajouter une note", placeholder="Entrez votre note ici...")

if note.strip() == "":
    st.error("Veuillez saisir une note avant de continuer.")
    st.stop()

# Récupération de l'image (capture ou upload)
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

# Bouton "Envoyer la note" visible en permanence
if st.button("Envoyer la note"):
    if image_data_uri is None:
        st.error("Aucune image n'a été fournie. Veuillez capturer ou uploader une photo de la carte.")
    else:
        try:
            # Extraction OCR via Mistral
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
                # Assistant 1 : Extraction & recherche
                ##################################################
                thread1 = client_openai.beta.threads.create()
                user_message_agent1 = (
                    f"Données extraites de la carte :\n"
                    f"Qualification : {qualification}\n"
                    f"Note : {note}\n"
                    f"Texte : {ocr_text}\n\n"
                    "Veuillez extraire les informations clés (Nom, Prénom, Téléphone, Mail) "
                    "et compléter par une recherche en ligne."
                )
                client_openai.beta.threads.messages.create(
                    thread_id=thread1.id, role="user", content=user_message_agent1
                )
                run1 = client_openai.beta.threads.runs.create(
                    thread_id=thread1.id, assistant_id=assistant_id
                )
                run1 = wait_for_run_completion(thread1.id, run1.id)
                if run1.status == 'requires_action':
                    run1 = submit_tool_outputs(thread1.id, run1.id, run1.required_action.submit_tool_outputs.tool_calls)
                    run1 = wait_for_run_completion(thread1.id, run1.id)
                response_agent1 = get_final_assistant_message(thread1.id)
                cleaned_response_agent1 = clean_response(response_agent1)
                st.subheader("Réponse agent 1 :")
                st.markdown(cleaned_response_agent1)
        
                # Extraction des champs via parsing
                parsed_data = parse_agent1_response(cleaned_response_agent1)
        
                ##################################################
                # Assistant 2 : Description des produits
                ##################################################
                thread2 = client_openai.beta.threads.create()
                user_message_agent2 = (
                    f"Informations sur l'entreprise extraites :\n{cleaned_response_agent1}\n\n"
                    f"Qualification : {qualification}\n"
                    f"Note : {note}\n\n"
                    "Veuillez rédiger un matching entre nos produits et les besoins du client, "
                    "en mettant en avant les avantages de nos offres."
                )
                client_openai.beta.threads.messages.create(
                    thread_id=thread2.id, role="user", content=user_message_agent2
                )
                run2 = client_openai.beta.threads.runs.create(
                    thread_id=thread2.id, assistant_id=product_assistant_id
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
                # Assistant 3 : Rédaction du mail
                ##################################################
                thread3 = client_openai.beta.threads.create()
                user_message_agent3 = (
                    f"Informations sur l'intervenant et son entreprise :\n{cleaned_response_agent1}\n\n"
                    f"Matching de notre offre :\n{cleaned_response_agent2}\n\n"
                    f"Qualification : {qualification}\n"
                    f"Note : {note}\n\n"
                    "Veuillez rédiger un mail de relance percutant pour convertir ce lead. "
                    "Le mail doit commencer par 'Bonjour [prénom]' et se terminer par 'Cordialement Emeline Boulange Co-dirigeante de Nin-IA'."
                )
                client_openai.beta.threads.messages.create(
                    thread_id=thread3.id, role="user", content=user_message_agent3
                )
                run3 = client_openai.beta.threads.runs.create(
                    thread_id=thread3.id, assistant_id=email_assistant_id
                )
                run3 = wait_for_run_completion(thread3.id, run3.id)
                if run3.status == 'requires_action':
                    run3 = submit_tool_outputs(thread3.id, run3.id, run3.required_action.submit_tool_outputs.tool_calls)
                    run3 = wait_for_run_completion(thread3.id, run3.id)
                response_agent3 = get_final_assistant_message(thread3.id)
                cleaned_response_agent3 = clean_response(response_agent3)
                st.subheader("Réponse agent 3 :")
                st.markdown(cleaned_response_agent3)
        
                ###########################################
                # Envoi automatique du lead dans la DB
                ###########################################
                cursor.execute(
                    "INSERT INTO leads (ocr_text, nom, prenom, telephone, mail, agent1, agent2, agent3, qualification, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ocr_text, parsed_data["nom"], parsed_data["prenom"], parsed_data["telephone"], parsed_data["mail"],
                     cleaned_response_agent1, cleaned_response_agent2, cleaned_response_agent3, qualification, note)
                )
                conn.commit()
                st.session_state["lead_sent"] = True
                st.success("Le lead a été envoyé automatiquement.")
        except Exception as e:
            st.error(f"Erreur lors du traitement OCR ou de l'analyse par les assistants : {e}")

