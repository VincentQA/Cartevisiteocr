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

client = Mistral(api_key=MISTRAL_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

##############################################
# Configuration de l'assistant avec function calling
##############################################

assistant_instructions = """
Vous êtes Chat IA, un assistant expert en analyse de cartes de visite.
Votre tâche est la suivante:
1. Extraire le nom, le prénom et le nom de l'entreprise à partir du texte OCR fourni.
2. Utiliser la fonction tavily_search pour effectuer une recherche en ligne et obtenir un maximum d'informations sur l'intervenant et son entreprise.
Répondez uniquement sous forme d'un objet JSON avec les clés "nom", "prenom", "entreprise" et "infos_en_ligne".
"""

tavily_function = {
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

assistant = client.assistants.create(
    instructions=assistant_instructions,
    model="mistral-small-latest",
    functions=[tavily_function]
)
assistant_id = assistant.id

##############################################
# Fonctions utilitaires
##############################################

def tavily_search(query):
    return tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)

def wait_for_run(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status in ["completed", "failed", "requires_action"]:
            return run

def submit_tool_outputs(thread_id, run_id, tool_calls):
    outputs = []
    for call in tool_calls:
        call_id = call.id
        func_name = call.function.name
        args = json.loads(call.function.arguments)
        if func_name == "tavily_search":
            result = tavily_search(args["query"])
            outputs.append({"tool_call_id": call_id, "output": result})
    return client.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=outputs
    )

def get_last_message(thread_id):
    messages = client.messages.list(thread_id=thread_id)
    assistant_msgs = [m for m in messages if m.role == "assistant"]
    if assistant_msgs:
        last = assistant_msgs[-1]
        text = ""
        for c in last.content:
            if isinstance(c, dict):
                text += c.get("text", "")
            else:
                text += str(c)
        return text.strip()
    return None

def extract_ocr_text(ocr_response):
    text = ""
    pages = ocr_response.pages if hasattr(ocr_response, "pages") else ocr_response if isinstance(ocr_response, list) else []
    for p in pages:
        if hasattr(p, "markdown") and p.markdown:
            lines = p.markdown.split("\n")
            filtered = [l.strip() for l in lines if not l.startswith("![")]
            if filtered:
                text += "\n".join(filtered) + "\n"
    return text.strip()

def format_output(raw):
    prefix = "json\n"
    cleaned = raw.strip()
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

# Affichage immédiat de la capture et des inputs supplémentaires
image_file = st.camera_input("Prenez une photo des cartes de visite")

st.markdown("### Sujets abordés")
s1 = st.checkbox("Rencontre courtoise mais potentiel de faire plus ensemble")
s2 = st.checkbox("Relancer sur le programme incubation collective")
s3 = st.checkbox("Relancer sur le programme d'incubation individuelle")
s4 = st.checkbox("Mettre en relation avec la transformation numérique car on ne peut pas l'accompagner")
comments = st.text_area("Commentaires additionnels", placeholder="Ajouter des commentaires ici...")

if st.button("Lancer l'analyse"):
    if image_file is None:
        st.error("Veuillez prendre une photo.")
    else:
        st.image(image_file, caption="Carte de visite capturée", use_column_width=True)
        img_bytes = image_file.getvalue()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        uri = f"data:image/jpeg;base64,{b64}"
        try:
            # Appel OCR
            ocr_resp = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "image_url", "image_url": uri}
            )
            st.subheader("OCR brut:")
            st.write(ocr_resp)
            ocr_text = extract_ocr_text(ocr_resp)
            if not ocr_text:
                st.warning("Aucun texte extrait.")
            else:
                st.subheader("Texte OCR:")
                st.text(ocr_text)
                # Création du thread et envoi du message à l'assistant
                thread = client.threads.create()
                msg = (
                    f"Voici le texte OCR extrait :\n{ocr_text}\n\n"
                    "Extrais les informations suivantes : nom, prenom, entreprise. "
                    "Ensuite, effectue une recherche en ligne pour obtenir un maximum d'informations sur l'intervenant et son entreprise."
                )
                client.messages.create(thread_id=thread.id, role="user", content=msg)
                run = client.runs.create(thread_id=thread.id, assistant_id=assistant_id)
                run = wait_for_run(thread.id, run.id)
                if run.status == "failed":
                    st.error(run.error)
                elif run.status == "requires_action":
                    run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
                    run = wait_for_run(thread.id, run.id)
                raw_msg = get_last_message(thread.id)
                st.subheader("Résultat assistant:")
                if raw_msg:
                    parsed = format_output(raw_msg)
                    if parsed:
                        st.json(parsed)
                    else:
                        st.text("Erreur de parsing, voici le message brut:")
                        st.text(raw_msg)
                else:
                    st.warning("Pas de message de l'assistant.")
                
                # Affichage des inputs utilisateurs supplémentaires
                st.markdown("---")
                st.subheader("Sujets et commentaires saisis")
                selected = []
                if s1: selected.append("Rencontre courtoise mais potentiel de faire plus ensemble")
                if s2: selected.append("Relancer sur le programme incubation collective")
                if s3: selected.append("Relancer sur le programme d'incubation individuelle")
                if s4: selected.append("Mettre en relation avec la transformation numérique car on ne peut pas l'accompagner")
                st.write("Sujets sélectionnés:", selected)
                st.write("Commentaires:", comments)
        except Exception as e:
            st.error(f"Erreur lors du traitement OCR ou de l'analyse: {e}")
