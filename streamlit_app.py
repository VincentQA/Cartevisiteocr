import streamlit as st
import os
import base64
from mistralai import Mistral

def extract_text_from_ocr_response(ocr_response):
    """
    Extrait le texte exploitable depuis la réponse OCR.
    Si la réponse possède un attribut 'pages' (liste d'OCRPageObject), on itère dessus.
    Pour chaque page, on supprime la ligne contenant l'image (commençant par "![") et on conserve le reste.
    """
    extracted_text = ""
    # Si l'objet a un attribut 'pages'
    if hasattr(ocr_response, "pages"):
        pages = ocr_response.pages
    elif isinstance(ocr_response, list):
        pages = ocr_response
    else:
        pages = []

    for page in pages:
        if hasattr(page, "markdown") and page.markdown:
            # Découper le markdown en lignes et filtrer celles qui commencent par "!["
            lines = page.markdown.split("\n")
            filtered_lines = [line.strip() for line in lines if not line.startswith("![")]
            if filtered_lines:
                extracted_text += "\n".join(filtered_lines) + "\n"
    return extracted_text.strip()

def extract_business_card_data_via_ai(ocr_text, client, model="mistral-small-latest"):
    """
    Envoie le texte OCR à l'agent pour extraire les informations de la carte de visite.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extrais les informations de la carte de visite suivantes. Donne-moi le nom, l'email, le téléphone et, si possible, le nom de l'entreprise."
                },
                {
                    "type": "text",
                    "text": ocr_text
                }
            ]
        }
    ]
    response = client.chat.complete(model=model, messages=messages)
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        return "Erreur dans la réponse de l'agent : " + str(e)

# Configuration de la page
st.set_page_config(page_title="Le charte visite 🐱", layout="centered")
st.title("Le charte visite 🐱")

# Capture de l'image via la caméra
image_file = st.camera_input("Prenez une photo des cartes de visite")

if image_file is not None:
    st.image(image_file, caption="Carte de visite capturée", use_column_width=True)
    
    # Encodage de l'image en base64 pour créer une data URI
    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"
    
    # Récupération de la clé API Mistral
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Clé API Mistral non trouvée. Veuillez définir MISTRAL_API_KEY dans vos variables d'environnement.")
    else:
        client = Mistral(api_key=api_key)
        try:
            # Appel à l'API OCR avec l'image encodée
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": image_data_uri
                }
            )
            st.subheader("Résultat brut de l'OCR :")
            st.write(ocr_response)
            
            # Extraction du texte exploitable depuis la réponse OCR
            ocr_text = extract_text_from_ocr_response(ocr_response)
            if not ocr_text:
                st.warning("Aucun texte exploitable extrait par l'OCR.")
            else:
                st.subheader("Texte OCR extrait :")
                st.text(ocr_text)
                
                st.subheader("Extraction initiale par l'IA :")
                initial_extraction = extract_business_card_data_via_ai(ocr_text, client)
                st.write(initial_extraction)
                
                # Mise en place de l'historique de la conversation dans st.session_state
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Contexte OCR extrait :\n\n{ocr_text}"
                                }
                            ]
                        }
                    ]
                
                st.subheader("Discussion avec l'agent")
                # Affichage de l'historique du chat
                for msg in st.session_state.chat_history:
                    role = msg.get("role", "inconnu")
                    text_content = "".join(c.get("text", "") for c in msg.get("content", []))
                    if role == "user":
                        st.markdown(f"**Vous :** {text_content}")
                    elif role == "assistant":
                        st.markdown(f"**Agent :** {text_content}")
                    elif role == "system":
                        st.markdown(f"**Contexte :** {text_content}")
                
                # Zone de saisie pour la question de l'utilisateur
                user_input = st.text_input("Votre question pour l'agent", key="user_input")
                if st.button("Envoyer") and user_input:
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": [{"type": "text", "text": user_input}]
                    })
                    
                    with st.spinner("L'agent réfléchit..."):
                        response = client.chat.complete(
                            model="mistral-small-latest",
                            messages=st.session_state.chat_history
                        )
                        try:
                            agent_reply = response.choices[0].message.content
                        except (AttributeError, IndexError) as e:
                            agent_reply = "Erreur lors de la récupération de la réponse de l'agent : " + str(e)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": agent_reply}]
                        })
                    
                    st.experimental_rerun()
        except Exception as e:
            st.error(f"Erreur lors du traitement OCR ou de l'analyse par l'IA : {e}")
